//! The paged KV cache: a block pool, block tables, a prefix cache,
//! copy-on-write and a host swap pool.
//!
//! **Blocks.** The device pool has a fixed number of blocks of `block_size`
//! tokens. Under tensor parallelism each rank stores 1/P of the KV heads of
//! the same tokens, so one logical block table serves every rank. Free
//! blocks sit on a LIFO free list (allocation pops the most recently freed
//! block; initially block 0 first), and every block has a reference count:
//! the number of block-table entries that point at it.
//!
//! **Block tables.** A sequence (one sample of one request) has an ordered
//! list of blocks and a token count; only its last block may be partly
//! full. Every block stores the token IDs it holds, so invariant checks can
//! compare a block's contents with what each sequence that references it
//! expects.
//!
//! **Prefix cache.** Each full block has a chain key: a hash of the previous
//! full block's key and this block's tokens. With prefix caching on, a newly
//! full block is registered under its key (unless another block already
//! holds that key). A new sequence adopts matching registered blocks for its
//! leading full blocks, leaving at least one prompt token to compute; a hit
//! is verified against the stored parent key and tokens, so a hash collision
//! is a miss, never a wrong block. A registered block whose reference count
//! drops to zero is **retained**: it stays reusable until the allocator needs
//! it, and then the least recently released one is evicted first, ties
//! broken by lower block ID. The free list is always used before evicting.
//!
//! **Copy-on-write.** [`KvCache::fork`] makes a child sequence share every
//! block of its parent. Appending to a sequence whose partly filled last
//! block is shared first copies that block into a new private block. Full
//! blocks are never written again, so they never need copying.
//!
//! **Swap.** [`KvCache::swap_out`] copies every block of a sequence into
//! private host swap blocks (a shared block gets its own copy) and releases
//! the device blocks; [`KvCache::swap_in`] copies them back into fresh
//! private device blocks, which are not re-registered in the prefix cache.
//!
//! **Atomicity.** Every operation checks capacity first and either completes
//! or returns an error with nothing changed.
//!
//! [`KvCache::check`] verifies every invariant (see [`Invariant`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::rng::SplitMix64;

pub type BlockId = u32;

/// One sample of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId {
    pub request: u32,
    pub sample: u32,
}

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}s{}", self.request, self.sample)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvConfig {
    /// Tokens per block (B).
    pub block_size: usize,
    /// Device blocks (the logical pool every rank mirrors).
    pub device_blocks: usize,
    /// Host swap blocks.
    pub swap_blocks: usize,
    pub prefix_caching: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvStats {
    /// Device blocks handed out (from the free list or by eviction).
    pub allocations: u64,
    pub cow_copies: u64,
    /// Retained prefix blocks evicted to satisfy an allocation.
    pub evictions: u64,
    /// Full prompt blocks a new sequence could have adopted.
    pub prefix_eligible_blocks: u64,
    /// Of those, blocks actually adopted from the prefix cache.
    pub prefix_hit_blocks: u64,
    pub swapped_out_blocks: u64,
    pub swapped_in_blocks: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
    /// Tokens held (the last block holds `len − (blocks − 1)·B` of them).
    pub len: usize,
    /// Chain key of each full block, in order.
    pub keys: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwappedTable {
    /// Host swap blocks, in sequence order.
    pub blocks: Vec<u32>,
    pub len: usize,
    pub keys: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppendReport {
    /// New device blocks taken (including a copy-on-write copy).
    pub new_blocks: usize,
    /// Whether the shared last block was copied first.
    pub cow: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvError {
    UnknownSeq(SeqId),
    SeqExists(SeqId),
    /// Prefix adoption needs a sequence with no tokens yet.
    NotEmpty(SeqId),
    OutOfBlocks {
        needed: usize,
        available: usize,
    },
    SwapFull {
        needed: usize,
        available: usize,
    },
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvError::UnknownSeq(s) => write!(f, "no sequence {s} in this pool"),
            KvError::SeqExists(s) => write!(f, "sequence {s} already exists"),
            KvError::NotEmpty(s) => write!(f, "sequence {s} already holds tokens"),
            KvError::OutOfBlocks { needed, available } => {
                write!(f, "needs {needed} device blocks, {available} available")
            }
            KvError::SwapFull { needed, available } => {
                write!(f, "needs {needed} swap blocks, {available} available")
            }
        }
    }
}

impl std::error::Error for KvError {}

/// The invariant a [`Violation`] breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Invariant {
    /// Free, retained and in-use blocks partition the device pool exactly
    /// (no duplicates, nothing lost, nothing beyond capacity); a sequence is
    /// on the device or in swap, not both.
    Conservation,
    /// A block's reference count equals the block-table entries pointing at it.
    Refcount,
    /// A table has ceil(len ÷ B) blocks and floor(len ÷ B) chain keys.
    Shape,
    /// Every block holds exactly the tokens each referencing sequence expects
    /// at those positions (length included).
    Content,
    /// Registered keys, block keys and table keys agree with recomputed chain hashes.
    PrefixCache,
    /// The swap pool is partitioned like the device pool, and swapped
    /// sequences have the right shape and contents.
    Swap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub invariant: Invariant,
    pub detail: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.invariant, self.detail)
    }
}

/// Deliberate bugs that tests plant to show the invariant checks catch them.
/// Never set outside tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    /// `free` skips releasing a sequence's last block (a leak on preemption).
    LeakOnFree,
    /// `append` writes into a shared partly filled block instead of copying it.
    SkipCow,
    /// `swap_out` keeps the device reference of blocks that are shared.
    SwapOutKeepsSharedRef,
    /// A block is treated as full one token early when appending.
    EarlyBlockFull,
    /// Evicting a retained block leaves its key in the prefix cache.
    StalePrefixKey,
}

const ROOT_KEY: u64 = 0x9b1d_c0de_0000_0001;

/// Chain key of a full block from its parent's key and its tokens: a
/// multiply–rotate hash over the tokens, then the SplitMix64 finalizer.
/// A hit is always verified against the stored parent key and tokens, so a
/// collision can only cause a miss.
pub fn chain_key(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = parent ^ 0x6b65_7973_6b65_7973;
    for &t in tokens {
        h = (h ^ u64::from(t)).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
    }
    SplitMix64::new(h).next_u64()
}

pub struct KvCache {
    config: KvConfig,
    free: Vec<BlockId>,
    refcount: Vec<u32>,
    content: Vec<Vec<u32>>,
    /// Key a block is registered under in the prefix cache.
    cached_key: Vec<Option<u64>>,
    /// Parent chain key of a registered block (to verify hits).
    cached_parent: Vec<u64>,
    by_key: BTreeMap<u64, BlockId>,
    /// Retained registered blocks with no references: (released at, block).
    lru: BTreeSet<(u64, BlockId)>,
    released_at: Vec<u64>,
    /// Logical clock: one tick per operation that can release blocks.
    tick: u64,
    tables: BTreeMap<SeqId, BlockTable>,
    swap_free: Vec<u32>,
    swap_content: Vec<Vec<u32>>,
    swapped: BTreeMap<SeqId, SwappedTable>,
    stats: KvStats,
    mutation: Option<Mutation>,
}

impl KvCache {
    pub fn new(config: KvConfig) -> KvCache {
        assert!(config.block_size > 0, "block size must be at least 1");
        assert!(config.device_blocks <= u32::MAX as usize && config.swap_blocks <= u32::MAX as usize);
        let n = config.device_blocks;
        KvCache {
            config,
            free: (0..n as BlockId).rev().collect(),
            refcount: vec![0; n],
            content: vec![Vec::new(); n],
            cached_key: vec![None; n],
            cached_parent: vec![0; n],
            by_key: BTreeMap::new(),
            lru: BTreeSet::new(),
            released_at: vec![0; n],
            tick: 0,
            tables: BTreeMap::new(),
            swap_free: (0..config.swap_blocks as u32).rev().collect(),
            swap_content: vec![Vec::new(); config.swap_blocks],
            swapped: BTreeMap::new(),
            stats: KvStats::default(),
            mutation: None,
        }
    }

    #[doc(hidden)]
    pub fn plant(&mut self, mutation: Mutation) {
        self.mutation = Some(mutation);
    }

    fn planted(&self, m: Mutation) -> bool {
        self.mutation == Some(m)
    }

    pub fn config(&self) -> &KvConfig {
        &self.config
    }

    pub fn stats(&self) -> &KvStats {
        &self.stats
    }

    pub fn table(&self, seq: SeqId) -> Option<&BlockTable> {
        self.tables.get(&seq)
    }

    pub fn swapped_table(&self, seq: SeqId) -> Option<&SwappedTable> {
        self.swapped.get(&seq)
    }

    /// Sequences on the device, in order.
    pub fn sequences(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.tables.keys().copied()
    }

    /// Sequences in host swap, in order.
    pub fn swapped_sequences(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.swapped.keys().copied()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Retained prefix blocks with no references (reusable or evictable).
    pub fn retained_blocks(&self) -> usize {
        self.lru.len()
    }

    /// Blocks an allocation can obtain: free plus retained.
    pub fn available(&self) -> usize {
        self.free.len() + self.lru.len()
    }

    /// Blocks referenced by at least one sequence.
    pub fn used_blocks(&self) -> usize {
        self.config.device_blocks - self.available()
    }

    pub fn swap_free_blocks(&self) -> usize {
        self.swap_free.len()
    }

    pub fn refcount(&self, block: BlockId) -> u32 {
        self.refcount[block as usize]
    }

    pub fn block_tokens(&self, block: BlockId) -> &[u32] {
        &self.content[block as usize]
    }

    /// Tokens a block holds before it counts as full (the planted
    /// `EarlyBlockFull` bug lowers it by one).
    fn full_at(&self) -> usize {
        if self.planted(Mutation::EarlyBlockFull) && self.config.block_size > 1 {
            self.config.block_size - 1
        } else {
            self.config.block_size
        }
    }

    fn copies_on_append(&self, table: &BlockTable) -> bool {
        match table.blocks.last() {
            Some(&last) => {
                let fill = self.content[last as usize].len();
                fill < self.full_at() && self.refcount[last as usize] > 1 && !self.planted(Mutation::SkipCow)
            }
            None => false,
        }
    }

    /// Device blocks appending `n` tokens to `seq` would take, including a
    /// copy-on-write copy (`None` for an unknown sequence).
    pub fn append_cost(&self, seq: SeqId, n: usize) -> Option<usize> {
        let table = self.tables.get(&seq)?;
        let full_at = self.full_at();
        let mut fill = table.blocks.last().map_or(full_at, |&b| self.content[b as usize].len());
        let mut blocks = usize::from(n > 0 && self.copies_on_append(table));
        for _ in 0..n {
            if fill >= full_at {
                blocks += 1;
                fill = 0;
            }
            fill += 1;
        }
        Some(blocks)
    }

    /// Device blocks appending `n_i` tokens to each `seq_i`, in order, would
    /// take in total. Exact when several sequences share a partly filled last
    /// block: the last of them to append owns it and does not copy.
    pub fn append_cost_batch(&self, work: &[(SeqId, usize)]) -> Option<usize> {
        let mut drops: BTreeMap<BlockId, u32> = BTreeMap::new();
        let mut total = 0;
        for &(seq, n) in work {
            let table = self.tables.get(&seq)?;
            let mut cost = self.append_cost(seq, n)?;
            if let Some(&last) = table.blocks.last() {
                let shared = self.refcount[last as usize] - drops.get(&last).copied().unwrap_or(0);
                let copies = n > 0 && self.copies_on_append(table);
                if copies && shared <= 1 {
                    // Earlier sequences in this batch already copied away from it.
                    cost -= 1;
                } else if copies {
                    *drops.entry(last).or_default() += 1;
                }
            }
            total += cost;
        }
        Some(total)
    }

    /// The registered blocks [`Self::adopt_prefix`] would adopt for `tokens`.
    fn match_prefix(&self, tokens: &[u32]) -> Vec<(BlockId, u64)> {
        let mut found = Vec::new();
        if !self.config.prefix_caching {
            return found;
        }
        let b = self.config.block_size;
        let eligible = tokens.len().saturating_sub(1) / b;
        let mut parent = ROOT_KEY;
        for chunk in tokens.chunks_exact(b).take(eligible) {
            let key = chain_key(parent, chunk);
            let Some(&block) = self.by_key.get(&key) else { break };
            let i = block as usize;
            if self.cached_key[i] != Some(key) || self.cached_parent[i] != parent || self.content[i] != chunk {
                break;
            }
            found.push((block, key));
            parent = key;
        }
        found
    }

    /// What adopting a prefix of `tokens` would do, without doing it:
    /// (tokens adopted, retained blocks it would take out of the reusable pool).
    pub fn peek_prefix(&self, tokens: &[u32]) -> (usize, usize) {
        let found = self.match_prefix(tokens);
        let retained = found.iter().filter(|(b, _)| self.refcount[*b as usize] == 0).count();
        (found.len() * self.config.block_size, retained)
    }

    /// Take a block: from the free list, else evict the least recently
    /// released retained block (ties: lower ID). The caller checked availability.
    fn take_block(&mut self) -> BlockId {
        self.stats.allocations += 1;
        if let Some(b) = self.free.pop() {
            return b;
        }
        let (at, b) = *self.lru.iter().next().expect("caller checked availability");
        self.lru.remove(&(at, b));
        let key = self.cached_key[b as usize]
            .take()
            .expect("retained blocks are registered");
        if !self.planted(Mutation::StalePrefixKey) {
            self.by_key.remove(&key);
        }
        self.content[b as usize].clear();
        self.stats.evictions += 1;
        b
    }

    fn release(&mut self, b: BlockId) {
        let i = b as usize;
        assert!(self.refcount[i] > 0, "release of unreferenced block {b}");
        self.refcount[i] -= 1;
        if self.refcount[i] == 0 {
            if self.cached_key[i].is_some() {
                self.released_at[i] = self.tick;
                self.lru.insert((self.tick, b));
            } else {
                self.content[i].clear();
                self.free.push(b);
            }
        }
    }

    fn exists(&self, seq: SeqId) -> bool {
        self.tables.contains_key(&seq) || self.swapped.contains_key(&seq)
    }

    /// Create an empty sequence on the device.
    pub fn start(&mut self, seq: SeqId) -> Result<(), KvError> {
        if self.exists(seq) {
            return Err(KvError::SeqExists(seq));
        }
        self.tables.insert(
            seq,
            BlockTable {
                blocks: Vec::new(),
                len: 0,
                keys: Vec::new(),
            },
        );
        Ok(())
    }

    /// For an empty sequence whose content will begin with `tokens`, adopt
    /// registered blocks for its leading full blocks, leaving at least one
    /// token to compute. Returns the tokens adopted (a multiple of B; 0 when
    /// prefix caching is off).
    pub fn adopt_prefix(&mut self, seq: SeqId, tokens: &[u32]) -> Result<usize, KvError> {
        let table = self.tables.get(&seq).ok_or(KvError::UnknownSeq(seq))?;
        if table.len > 0 {
            return Err(KvError::NotEmpty(seq));
        }
        if !self.config.prefix_caching {
            return Ok(0);
        }
        let b = self.config.block_size;
        self.stats.prefix_eligible_blocks += (tokens.len().saturating_sub(1) / b) as u64;
        let mut adopted = BlockTable {
            blocks: Vec::new(),
            len: 0,
            keys: Vec::new(),
        };
        for (block, key) in self.match_prefix(tokens) {
            let i = block as usize;
            if self.refcount[i] == 0 {
                self.lru.remove(&(self.released_at[i], block));
            }
            self.refcount[i] += 1;
            adopted.blocks.push(block);
            adopted.keys.push(key);
            adopted.len += b;
        }
        let n = adopted.len;
        self.stats.prefix_hit_blocks += adopted.blocks.len() as u64;
        self.tables.insert(seq, adopted);
        Ok(n)
    }

    /// Append `tokens` to `seq`, taking blocks as needed (copying a shared
    /// last block first). Fails with nothing changed if blocks run out.
    /// Appending nothing changes nothing.
    pub fn append(&mut self, seq: SeqId, tokens: &[u32]) -> Result<AppendReport, KvError> {
        let needed = self.append_cost(seq, tokens.len()).ok_or(KvError::UnknownSeq(seq))?;
        if needed > self.available() {
            return Err(KvError::OutOfBlocks {
                needed,
                available: self.available(),
            });
        }
        let mut table = self.tables.remove(&seq).expect("checked above");
        let cow = !tokens.is_empty() && self.copies_on_append(&table);
        if cow {
            let last = table.blocks.len() - 1;
            let old = table.blocks[last];
            let copy = self.take_block();
            self.content[copy as usize] = self.content[old as usize].clone();
            self.refcount[copy as usize] = 1;
            self.release(old);
            table.blocks[last] = copy;
            self.stats.cow_copies += 1;
        }
        let (b, full_at) = (self.config.block_size, self.full_at());
        for &t in tokens {
            let start_new = table
                .blocks
                .last()
                .map_or(true, |&last| self.content[last as usize].len() >= full_at);
            if start_new {
                let block = self.take_block();
                self.refcount[block as usize] = 1;
                table.blocks.push(block);
            }
            let last = *table.blocks.last().expect("a block was just ensured") as usize;
            self.content[last].push(t);
            table.len += 1;
            if self.content[last].len() == b {
                let parent = table.keys.last().copied().unwrap_or(ROOT_KEY);
                let key = chain_key(parent, &self.content[last]);
                table.keys.push(key);
                if self.config.prefix_caching && !self.by_key.contains_key(&key) {
                    self.by_key.insert(key, last as BlockId);
                    self.cached_key[last] = Some(key);
                    self.cached_parent[last] = parent;
                }
            }
        }
        self.tables.insert(seq, table);
        Ok(AppendReport {
            new_blocks: needed,
            cow,
        })
    }

    /// Make `child` share every block of `parent`.
    pub fn fork(&mut self, parent: SeqId, child: SeqId) -> Result<(), KvError> {
        if self.exists(child) {
            return Err(KvError::SeqExists(child));
        }
        let table = self.tables.get(&parent).ok_or(KvError::UnknownSeq(parent))?.clone();
        for &b in &table.blocks {
            self.refcount[b as usize] += 1;
        }
        self.tables.insert(child, table);
        Ok(())
    }

    /// Release every block of `seq` and forget it.
    pub fn free(&mut self, seq: SeqId) -> Result<(), KvError> {
        let table = self.tables.remove(&seq).ok_or(KvError::UnknownSeq(seq))?;
        self.tick += 1;
        let last = table.blocks.len().saturating_sub(1);
        for (i, &b) in table.blocks.iter().enumerate() {
            if i == last && self.planted(Mutation::LeakOnFree) {
                continue;
            }
            self.release(b);
        }
        Ok(())
    }

    /// Copy `seq` into private swap blocks and release its device blocks.
    /// Returns the blocks copied.
    pub fn swap_out(&mut self, seq: SeqId) -> Result<usize, KvError> {
        let table = self.tables.get(&seq).ok_or(KvError::UnknownSeq(seq))?;
        let n = table.blocks.len();
        if n > self.swap_free.len() {
            return Err(KvError::SwapFull {
                needed: n,
                available: self.swap_free.len(),
            });
        }
        let table = self.tables.remove(&seq).expect("checked above");
        self.tick += 1;
        let mut host = Vec::with_capacity(n);
        for &b in &table.blocks {
            let s = self.swap_free.pop().expect("checked above");
            self.swap_content[s as usize] = self.content[b as usize].clone();
            host.push(s);
            if self.planted(Mutation::SwapOutKeepsSharedRef) && self.refcount[b as usize] > 1 {
                continue;
            }
            self.release(b);
        }
        self.stats.swapped_out_blocks += n as u64;
        self.swapped.insert(
            seq,
            SwappedTable {
                blocks: host,
                len: table.len,
                keys: table.keys,
            },
        );
        Ok(n)
    }

    /// Copy `seq` back into fresh private device blocks. Returns the blocks copied.
    pub fn swap_in(&mut self, seq: SeqId) -> Result<usize, KvError> {
        let host = self.swapped.get(&seq).ok_or(KvError::UnknownSeq(seq))?;
        let n = host.blocks.len();
        if n > self.available() {
            return Err(KvError::OutOfBlocks {
                needed: n,
                available: self.available(),
            });
        }
        let host = self.swapped.remove(&seq).expect("checked above");
        let mut blocks = Vec::with_capacity(n);
        for &s in &host.blocks {
            let b = self.take_block();
            self.content[b as usize] = std::mem::take(&mut self.swap_content[s as usize]);
            self.refcount[b as usize] = 1;
            self.swap_free.push(s);
            blocks.push(b);
        }
        self.stats.swapped_in_blocks += n as u64;
        self.tables.insert(
            seq,
            BlockTable {
                blocks,
                len: host.len,
                keys: host.keys,
            },
        );
        Ok(n)
    }

    /// Forget a swapped sequence and free its swap blocks.
    pub fn drop_swapped(&mut self, seq: SeqId) -> Result<(), KvError> {
        let host = self.swapped.remove(&seq).ok_or(KvError::UnknownSeq(seq))?;
        for s in host.blocks {
            self.swap_content[s as usize].clear();
            self.swap_free.push(s);
        }
        Ok(())
    }

    /// Lose every device block (a rank failure invalidates all of them):
    /// every device sequence is dropped and the prefix cache emptied. The
    /// swap pool survives. Returns the dropped sequences, in order.
    pub fn reset_device(&mut self) -> Vec<SeqId> {
        let dropped: Vec<SeqId> = self.tables.keys().copied().collect();
        let n = self.config.device_blocks;
        self.tables.clear();
        self.free = (0..n as BlockId).rev().collect();
        self.refcount.iter_mut().for_each(|r| *r = 0);
        self.content.iter_mut().for_each(Vec::clear);
        self.cached_key.iter_mut().for_each(|k| *k = None);
        self.by_key.clear();
        self.lru.clear();
        self.tick += 1;
        dropped
    }

    /// Check every invariant. `expected(seq, position)` is the token the
    /// sequence must hold at that position (device and swapped sequences).
    pub fn check(&self, expected: &dyn Fn(SeqId, usize) -> u32) -> Result<(), Violation> {
        let fail = |invariant, detail: String| Err(Violation { invariant, detail });
        let (b, n) = (self.config.block_size, self.config.device_blocks);
        if self.refcount.len() != n || self.content.len() != n {
            return fail(Invariant::Conservation, "pool size changed".into());
        }

        // Conservation: free, retained and referenced blocks partition the pool.
        #[derive(Clone, Copy, PartialEq)]
        enum State {
            Unseen,
            Free,
            Retained,
            InUse,
        }
        let mut state = vec![State::Unseen; n];
        for &blk in &self.free {
            let i = blk as usize;
            if i >= n || state[i] != State::Unseen {
                return fail(
                    Invariant::Conservation,
                    format!("block {blk} listed twice or out of range"),
                );
            }
            state[i] = State::Free;
            if self.refcount[i] != 0 || self.cached_key[i].is_some() || !self.content[i].is_empty() {
                return fail(Invariant::Conservation, format!("free block {blk} is still in use"));
            }
        }
        for &(at, blk) in &self.lru {
            let i = blk as usize;
            if i >= n || state[i] != State::Unseen {
                return fail(Invariant::Conservation, format!("retained block {blk} listed twice"));
            }
            state[i] = State::Retained;
            if self.refcount[i] != 0 || self.cached_key[i].is_none() || self.released_at[i] != at {
                return fail(Invariant::Conservation, format!("retained block {blk} is inconsistent"));
            }
        }
        for (i, s) in state.iter_mut().enumerate() {
            if self.refcount[i] > 0 {
                if *s != State::Unseen {
                    return fail(Invariant::Conservation, format!("block {i} is referenced and free"));
                }
                *s = State::InUse;
            }
        }
        if let Some(i) = state.iter().position(|s| *s == State::Unseen) {
            return fail(Invariant::Conservation, format!("block {i} is in no state (lost)"));
        }
        if let Some(seq) = self.tables.keys().find(|s| self.swapped.contains_key(s)) {
            return fail(
                Invariant::Conservation,
                format!("{seq} is both on the device and swapped"),
            );
        }

        // Refcounts.
        let mut refs = vec![0u32; n];
        for table in self.tables.values() {
            for &blk in &table.blocks {
                refs[blk as usize] += 1;
            }
        }
        if let Some(i) = (0..n).find(|&i| refs[i] != self.refcount[i]) {
            return fail(
                Invariant::Refcount,
                format!("block {i}: refcount {} but {} references", self.refcount[i], refs[i]),
            );
        }

        // Shape.
        for (seq, table) in &self.tables {
            if table.blocks.len() != table.len.div_ceil(b) || table.keys.len() != table.len / b {
                return fail(
                    Invariant::Shape,
                    format!(
                        "{seq}: {} tokens in {} blocks with {} keys",
                        table.len,
                        table.blocks.len(),
                        table.keys.len()
                    ),
                );
            }
        }

        // Content.
        for (seq, table) in &self.tables {
            for (k, &blk) in table.blocks.iter().enumerate() {
                let start = k * b;
                let want: Vec<u32> = (start..table.len.min(start + b)).map(|p| expected(*seq, p)).collect();
                if self.content[blk as usize] != want {
                    return fail(
                        Invariant::Content,
                        format!("{seq} block {k} (#{blk}) does not hold the expected tokens"),
                    );
                }
            }
        }

        // Prefix cache.
        for (&key, &blk) in &self.by_key {
            let i = blk as usize;
            if self.cached_key[i] != Some(key)
                || self.content[i].len() != b
                || chain_key(self.cached_parent[i], &self.content[i]) != key
            {
                return fail(Invariant::PrefixCache, format!("key {key:#x} → block {blk} is stale"));
            }
        }
        for (i, key) in self.cached_key.iter().enumerate() {
            if let Some(key) = key {
                if self.by_key.get(key) != Some(&(i as BlockId)) {
                    return fail(Invariant::PrefixCache, format!("block {i} registered but not indexed"));
                }
            }
        }
        for (seq, table) in &self.tables {
            let mut parent = ROOT_KEY;
            for (k, &key) in table.keys.iter().enumerate() {
                parent = chain_key(parent, &self.content[table.blocks[k] as usize]);
                if parent != key {
                    return fail(Invariant::PrefixCache, format!("{seq}: chain key {k} is wrong"));
                }
            }
        }

        // Swap pool.
        let mut swap_seen = vec![false; self.config.swap_blocks];
        let mut mark = |s: u32| -> bool {
            let i = s as usize;
            if i >= swap_seen.len() || swap_seen[i] {
                return false;
            }
            swap_seen[i] = true;
            true
        };
        if !self.swap_free.iter().all(|&s| mark(s)) {
            return fail(Invariant::Swap, "swap block listed twice".into());
        }
        for (seq, host) in &self.swapped {
            if !host.blocks.iter().all(|&s| mark(s)) {
                return fail(Invariant::Swap, format!("{seq}: swap block shared or free"));
            }
            if host.blocks.len() != host.len.div_ceil(b) || host.keys.len() != host.len / b {
                return fail(Invariant::Swap, format!("{seq}: swapped table has the wrong shape"));
            }
            for (k, &s) in host.blocks.iter().enumerate() {
                let start = k * b;
                let want: Vec<u32> = (start..host.len.min(start + b)).map(|p| expected(*seq, p)).collect();
                if self.swap_content[s as usize] != want {
                    return fail(
                        Invariant::Swap,
                        format!("{seq}: swapped block {k} has the wrong tokens"),
                    );
                }
            }
        }
        if !swap_seen.iter().all(|&s| s) {
            return fail(Invariant::Swap, "a swap block is lost".into());
        }
        Ok(())
    }
}

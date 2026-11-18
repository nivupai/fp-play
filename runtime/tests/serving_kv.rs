//! R2: the paged KV cache.
//!
//! * Hand-derived cases: allocation order, prefix hits, LRU retention and
//!   eviction order, copy-on-write, swap of shared blocks, atomic failure.
//! * A model-based property test: random operation sequences checked after
//!   every operation against a plain map of each sequence's tokens, with
//!   every invariant of `KvCache::check`.
//! * Planted bugs: each deliberate bug must be caught by the invariant this
//!   file names for it.

use std::collections::BTreeMap;

use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::kv::{Invariant, KvCache, KvConfig, KvError, Mutation, SeqId, Violation};

fn seq(request: u32, sample: u32) -> SeqId {
    SeqId { request, sample }
}

fn config(block_size: usize, device_blocks: usize, prefix_caching: bool) -> KvConfig {
    KvConfig {
        block_size,
        device_blocks,
        swap_blocks: 8,
        prefix_caching,
    }
}

/// Tokens of a sequence for the hand-derived tests.
fn toks(base: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

/// Check against a map of each sequence's tokens.
fn check(kv: &KvCache, model: &BTreeMap<SeqId, Vec<u32>>) -> Result<(), Violation> {
    kv.check(&|s, p| model[&s][p])
}

#[test]
fn lifo_allocation_order() {
    let mut kv = KvCache::new(config(4, 6, false));
    let mut model = BTreeMap::new();
    kv.start(seq(0, 0)).unwrap();
    kv.append(seq(0, 0), &toks(100, 9)).unwrap(); // 3 blocks: 0, 1, 2
    model.insert(seq(0, 0), toks(100, 9));
    assert_eq!(kv.table(seq(0, 0)).unwrap().blocks, vec![0, 1, 2]);
    kv.start(seq(1, 0)).unwrap();
    kv.append(seq(1, 0), &toks(200, 4)).unwrap(); // block 3
    model.insert(seq(1, 0), toks(200, 4));
    kv.free(seq(0, 0)).unwrap(); // pushes 0, 1, 2: next pop is 2
    model.remove(&seq(0, 0));
    kv.append(seq(1, 0), &[7, 8]).unwrap();
    model.get_mut(&seq(1, 0)).unwrap().extend([7, 8]);
    assert_eq!(kv.table(seq(1, 0)).unwrap().blocks, vec![3, 2]);
    // Free list now [5, 4, 0, 1] (bottom to top).
    assert_eq!(kv.free_blocks(), 4);
    check(&kv, &model).unwrap();
}

#[test]
fn prefix_hits_leave_one_token_and_are_retained() {
    let mut kv = KvCache::new(config(4, 8, true));
    let prompt = toks(500, 12); // exactly 3 full blocks
    kv.start(seq(0, 0)).unwrap();
    assert_eq!(kv.adopt_prefix(seq(0, 0), &prompt).unwrap(), 0);
    kv.append(seq(0, 0), &prompt).unwrap();
    kv.free(seq(0, 0)).unwrap();
    // The three registered blocks are retained, not freed.
    assert_eq!((kv.retained_blocks(), kv.free_blocks()), (3, 5));
    // Same prompt again: two blocks adopted, the last block's 4 tokens recomputed.
    kv.start(seq(1, 0)).unwrap();
    assert_eq!(kv.adopt_prefix(seq(1, 0), &prompt).unwrap(), 8);
    assert_eq!(kv.table(seq(1, 0)).unwrap().blocks, vec![0, 1]);
    // A longer prompt with the same start adopts all three.
    kv.start(seq(2, 0)).unwrap();
    let mut longer = prompt.clone();
    longer.push(9);
    assert_eq!(kv.adopt_prefix(seq(2, 0), &longer).unwrap(), 12);
    assert_eq!(kv.refcount(0), 2);
    assert_eq!(kv.refcount(2), 1);
    // A different second block stops matching after block 0.
    kv.start(seq(3, 0)).unwrap();
    let mut diverging = prompt.clone();
    diverging[5] = 0;
    assert_eq!(kv.adopt_prefix(seq(3, 0), &diverging).unwrap(), 4);
    assert_eq!(kv.stats().prefix_hit_blocks, 2 + 3 + 1);
    assert_eq!(kv.stats().prefix_eligible_blocks, 2 + 2 + 3 + 2);
}

#[test]
fn eviction_is_lru_with_ties_by_block_id() {
    let mut kv = KvCache::new(config(2, 6, true));
    let mut model = BTreeMap::new();
    // Sequence A fills blocks 0, 1; sequence B fills 2, 3; both registered.
    for (i, base) in [(0, 10), (1, 20)] {
        kv.start(seq(i, 0)).unwrap();
        kv.append(seq(i, 0), &toks(base, 4)).unwrap();
    }
    kv.free(seq(1, 0)).unwrap(); // B released first (blocks 2, 3, same tick)
    kv.free(seq(0, 0)).unwrap(); // then A (blocks 0, 1)
    assert_eq!((kv.free_blocks(), kv.retained_blocks()), (2, 4));
    // Take 5 blocks: the free list [5, 4] first (top is 4), then evict 2, 3
    // (B, released first; lower ID first), then 0.
    kv.start(seq(9, 0)).unwrap();
    kv.append(seq(9, 0), &toks(900, 9)).unwrap();
    model.insert(seq(9, 0), toks(900, 9));
    assert_eq!(kv.table(seq(9, 0)).unwrap().blocks, vec![4, 5, 2, 3, 0]);
    assert_eq!(kv.stats().evictions, 3);
    assert_eq!(kv.retained_blocks(), 1);
    check(&kv, &model).unwrap();
    // A's first block was evicted, so A's prompt no longer hits at all.
    kv.start(seq(0, 1)).unwrap();
    assert_eq!(kv.adopt_prefix(seq(0, 1), &toks(10, 5)).unwrap(), 0);
}

#[test]
fn fork_then_copy_on_write() {
    let mut kv = KvCache::new(config(4, 8, false));
    let mut model = BTreeMap::new();
    let prompt = toks(1, 6); // one full block + 2 tokens
    kv.start(seq(0, 0)).unwrap();
    kv.append(seq(0, 0), &prompt).unwrap();
    for s in 1..4 {
        kv.fork(seq(0, 0), seq(0, s)).unwrap();
    }
    assert_eq!(kv.refcount(0), 4);
    assert_eq!(kv.refcount(1), 4);
    for s in 0..4 {
        model.insert(seq(0, s), prompt.clone());
    }
    // Appending nothing never copies.
    assert_eq!(kv.append_cost(seq(0, 0), 0), Some(0));
    assert!(!kv.append(seq(0, 0), &[]).unwrap().cow);
    // The first three samples copy the shared partial block; the last owns it.
    for s in 0..4 {
        assert_eq!(kv.append_cost(seq(0, s), 1), Some(usize::from(s < 3)));
        let report = kv.append(seq(0, s), &[1000 + s]).unwrap();
        assert_eq!(report.cow, s < 3, "sample {s}");
        model.get_mut(&seq(0, s)).unwrap().push(1000 + s);
        check(&kv, &model).unwrap();
    }
    assert_eq!(kv.stats().cow_copies, 3);
    assert_eq!(kv.refcount(0), 4); // the full block stays shared
    assert_eq!(kv.table(seq(0, 3)).unwrap().blocks, vec![0, 1]);
    assert_eq!(kv.used_blocks(), 5);
}

#[test]
fn swap_copies_shared_blocks_privately() {
    let mut kv = KvCache::new(config(4, 8, false));
    let mut model = BTreeMap::new();
    kv.start(seq(0, 0)).unwrap();
    kv.append(seq(0, 0), &toks(1, 6)).unwrap();
    kv.fork(seq(0, 0), seq(0, 1)).unwrap();
    model.insert(seq(0, 0), toks(1, 6));
    model.insert(seq(0, 1), toks(1, 6));
    assert_eq!(kv.swap_out(seq(0, 0)).unwrap(), 2);
    assert_eq!(kv.swap_free_blocks(), 6);
    // Sample 1 still holds both device blocks alone.
    assert_eq!((kv.refcount(0), kv.refcount(1)), (1, 1));
    check(&kv, &model).unwrap();
    assert_eq!(kv.swap_in(seq(0, 0)).unwrap(), 2);
    assert_eq!(kv.used_blocks(), 4);
    assert_eq!(kv.swap_free_blocks(), 8);
    check(&kv, &model).unwrap();
}

#[test]
fn failures_change_nothing() {
    let mut kv = KvCache::new(KvConfig {
        swap_blocks: 1,
        ..config(4, 3, true)
    });
    let mut model = BTreeMap::new();
    kv.start(seq(0, 0)).unwrap();
    kv.append(seq(0, 0), &toks(1, 10)).unwrap();
    model.insert(seq(0, 0), toks(1, 10));
    let before = kv.table(seq(0, 0)).unwrap().clone();
    assert_eq!(
        kv.append(seq(0, 0), &toks(50, 3)),
        Err(KvError::OutOfBlocks {
            needed: 1,
            available: 0
        })
    );
    assert_eq!(kv.table(seq(0, 0)), Some(&before));
    assert_eq!(
        kv.swap_out(seq(0, 0)),
        Err(KvError::SwapFull {
            needed: 3,
            available: 1
        })
    );
    assert_eq!(kv.table(seq(0, 0)), Some(&before));
    assert_eq!(kv.start(seq(0, 0)), Err(KvError::SeqExists(seq(0, 0))));
    assert_eq!(kv.fork(seq(5, 0), seq(5, 1)), Err(KvError::UnknownSeq(seq(5, 0))));
    check(&kv, &model).unwrap();
    // A device reset drops everything on the device.
    assert_eq!(kv.reset_device(), vec![seq(0, 0)]);
    assert_eq!(kv.free_blocks(), 3);
    check(&kv, &BTreeMap::new()).unwrap();
}

// ---------------------------------------------------------------------------
// Model-based random operations
// ---------------------------------------------------------------------------

/// A random operation driver over one cache and a plain model of every
/// sequence's tokens. Returns the first violation with the seed and step.
struct Driver {
    rng: SplitMix64,
    kv: KvCache,
    device: BTreeMap<SeqId, Vec<u32>>,
    swapped: BTreeMap<SeqId, Vec<u32>>,
    prompts: Vec<Vec<u32>>,
    next_request: u32,
    next_token: u32,
}

#[derive(Debug)]
struct Failure {
    seed: u64,
    step: usize,
    op: &'static str,
    violation: Violation,
}

impl Driver {
    fn new(seed: u64, mutation: Option<Mutation>) -> Driver {
        let mut rng = SplitMix64::new(seed);
        let block_size = [1, 2, 4, 16][rng.below(4) as usize];
        let config = KvConfig {
            block_size,
            device_blocks: 12 + rng.below(40) as usize,
            swap_blocks: 4 + rng.below(30) as usize,
            prefix_caching: rng.below(4) != 0,
        };
        let mut kv = KvCache::new(config);
        if let Some(m) = mutation {
            kv.plant(m);
        }
        // Three shared system prompts, each a few blocks long.
        let prompts = (0..3)
            .map(|p| {
                (0..(3 * block_size + 3) as u32)
                    .map(|i| 1_000_000 * (p + 1) + i)
                    .collect()
            })
            .collect();
        Driver {
            rng,
            kv,
            device: BTreeMap::new(),
            swapped: BTreeMap::new(),
            prompts,
            next_request: 0,
            next_token: 1,
        }
    }

    fn fresh_tokens(&mut self, n: usize) -> Vec<u32> {
        let start = self.next_token;
        self.next_token += n as u32;
        (start..start + n as u32).collect()
    }

    fn pick(&mut self, from_swapped: bool) -> Option<SeqId> {
        let keys: Vec<SeqId> = if from_swapped {
            self.swapped.keys().copied().collect()
        } else {
            self.device.keys().copied().collect()
        };
        (!keys.is_empty()).then(|| keys[self.rng.below(keys.len() as u64) as usize])
    }

    /// One random operation; returns its name.
    fn step(&mut self) -> &'static str {
        let b = self.kv.config().block_size;
        match self.rng.below(100) {
            0..=24 => {
                // New sequence: maybe a shared prompt prefix, then its own tokens.
                let mut tokens = Vec::new();
                if self.rng.below(10) < 6 {
                    let p = self.rng.below(3) as usize;
                    let take = 1 + self.rng.below(self.prompts[p].len() as u64) as usize;
                    tokens.extend_from_slice(&self.prompts[p][..take]);
                }
                let own = self.rng.below(2 * b as u64 + 2) as usize;
                tokens.extend(self.fresh_tokens(own));
                if tokens.is_empty() {
                    tokens.extend(self.fresh_tokens(1));
                }
                let id = seq(self.next_request, 0);
                self.next_request += 1;
                self.kv.start(id).unwrap();
                let adopted = self.kv.adopt_prefix(id, &tokens).unwrap();
                assert!(adopted < tokens.len() && adopted % b == 0);
                match self.kv.append(id, &tokens[adopted..]) {
                    Ok(_) => {
                        self.device.insert(id, tokens);
                    }
                    Err(KvError::OutOfBlocks { .. }) => {
                        self.kv.free(id).unwrap();
                    }
                    Err(e) => panic!("{e}"),
                }
                "start"
            }
            25..=54 => {
                if let Some(id) = self.pick(false) {
                    let n = 1 + self.rng.below(2 * b as u64) as usize;
                    let tokens = self.fresh_tokens(n);
                    let before = self.kv.table(id).cloned();
                    let available = self.kv.available();
                    let cost = self.kv.append_cost(id, n).unwrap();
                    match self.kv.append(id, &tokens) {
                        Ok(report) => {
                            assert!(cost <= available);
                            assert_eq!(report.new_blocks, cost);
                            self.device.get_mut(&id).unwrap().extend(tokens);
                        }
                        Err(KvError::OutOfBlocks { needed, .. }) => {
                            assert!(needed > available);
                            assert_eq!(self.kv.table(id).cloned(), before, "failed append changed the table");
                        }
                        Err(e) => panic!("{e}"),
                    }
                }
                "append"
            }
            55..=64 => {
                if let Some(id) = self.pick(false) {
                    let child = seq(id.request, 1000 + self.rng.below(1_000_000) as u32);
                    if self.kv.fork(id, child).is_ok() {
                        let tokens = self.device[&id].clone();
                        self.device.insert(child, tokens);
                    }
                }
                "fork"
            }
            65..=79 => {
                if let Some(id) = self.pick(false) {
                    self.kv.free(id).unwrap();
                    self.device.remove(&id);
                }
                "free"
            }
            80..=87 => {
                if let Some(id) = self.pick(false) {
                    match self.kv.swap_out(id) {
                        Ok(_) => {
                            let tokens = self.device.remove(&id).unwrap();
                            self.swapped.insert(id, tokens);
                        }
                        Err(KvError::SwapFull { .. }) => {}
                        Err(e) => panic!("{e}"),
                    }
                }
                "swap_out"
            }
            88..=95 => {
                if let Some(id) = self.pick(true) {
                    match self.kv.swap_in(id) {
                        Ok(_) => {
                            let tokens = self.swapped.remove(&id).unwrap();
                            self.device.insert(id, tokens);
                        }
                        Err(KvError::OutOfBlocks { .. }) => {}
                        Err(e) => panic!("{e}"),
                    }
                }
                "swap_in"
            }
            96..=98 => {
                if let Some(id) = self.pick(true) {
                    self.kv.drop_swapped(id).unwrap();
                    self.swapped.remove(&id);
                }
                "drop_swapped"
            }
            _ => {
                let dropped = self.kv.reset_device();
                assert!(dropped.iter().copied().eq(self.device.keys().copied()));
                self.device.clear();
                "reset_device"
            }
        }
    }

    fn check(&self) -> Result<(), Violation> {
        let (device, swapped) = (&self.device, &self.swapped);
        self.kv
            .check(&|s, p| device.get(&s).or_else(|| swapped.get(&s)).expect("known sequence")[p])
    }
}

/// Run `seeds` random sequences of `steps` operations, checking after each.
fn run(seeds: std::ops::Range<u64>, steps: usize, mutation: Option<Mutation>) -> Result<(), Failure> {
    for seed in seeds {
        let mut d = Driver::new(seed, mutation);
        for step in 0..steps {
            let op = d.step();
            d.check().map_err(|violation| Failure {
                seed,
                step,
                op,
                violation,
            })?;
        }
    }
    Ok(())
}

#[test]
fn random_operations_keep_every_invariant() {
    if let Err(f) = run(0..400, 400, None) {
        panic!(
            "replay with seed {}: step {} ({}) broke {}",
            f.seed, f.step, f.op, f.violation
        );
    }
}

#[test]
fn random_operations_exercise_every_path() {
    let mut totals = [0u64; 5];
    for seed in 0..100 {
        let mut d = Driver::new(seed, None);
        for _ in 0..400 {
            d.step();
        }
        let s = d.kv.stats();
        for (t, v) in totals.iter_mut().zip([
            s.cow_copies,
            s.evictions,
            s.prefix_hit_blocks,
            s.swapped_out_blocks,
            s.swapped_in_blocks,
        ]) {
            *t += v;
        }
    }
    assert!(totals.iter().all(|&t| t > 100), "{totals:?}");
}

// ---------------------------------------------------------------------------
// Planted bugs
// ---------------------------------------------------------------------------

/// Each planted bug and the invariant that must catch it.
const PLANTED: [(Mutation, Invariant); 5] = [
    (Mutation::LeakOnFree, Invariant::Refcount),
    (Mutation::SkipCow, Invariant::Content),
    (Mutation::SwapOutKeepsSharedRef, Invariant::Refcount),
    (Mutation::EarlyBlockFull, Invariant::Shape),
    (Mutation::StalePrefixKey, Invariant::PrefixCache),
];

#[test]
fn every_planted_bug_is_caught_by_its_invariant() {
    for (mutation, invariant) in PLANTED {
        let failure = run(0..400, 400, Some(mutation)).expect_err(&format!("{mutation:?} was not caught"));
        assert_eq!(
            failure.violation.invariant, invariant,
            "{mutation:?} caught by {} (seed {}, step {}, {})",
            failure.violation, failure.seed, failure.step, failure.op
        );
        eprintln!(
            "{mutation:?}: caught by {:?} at seed {} step {} ({})",
            invariant, failure.seed, failure.step, failure.op
        );
    }
}

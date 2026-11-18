//! The decoder inside the R2 engine.
//!
//! [`serve`] runs a workload through [`crate::serving::engine::run_model`]:
//! the scheduler decides every step exactly as in R2 (continuous batching,
//! chunked prefill, prefix caching, preemption, failures), and a
//! [`ModelExecutor`] computes each completed step with the real model. Token
//! values are the model's greedy outputs.
//!
//! # The K/V tensor store
//!
//! K and V live in a [`TensorStore`] indexed by the same physical blocks as
//! the R2 block tables: device block `b` holds, for each of its `B` token
//! slots, every layer's K and V for every KV head (all ranks' shards of
//! that block together). Before each step the executor replays the KV
//! cache's block copies (copy-on-write, swap-out, swap-in) in order, then the
//! model writes each computed position into its sequence's block and reads
//! every earlier position from wherever the block table points. So a
//! sequence that adopted a cached prefix reads K/V that another request
//! computed in another step, and a swapped-in sequence reads exactly what was
//! swapped out.
//!
//! **Tensor invariant** (checked after every step with checks on): every
//! slot a device or swapped sequence covers carries the tag of the token
//! prefix that sequence holds at that position (a hash of its tokens up to
//! and including it). A missed or misdirected copy, or a write to a shared
//! block, breaks it.

use std::collections::BTreeMap;

use super::decode::{fnv1a, Generation, FNV_OFFSET};
use super::forward::{KvStore, Model, RowReduce, StepToken, Top2};
use crate::serving::engine::{run_model, EngineConfig, EngineError, RunResult, StepExecutor};
use crate::serving::kv::{BlockCopy, BlockLoc, KvCache, KvConfig, SeqId};
use crate::serving::scheduler::{Scheduler, StepPlan};
use crate::serving::step_time::StepTimer;
use crate::serving::workload::Request;

/// Physical K/V storage mirroring the KV cache's device and host blocks.
#[derive(Clone, Debug)]
pub struct TensorStore {
    block_size: usize,
    width: usize,
    per_block: usize,
    device: Vec<u16>,
    host: Vec<u16>,
    device_tags: Vec<u64>,
    host_tags: Vec<u64>,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<super::DecoderMutation>,
}

impl TensorStore {
    pub fn new(layers: usize, kv_width: usize, kv: &KvConfig) -> TensorStore {
        let per_block = layers * 2 * kv.block_size * kv_width;
        TensorStore {
            block_size: kv.block_size,
            width: kv_width,
            per_block,
            device: vec![0; kv.device_blocks * per_block],
            host: vec![0; kv.swap_blocks * per_block],
            device_tags: vec![0; kv.device_blocks * kv.block_size],
            host_tags: vec![0; kv.swap_blocks * kv.block_size],
            #[cfg(feature = "planted-bugs")]
            planted: None,
        }
    }

    /// Offset of (layer, K or V, slot) inside a block.
    fn offset(&self, layer: usize, kind: usize, slot: usize) -> usize {
        ((layer * 2 + kind) * self.block_size + slot) * self.width
    }

    /// Replay one block copy.
    pub fn apply(&mut self, c: BlockCopy) {
        #[cfg(feature = "planted-bugs")]
        if self.planted == Some(super::DecoderMutation::SkipKvTensorCow)
            && matches!((c.from, c.to), (BlockLoc::Device(_), BlockLoc::Device(_)))
        {
            return;
        }
        let (pb, b) = (self.per_block, self.block_size);
        let read = |loc: BlockLoc, s: &TensorStore| -> (Vec<u16>, Vec<u64>) {
            match loc {
                BlockLoc::Device(i) => {
                    let i = i as usize;
                    (
                        s.device[i * pb..(i + 1) * pb].to_vec(),
                        s.device_tags[i * b..(i + 1) * b].to_vec(),
                    )
                }
                BlockLoc::Host(i) => {
                    let i = i as usize;
                    (
                        s.host[i * pb..(i + 1) * pb].to_vec(),
                        s.host_tags[i * b..(i + 1) * b].to_vec(),
                    )
                }
            }
        };
        let (data, tags) = read(c.from, self);
        let (store, tag_store, i) = match c.to {
            BlockLoc::Device(i) => (&mut self.device, &mut self.device_tags, i as usize),
            BlockLoc::Host(i) => (&mut self.host, &mut self.host_tags, i as usize),
        };
        store[i * pb..(i + 1) * pb].copy_from_slice(&data);
        tag_store[i * b..(i + 1) * b].copy_from_slice(&tags);
    }

    /// Check the tensor invariant for every sequence in `s`'s KV cache.
    pub fn verify(&self, s: &Scheduler) -> Result<(), String> {
        let b = self.block_size;
        let check = |seq: SeqId, blocks: &[u32], len: usize, tags: &[u64], where_: &str| -> Result<(), String> {
            let r = &s.requests[seq.request as usize];
            let stream: Vec<u32> = (0..len).map(|p| r.token_at(seq.sample as usize, p as u64)).collect();
            for (pos, &want) in prefix_tags(&stream).iter().enumerate() {
                let got = tags[blocks[pos / b] as usize * b + pos % b];
                if got != want {
                    return Err(format!(
                        "tensor: {seq} position {pos} ({where_} block {}) holds K/V of a different token prefix",
                        blocks[pos / b]
                    ));
                }
            }
            Ok(())
        };
        for seq in s.kv.sequences() {
            let t = s.kv.table(seq).expect("listed");
            check(seq, &t.blocks, t.len, &self.device_tags, "device")?;
        }
        for seq in s.kv.swapped_sequences() {
            let t = s.kv.swapped_table(seq).expect("listed");
            check(seq, &t.blocks, t.len, &self.host_tags, "host")?;
        }
        Ok(())
    }
}

/// Tag of every prefix of `tokens`: a chained SplitMix-style hash, never 0.
pub fn prefix_tags(tokens: &[u32]) -> Vec<u64> {
    let mut h = 0x7a95_0000_0000_0001u64;
    tokens
        .iter()
        .map(|&t| {
            h = (h ^ u64::from(t)).wrapping_add(0x9e37_79b9_7f4a_7c15);
            h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            h ^= h >> 31;
            h | 1
        })
        .collect()
}

/// The model's view of the store for one step: positions resolve through
/// the block tables.
struct PagedView<'a> {
    store: &'a mut TensorStore,
    kv: &'a KvCache,
    tags: &'a BTreeMap<SeqId, Vec<u64>>,
}

impl PagedView<'_> {
    fn locate(&self, seq: SeqId, pos: usize) -> usize {
        let table = self.kv.table(seq).expect("a computed sequence is on the device");
        table.blocks[pos / self.store.block_size] as usize
    }
}

impl KvStore for PagedView<'_> {
    fn write(&mut self, seq: SeqId, layer: usize, pos: usize, k: &[u16], v: &[u16]) {
        let block = self.locate(seq, pos);
        let slot = pos % self.store.block_size;
        let base = block * self.store.per_block;
        let w = self.store.width;
        for (kind, data) in [(0, k), (1, v)] {
            let at = base + self.store.offset(layer, kind, slot);
            self.store.device[at..at + w].copy_from_slice(data);
        }
        self.store.device_tags[block * self.store.block_size + slot] = self.tags[&seq][pos];
    }

    fn k(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16] {
        let at =
            self.locate(seq, pos) * self.store.per_block + self.store.offset(layer, 0, pos % self.store.block_size);
        &self.store.device[at..at + self.store.width]
    }

    fn v(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16] {
        let at =
            self.locate(seq, pos) * self.store.per_block + self.store.offset(layer, 1, pos % self.store.block_size);
        &self.store.device[at..at + self.store.width]
    }
}

/// Counters of what the executor computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecStats {
    pub steps: u64,
    /// Token positions computed (including recomputation).
    pub forward_tokens: u64,
    /// Of those, positions the same sequence had computed before.
    pub recomputed_tokens: u64,
    /// Block copies replayed.
    pub block_copies: u64,
}

/// Computes each completed engine step with the model.
pub struct ModelExecutor<'a> {
    model: &'a mut Model,
    reduce: &'a mut dyn RowReduce,
    pub store: TensorStore,
    prompt_len: Vec<usize>,
    /// Logits of this step's committing tokens: seq → (position, row).
    step_logits: BTreeMap<SeqId, (usize, Vec<f32>)>,
    /// Highest position + 1 each sequence has computed.
    peak: BTreeMap<SeqId, usize>,
    pub generations: BTreeMap<SeqId, Generation>,
    pub stats: ExecStats,
    /// Verify the tensor invariant after every step.
    pub verify: bool,
    keep_logits: bool,
}

impl<'a> ModelExecutor<'a> {
    pub fn new(
        model: &'a mut Model,
        reduce: &'a mut dyn RowReduce,
        kv: &KvConfig,
        prompt_len: Vec<usize>,
        keep_logits: bool,
    ) -> ModelExecutor<'a> {
        let store = TensorStore::new(model.config.layers, model.config.kv_width(), kv);
        ModelExecutor {
            model,
            reduce,
            store,
            prompt_len,
            step_logits: BTreeMap::new(),
            peak: BTreeMap::new(),
            generations: BTreeMap::new(),
            stats: ExecStats::default(),
            verify: true,
            keep_logits,
        }
    }
}

impl StepExecutor for ModelExecutor<'_> {
    fn execute(&mut self, _index: u64, plan: &StepPlan, s: &mut Scheduler) -> Result<(), String> {
        for c in s.kv.take_copies() {
            self.store.apply(c);
            self.stats.block_copies += 1;
        }
        let mut tokens = Vec::new();
        let mut tags = BTreeMap::new();
        for w in &plan.work {
            let r = &s.requests[w.seq.request as usize];
            let sample = w.seq.sample as usize;
            let end = w.context + w.tokens;
            let commits = end == r.known_len(sample);
            let stream: Vec<u32> = (0..end).map(|p| r.token_at(sample, p)).collect();
            let peak = self.peak.entry(w.seq).or_insert(0);
            for (p, &token) in stream.iter().enumerate().skip(w.context as usize) {
                tokens.push(StepToken {
                    seq: w.seq,
                    pos: p,
                    token,
                    logits: commits && p + 1 == end as usize,
                    recompute: p < *peak,
                });
            }
            self.stats.recomputed_tokens += (*peak).min(end as usize).saturating_sub(w.context as usize) as u64;
            *peak = (*peak).max(end as usize);
            tags.insert(w.seq, prefix_tags(&stream));
        }
        self.stats.steps += 1;
        self.stats.forward_tokens += tokens.len() as u64;
        let mut view = PagedView {
            store: &mut self.store,
            kv: &s.kv,
            tags: &tags,
        };
        let rows = self.model.step(&tokens, &mut view, self.reduce);
        self.step_logits = rows.into_iter().map(|r| (r.seq, (r.pos, r.logits))).collect();
        if self.verify {
            self.store.verify(s)?;
        }
        Ok(())
    }

    fn output(&mut self, seq: SeqId, k: u32) -> Result<u32, String> {
        let prompt = self.prompt_len[seq.request as usize];
        // A forked sibling takes its first token from the carrier's prompt logits.
        let entry = self.step_logits.get(&seq).or_else(|| {
            self.step_logits
                .iter()
                .find(|(other, (pos, _))| other.request == seq.request && pos + 1 == prompt)
                .map(|(_, e)| e)
        });
        let (pos, row) = entry.ok_or_else(|| format!("{seq} committed output {k} without logits"))?;
        if pos + 1 != prompt + k as usize {
            return Err(format!("{seq} output {k} chosen from the logits of position {pos}"));
        }
        let top = Top2::of(row);
        let g = self.generations.entry(seq).or_insert_with(|| Generation {
            tokens: Vec::new(),
            top: Vec::new(),
            logits_hash: FNV_OFFSET,
            logits: Vec::new(),
        });
        g.tokens.push(top.token);
        g.top.push(top);
        g.logits_hash = fnv1a(g.logits_hash, row);
        if self.keep_logits {
            g.logits.push(row.clone());
        }
        Ok(top.token)
    }
}

/// A workload for [`serve`]: request `i` carries `prompts[order[i]]` and
/// arrives at `arrival_ns[i]`, with one sample of `output_len` tokens.
#[derive(Clone, Debug)]
pub struct ServeSpec {
    pub prompts: Vec<Vec<u32>>,
    pub order: Vec<usize>,
    pub arrival_ns: Vec<u64>,
    pub output_len: u32,
    pub engine: EngineConfig,
}

pub struct Served {
    /// One per request: request i served `prompts[order[i]]`.
    pub generations: Vec<Generation>,
    pub run: RunResult,
    pub stats: ExecStats,
}

/// Serve `spec` with `model` and `reduce` through the R2 engine.
pub fn serve(
    model: &mut Model,
    reduce: &mut dyn RowReduce,
    spec: &ServeSpec,
    timer: &dyn StepTimer,
    keep_logits: bool,
) -> Result<Served, EngineError> {
    let n = spec.order.len();
    if spec.arrival_ns.len() != n {
        return Err(EngineError::Config("one arrival time per request".into()));
    }
    let prompts: Vec<Vec<u32>> = spec.order.iter().map(|&i| spec.prompts[i].clone()).collect();
    let requests: Vec<Request> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| Request {
            id: i as u32,
            arrival_ns: spec.arrival_ns[i],
            priority: 0,
            prefix_group: None,
            prefix_len: 0,
            prompt_len: p.len() as u32,
            output_len: spec.output_len,
            samples: 1,
        })
        .collect();
    let mut executor = ModelExecutor::new(
        model,
        reduce,
        &spec.engine.kv,
        prompts.iter().map(Vec::len).collect(),
        keep_logits,
    );
    executor.verify = spec.engine.kv_check_interval > 0;
    let run = run_model(&requests, &prompts, &spec.engine, timer, &mut executor)?;
    let generations = (0..n)
        .map(|i| {
            let seq = SeqId {
                request: i as u32,
                sample: 0,
            };
            executor
                .generations
                .remove(&seq)
                .ok_or_else(|| EngineError::Config(format!("request {i} produced no tokens")))
        })
        .collect::<Result<_, _>>()?;
    Ok(Served {
        generations,
        run,
        stats: executor.stats,
    })
}

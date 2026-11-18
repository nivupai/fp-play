//! Greedy decoding with a simple static batch (R3a): one prefill step for
//! every prompt, then one decode step per generated token for every
//! sequence, in sequence order. The engine-driven schedules come in R3b.

use std::collections::BTreeMap;

use super::forward::{KvStore, Model, RowReduce, StepToken, Top2};
use crate::serving::kv::SeqId;

/// Contiguous per-sequence KV storage: `[layer][pos]` → K and V slices.
#[derive(Clone, Debug, Default)]
pub struct ContiguousKv {
    width: usize,
    k: BTreeMap<(SeqId, usize), Vec<u16>>,
    v: BTreeMap<(SeqId, usize), Vec<u16>>,
}

impl ContiguousKv {
    pub fn new(width: usize) -> ContiguousKv {
        ContiguousKv {
            width,
            ..ContiguousKv::default()
        }
    }

    fn slot(&self, pos: usize) -> std::ops::Range<usize> {
        pos * self.width..(pos + 1) * self.width
    }
}

impl KvStore for ContiguousKv {
    fn write(&mut self, seq: SeqId, layer: usize, pos: usize, k: &[u16], v: &[u16]) {
        assert_eq!((k.len(), v.len()), (self.width, self.width));
        for (store, data) in [(&mut self.k, k), (&mut self.v, v)] {
            let buf = store.entry((seq, layer)).or_default();
            assert_eq!(
                buf.len(),
                pos * self.width,
                "{seq} layer {layer}: write at {pos} out of order"
            );
            buf.extend_from_slice(data);
        }
    }

    fn k(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16] {
        &self.k[&(seq, layer)][self.slot(pos)]
    }

    fn v(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16] {
        &self.v[&(seq, layer)][self.slot(pos)]
    }
}

/// One sequence's greedy output: the generated tokens and, for each, the
/// top-2 summary of the logits it was chosen from, plus an FNV-1a hash of
/// every logit's bits in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub top: Vec<Top2>,
    pub logits_hash: u64,
    /// Every logits row, kept only when requested (for vector files).
    pub logits: Vec<Vec<f32>>,
}

pub fn fnv1a(hash: u64, logits: &[f32]) -> u64 {
    let mut h = hash;
    for z in logits {
        for b in z.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Generate `steps` tokens for every prompt with one static batch.
pub fn generate_static(
    model: &mut Model,
    reduce: &mut dyn RowReduce,
    prompts: &[Vec<u32>],
    steps: usize,
    keep_logits: bool,
) -> Vec<Generation> {
    assert!(prompts.iter().all(|p| !p.is_empty()) && steps > 0);
    let seq = |i: usize| SeqId {
        request: i as u32,
        sample: 0,
    };
    let mut kv = ContiguousKv::new(model.config.kv_width());
    let mut out: Vec<Generation> = prompts
        .iter()
        .map(|_| Generation {
            tokens: Vec::new(),
            top: Vec::new(),
            logits_hash: FNV_OFFSET,
            logits: Vec::new(),
        })
        .collect();
    let mut work: Vec<StepToken> = prompts
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            p.iter().enumerate().map(move |(pos, &token)| StepToken {
                seq: seq(i),
                pos,
                token,
                logits: pos + 1 == p.len(),
                recompute: false,
            })
        })
        .collect();
    for step in 0..steps {
        let logits = model.step(&work, &mut kv, reduce);
        assert_eq!(logits.len(), prompts.len());
        for (i, row) in logits.into_iter().enumerate() {
            let g = &mut out[i];
            let top = Top2::of(&row.logits);
            g.tokens.push(top.token);
            g.top.push(top);
            g.logits_hash = fnv1a(g.logits_hash, &row.logits);
            if keep_logits {
                g.logits.push(row.logits);
            }
        }
        if step + 1 < steps {
            work = (0..prompts.len())
                .map(|i| StepToken {
                    seq: seq(i),
                    pos: prompts[i].len() + step,
                    token: out[i].tokens[step],
                    logits: true,
                    recompute: false,
                })
                .collect();
        }
    }
    out
}

/// Deterministic prompts: `count` prompts of `len` tokens, uniform over the vocabulary.
pub fn prompts(seed: u64, count: usize, len: usize, vocab: usize) -> Vec<Vec<u32>> {
    let mut rng = crate::rng::SplitMix64::new(seed);
    (0..count)
        .map(|_| (0..len).map(|_| rng.below(vocab as u64) as u32).collect())
        .collect()
}

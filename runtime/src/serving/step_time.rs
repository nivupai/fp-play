//! How long one engine step takes: a roofline compute model plus the
//! tensor-parallel AllReduces, costed from their real schedules.
//!
//! **Compute.** Each of the P devices holds 1/P of the weights and of every
//! token's KV heads. For a step that processes T tokens, a device performs
//! `2 × (layer parameters + output head) × T ÷ P` dense FLOPs (the output
//! head only for tokens whose logits are sampled) plus attention FLOPs
//! `4 × layers × q_heads × head_dim × (positions attended) ÷ P`, and moves
//! its weight shard (excluding the embedding table, which is only indexed)
//! plus the KV it reads and writes. Its time is the larger of FLOPs ÷ peak
//! and bytes ÷ memory bandwidth (a roofline). The constants are illustrative.
//!
//! **Communication.** Every layer ends its attention block and its MLP with
//! an AllReduce of `T × hidden` BF16 activations. Each is costed by
//! [`cost::estimate_recorded`] on the schedule the chosen algorithm (or the
//! shard tree) actually records for that length and P, obtained by a dry run
//! ([`collectives::schedule_rounds`], [`shard_tree::schedule_rounds`]), so
//! the collective's rounds, message sizes and wire policy all show up in the
//! step time. Compute and communication do not overlap.
//!
//! **Extras.** A fixed per-step overhead, plus copy-on-write copies (each a
//! read and a write of one block's per-device bytes at memory bandwidth) and
//! swap traffic (per-device block bytes over the host link).
//!
//! Every quantity is computed with basic IEEE binary64 operations and the
//! total is rounded once to whole nanoseconds, so step times are identical
//! on every platform.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::collectives::{schedule_rounds, Algorithm, WirePolicy};
use crate::cost::{estimate_recorded, LinkModel};
use crate::shard_tree;

use super::model::{DeviceConfig, ModelConfig, ServingError};

/// The collective used for the per-layer AllReduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collective {
    Schedule(Algorithm),
    /// The fixed tree over this many logical shards.
    ShardTree(usize),
}

impl Collective {
    pub fn name(&self) -> String {
        match self {
            Collective::Schedule(a) => a.name().to_string(),
            Collective::ShardTree(s) => format!("shard-tree (S = {s})"),
        }
    }
}

/// The work of one sequence in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqWork {
    /// Tokens already in the KV cache before this step.
    pub context: u64,
    /// Tokens processed this step (1 for a decode).
    pub tokens: u64,
    /// Whether the last token's logits are sampled this step.
    pub samples: bool,
}

/// Everything a step does that costs time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepWork {
    pub seqs: Vec<SeqWork>,
    pub cow_blocks: u64,
    pub swap_blocks: u64,
}

impl StepWork {
    pub fn tokens(&self) -> u64 {
        self.seqs.iter().map(|s| s.tokens).sum()
    }
}

/// A step's duration and its parts (nanoseconds; the parts are unrounded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepTime {
    pub total_ns: u64,
    pub compute_ns: f64,
    pub comm_ns: f64,
    pub overhead_ns: f64,
    pub copy_ns: f64,
}

/// Anything that can time a step: the roofline model below, or a simple
/// linear model for hand-derived tests.
pub trait StepTimer {
    fn time(&self, work: &StepWork) -> StepTime;
}

#[derive(Clone, Debug)]
pub struct StepModel {
    pub model: ModelConfig,
    pub device: DeviceConfig,
    pub world: u64,
    pub block_size: u64,
    pub collective: Collective,
    pub policy: WirePolicy,
    pub link: LinkModel,
    /// Fixed per-step overhead, nanoseconds.
    pub overhead_ns: f64,
    /// Communication time of one AllReduce, by element count (memoized).
    allreduce_ns: RefCell<BTreeMap<u64, f64>>,
}

impl StepModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelConfig,
        device: DeviceConfig,
        world: u64,
        block_size: u64,
        collective: Collective,
        policy: WirePolicy,
        link: LinkModel,
        overhead_ns: f64,
    ) -> Result<StepModel, ServingError> {
        model.check_world(world)?;
        let m = StepModel {
            model,
            device,
            world,
            block_size,
            collective,
            policy,
            link,
            overhead_ns,
            allreduce_ns: RefCell::new(BTreeMap::new()),
        };
        // Reject an unusable collective now rather than mid-run.
        m.try_allreduce_ns(1)?;
        Ok(m)
    }

    fn try_allreduce_ns(&self, elements: u64) -> Result<f64, ServingError> {
        if let Some(&ns) = self.allreduce_ns.borrow().get(&elements) {
            return Ok(ns);
        }
        let (p, len) = (self.world as usize, elements as usize);
        let rounds = match self.collective {
            Collective::Schedule(a) => schedule_rounds(a, self.policy, p, len).map_err(|e| e.to_string()),
            Collective::ShardTree(s) => shard_tree::schedule_rounds(self.policy, p, s, len)
                .map(|t| t.rounds)
                .map_err(|e| e.to_string()),
        }
        .map_err(ServingError::InvalidConfig)?;
        let ns = estimate_recorded(&rounds, p, self.link).estimated_us * 1e3;
        self.allreduce_ns.borrow_mut().insert(elements, ns);
        Ok(ns)
    }

    /// Communication time of one AllReduce of `elements` activations.
    pub fn allreduce_ns(&self, elements: u64) -> f64 {
        self.try_allreduce_ns(elements).expect("checked at construction")
    }

    /// KV bytes of one block on one device.
    pub fn block_bytes_per_device(&self) -> u64 {
        self.block_size * self.model.kv_bytes_per_token_per_device(self.world)
    }
}

impl StepTimer for StepModel {
    fn time(&self, work: &StepWork) -> StepTime {
        let m = &self.model;
        let p = self.world as f64;
        let tokens = work.tokens();
        let sampled = work.seqs.iter().filter(|s| s.samples).count() as u64;
        // Dense FLOPs: every layer matrix for every token, the output head
        // for sampled tokens.
        let layer_flops = 2.0 * (m.layers * m.layer_params()) as f64 * tokens as f64;
        let head_flops = 2.0 * (m.vocab * m.hidden) as f64 * sampled as f64;
        // Attention: token i of a chunk starting at context c attends to c + i + 1 positions.
        let attended: u64 = work
            .seqs
            .iter()
            .map(|s| s.tokens * s.context + s.tokens * (s.tokens + 1) / 2)
            .sum();
        let attn_flops = 4.0 * (m.layers * m.q_heads * m.head_dim) as f64 * attended as f64;
        let flops = (layer_flops + head_flops + attn_flops) / p;
        // Bytes: the weight shard (without the embedding table) once, the KV
        // each sequence reads (its context) and writes (its new tokens).
        let weights = (m.params() - m.vocab * m.hidden) * m.weight_bytes;
        let kv_tokens: u64 = work.seqs.iter().map(|s| s.context + 2 * s.tokens).sum();
        let bytes = weights as f64 / p + (kv_tokens * m.kv_bytes_per_token_per_device(self.world)) as f64;
        let compute_ns = (flops / self.device.peak_flops).max(bytes / self.device.hbm_bytes_per_s) * 1e9;
        let comm_ns = if tokens == 0 {
            0.0
        } else {
            (2 * m.layers) as f64 * self.allreduce_ns(tokens * m.hidden)
        };
        let block = self.block_bytes_per_device() as f64;
        let copy_ns = (2 * work.cow_blocks) as f64 * block / self.device.hbm_bytes_per_s * 1e9
            + work.swap_blocks as f64 * block / self.device.host_bytes_per_s * 1e9;
        let total = self.overhead_ns + compute_ns + comm_ns + copy_ns;
        StepTime {
            total_ns: total.round() as u64,
            compute_ns,
            comm_ns,
            overhead_ns: self.overhead_ns,
            copy_ns,
        }
    }
}

/// `base + per_token × tokens` nanoseconds: for hand-derived tests.
#[derive(Clone, Copy, Debug)]
pub struct LinearTimer {
    pub base_ns: u64,
    pub per_token_ns: u64,
}

impl StepTimer for LinearTimer {
    fn time(&self, work: &StepWork) -> StepTime {
        let total_ns = self.base_ns + self.per_token_ns * work.tokens();
        StepTime {
            total_ns,
            compute_ns: (total_ns - self.base_ns) as f64,
            comm_ns: 0.0,
            overhead_ns: self.base_ns as f64,
            copy_ns: 0.0,
        }
    }
}

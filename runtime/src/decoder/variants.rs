//! The R3 workload and its batch-composition variants.
//!
//! Every variant serves the same prompts with the same greedy, fixed-length
//! decoding; only how the engine batches them changes. Comparing a prompt's
//! logits across variants under one numerical configuration (P, schedule,
//! wire policy) tests batch invariance.
//!
//! The prompts come in groups that share a leading block-aligned prefix, so
//! the prefix-caching variant has something to reuse; every variant uses the
//! same prompts.

use super::decode::Generation;
use super::forward::{Model, RowReduce};
use super::serve::{serve, ExecStats, ServeSpec};
use crate::rng::SplitMix64;
use crate::serving::engine::{EngineConfig, EngineError, Fault, FaultConfig};
use crate::serving::kv::KvConfig;
use crate::serving::scheduler::{FaultPolicy, PreemptionMode, SchedulerConfig};
use crate::serving::step_time::LinearTimer;

/// The workload: `groups × per_group` prompts of `shared + own` tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workload {
    pub seed: u64,
    pub groups: usize,
    pub per_group: usize,
    /// Tokens every prompt in a group shares (a multiple of the block size).
    pub shared: usize,
    pub own: usize,
    pub output_len: u32,
    pub block_size: usize,
}

impl Workload {
    /// 32 prompts of 16 tokens (4 groups of 8 sharing 8 tokens), 48 greedy tokens each.
    pub const R3: Workload = Workload {
        seed: 0x0003_9e0d_0001,
        groups: 4,
        per_group: 8,
        shared: 8,
        own: 8,
        output_len: 48,
        block_size: 4,
    };

    pub fn count(&self) -> usize {
        self.groups * self.per_group
    }

    pub fn prompts(&self, vocab: usize) -> Vec<Vec<u32>> {
        let mut rng = SplitMix64::new(self.seed);
        let prefixes: Vec<Vec<u32>> = (0..self.groups)
            .map(|_| (0..self.shared).map(|_| rng.below(vocab as u64) as u32).collect())
            .collect();
        (0..self.count())
            .map(|i| {
                let mut p = prefixes[i % self.groups].clone();
                p.extend((0..self.own).map(|_| rng.below(vocab as u64) as u32));
                p
            })
            .collect()
    }
}

/// How the engine batches the workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variant {
    /// Each prompt served by itself (the reference).
    Alone,
    /// Every prompt arrives at once; every prefill fits one step.
    Together,
    /// As `Together`, with the arrival order reversed.
    Reversed,
    /// An 8-token step budget: prefills in 8-token chunks, at most 8 decodes per step.
    Chunk8,
    /// A 64-token step budget.
    Chunk64,
    /// `Chunk64` with prefix caching: later prompts reuse shared-prefix K/V
    /// computed in earlier steps.
    PrefixCache,
    /// `Together` with too few KV blocks: recompute preemptions.
    Pressure,
    /// `Together` with one rank failure mid-decode, handled by retry.
    FailureRetry,
}

impl Variant {
    pub const ALL: [Variant; 8] = [
        Variant::Alone,
        Variant::Together,
        Variant::Reversed,
        Variant::Chunk8,
        Variant::Chunk64,
        Variant::PrefixCache,
        Variant::Pressure,
        Variant::FailureRetry,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Variant::Alone => "alone",
            Variant::Together => "together",
            Variant::Reversed => "reversed arrival",
            Variant::Chunk8 => "chunk 8",
            Variant::Chunk64 => "chunk 64",
            Variant::PrefixCache => "prefix cache",
            Variant::Pressure => "recompute preemption",
            Variant::FailureRetry => "failure + retry",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Variant::Alone => "each prompt served by itself",
            Variant::Together => "all prompts arrive at once; budget 1024 tokens, so every prefill runs in one step",
            Variant::Reversed => "as together, arrival order reversed",
            Variant::Chunk8 => "budget 8 tokens: 8-token prefill chunks, at most 8 tokens per step",
            Variant::Chunk64 => "budget 64 tokens",
            Variant::PrefixCache => "budget 64 with prefix caching: shared-prefix K/V reused from earlier steps",
            Variant::Pressure => "as together with 5/8 of the KV blocks the workload needs: recompute preemptions",
            Variant::FailureRetry => {
                "as together; a rank fails a quarter of the way through decoding (last layer, MLP AllReduce); retry recomputes"
            }
        }
    }
}

/// Step and engine settings shared by the variants.
const BIG_BUDGET: usize = 1024;
const MAX_SEQS: usize = 64;
const DEVICE_BLOCKS: usize = 1024;

impl Workload {
    /// KV blocks every sequence needs at its longest, summed.
    pub fn blocks_needed(&self) -> usize {
        self.count() * (self.shared + self.own + self.output_len as usize).div_ceil(self.block_size)
    }

    /// Device blocks of the pressure variant: 5/8 of what the workload needs.
    pub fn pressure_blocks(&self) -> usize {
        self.blocks_needed() * 5 / 8
    }

    /// The failing step of the failure variant: a quarter of the way through decoding.
    pub fn failure_step(&self) -> u64 {
        u64::from(self.output_len / 4).max(1)
    }
}

/// The engine timer. Time does not affect the numerics: every request
/// arrives at 0, so only the step order matters.
pub const TIMER: LinearTimer = LinearTimer {
    base_ns: 1_000,
    per_token_ns: 10,
};

fn engine(w: &Workload, budget: usize, blocks: usize, prefix_caching: bool, layers: usize) -> EngineConfig {
    EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: budget,
            max_seqs: MAX_SEQS,
            watermark_blocks: 0,
            preemption: PreemptionMode::Recompute,
        },
        kv: KvConfig {
            block_size: w.block_size,
            device_blocks: blocks,
            swap_blocks: 0,
            prefix_caching,
        },
        faults: FaultConfig {
            layers: layers as u32,
            ..FaultConfig::none()
        },
        kv_check_interval: 1,
    }
}

/// The engine runs a variant needs (one per prompt for `Alone`).
pub fn specs(w: &Workload, prompts: &[Vec<u32>], variant: Variant, layers: usize) -> Vec<ServeSpec> {
    let n = prompts.len();
    let spec = |order: Vec<usize>, engine: EngineConfig| ServeSpec {
        prompts: prompts.to_vec(),
        arrival_ns: vec![0; order.len()],
        order,
        output_len: w.output_len,
        engine,
    };
    let forward: Vec<usize> = (0..n).collect();
    match variant {
        Variant::Alone => (0..n)
            .map(|i| spec(vec![i], engine(w, BIG_BUDGET, DEVICE_BLOCKS, false, layers)))
            .collect(),
        Variant::Together => vec![spec(forward, engine(w, BIG_BUDGET, DEVICE_BLOCKS, false, layers))],
        Variant::Reversed => vec![spec(
            (0..n).rev().collect(),
            engine(w, BIG_BUDGET, DEVICE_BLOCKS, false, layers),
        )],
        Variant::Chunk8 => vec![spec(forward, engine(w, 8, DEVICE_BLOCKS, false, layers))],
        Variant::Chunk64 => vec![spec(forward, engine(w, 64, DEVICE_BLOCKS, false, layers))],
        Variant::PrefixCache => vec![spec(forward, engine(w, 64, DEVICE_BLOCKS, true, layers))],
        Variant::Pressure => vec![spec(forward, engine(w, BIG_BUDGET, w.pressure_blocks(), false, layers))],
        Variant::FailureRetry => {
            let mut e = engine(w, BIG_BUDGET, DEVICE_BLOCKS, false, layers);
            e.faults = FaultConfig {
                plan: vec![Fault {
                    step: w.failure_step(),
                    layer: (layers - 1) as u32,
                    allreduce: 1,
                }],
                policy: FaultPolicy::Retry,
                detection_ns: 1_000_000,
                recovery_ns: 5_000_000,
                retry_limit: 1,
                layers: layers as u32,
            };
            vec![spec(forward, e)]
        }
    }
}

/// What one variant produced.
#[derive(Clone, Debug)]
pub struct VariantRun {
    pub variant: Variant,
    /// One per prompt.
    pub generations: Vec<Generation>,
    pub exec: ExecStats,
    pub preemptions: u64,
    pub prefix_hit_blocks: u64,
    pub failed_steps: usize,
    pub engine_steps: usize,
}

/// Serve the workload as `variant`. `reduce` is called once per engine run
/// (a fresh reducer each time, so counters do not mix).
pub fn run_variant(
    model: &mut Model,
    reduce: &mut dyn FnMut() -> Box<dyn RowReduce>,
    w: &Workload,
    prompts: &[Vec<u32>],
    variant: Variant,
) -> Result<VariantRun, EngineError> {
    let mut out = VariantRun {
        variant,
        generations: vec![Generation::default(); prompts.len()],
        exec: ExecStats::default(),
        preemptions: 0,
        prefix_hit_blocks: 0,
        failed_steps: 0,
        engine_steps: 0,
    };
    for spec in specs(w, prompts, variant, model.config.layers) {
        let mut r = reduce();
        let served = serve(model, r.as_mut(), &spec, &TIMER, false)?;
        for (i, &p) in spec.order.iter().enumerate() {
            out.generations[p] = served.generations[i].clone();
        }
        let s = &served.stats;
        out.exec.steps += s.steps;
        out.exec.forward_tokens += s.forward_tokens;
        out.exec.recomputed_tokens += s.recomputed_tokens;
        out.exec.block_copies += s.block_copies;
        let c = &served.run.counters;
        out.preemptions += c.preemptions_recompute + c.preemptions_swap;
        out.prefix_hit_blocks += served.run.kv_stats.prefix_hit_blocks;
        out.failed_steps += served.run.steps.iter().filter(|x| x.fault.is_some()).count();
        out.engine_steps += served.run.steps.len();
    }
    Ok(out)
}

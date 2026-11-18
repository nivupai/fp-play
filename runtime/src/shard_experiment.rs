//! R1.1 experiment: the fixed shard tree on the R1 problem
//! (results/r1.1-shard-tree.*).
//!
//! The hidden dimension is split into S logical shards (default 64). Each
//! shard's FP32 partial is the same fused-chain kernel over its slice, so the
//! partials depend on S only. They are combined by [`shard_tree::all_reduce`]
//! for every P from 1 to S, and the output bits are checked to be identical
//! for all of them. The result is compared with ring and exact-gather on the
//! per-device partials of `config.world_sizes`.

use crate::collectives::{all_reduce_with, Algorithm, WirePolicy};
use crate::cost::{estimate_recorded, CostEstimate};
use crate::exact::ExactSum;
use crate::format::Format;
use crate::ops::convert;
use crate::shard_tree;
use crate::tensor_parallel::{
    argmax_changes, compare, generate, split_partials, Comparison, ExperimentConfig, Prepared, LOCAL_ACCUMULATOR,
};

/// Output format of every run.
pub const OUTPUT: Format = Format::Bf16;

/// Policies the shard tree accepts with a BF16 output (`bf16-wire/fp32-add`
/// is rejected: its reduce wire would round a child that crosses ranks).
pub const POLICIES: [WirePolicy; 2] = [WirePolicy::uniform(Format::Bf16), WirePolicy::FP32_RS_BF16_AG];

/// Schedules the shard tree is compared with.
pub const BASELINES: [Algorithm; 2] = [Algorithm::Ring, Algorithm::ExactGather];

#[derive(Clone, Debug)]
pub struct WorldCost {
    pub world: usize,
    pub reduce_rounds: usize,
    pub broadcast_rounds: usize,
    pub levels_with_messages: usize,
    pub serialization_extra_rounds: usize,
    pub cost: CostEstimate,
}

#[derive(Clone, Debug)]
pub struct ShardPolicy {
    pub policy: WirePolicy,
    /// Roundings per element before the tree: one per shard partial converted
    /// to a narrower accumulator, zero otherwise.
    pub input_roundings: u32,
    pub collective_roundings: (u32, u32),
    pub vs_exact: Comparison,
    /// Against the ideal for these shard partials (their exact sum, rounded once).
    pub vs_ideal: Comparison,
    pub vs_single_device: Comparison,
    pub argmax_changes_vs_single_device: usize,
    /// Every P in `1..=S` gave bit-identical outputs to P = 1.
    pub p_invariant: bool,
    /// Every rank agreed for every P.
    pub consistent_across_ranks: bool,
    /// One entry per P in `1..=S`.
    pub per_world: Vec<WorldCost>,
    pub output: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Baseline {
    pub world: usize,
    pub policy: WirePolicy,
    pub algorithm: Algorithm,
    pub vs_exact: Comparison,
    pub cost: CostEstimate,
}

#[derive(Clone, Debug)]
pub struct ShardResult {
    pub config: ExperimentConfig,
    pub shards: usize,
    pub elements: usize,
    /// The ideal for the shard partials against the exact dot product.
    pub ideal_vs_exact: Comparison,
    pub single_device_vs_exact: Comparison,
    pub policies: Vec<ShardPolicy>,
    pub baselines: Vec<Baseline>,
}

impl ShardResult {
    pub fn policy(&self, policy: WirePolicy) -> Option<&ShardPolicy> {
        self.policies.iter().find(|p| p.policy == policy)
    }
}

pub fn run_prepared(config: &ExperimentConfig, prepared: &Prepared, shards: usize) -> Result<ShardResult, String> {
    if shards == 0 || shards > config.hidden {
        return Err(format!(
            "shards must be between 1 and the hidden dimension ({})",
            config.hidden
        ));
    }
    if let Some(&p) = config.world_sizes.iter().find(|&&p| p > shards) {
        return Err(format!("world size {p} exceeds the {shards} shards"));
    }
    let partials = split_partials(config, &generate(config), shards);
    let exact = prepared.exact_in(OUTPUT);
    let single: Vec<u32> = prepared
        .single_fp32
        .iter()
        .map(|&v| convert(LOCAL_ACCUMULATOR, OUTPUT, v))
        .collect();
    let ideal: Vec<u32> = (0..exact.len())
        .map(|e| {
            let mut sum = ExactSum::new();
            for p in &partials {
                sum.add_raw(LOCAL_ACCUMULATOR, p[e]);
            }
            sum.round(OUTPUT)
        })
        .collect();
    let to_acc = |parts: &[Vec<u32>], acc: Format| -> Vec<Vec<u32>> {
        parts
            .iter()
            .map(|p| p.iter().map(|&v| convert(LOCAL_ACCUMULATOR, acc, v)).collect())
            .collect()
    };

    let mut policies = Vec::new();
    for policy in POLICIES {
        let inputs = to_acc(&partials, policy.accumulator);
        let reference = shard_tree::all_reduce(policy, 1, &inputs).map_err(|e| e.to_string())?;
        let output = reference.outcome.results[0].clone();
        let mut p_invariant = true;
        let mut consistent = true;
        let mut per_world = Vec::new();
        for world in 1..=shards {
            let got = shard_tree::all_reduce(policy, world, &inputs).map_err(|e| e.to_string())?;
            p_invariant &= got.outcome.results.iter().all(|r| *r == output);
            consistent &= got.outcome.consistent();
            per_world.push(WorldCost {
                world,
                reduce_rounds: got.reduce_rounds,
                broadcast_rounds: got.broadcast_rounds,
                levels_with_messages: got.levels_with_messages,
                serialization_extra_rounds: got.serialization_extra_rounds(),
                cost: estimate_recorded(&got.outcome.rounds, world, config.link),
            });
        }
        policies.push(ShardPolicy {
            policy,
            input_roundings: if policy.accumulator.contains(LOCAL_ACCUMULATOR) {
                0
            } else {
                shards as u32
            },
            collective_roundings: reference.outcome.roundings_range().unwrap_or((0, 0)),
            vs_exact: compare(OUTPUT, &output, &exact),
            vs_ideal: compare(OUTPUT, &output, &ideal),
            vs_single_device: compare(OUTPUT, &output, &single),
            argmax_changes_vs_single_device: argmax_changes(OUTPUT, config.outputs, &output, &single),
            p_invariant,
            consistent_across_ranks: consistent,
            per_world,
            output,
        });
    }

    let mut baselines = Vec::new();
    for (world, parts) in &prepared.partials {
        for policy in POLICIES {
            let inputs = to_acc(parts, policy.accumulator);
            for algorithm in BASELINES {
                let outcome = all_reduce_with(algorithm, policy, &inputs).map_err(|e| e.to_string())?;
                baselines.push(Baseline {
                    world: *world,
                    policy,
                    algorithm,
                    vs_exact: compare(OUTPUT, &outcome.results[0], &exact),
                    cost: estimate_recorded(&outcome.rounds, *world, config.link),
                });
            }
        }
    }
    Ok(ShardResult {
        config: config.clone(),
        shards,
        elements: exact.len(),
        ideal_vs_exact: compare(OUTPUT, &ideal, &exact),
        single_device_vs_exact: compare(OUTPUT, &single, &exact),
        policies,
        baselines,
    })
}

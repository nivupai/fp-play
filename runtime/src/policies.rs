//! R1.1 experiment: BF16 output with FP32 kept where it can help, and
//! reduce-scatter on its own (results/r1.1-policies.*).
//!
//! Same problem as the R1 experiment (`tensor_parallel`): each device's FP32
//! partial of `Y = X·W` is combined across devices, and the output is BF16.
//! Three wire policies are compared under every applicable schedule:
//!
//! * `bf16` — R1's BF16 wire: each FP32 partial is rounded to BF16 before the
//!   collective, and every combine is a BF16 addition;
//! * `fp32-rs/bf16-ag` — FP32 partials, FP32 reduce phases, one rounding to
//!   BF16, BF16 gather phases ([`WirePolicy::FP32_RS_BF16_AG`]);
//! * `bf16-wire/fp32-add` — FP32 partials and FP32 additions, but every
//!   message is BF16 ([`WirePolicy::BF16_WIRE_FP32_ADD`]).
//!
//! Each result is compared with the exact dot product rounded once to BF16,
//! with the **ideal for these partials** (the exact sum of the P FP32 partials
//! rounded once to BF16 — the best any collective can do once the local
//! kernels have run), and with the unsharded FP32 kernel rounded to BF16.

use crate::collectives::{all_reduce_with, reduce_scatter, Algorithm, WirePolicy};
use crate::cost::{estimate_recorded, CostEstimate};
use crate::exact::ExactSum;
use crate::format::Format;
use crate::ops::convert;
use crate::tensor_parallel::{
    argmax_changes, compare, prepare, Comparison, ExperimentConfig, Prepared, LOCAL_ACCUMULATOR,
};

/// Output format of every policy compared here.
pub const OUTPUT: Format = Format::Bf16;

/// The policies compared, in report order.
pub const POLICIES: [WirePolicy; 3] = [
    WirePolicy::uniform(Format::Bf16),
    WirePolicy::FP32_RS_BF16_AG,
    WirePolicy::BF16_WIRE_FP32_ADD,
];

/// Schedules that have separate reduce-scatter and all-gather phases.
pub const SPLIT_ALGORITHMS: [Algorithm; 2] = [Algorithm::Ring, Algorithm::HalvingDoubling];

#[derive(Clone, Debug)]
pub struct PolicyRun {
    pub policy: WirePolicy,
    pub algorithm: Algorithm,
    pub consistent_across_ranks: bool,
    /// Roundings per element before the collective: one per FP32 partial
    /// converted to a narrower accumulator format, zero otherwise.
    pub input_roundings: u32,
    /// Measured roundings per element inside the collective (min, max over all
    /// ranks and elements; equal in every run here).
    pub collective_roundings: (u32, u32),
    pub vs_exact: Comparison,
    pub vs_ideal: Comparison,
    pub vs_single_device: Comparison,
    pub argmax_changes_vs_single_device: usize,
    pub cost: CostEstimate,
    pub output: Vec<u32>,
}

/// How much of the gap between R1's BF16 wire and BF16 exact-gather a mixed
/// policy closes for one schedule: `(baseline - policy) / (baseline - exact_gather)`.
#[derive(Clone, Debug)]
pub struct Recovery {
    pub algorithm: Algorithm,
    pub policy: WirePolicy,
    /// Elements ≠ exact for the schedule under R1's BF16 wire.
    pub baseline_mismatched: usize,
    /// Elements ≠ exact for the schedule under the policy.
    pub policy_mismatched: usize,
    /// Elements ≠ exact for exact-gather under R1's BF16 wire.
    pub exact_gather_mismatched: usize,
    /// Fraction of the mismatch gap closed (`None` when there is no gap).
    pub mismatch_fraction: Option<f64>,
    /// The same fraction computed on relative RMS error.
    pub rms_fraction: Option<f64>,
}

/// Cost of the reduce-scatter alone against the whole AllReduce.
#[derive(Clone, Debug)]
pub struct SplitCost {
    pub algorithm: Algorithm,
    pub policy: WirePolicy,
    /// The owned shards, placed together, equal the AllReduce output bit for bit.
    pub bits_match_all_reduce: bool,
    pub reduce_scatter: CostEstimate,
    pub all_reduce: CostEstimate,
}

#[derive(Clone, Debug)]
pub struct PolicyWorld {
    pub world: usize,
    /// The ideal for these partials against the exact dot product.
    pub ideal_vs_exact: Comparison,
    pub runs: Vec<PolicyRun>,
    pub recovery: Vec<Recovery>,
    pub split: Vec<SplitCost>,
}

#[derive(Clone, Debug)]
pub struct PoliciesResult {
    pub config: ExperimentConfig,
    pub elements: usize,
    /// The unsharded FP32 kernel rounded to BF16, against the exact dot product.
    pub single_device_vs_exact: Comparison,
    pub worlds: Vec<PolicyWorld>,
}

impl PolicyWorld {
    pub fn run(&self, policy: WirePolicy, algorithm: Algorithm) -> Option<&PolicyRun> {
        self.runs
            .iter()
            .find(|r| r.policy == policy && r.algorithm == algorithm)
    }
}

pub fn run(config: &ExperimentConfig) -> Result<PoliciesResult, String> {
    run_prepared(config, &prepare(config)?)
}

/// Gap-closing fraction, defined only when the baseline is worse than the target.
fn fraction(baseline: f64, policy: f64, target: f64) -> Option<f64> {
    (baseline > target).then(|| (baseline - policy) / (baseline - target))
}

pub fn run_prepared(config: &ExperimentConfig, prepared: &Prepared) -> Result<PoliciesResult, String> {
    let outputs = config.outputs;
    let exact = prepared.exact_in(OUTPUT);
    let single: Vec<u32> = prepared
        .single_fp32
        .iter()
        .map(|&v| convert(LOCAL_ACCUMULATOR, OUTPUT, v))
        .collect();
    let mut worlds = Vec::new();
    for (world, partials) in &prepared.partials {
        let world = *world;
        let ideal: Vec<u32> = (0..exact.len())
            .map(|e| {
                let mut sum = ExactSum::new();
                for p in partials {
                    sum.add_raw(LOCAL_ACCUMULATOR, p[e]);
                }
                sum.round(OUTPUT)
            })
            .collect();
        let mut runs = Vec::new();
        let mut split = Vec::new();
        for policy in POLICIES {
            let acc = policy.accumulator;
            let inputs: Vec<Vec<u32>> = partials
                .iter()
                .map(|p| p.iter().map(|&v| convert(LOCAL_ACCUMULATOR, acc, v)).collect())
                .collect();
            let input_roundings = if acc.contains(LOCAL_ACCUMULATOR) {
                0
            } else {
                world as u32
            };
            for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                let outcome = all_reduce_with(algorithm, policy, &inputs).map_err(|e| e.to_string())?;
                let output = outcome.results[0].clone();
                let cost = estimate_recorded(&outcome.rounds, world, config.link);
                if SPLIT_ALGORITHMS.contains(&algorithm) {
                    let rs = reduce_scatter(algorithm, policy, &inputs).map_err(|e| e.to_string())?;
                    split.push(SplitCost {
                        algorithm,
                        policy,
                        bits_match_all_reduce: rs.assemble(output.len()) == output,
                        reduce_scatter: estimate_recorded(&rs.rounds, world, config.link),
                        all_reduce: cost.clone(),
                    });
                }
                runs.push(PolicyRun {
                    policy,
                    algorithm,
                    consistent_across_ranks: outcome.consistent(),
                    input_roundings,
                    collective_roundings: outcome.roundings_range().unwrap_or((0, 0)),
                    vs_exact: compare(OUTPUT, &output, &exact),
                    vs_ideal: compare(OUTPUT, &output, &ideal),
                    vs_single_device: compare(OUTPUT, &output, &single),
                    argmax_changes_vs_single_device: argmax_changes(OUTPUT, outputs, &output, &single),
                    cost,
                    output,
                });
            }
        }
        let mut world_result = PolicyWorld {
            world,
            ideal_vs_exact: compare(OUTPUT, &ideal, &exact),
            runs,
            recovery: Vec::new(),
            split,
        };
        let bf16 = POLICIES[0];
        let target = world_result
            .run(bf16, Algorithm::ExactGather)
            .expect("exact-gather supports every world")
            .vs_exact
            .clone();
        let mut recovery = Vec::new();
        for algorithm in Algorithm::ALL
            .into_iter()
            .filter(|a| *a != Algorithm::ExactGather && a.supports(world))
        {
            let baseline = &world_result.run(bf16, algorithm).expect("run exists").vs_exact;
            for policy in &POLICIES[1..] {
                let got = &world_result.run(*policy, algorithm).expect("run exists").vs_exact;
                recovery.push(Recovery {
                    algorithm,
                    policy: *policy,
                    baseline_mismatched: baseline.mismatched,
                    policy_mismatched: got.mismatched,
                    exact_gather_mismatched: target.mismatched,
                    mismatch_fraction: fraction(
                        baseline.mismatched as f64,
                        got.mismatched as f64,
                        target.mismatched as f64,
                    ),
                    rms_fraction: fraction(
                        baseline.relative_rms_error,
                        got.relative_rms_error,
                        target.relative_rms_error,
                    ),
                });
            }
        }
        world_result.recovery = recovery;
        worlds.push(world_result);
    }
    Ok(PoliciesResult {
        config: config.clone(),
        elements: exact.len(),
        single_device_vs_exact: compare(OUTPUT, &single, &exact),
        worlds,
    })
}

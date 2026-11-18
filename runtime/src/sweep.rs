//! R1.1 sensitivity sweep (results/r1.1-sweep.*): the spread across seeds.
//!
//! The R1 problem at reduced size (4 tokens × 256 outputs = 1,024 elements
//! per run) over 8 seeds, three hidden sizes, P up to 64, four wire policies
//! and every applicable schedule plus the shard tree. Every metric is reported
//! as min / median / max over the seeds, so no conclusion rests on one run.
//! The median of an even number of values is the mean of the two middle ones.

use std::thread;

use crate::collectives::{all_reduce_with, Algorithm, WirePolicy};
use crate::format::{ulp_distance, Format};
use crate::ops::convert;
use crate::shard_tree::{self, DEFAULT_SHARDS};
use crate::tensor_parallel::{
    argmax_changes, compare, generate, prepare, split_partials, ExperimentConfig, LOCAL_ACCUMULATOR,
};

/// The wire policies swept, in report order.
pub const POLICIES: [WirePolicy; 4] = [
    WirePolicy::uniform(Format::Fp32),
    WirePolicy::uniform(Format::Bf16),
    WirePolicy::FP32_RS_BF16_AG,
    WirePolicy::BF16_WIRE_FP32_ADD,
];

#[derive(Clone, Debug, PartialEq)]
pub struct SweepConfig {
    pub tokens: usize,
    pub outputs: usize,
    pub hiddens: Vec<usize>,
    pub world_sizes: Vec<usize>,
    pub seeds: Vec<u64>,
    pub shards: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        let base = ExperimentConfig::default().seed;
        SweepConfig {
            tokens: 4,
            outputs: 256,
            hiddens: vec![1024, 4096, 16384],
            world_sizes: vec![2, 3, 4, 8, 16, 32, 64],
            seeds: (0..8).map(|i| base + i).collect(),
            shards: DEFAULT_SHARDS,
        }
    }
}

/// A collective schedule or the shard tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Collective(Algorithm),
    ShardTree,
}

impl Schedule {
    pub const ALL: [Schedule; 6] = [
        Schedule::Collective(Algorithm::Sequential),
        Schedule::Collective(Algorithm::Ring),
        Schedule::Collective(Algorithm::BinomialTree),
        Schedule::Collective(Algorithm::HalvingDoubling),
        Schedule::Collective(Algorithm::ExactGather),
        Schedule::ShardTree,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Schedule::Collective(a) => a.name(),
            Schedule::ShardTree => "shard-tree",
        }
    }

    /// Whether the schedule runs for this world size and policy.
    pub fn applies(self, world: usize, policy: WirePolicy, shards: usize) -> bool {
        match self {
            Schedule::Collective(a) => a.supports(world),
            Schedule::ShardTree => world <= shards && policy.reduce_wire.contains(policy.accumulator),
        }
    }
}

/// Min, median and max of a sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

/// Exact order statistics of `values` (median of an even count: mean of the two middle values).
pub fn stats(values: &[f64]) -> Stats {
    assert!(!values.is_empty(), "stats of an empty sample");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    Stats {
        min: sorted[0],
        median,
        max: sorted[n - 1],
    }
}

/// One run's metrics.
#[derive(Clone, Debug)]
struct RunMetrics {
    hidden: usize,
    world: usize,
    policy: WirePolicy,
    schedule: Schedule,
    mismatched_fraction: f64,
    relative_rms_error: f64,
    max_ulp: f64,
    vs_ring_fraction: f64,
    top1_changes: f64,
}

#[derive(Clone, Debug)]
pub struct SweepRow {
    pub hidden: usize,
    pub world: usize,
    pub policy: WirePolicy,
    pub schedule: Schedule,
    pub runs: usize,
    /// Fraction of elements ≠ the exact result rounded once to the policy's output format.
    pub mismatched_fraction: Stats,
    pub relative_rms_error: Stats,
    pub max_ulp: Stats,
    /// Fraction of elements whose bits differ from ring under the same policy.
    pub vs_ring_fraction: Stats,
    /// Tokens whose top-1 output changes vs the unsharded kernel (of `tokens`).
    pub top1_changes: Stats,
}

#[derive(Clone, Debug)]
pub struct SweepResult {
    pub config: SweepConfig,
    pub elements: usize,
    pub rows: Vec<SweepRow>,
    /// Shard-tree outputs compared across every swept P, per (seed, hidden, policy).
    pub shard_tree_invariance_checks: usize,
    /// All of those comparisons found identical bits.
    pub shard_tree_p_invariant: bool,
}

impl SweepConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.tokens == 0 || self.outputs == 0 || self.seeds.is_empty() || self.hiddens.is_empty() {
            return Err("tokens, outputs, seeds and hidden sizes must be non-empty".into());
        }
        if self.world_sizes.is_empty() {
            return Err("at least one world size is required".into());
        }
        if let Some(&h) = self.hiddens.iter().find(|&&h| h < self.shards) {
            return Err(format!("hidden size {h} is smaller than the {} shards", self.shards));
        }
        Ok(())
    }

    fn experiment(&self, seed: u64, hidden: usize) -> ExperimentConfig {
        ExperimentConfig {
            tokens: self.tokens,
            hidden,
            outputs: self.outputs,
            world_sizes: self.world_sizes.clone(),
            seed,
            ..ExperimentConfig::default()
        }
    }
}

/// One (seed, hidden) problem's runs, shard-tree invariance checks, and whether all passed.
type JobResult = Result<(Vec<RunMetrics>, usize, bool), String>;

/// Every run of one (seed, hidden) problem, and the shard-tree invariance checks.
fn job(config: &SweepConfig, seed: u64, hidden: usize) -> JobResult {
    let experiment = config.experiment(seed, hidden);
    let prepared = prepare(&experiment)?;
    let shard_partials = split_partials(&experiment, &generate(&experiment), config.shards);
    let to_acc = |parts: &[Vec<u32>], acc: Format| -> Vec<Vec<u32>> {
        parts
            .iter()
            .map(|p| p.iter().map(|&v| convert(LOCAL_ACCUMULATOR, acc, v)).collect())
            .collect()
    };
    let mut runs = Vec::new();
    let (mut checks, mut invariant) = (0, true);
    for policy in POLICIES {
        let out = policy.output;
        let exact = prepared.exact_in(out);
        let single: Vec<u32> = prepared
            .single_fp32
            .iter()
            .map(|&v| convert(LOCAL_ACCUMULATOR, out, v))
            .collect();
        let shard_inputs = to_acc(&shard_partials, policy.accumulator);
        let mut shard_reference: Option<Vec<u32>> = None;
        for (world, partials) in &prepared.partials {
            let world = *world;
            let inputs = to_acc(partials, policy.accumulator);
            let mut outputs: Vec<(Schedule, Vec<u32>)> = Vec::new();
            for schedule in Schedule::ALL {
                if !schedule.applies(world, policy, config.shards) {
                    continue;
                }
                let output = match schedule {
                    Schedule::Collective(a) => {
                        let o = all_reduce_with(a, policy, &inputs).map_err(|e| e.to_string())?;
                        o.results[0].clone()
                    }
                    Schedule::ShardTree => {
                        let o = shard_tree::all_reduce(policy, world, &shard_inputs).map_err(|e| e.to_string())?;
                        let bits = o.outcome.results[0].clone();
                        match &shard_reference {
                            None => shard_reference = Some(bits.clone()),
                            Some(r) => {
                                checks += 1;
                                invariant &= *r == bits && o.outcome.consistent();
                            }
                        }
                        bits
                    }
                };
                outputs.push((schedule, output));
            }
            let ring = outputs
                .iter()
                .find(|(s, _)| *s == Schedule::Collective(Algorithm::Ring))
                .map(|(_, o)| o.clone())
                .expect("ring supports every world");
            for (schedule, output) in outputs {
                let vs_exact = compare(out, &output, &exact);
                let differs_from_ring = output.iter().zip(&ring).filter(|(a, b)| a != b).count();
                let max_ulp = output
                    .iter()
                    .zip(&exact)
                    .map(|(&a, &b)| ulp_distance(out, a, b).expect("outputs are finite"))
                    .max()
                    .unwrap_or(0);
                runs.push(RunMetrics {
                    hidden,
                    world,
                    policy,
                    schedule,
                    mismatched_fraction: vs_exact.mismatched as f64 / exact.len() as f64,
                    relative_rms_error: vs_exact.relative_rms_error,
                    max_ulp: max_ulp as f64,
                    vs_ring_fraction: differs_from_ring as f64 / exact.len() as f64,
                    top1_changes: argmax_changes(out, config.outputs, &output, &single) as f64,
                });
            }
        }
    }
    Ok((runs, checks, invariant))
}

pub fn run(config: &SweepConfig) -> Result<SweepResult, String> {
    config.validate()?;
    let jobs: Vec<(u64, usize)> = config
        .seeds
        .iter()
        .flat_map(|&s| config.hiddens.iter().map(move |&h| (s, h)))
        .collect();
    // Jobs run in parallel; results are collected in job order, so the output
    // does not depend on scheduling.
    let threads = thread::available_parallelism().map_or(1, |n| n.get()).min(jobs.len());
    let mut results: Vec<Option<JobResult>> = vec![None; jobs.len()];
    thread::scope(|scope| {
        let jobs = &jobs;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t..jobs.len())
                        .step_by(threads)
                        .map(|i| (i, job(config, jobs[i].0, jobs[i].1)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, r) in h.join().expect("sweep worker panicked") {
                results[i] = Some(r);
            }
        }
    });
    let mut all = Vec::new();
    let (mut checks, mut invariant) = (0, true);
    for r in results {
        let (runs, c, ok) = r.expect("every job ran")?;
        all.extend(runs);
        checks += c;
        invariant &= ok;
    }
    let mut rows = Vec::new();
    for &hidden in &config.hiddens {
        for &world in &config.world_sizes {
            for policy in POLICIES {
                for schedule in Schedule::ALL {
                    let sample: Vec<&RunMetrics> = all
                        .iter()
                        .filter(|m| {
                            m.hidden == hidden && m.world == world && m.policy == policy && m.schedule == schedule
                        })
                        .collect();
                    if sample.is_empty() {
                        continue;
                    }
                    let of = |f: fn(&RunMetrics) -> f64| stats(&sample.iter().map(|m| f(m)).collect::<Vec<_>>());
                    rows.push(SweepRow {
                        hidden,
                        world,
                        policy,
                        schedule,
                        runs: sample.len(),
                        mismatched_fraction: of(|m| m.mismatched_fraction),
                        relative_rms_error: of(|m| m.relative_rms_error),
                        max_ulp: of(|m| m.max_ulp),
                        vs_ring_fraction: of(|m| m.vs_ring_fraction),
                        top1_changes: of(|m| m.top1_changes),
                    });
                }
            }
        }
    }
    Ok(SweepResult {
        config: config.clone(),
        elements: config.tokens * config.outputs,
        rows,
        shard_tree_invariance_checks: checks,
        shard_tree_p_invariant: invariant,
    })
}

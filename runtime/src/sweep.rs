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
    changed_top1, compare, generate, prepare, split_partials, top1_lead_ulps, ExperimentConfig, LOCAL_ACCUMULATOR,
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
    seed: u64,
    hidden: usize,
    world: usize,
    policy: WirePolicy,
    schedule: Schedule,
    mismatched_fraction: f64,
    relative_rms_error: f64,
    max_ulp: f64,
    vs_ring_fraction: f64,
    top1_changes: f64,
    /// The tokens whose top-1 changed, with the unsharded kernel's lead.
    changed: Vec<TokenLead>,
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

/// A token whose top-1 output changed, and how far the unsharded kernel's
/// top-1 output led its runner-up, in ulps of the output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenLead {
    pub token: usize,
    pub lead_ulps: u64,
}

/// One run whose top-1 output changed for at least one token.
#[derive(Clone, Debug, PartialEq)]
pub struct Top1Change {
    pub seed: u64,
    pub hidden: usize,
    pub world: usize,
    pub policy: WirePolicy,
    pub schedule: Schedule,
    /// Tokens whose top-1 output differs from the unsharded kernel.
    pub tokens: Vec<TokenLead>,
}

/// The unsharded kernel's top-1 lead for every token of one problem, in ulps of `format`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProblemLeads {
    pub seed: u64,
    pub hidden: usize,
    pub format: Format,
    pub leads: Vec<u64>,
}

/// The distribution of top-1 leads over every token of every problem, for one output format.
#[derive(Clone, Debug, PartialEq)]
pub struct LeadSummary {
    pub format: Format,
    pub tokens: usize,
    pub leads: Stats,
}

#[derive(Clone, Debug)]
pub struct SweepResult {
    pub config: SweepConfig,
    pub elements: usize,
    pub rows: Vec<SweepRow>,
    /// Every run that changed a top-1 output, ordered by hidden, P, policy, schedule and seed.
    pub top1_changes: Vec<Top1Change>,
    /// Top-1 leads of the unsharded kernel, per problem and output format (in job order).
    pub reference_leads: Vec<ProblemLeads>,
    /// Their distribution per output format.
    pub lead_summary: Vec<LeadSummary>,
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

/// What one (seed, hidden) problem contributes.
struct JobOutput {
    runs: Vec<RunMetrics>,
    /// Shard-tree comparisons across world sizes, and whether all found identical bits.
    checks: usize,
    invariant: bool,
    leads: Vec<ProblemLeads>,
}

type JobResult = Result<JobOutput, String>;

/// Output formats of the swept policies, in first-use order.
fn output_formats() -> Vec<Format> {
    let mut formats = Vec::new();
    for p in POLICIES {
        if !formats.contains(&p.output) {
            formats.push(p.output);
        }
    }
    formats
}

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
    let single_in = |format: Format| -> Vec<u32> {
        prepared
            .single_fp32
            .iter()
            .map(|&v| convert(LOCAL_ACCUMULATOR, format, v))
            .collect()
    };
    let lead_of = |format: Format, single: &[u32], token: usize| -> u64 {
        top1_lead_ulps(format, &single[token * config.outputs..(token + 1) * config.outputs])
            .expect("at least two outputs and finite values")
    };
    let leads: Vec<ProblemLeads> = output_formats()
        .into_iter()
        .map(|format| {
            let single = single_in(format);
            ProblemLeads {
                seed,
                hidden,
                format,
                leads: (0..config.tokens).map(|m| lead_of(format, &single, m)).collect(),
            }
        })
        .collect();
    for policy in POLICIES {
        let out = policy.output;
        let exact = prepared.exact_in(out);
        let single = single_in(out);
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
                let changed: Vec<TokenLead> = changed_top1(out, config.outputs, &output, &single)
                    .into_iter()
                    .map(|token| TokenLead {
                        token,
                        lead_ulps: lead_of(out, &single, token),
                    })
                    .collect();
                runs.push(RunMetrics {
                    seed,
                    hidden,
                    world,
                    policy,
                    schedule,
                    mismatched_fraction: vs_exact.mismatched as f64 / exact.len() as f64,
                    relative_rms_error: vs_exact.relative_rms_error,
                    max_ulp: max_ulp as f64,
                    vs_ring_fraction: differs_from_ring as f64 / exact.len() as f64,
                    top1_changes: changed.len() as f64,
                    changed,
                });
            }
        }
    }
    Ok(JobOutput {
        runs,
        checks,
        invariant,
        leads,
    })
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
    let mut results: Vec<Option<JobResult>> = (0..jobs.len()).map(|_| None).collect();
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
    let mut reference_leads = Vec::new();
    for r in results {
        let job = r.expect("every job ran")?;
        all.extend(job.runs);
        checks += job.checks;
        invariant &= job.invariant;
        reference_leads.extend(job.leads);
    }
    let lead_summary = output_formats()
        .into_iter()
        .map(|format| {
            let leads: Vec<f64> = reference_leads
                .iter()
                .filter(|p| p.format == format)
                .flat_map(|p| p.leads.iter().map(|&l| l as f64))
                .collect();
            LeadSummary {
                format,
                tokens: leads.len(),
                leads: stats(&leads),
            }
        })
        .collect();
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
    let mut top1_changes = Vec::new();
    for row in &rows {
        for &seed in &config.seeds {
            if let Some(m) = all.iter().find(|m| {
                m.seed == seed
                    && m.hidden == row.hidden
                    && m.world == row.world
                    && m.policy == row.policy
                    && m.schedule == row.schedule
            }) {
                if !m.changed.is_empty() {
                    top1_changes.push(Top1Change {
                        seed,
                        hidden: m.hidden,
                        world: m.world,
                        policy: m.policy,
                        schedule: m.schedule,
                        tokens: m.changed.clone(),
                    });
                }
            }
        }
    }
    Ok(SweepResult {
        config: config.clone(),
        elements: config.tokens * config.outputs,
        rows,
        top1_changes,
        reference_leads,
        lead_summary,
        shard_tree_invariance_checks: checks,
        shard_tree_p_invariant: invariant,
    })
}

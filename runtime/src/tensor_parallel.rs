//! Experiment: a tensor-parallel linear layer whose partial sums are combined by AllReduce.
//!
//! `Y = X · W` with `X` of shape tokens × hidden and `W` of shape hidden ×
//! outputs, both BF16. The hidden (reduction) dimension is split into
//! contiguous slices across P virtual devices, as in row-parallel linear
//! layers. Each device computes its partial `Y_r` with an FP32 fused chain
//! (exact products, one FP32 rounding per term), converts it to the wire
//! format, and the devices AllReduce the partials.
//!
//! Each AllReduce result is compared with two references:
//! * **exact** — the exact full dot product rounded once to the wire format;
//! * **single-device** — the same FP32 fused chain over the whole hidden
//!   dimension on one device, converted once to the wire format. This is what
//!   a non-sharded run of this kernel order would produce.

use std::thread;

use crate::collectives::{all_reduce, chunk_range, Algorithm};
use crate::cost::{estimate, CostEstimate, LinkModel};
use crate::exact::ExactSum;
use crate::format::{to_f64, ulp_distance, Format};
use crate::ops::{convert, exact_product, from_f64, fused_chain_dot};
use crate::rng::SplitMix64;

pub const INPUT_FORMAT: Format = Format::Bf16;
pub const LOCAL_ACCUMULATOR: Format = Format::Fp32;

#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub tokens: usize,
    pub hidden: usize,
    pub outputs: usize,
    pub world_sizes: Vec<usize>,
    pub wire_formats: Vec<Format>,
    pub seed: u64,
    pub link: LinkModel,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig {
            tokens: 8,
            hidden: 4096,
            outputs: 512,
            world_sizes: vec![2, 3, 4, 8],
            wire_formats: vec![Format::Fp32, Format::Bf16],
            seed: 0x00f9_91a7,
            link: LinkModel::default(),
        }
    }
}

impl ExperimentConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.tokens == 0 || self.hidden == 0 || self.outputs == 0 {
            return Err("tokens, hidden and outputs must be positive".into());
        }
        if self.world_sizes.is_empty() || self.world_sizes.iter().any(|&p| p < 2 || p > self.hidden) {
            return Err("world sizes must be between 2 and the hidden dimension".into());
        }
        if self.wire_formats.is_empty() {
            return Err("at least one wire format is required".into());
        }
        if !(self.link.latency_us >= 0.0 && self.link.bandwidth_gbps > 0.0) {
            return Err("latency must be ≥ 0 and bandwidth > 0".into());
        }
        Ok(())
    }
}

/// Synthetic BF16 operands. `weights` is stored column-major (one column per output).
pub struct Problem {
    pub activations: Vec<u32>,
    pub weights: Vec<u32>,
}

pub fn generate(config: &ExperimentConfig) -> Problem {
    let mut rng = SplitMix64::new(config.seed);
    let activations = (0..config.tokens * config.hidden)
        .map(|_| from_f64(INPUT_FORMAT, rng.next_normalish()))
        .collect();
    let scale = 1.0 / (config.hidden as f64).sqrt();
    let weights = (0..config.outputs * config.hidden)
        .map(|_| from_f64(INPUT_FORMAT, rng.next_normalish() * scale))
        .collect();
    Problem { activations, weights }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub mismatched: usize,
    pub max_ulp: u64,
    pub mean_ulp: f64,
    /// ‖got − expected‖₂ / ‖expected‖₂, evaluated in binary64 (a display estimate).
    pub relative_rms_error: f64,
}

pub fn compare(format: Format, got: &[u32], expected: &[u32]) -> Comparison {
    assert_eq!(got.len(), expected.len());
    let mut mismatched = 0;
    let mut max_ulp = 0;
    let mut total: u128 = 0;
    let (mut err2, mut ref2) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(expected) {
        let (x, y) = (to_f64(format, a), to_f64(format, b));
        err2 += (x - y) * (x - y);
        ref2 += y * y;
        let d = ulp_distance(format, a, b).expect("experiment outputs are never NaN");
        if a != b {
            mismatched += 1;
        }
        max_ulp = max_ulp.max(d);
        total += d as u128;
    }
    let relative_rms_error = if ref2 > 0.0 { (err2 / ref2).sqrt() } else { 0.0 };
    Comparison {
        mismatched,
        max_ulp,
        mean_ulp: total as f64 / got.len().max(1) as f64,
        relative_rms_error,
    }
}

/// Index of the largest output in `row`. Ties resolve to the lowest index.
pub fn argmax(format: Format, row: &[u32]) -> usize {
    let mut best = 0;
    for (i, &raw) in row.iter().enumerate() {
        if to_f64(format, raw) > to_f64(format, row[best]) {
            best = i;
        }
    }
    best
}

/// How far the top-1 output of `row` leads the runner-up, in ulps of `format`
/// (representable values between them; 0 is an exact tie). `None` for fewer
/// than two outputs.
pub fn top1_lead_ulps(format: Format, row: &[u32]) -> Option<u64> {
    let top = argmax(format, row);
    let runner_up = (0..row.len())
        .filter(|&i| i != top)
        .max_by(|&a, &b| to_f64(format, row[a]).total_cmp(&to_f64(format, row[b])))?;
    ulp_distance(format, row[top], row[runner_up])
}

/// Rows (tokens) whose top-1 output index differs.
pub fn changed_top1(format: Format, outputs: usize, got: &[u32], expected: &[u32]) -> Vec<usize> {
    got.chunks(outputs)
        .zip(expected.chunks(outputs))
        .enumerate()
        .filter(|(_, (a, b))| argmax(format, a) != argmax(format, b))
        .map(|(m, _)| m)
        .collect()
}

/// Number of rows (tokens) whose top-1 output index differs. Ties resolve to the lowest index.
pub fn argmax_changes(format: Format, outputs: usize, got: &[u32], expected: &[u32]) -> usize {
    changed_top1(format, outputs, got, expected).len()
}

#[derive(Clone, Debug)]
pub struct AlgorithmResult {
    pub algorithm: Algorithm,
    pub consistent_across_ranks: bool,
    pub roundings_per_element: usize,
    pub vs_exact: Comparison,
    pub vs_single_device: Comparison,
    pub argmax_changes_vs_single_device: usize,
    pub cost: CostEstimate,
    pub output: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct WorldResult {
    pub world: usize,
    pub wire: Format,
    pub algorithms: Vec<AlgorithmResult>,
    /// (a, b, elements whose bits differ between algorithm a and b)
    pub pairwise_mismatches: Vec<(Algorithm, Algorithm, usize)>,
}

#[derive(Clone, Debug)]
pub struct WireBaseline {
    pub wire: Format,
    pub single_device_vs_exact: Comparison,
}

#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub config: ExperimentConfig,
    pub elements: usize,
    pub baselines: Vec<WireBaseline>,
    pub worlds: Vec<WorldResult>,
}

/// Evaluate `f(token)` for every token in parallel; output is token-ordered, so
/// results do not depend on thread scheduling.
fn per_token<T, F>(tokens: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let threads = thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(tokens)
        .max(1);
    let mut rows: Vec<Option<T>> = (0..tokens).map(|_| None).collect();
    thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = (0..threads)
            .map(|t| scope.spawn(move || (t..tokens).step_by(threads).map(|m| (m, f(m))).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            for (m, row) in h.join().expect("worker panicked") {
                rows[m] = Some(row);
            }
        }
    });
    rows.into_iter().map(|r| r.expect("every token computed")).collect()
}

/// The expensive part of an experiment on one configuration, shared by every
/// analysis of it: the exact dot products, the unsharded FP32 kernel, and the
/// FP32 partial sums for every world size. All are token-major
/// (`tokens × outputs` elements).
pub struct Prepared {
    /// Exact dot product of every output element (round with [`ExactSum::round`]).
    pub exact: Vec<ExactSum>,
    /// The FP32 fused-chain kernel run over the whole hidden dimension.
    pub single_fp32: Vec<u32>,
    /// `(world, partials)`, in `config.world_sizes` order; `partials[r]` is
    /// rank `r`'s FP32 partial over its contiguous slice of the hidden dimension.
    pub partials: Vec<(usize, Vec<Vec<u32>>)>,
}

impl Prepared {
    /// The exact dot products rounded once to `format`.
    pub fn exact_in(&self, format: Format) -> Vec<u32> {
        self.exact.iter().map(|s| s.round(format)).collect()
    }
}

pub fn prepare(config: &ExperimentConfig) -> Result<Prepared, String> {
    config.validate()?;
    let problem = generate(config);
    let (tokens, hidden, outputs) = (config.tokens, config.hidden, config.outputs);
    let x_row = |m: usize| &problem.activations[m * hidden..(m + 1) * hidden];
    let w_col = |n: usize| &problem.weights[n * hidden..(n + 1) * hidden];

    // One ExactSum per element (rounded later to whatever format is needed).
    let exact: Vec<ExactSum> = per_token(tokens, |m| {
        (0..outputs)
            .map(|n| {
                let mut sum = ExactSum::new();
                for (&a, &b) in x_row(m).iter().zip(w_col(n)) {
                    sum.add_value(exact_product(INPUT_FORMAT, a, INPUT_FORMAT, b));
                }
                sum
            })
            .collect::<Vec<_>>()
    })
    .concat();
    let single_fp32: Vec<u32> = per_token(tokens, |m| {
        (0..outputs)
            .map(|n| fused_chain_dot(LOCAL_ACCUMULATOR, INPUT_FORMAT, x_row(m), w_col(n)))
            .collect::<Vec<u32>>()
    })
    .concat();
    let partials = config
        .world_sizes
        .iter()
        .map(|&world| (world, split_partials(config, &problem, world)))
        .collect();
    Ok(Prepared {
        exact,
        single_fp32,
        partials,
    })
}

/// FP32 partials of every output element when the hidden dimension is split
/// into `parts` contiguous slices: `partials[i]` is the fused-chain kernel over
/// slice `i` (token-major, `tokens × outputs` elements).
pub fn split_partials(config: &ExperimentConfig, problem: &Problem, parts: usize) -> Vec<Vec<u32>> {
    let (tokens, hidden, outputs) = (config.tokens, config.hidden, config.outputs);
    let x_row = |m: usize| &problem.activations[m * hidden..(m + 1) * hidden];
    let w_col = |n: usize| &problem.weights[n * hidden..(n + 1) * hidden];
    (0..parts)
        .map(|i| {
            let slice = chunk_range(hidden, parts, i);
            per_token(tokens, |m| {
                (0..outputs)
                    .map(|n| {
                        fused_chain_dot(
                            LOCAL_ACCUMULATOR,
                            INPUT_FORMAT,
                            &x_row(m)[slice.clone()],
                            &w_col(n)[slice.clone()],
                        )
                    })
                    .collect::<Vec<u32>>()
            })
            .concat()
        })
        .collect()
}

pub fn run(config: &ExperimentConfig) -> Result<ExperimentResult, String> {
    run_prepared(config, &prepare(config)?)
}

/// The R1 experiment (results/tp-allreduce.*) on prepared data.
pub fn run_prepared(config: &ExperimentConfig, prepared: &Prepared) -> Result<ExperimentResult, String> {
    let (tokens, outputs) = (config.tokens, config.outputs);
    let mut baselines = Vec::new();
    let mut references = Vec::new();
    for &wire in &config.wire_formats {
        let exact = prepared.exact_in(wire);
        let single: Vec<u32> = prepared
            .single_fp32
            .iter()
            .map(|&v| convert(LOCAL_ACCUMULATOR, wire, v))
            .collect();
        baselines.push(WireBaseline {
            wire,
            single_device_vs_exact: compare(wire, &single, &exact),
        });
        references.push((exact, single));
    }

    let mut worlds = Vec::new();
    for (world, partials) in &prepared.partials {
        let world = *world;
        for (wi, &wire) in config.wire_formats.iter().enumerate() {
            let (exact, single) = &references[wi];
            let inputs: Vec<Vec<u32>> = partials
                .iter()
                .map(|p| p.iter().map(|&v| convert(LOCAL_ACCUMULATOR, wire, v)).collect())
                .collect();
            let mut algorithms = Vec::new();
            for algorithm in Algorithm::ALL {
                if !algorithm.supports(world) {
                    continue;
                }
                let outcome = all_reduce(algorithm, wire, &inputs).map_err(|e| e.to_string())?;
                let output = outcome.results[0].clone();
                algorithms.push(AlgorithmResult {
                    algorithm,
                    consistent_across_ranks: outcome.consistent(),
                    roundings_per_element: algorithm.roundings_per_element(world),
                    vs_exact: compare(wire, &output, exact),
                    vs_single_device: compare(wire, &output, single),
                    argmax_changes_vs_single_device: argmax_changes(wire, outputs, &output, single),
                    cost: estimate(&outcome.rounds, world, wire.bytes(), config.link),
                    output,
                });
            }
            let mut pairwise_mismatches = Vec::new();
            for i in 0..algorithms.len() {
                for j in i + 1..algorithms.len() {
                    let d = compare(wire, &algorithms[i].output, &algorithms[j].output).mismatched;
                    pairwise_mismatches.push((algorithms[i].algorithm, algorithms[j].algorithm, d));
                }
            }
            worlds.push(WorldResult {
                world,
                wire,
                algorithms,
                pairwise_mismatches,
            });
        }
    }

    Ok(ExperimentResult {
        config: config.clone(),
        elements: tokens * outputs,
        baselines,
        worlds,
    })
}

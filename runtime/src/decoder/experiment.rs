//! The recorded R3 experiment (results/r3-decode.*).
//!
//! 1. **Divergence grid.** The workload served in one batch ("together")
//!    under every P ∈ {2, 4, 8}, AllReduce schedule and wire policy, plus
//!    the shard tree at P = 1, against the P = 1 baseline with the same
//!    policy (P = 1 runs no collective: each projection's binary32 sum is
//!    rounded once to the policy's output format). Repeated at two logit
//!    scales.
//! 2. **Margins.** For every token whose context still matches the
//!    baseline, the baseline's top-2 margin, whether the run picked a
//!    different token there, and how far the run's logits moved.
//! 3. **Batch invariance.** At P = 4, every schedule and policy serves the
//!    workload in each of seven batch compositions; each prompt's logits are
//!    compared bit for bit with the same prompt served alone. A P = 1
//!    control runs the same variants.
//!
//! Every run checks the engine's scheduler, token-stream and KV invariants
//! and the tensor invariant after every step.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::config::DecoderConfig;
use super::decode::Generation;
use super::forward::{Model, RowReduce};
use super::parallel::{ReduceStats, Schedule, TensorParallel, Unsharded};
use super::serve::serve;
use super::variants::{run_variant, specs, Variant, VariantRun, Workload, TIMER};
use super::weights::Weights;
use crate::collectives::{Algorithm, WirePolicy};
use crate::format::Format;
use crate::shard_tree::DEFAULT_SHARDS;

#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub model: DecoderConfig,
    pub workload: Workload,
    pub worlds: Vec<usize>,
    pub schedules: Vec<Schedule>,
    pub policies: Vec<WirePolicy>,
    /// Logit scales; the first is the model's default.
    pub scales: Vec<f32>,
    pub matrix_world: usize,
    /// Upper edges of the margin bins, in ulps of the top logit (the last bin is open).
    pub margin_bins: Vec<f64>,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        let mut schedules: Vec<Schedule> = Algorithm::ALL.into_iter().map(Schedule::Collective).collect();
        schedules.push(Schedule::ShardTree { shards: DEFAULT_SHARDS });
        DecodeConfig {
            model: DecoderConfig::TINY,
            workload: Workload::R3,
            worlds: vec![2, 4, 8],
            schedules,
            policies: vec![WirePolicy::uniform(Format::Bf16), WirePolicy::FP32_RS_BF16_AG],
            scales: vec![1.0, 8.0],
            matrix_world: 4,
            margin_bins: (12..=22).step_by(2).map(|k| f64::from(1u32 << k)).collect(),
        }
    }
}

/// Unit in the last place of a binary32 value, as a binary64.
pub fn ulp_f32(x: f32) -> f64 {
    let field = (x.to_bits() >> 23) & 0xff;
    let exponent = if field == 0 { -149 } else { field as i32 - 127 - 23 };
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// A token at which a run's context still matched the baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtRisk {
    /// Baseline top-2 margin, absolute and in ulps of the top logit.
    pub margin: f64,
    pub margin_ulps: f64,
    /// The run chose a different token here.
    pub flipped: bool,
    /// max over the vocabulary of |run logit − baseline logit|, in ulps of
    /// the baseline's top logit.
    pub perturbation_ulps: f64,
}

/// One divergence-grid run compared with its baseline.
#[derive(Clone, Debug)]
pub struct GridRun {
    pub world: usize,
    pub schedule: Schedule,
    pub policy: WirePolicy,
    pub scale: f32,
    /// Per prompt: index of the first generated token that differs from the baseline.
    pub first_divergence: Vec<Option<usize>>,
    /// Prompts whose logits match the baseline bit for bit at every step.
    pub logits_identical: usize,
    pub at_risk: Vec<AtRisk>,
    pub reduce: ReduceStats,
    /// Every generated token, per prompt (to compare across scales).
    pub tokens: Vec<Vec<u32>>,
}

impl GridRun {
    pub fn diverged(&self) -> usize {
        self.first_divergence.iter().filter(|d| d.is_some()).count()
    }

    /// Prompts whose first `t` generated tokens all match the baseline.
    pub fn surviving(&self, t: usize) -> usize {
        self.first_divergence
            .iter()
            .filter(|d| d.map_or(true, |i| i >= t))
            .count()
    }
}

/// The baseline of one (policy, scale): P = 1, no collective.
#[derive(Clone, Debug)]
pub struct Baseline {
    pub policy: WirePolicy,
    pub scale: f32,
    pub generations: Vec<Generation>,
}

/// One row of the batch-invariance matrix.
#[derive(Clone, Debug)]
pub struct MatrixRow {
    pub world: usize,
    pub schedule: Schedule,
    pub policy: WirePolicy,
    /// `runs[0]` is `Alone`; the rest follow [`Variant::ALL`].
    pub runs: Vec<VariantRun>,
}

impl MatrixRow {
    /// Prompts of `runs[i]` whose logits (at any step) and whose tokens differ from `Alone`.
    pub fn differing(&self, i: usize) -> (usize, usize) {
        let alone = &self.runs[0].generations;
        let r = &self.runs[i].generations;
        let logits = r
            .iter()
            .zip(alone)
            .filter(|(a, b)| a.logits_hash != b.logits_hash)
            .count();
        let tokens = r.iter().zip(alone).filter(|(a, b)| a.tokens != b.tokens).count();
        (logits, tokens)
    }

    /// The prediction: only the ring's bits may depend on batch composition.
    pub fn predicted_invariant(&self) -> bool {
        self.world == 1 || self.schedule != Schedule::Collective(Algorithm::Ring)
    }
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub config: DecodeConfig,
    pub weights_checksum: u64,
    pub baselines: Vec<Baseline>,
    pub grid: Vec<GridRun>,
    pub matrix: Vec<MatrixRow>,
}

impl DecodeResult {
    pub fn baseline(&self, policy: WirePolicy, scale: f32) -> &Baseline {
        self.baselines
            .iter()
            .find(|b| b.policy == policy && b.scale == scale)
            .expect("every policy and scale has a baseline")
    }

    /// Whether the batch-invariance prediction held exactly: every predicted
    /// row bit-identical in every variant, and the ring varying somewhere.
    pub fn prediction_held(&self) -> bool {
        let invariant_ok = self
            .matrix
            .iter()
            .filter(|r| r.predicted_invariant())
            .all(|r| (1..r.runs.len()).all(|i| r.differing(i).0 == 0));
        let ring_varies = self
            .matrix
            .iter()
            .filter(|r| !r.predicted_invariant())
            .any(|r| (1..r.runs.len()).any(|i| r.differing(i).0 > 0));
        invariant_ok && ring_varies
    }
}

/// Run `f` over `items` on the available cores; results keep item order.
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<R>>> = Mutex::new(items.iter().map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                slots.lock().expect("no worker panicked")[i] = Some(r);
            });
        }
    });
    slots
        .into_inner()
        .expect("no worker panicked")
        .into_iter()
        .map(|r| r.expect("every item ran"))
        .collect()
}

/// Serve the workload in one batch and keep every logits row.
fn together(
    model: &mut Model,
    reduce: &mut dyn RowReduce,
    w: &Workload,
    prompts: &[Vec<u32>],
) -> Result<Vec<Generation>, String> {
    let spec = &specs(w, prompts, Variant::Together, model.config.layers)[0];
    serve(model, reduce, spec, &TIMER, true)
        .map(|s| s.generations)
        .map_err(|e| e.to_string())
}

fn compare(run: &[Generation], base: &[Generation]) -> (Vec<Option<usize>>, usize, Vec<AtRisk>) {
    let mut first = Vec::new();
    let mut identical = 0;
    let mut at_risk = Vec::new();
    for (r, b) in run.iter().zip(base) {
        let d = r.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
        first.push(d);
        identical += usize::from(r.logits_hash == b.logits_hash);
        let last = d.unwrap_or(b.tokens.len() - 1);
        for t in 0..=last {
            let top = b.top[t];
            let ulp = ulp_f32(top.top);
            let moved = r.logits[t]
                .iter()
                .zip(&b.logits[t])
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
                .fold(0.0, f64::max);
            at_risk.push(AtRisk {
                margin: top.margin(),
                margin_ulps: top.margin() / ulp,
                flipped: Some(t) == d,
                perturbation_ulps: moved / ulp,
            });
        }
    }
    (first, identical, at_risk)
}

pub fn run(config: &DecodeConfig) -> Result<DecodeResult, String> {
    config.model.validate()?;
    let weights = Weights::generate(&config.model);
    let prompts = config.workload.prompts(config.model.vocab);
    let model_at = |scale: f32| {
        Model::with_weights(
            DecoderConfig {
                logit_scale: scale,
                ..config.model
            },
            weights.clone(),
        )
    };
    // Baselines: P = 1, every policy and scale.
    let pairs: Vec<(WirePolicy, f32)> = config
        .scales
        .iter()
        .flat_map(|&s| config.policies.iter().map(move |&p| (p, s)))
        .collect();
    let baselines = parallel(&pairs, |&(policy, scale)| -> Result<Baseline, String> {
        let mut m = model_at(scale);
        let generations = together(&mut m, &mut Unsharded { policy }, &config.workload, &prompts)?;
        Ok(Baseline {
            policy,
            scale,
            generations,
        })
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    // The divergence grid.
    let mut jobs: Vec<(usize, Schedule, WirePolicy, f32)> = Vec::new();
    for &scale in &config.scales {
        for &policy in &config.policies {
            for &schedule in &config.schedules {
                let shard_tree = matches!(schedule, Schedule::ShardTree { .. });
                let worlds = std::iter::once(1)
                    .filter(|_| shard_tree)
                    .chain(config.worlds.iter().copied());
                for world in worlds {
                    if TensorParallel::new(world, schedule, policy).is_ok() && config.model.supports_world(world) {
                        jobs.push((world, schedule, policy, scale));
                    }
                }
            }
        }
    }
    let grid = parallel(&jobs, |&(world, schedule, policy, scale)| -> Result<GridRun, String> {
        let mut m = model_at(scale);
        let mut tp = TensorParallel::new(world, schedule, policy)?;
        let generations = together(&mut m, &mut tp, &config.workload, &prompts)?;
        let base = &baselines
            .iter()
            .find(|b| b.policy == policy && b.scale == scale)
            .expect("baseline")
            .generations;
        let (first_divergence, logits_identical, at_risk) = compare(&generations, base);
        Ok(GridRun {
            world,
            schedule,
            policy,
            scale,
            first_divergence,
            logits_identical,
            at_risk,
            reduce: tp.stats,
            tokens: generations.into_iter().map(|g| g.tokens).collect(),
        })
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    // The batch-invariance matrix (default scale).
    let mut rows: Vec<(usize, Schedule, WirePolicy)> = config
        .policies
        .iter()
        .map(|&p| (1, Schedule::Collective(Algorithm::Ring), p))
        .collect();
    for &policy in &config.policies {
        for &schedule in &config.schedules {
            rows.push((config.matrix_world, schedule, policy));
        }
    }
    let matrix = parallel(&rows, |&(world, schedule, policy)| -> Result<MatrixRow, String> {
        let mut m = model_at(config.scales[0]);
        let runs = Variant::ALL
            .iter()
            .map(|&v| {
                let mut make = || -> Box<dyn RowReduce> {
                    Box::new(TensorParallel::new(world, schedule, policy).expect("valid row"))
                };
                run_variant(&mut m, &mut make, &config.workload, &prompts, v).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MatrixRow {
            world,
            schedule,
            policy,
            runs,
        })
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    Ok(DecodeResult {
        config: config.clone(),
        weights_checksum: weights.checksum(),
        baselines,
        grid,
        matrix,
    })
}

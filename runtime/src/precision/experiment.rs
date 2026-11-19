//! The recorded R4 experiment (results/r4-precision.*).
//!
//! The reference is the all-FP32 policy. It decodes each prompt greedily;
//! every policy is then run teacher-forced on those same tokens (so every
//! policy sees the same contexts) and compared token by token with the
//! reference, and also decodes greedily on its own to count divergence.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::metrics::{compare, Drift};
use super::model::{MasterWeights, PolicyModel};
use super::policy::{Level, Policy, Site};
use crate::decoder::config::DecoderConfig;
use crate::decoder::decode::prompts;

#[derive(Clone, Debug)]
pub struct PrecisionConfig {
    pub model: DecoderConfig,
    pub prompt_seed: u64,
    pub prompts: usize,
    pub prompt_len: usize,
    pub continuation: usize,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        PrecisionConfig {
            model: DecoderConfig::TINY,
            prompt_seed: 0x0004_9e0d_0001,
            prompts: 8,
            prompt_len: 16,
            continuation: 32,
        }
    }
}

/// What a policy is, for grouping in the report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Uniform(Level),
    OneSite(Site, Level),
    Mixed,
}

impl PrecisionConfig {
    /// The policies: every uniform level, every site alone at every lower
    /// level (the KV cache at its two storage levels only), and two mixed policies.
    pub fn policies(&self) -> Vec<(Kind, Policy)> {
        let layers = self.model.layers;
        let mut out: Vec<(Kind, Policy)> = Level::ALL
            .iter()
            .map(|&l| (Kind::Uniform(l), Policy::uniform(layers, l)))
            .collect();
        for site in Site::all(layers) {
            for level in Level::LOW {
                // The KV cache only stores values: its -acc levels equal its storage levels.
                if site == Site::KvCache && level.accumulator() != Level::Fp32.accumulator() {
                    continue;
                }
                out.push((Kind::OneSite(site, level), Policy::only(layers, site, level)));
            }
        }
        let mut a = Policy::uniform(layers, Level::Bf16);
        for l in 0..layers {
            a.set(Site::Mlp(l), Level::Bf16Acc);
        }
        a.name = "BF16 storage, BF16 accumulation in the MLPs only".into();
        out.push((Kind::Mixed, a));
        let mut b = Policy::uniform(layers, Level::Fp16Acc);
        b.set(Site::Head, Level::Fp16);
        b.name = "FP16 storage, FP16 accumulation except the head".into();
        out.push((Kind::Mixed, b));
        out
    }
}

#[derive(Clone, Debug)]
pub struct PolicyRun {
    pub kind: Kind,
    pub policy: Policy,
    /// Teacher-forced comparison over every continuation token.
    pub drift: Drift,
    /// Free-running greedy decoding: first token differing from the
    /// reference's continuation, per prompt.
    pub first_divergence: Vec<Option<usize>>,
}

impl PolicyRun {
    pub fn diverged(&self) -> usize {
        self.first_divergence.iter().filter(|d| d.is_some()).count()
    }
}

#[derive(Clone, Debug)]
pub struct PrecisionResult {
    pub config: PrecisionConfig,
    /// The reference's own perplexity of its continuation.
    pub reference_perplexity: f64,
    pub runs: Vec<PolicyRun>,
}

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

/// Positions whose logits predict the continuation: `prompt_len − 1 … len − 2`.
fn scored(rows: &[Vec<f32>], prompt_len: usize) -> &[Vec<f32>] {
    &rows[prompt_len - 1..rows.len() - 1]
}

pub fn run(config: &PrecisionConfig) -> Result<PrecisionResult, String> {
    let c = config.model;
    c.validate()?;
    let w = MasterWeights::generate(&c);
    let ps = prompts(config.prompt_seed, config.prompts, config.prompt_len, c.vocab);
    let mut reference = PolicyModel::new(c, &w, Policy::uniform(c.layers, Level::Fp32));
    let continuations: Vec<Vec<u32>> = ps.iter().map(|p| reference.greedy(p, config.continuation)).collect();
    let sequences: Vec<Vec<u32>> = ps
        .iter()
        .zip(&continuations)
        .map(|(p, t)| [p.clone(), t.clone()].concat())
        .collect();
    let ref_logits: Vec<Vec<Vec<f32>>> = sequences.iter().map(|s| reference.teacher_forced(s)).collect();
    let merge = |parts: Vec<Drift>| -> Drift {
        let mut d = Drift::default();
        let (mut rms, mut kl, mut nll) = (0.0, 0.0, 0.0);
        for p in parts {
            d.tokens += p.tokens;
            d.scored += p.scored;
            d.nonfinite += p.nonfinite;
            d.top1_agree += p.top1_agree;
            d.max_abs = d.max_abs.max(p.max_abs);
            rms += p.mean_rms * p.scored as f64;
            kl += p.mean_kl * p.scored as f64;
            nll += p.nll * p.scored as f64;
        }
        let n = d.scored.max(1) as f64;
        d.mean_rms = rms / n;
        d.mean_kl = kl / n;
        d.nll = nll / n;
        d
    };
    let reference_drift = merge(
        sequences
            .iter()
            .zip(&ref_logits)
            .map(|(s, r)| {
                compare(
                    scored(r, config.prompt_len),
                    scored(r, config.prompt_len),
                    &s[config.prompt_len..],
                )
            })
            .collect(),
    );
    let policies = config.policies();
    let runs = parallel(&policies, |(kind, policy)| {
        let mut m = PolicyModel::new(c, &w, policy.clone());
        let drift = merge(
            sequences
                .iter()
                .zip(&ref_logits)
                .map(|(s, r)| {
                    let l = m.teacher_forced(s);
                    compare(
                        scored(r, config.prompt_len),
                        scored(&l, config.prompt_len),
                        &s[config.prompt_len..],
                    )
                })
                .collect(),
        );
        let first_divergence = ps
            .iter()
            .zip(&continuations)
            .map(|(p, want)| {
                let got = m.greedy(p, config.continuation);
                got.iter().zip(want).position(|(a, b)| a != b)
            })
            .collect();
        PolicyRun {
            kind: *kind,
            policy: policy.clone(),
            drift,
            first_divergence,
        }
    });
    Ok(PrecisionResult {
        config: config.clone(),
        reference_perplexity: reference_drift.perplexity(),
        runs,
    })
}

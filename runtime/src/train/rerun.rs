//! R4 completion: R3's divergence and batch-invariance experiment and R4's
//! precision study rerun on the trained model, each beside the original
//! experiment's random-weight model on the same copy-task prompts
//! (results/r3-trained.*, results/r4-trained.*).
//!
//! Both models have R3's architecture and dimensions, so the comparison
//! isolates the weights: R3's random BF16 weights (`Weights::generate(TINY)`)
//! or R4's random FP32 master weights (`MasterWeights::generate(TINY)`),
//! against the trained weights (rounded once to BF16 for R3).

use std::sync::Arc;

use super::task::{CopyTask, BOS, SEP};
use crate::decoder::config::DecoderConfig;
use crate::decoder::experiment::{run as run_decode, DecodeConfig, DecodeResult};
use crate::decoder::variants::Workload;
use crate::decoder::weights::Weights;
use crate::precision::experiment::{run as run_precision, PrecisionConfig, PrecisionResult};
use crate::precision::model::{MasterWeights, PolicyModel};
use crate::precision::policy::{Level, Policy};
use crate::rng::SplitMix64;

/// The copy task the model was trained on.
pub const TASK: CopyTask = CopyTask {
    len: 24,
    vocab: DecoderConfig::TINY.vocab,
};

/// R3's workload shape with copy prompts: 4 groups of 8 prompts, each
/// group sharing its first 8 tokens (`BOS` and 7 symbols), then 17 own
/// symbols and `SEP` (26 tokens), 24 greedy tokens each (the copy).
pub const COPY_WORKLOAD: Workload = Workload {
    seed: 0x0003_c09e_0001,
    groups: 4,
    per_group: 8,
    shared: 8,
    own: 18,
    output_len: 24,
    block_size: 4,
};

/// The prompts of [`COPY_WORKLOAD`] and each prompt's symbols (the expected copy).
pub fn copy_workload_prompts() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let w = COPY_WORKLOAD;
    let mut rng = SplitMix64::new(w.seed);
    let shared_symbols = w.shared - 1;
    let prefixes: Vec<Vec<u32>> = (0..w.groups)
        .map(|_| TASK.symbols(&mut rng)[..shared_symbols].to_vec())
        .collect();
    let mut prompts = Vec::new();
    let mut symbols = Vec::new();
    for i in 0..w.count() {
        let mut s = prefixes[i % w.groups].clone();
        s.extend_from_slice(&TASK.symbols(&mut rng)[..TASK.len - shared_symbols]);
        let p = TASK.prompt(&s);
        debug_assert!(p[0] == BOS && p[p.len() - 1] == SEP && p.len() == w.shared + w.own);
        prompts.push(p);
        symbols.push(s);
    }
    (prompts, symbols)
}

/// R3 rerun: the random-weight model of `r3-decode` and the trained model.
#[derive(Clone, Debug)]
pub struct R3Rerun {
    pub symbols: Vec<Vec<u32>>,
    pub random: DecodeResult,
    pub trained: DecodeResult,
}

fn decode_config(model: DecoderConfig, weights: Weights, prompts: &[Vec<u32>]) -> DecodeConfig {
    DecodeConfig {
        model,
        workload: COPY_WORKLOAD,
        scales: vec![1.0],
        weights: Some(Arc::new(weights)),
        prompts: Some(prompts.to_vec()),
        ..DecodeConfig::default()
    }
}

pub fn run_r3(trained: &MasterWeights) -> Result<R3Rerun, String> {
    let (prompts, symbols) = copy_workload_prompts();
    let tiny = DecoderConfig::TINY;
    let random = run_decode(&decode_config(tiny, Weights::generate(&tiny), &prompts))?;
    let trained = run_decode(&decode_config(
        DecoderConfig::TRAINED,
        Weights::from_master(trained),
        &prompts,
    ))?;
    Ok(R3Rerun {
        symbols,
        random,
        trained,
    })
}

/// R4 rerun: R4's random master weights and the trained model, every level.
#[derive(Clone, Debug)]
pub struct R4Rerun {
    pub symbols: Vec<Vec<u32>>,
    pub random: PrecisionResult,
    pub trained: PrecisionResult,
    /// Tokens of the copy the all-FP32 reference gets right, per model (random, trained).
    pub reference_copy_correct: (usize, usize),
}

/// Prompts for the precision rerun.
pub const R4_PROMPTS: usize = 8;
const R4_PROMPT_SEED: u64 = 0x0004_c09e_0001;

fn precision_config(model: DecoderConfig, w: MasterWeights, prompts: &[Vec<u32>]) -> PrecisionConfig {
    PrecisionConfig {
        model,
        prompt_seed: R4_PROMPT_SEED,
        prompts: prompts.len(),
        prompt_len: TASK.len + 2,
        continuation: TASK.len,
        weights: Some(Arc::new(w)),
        prompt_set: Some(prompts.to_vec()),
        lower_levels: Level::EVERY[1..].to_vec(),
    }
}

fn copy_correct(model: DecoderConfig, w: &MasterWeights, prompts: &[Vec<u32>], symbols: &[Vec<u32>]) -> usize {
    let mut m = PolicyModel::new(model, w, Policy::uniform(model.layers, Level::Fp32));
    prompts
        .iter()
        .zip(symbols)
        .map(|(p, s)| m.greedy(p, TASK.len).iter().zip(s).filter(|(a, b)| a == b).count())
        .sum()
}

pub fn run_r4(trained: &MasterWeights) -> Result<R4Rerun, String> {
    let symbols = TASK.batch(R4_PROMPT_SEED, 0, R4_PROMPTS);
    let prompts: Vec<Vec<u32>> = symbols.iter().map(|s| TASK.prompt(s)).collect();
    let tiny = DecoderConfig::TINY;
    let random_w = MasterWeights::generate(&tiny);
    let reference_copy_correct = (
        copy_correct(tiny, &random_w, &prompts, &symbols),
        copy_correct(DecoderConfig::TRAINED, trained, &prompts, &symbols),
    );
    let random = run_precision(&precision_config(tiny, random_w, &prompts))?;
    let trained = run_precision(&precision_config(DecoderConfig::TRAINED, trained.clone(), &prompts))?;
    Ok(R4Rerun {
        symbols,
        random,
        trained,
        reference_copy_correct,
    })
}

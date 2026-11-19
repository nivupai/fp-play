//! The recorded training run (results/r4-training.*) and the trained
//! model's greedy task accuracy through the inference models.

use super::task::CopyTask;
use super::{train, TrainConfig, TrainResult};
use crate::collectives::WirePolicy;
use crate::decoder::decode::generate_static;
use crate::decoder::forward::Model;
use crate::decoder::parallel::Unsharded;
use crate::decoder::weights::Weights;
use crate::format::Format;
use crate::json::Json;
use crate::precision::model::{MasterWeights, PolicyModel};
use crate::precision::policy::{Level, Policy};

/// Held-out greedy decoding of the copy task by one inference model.
#[derive(Clone, Debug, PartialEq)]
pub struct GreedyEval {
    pub model: String,
    pub sequences: usize,
    pub correct_sequences: usize,
    pub tokens: usize,
    pub correct_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingRecord {
    pub result: TrainResult,
    pub greedy: Vec<GreedyEval>,
}

/// Held-out prompts for greedy evaluation (the eval seed's first batch).
pub fn eval_symbols(c: &TrainConfig, count: usize) -> Vec<Vec<u32>> {
    c.task.batch(c.eval_seed, 1, count)
}

fn score(model: String, task: &CopyTask, symbols: &[Vec<u32>], outputs: &[Vec<u32>]) -> GreedyEval {
    let mut e = GreedyEval {
        model,
        sequences: symbols.len(),
        correct_sequences: 0,
        tokens: symbols.len() * task.len,
        correct_tokens: 0,
    };
    for (s, o) in symbols.iter().zip(outputs) {
        let ok = s.iter().zip(o).filter(|(a, b)| a == b).count();
        e.correct_tokens += ok;
        e.correct_sequences += usize::from(ok == task.len);
    }
    e
}

/// Greedy copy accuracy of `w` through the R3 decoder (BF16 weights and KV,
/// P = 1) under both R3 wire policies, and the R4 all-FP32 policy model.
pub fn greedy_accuracy(c: &TrainConfig, w: &MasterWeights, count: usize) -> Vec<GreedyEval> {
    let symbols = eval_symbols(c, count);
    let prompts: Vec<Vec<u32>> = symbols.iter().map(|s| c.task.prompt(s)).collect();
    let mut out = Vec::new();
    for policy in [WirePolicy::uniform(Format::Bf16), WirePolicy::FP32_RS_BF16_AG] {
        let mut m = Model::with_weights(c.model, Weights::from_master(w));
        let g = generate_static(&mut m, &mut Unsharded { policy }, &prompts, c.task.len, false);
        let outputs: Vec<Vec<u32>> = g.into_iter().map(|x| x.tokens).collect();
        out.push(score(
            format!("R3 decoder: BF16 weights and KV cache, P = 1, {} output", policy.name()),
            &c.task,
            &symbols,
            &outputs,
        ));
    }
    let mut pm = PolicyModel::new(c.model, w, Policy::uniform(c.model.layers, Level::Fp32));
    let outputs: Vec<Vec<u32>> = prompts.iter().map(|p| pm.greedy(p, c.task.len)).collect();
    out.push(score(
        "R4 policy model, all FP32 (fused FP32 chains)".into(),
        &c.task,
        &symbols,
        &outputs,
    ));
    out
}

/// Greedy prompts evaluated per inference model.
pub const GREEDY_PROMPTS: usize = 64;

pub fn run(c: &TrainConfig) -> TrainingRecord {
    let result = train(c);
    let greedy = greedy_accuracy(c, &result.master(), GREEDY_PROMPTS);
    TrainingRecord { result, greedy }
}

fn num(x: f64) -> Json {
    Json::Num(x)
}

pub fn to_json(r: &TrainingRecord, command: &str) -> Json {
    let t = &r.result;
    let c = &t.config;
    let m = &c.model;
    Json::obj(vec![
        ("command", Json::str(command)),
        (
            "model",
            Json::obj(vec![
                ("hidden", Json::uint(m.hidden)),
                ("layers", Json::uint(m.layers)),
                ("q_heads", Json::uint(m.q_heads)),
                ("kv_heads", Json::uint(m.kv_heads)),
                ("head_dim", Json::uint(m.head_dim)),
                ("ffn", Json::uint(m.ffn)),
                ("vocab", Json::uint(m.vocab)),
                ("parameters", Json::uint(t.parameters)),
                ("init_seed", Json::str(format!("{:#x}", m.seed))),
            ]),
        ),
        (
            "task",
            Json::obj(vec![
                ("name", Json::str("copy")),
                ("symbols", Json::uint(c.task.len)),
                (
                    "format",
                    Json::str("BOS s1 … sL SEP s1 … sL; scored on the L tokens after SEP"),
                ),
                ("bos", Json::uint(super::task::BOS as usize)),
                ("sep", Json::uint(super::task::SEP as usize)),
            ]),
        ),
        (
            "optimizer",
            Json::obj(vec![
                ("name", Json::str("Adam, binary32")),
                ("steps", Json::uint(c.steps)),
                ("batch", Json::uint(c.batch)),
                ("peak_lr", num(f64::from(c.peak_lr))),
                ("warmup", Json::uint(c.warmup)),
                ("final_lr_fraction", num(f64::from(c.final_lr_fraction))),
                ("beta1", num(f64::from(c.beta1))),
                ("beta2", num(f64::from(c.beta2))),
                ("eps", num(f64::from(c.adam_eps))),
                ("clip", num(f64::from(c.clip))),
                ("data_seed", Json::str(format!("{:#x}", c.data_seed))),
            ]),
        ),
        ("init_sha256", Json::str(&t.init_sha256)),
        ("weights_file", Json::str("r4-trained-weights.bin")),
        ("weights_sha256", Json::str(&t.sha256)),
        (
            "steps",
            Json::Arr(
                t.steps
                    .iter()
                    .map(|s| {
                        Json::obj(vec![
                            ("step", Json::uint(s.step)),
                            ("lr", num(f64::from(s.lr))),
                            ("loss", num(s.loss)),
                            ("accuracy", num(s.accuracy)),
                            ("grad_norm", num(f64::from(s.grad_norm))),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "held_out",
            Json::obj(vec![
                ("sequences", Json::uint(c.eval_count)),
                ("teacher_forced_loss", num(t.eval_loss)),
                ("teacher_forced_accuracy", num(t.eval_accuracy)),
                (
                    "greedy",
                    Json::Arr(
                        r.greedy
                            .iter()
                            .map(|g| {
                                Json::obj(vec![
                                    ("model", Json::str(&g.model)),
                                    ("sequences", Json::uint(g.sequences)),
                                    ("correct_sequences", Json::uint(g.correct_sequences)),
                                    ("tokens", Json::uint(g.tokens)),
                                    ("correct_tokens", Json::uint(g.correct_tokens)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
    ])
}

pub fn to_markdown(r: &TrainingRecord, command: &str) -> String {
    let t = &r.result;
    let c = &t.config;
    let m = &c.model;
    let mut md = String::new();
    md.push_str("# R4 completion: a tiny model trained deterministically on a copy task\n\n");
    md.push_str(&format!(
        "Generated by `{command}`. Every number below comes from that run; the weights it produced are `r4-trained-weights.bin`.\n\n"
    ));
    md.push_str("## Setup\n\n");
    md.push_str(&format!(
        "- Model: the R3 decoder's architecture and dimensions (hidden {}, {} layers, {} query and {} KV heads of dimension {}, FFN {}, vocabulary {}; {} parameters), initialized by the R4 FP32 master-weight generator with seed `{:#x}`.\n",
        m.hidden, m.layers, m.q_heads, m.kv_heads, m.head_dim, m.ffn, m.vocab, t.parameters, m.seed
    ));
    md.push_str(&format!(
        "- Task: copy. `BOS s1 … s{L} SEP s1 … s{L}` with {L} symbols drawn uniformly from IDs 2–{}; the loss is the mean cross-entropy of the {L} tokens after `SEP`.\n",
        m.vocab - 1,
        L = c.task.len
    ));
    md.push_str(&format!(
        "- Training: {} steps of Adam (β1 {}, β2 {}, ε {:e}) on batches of {} fresh sequences, learning rate warmed up linearly to {:e} over {} steps then decayed linearly to {} of the peak, gradients clipped to global norm {}. Every operation is an IEEE 754 binary32 basic operation or the crate's deterministic `exp`, in a fixed order (each sequence's gradient on its own, summed in sequence order), so the run does not depend on the platform or the number of threads; CI retrains on Linux x86-64 and requires the pinned SHA-256, which was first recorded on macOS arm64.\n",
        c.steps, c.beta1, c.beta2, c.adam_eps, c.batch, c.peak_lr, c.warmup, c.final_lr_fraction, c.clip
    ));
    md.push_str(&format!(
        "- SHA-256 of the initial weights: `{}`; of the trained weights: `{}`.\n\n",
        t.init_sha256, t.sha256
    ));
    md.push_str("## Loss curve\n\n| Step | Learning rate | Batch loss (nats per token) | Batch accuracy | Gradient norm |\n|---:|---:|---:|---:|---:|\n");
    for s in t.steps.iter().filter(|s| s.step % 50 == 0 || s.step + 1 == c.steps) {
        md.push_str(&format!(
            "| {} | {:.3e} | {:.5} | {:.1}% | {:.4} |\n",
            s.step,
            s.lr,
            s.loss,
            100.0 * s.accuracy,
            s.grad_norm
        ));
    }
    let first_perfect = t.steps.iter().find(|s| s.accuracy == 1.0).map(|s| s.step);
    md.push_str(&format!(
        "\nThe first batch scored 100% at step {}. Every step's record is in the JSON.\n\n",
        first_perfect.map_or("—".to_string(), |s| s.to_string())
    ));
    md.push_str("## Held-out accuracy\n\n");
    md.push_str(&format!(
        "Teacher-forced on {} held-out sequences with the training arithmetic: loss {:.6} nats per token, accuracy {:.2}%.\n\n",
        c.eval_count,
        t.eval_loss,
        100.0 * t.eval_accuracy
    ));
    md.push_str("Greedy decoding of held-out prompts (`BOS s1 … sL SEP`, L tokens generated) through the inference models:\n\n| Model | Sequences copied exactly | Tokens correct |\n|---|---:|---:|\n");
    for g in &r.greedy {
        md.push_str(&format!(
            "| {} | {} / {} | {} / {} |\n",
            g.model, g.correct_sequences, g.sequences, g.correct_tokens, g.tokens
        ));
    }
    md.push_str("\n## What this does and does not show\n\n");
    md.push_str("- It gives the R3/R4 experiments a model that solves its task with a held-out loss near zero, so their divergence and precision results can be compared between random and trained weights of one architecture; its top-2 margins are measured in `r3-trained`.\n");
    md.push_str("- The task is synthetic and trivial for this architecture; this is not a language model and its loss is not a language-modeling result.\n");
    md.push_str("- Retraining is checked bit for bit by the stale test (the weights' SHA-256 is pinned in the source and checked again with Python's `hashlib` in CI); the backward pass is checked against central finite differences in binary64 for every parameter of a small model, and the training forward pass against the inference model.\n");
    md
}

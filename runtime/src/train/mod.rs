//! R4 completion: a tiny Llama-style model trained deterministically in
//! binary32 on the copy task ([`task`]), so the precision and divergence
//! experiments can run on a model with realistic, confident margins.
//!
//! * [`net`]: the R3 decoder's forward pass and a hand-written backward pass,
//!   generic over binary32 (training) and binary64 (the finite-difference
//!   gradient check in `tests/train.rs`).
//! * [`train`]: Adam with linear warm-up and decay, global-norm gradient
//!   clipping, a fixed batch per step. Each sequence's gradient is computed
//!   on its own (possibly on another thread) and the batch gradient is their
//!   sum in sequence order, so the result does not depend on the number of
//!   threads.
//!
//! Every operation is an IEEE 754 basic operation or the crate's
//! deterministic `exp`, in a written order, so the trained weights are
//! bit-identical on every platform. Their SHA-256 is pinned in
//! [`TRAINED_WEIGHTS_SHA256`]; the recorded artifact `results/r4-training.*`
//! and the weights file `results/r4-trained-weights.bin` are rewritten only
//! by the documented experiment command, and CI retrains and compares both.

pub mod checks;
pub mod net;
pub mod real;
pub mod report;
pub mod task;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use self::net::{Layout, Net, RopeTable};
use self::task::CopyTask;
use crate::decoder::config::DecoderConfig;
use crate::precision::model::MasterWeights;

/// SHA-256 of `results/r4-trained-weights.bin` (binary32 little-endian, in
/// [`Layout`] order) produced by [`train`] with [`TrainConfig::default`].
pub const TRAINED_WEIGHTS_SHA256: &str = "9de08b23ec3ca33cf6650f921de2735188f223cd9582ad5fd8089aa704b15b65";

#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub model: DecoderConfig,
    pub task: CopyTask,
    pub data_seed: u64,
    pub steps: usize,
    pub batch: usize,
    pub peak_lr: f32,
    pub warmup: usize,
    /// The learning rate decays linearly from the peak to this fraction of it.
    pub final_lr_fraction: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub adam_eps: f32,
    /// Global gradient-norm clip.
    pub clip: f32,
    pub eval_seed: u64,
    pub eval_count: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            model: DecoderConfig::TRAINED,
            task: CopyTask {
                len: 24,
                vocab: DecoderConfig::TRAINED.vocab,
            },
            data_seed: 0x0074_7261_696e_0001,
            steps: 600,
            batch: 16,
            peak_lr: 3e-3,
            warmup: 50,
            final_lr_fraction: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            adam_eps: 1e-8,
            clip: 1.0,
            eval_seed: 0x0074_7261_696e_00e7,
            eval_count: 256,
        }
    }
}

/// One optimizer step's record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepRecord {
    pub step: usize,
    pub lr: f32,
    /// Mean cross-entropy (nats per scored token) of this step's batch, before the update.
    pub loss: f64,
    /// Fraction of this batch's scored tokens whose argmax was right.
    pub accuracy: f64,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainResult {
    pub config: TrainConfig,
    pub parameters: usize,
    pub init_sha256: String,
    pub weights: Vec<f32>,
    pub sha256: String,
    pub steps: Vec<StepRecord>,
    /// Teacher-forced loss and accuracy on the held-out set, after training.
    pub eval_loss: f64,
    pub eval_accuracy: f64,
}

impl TrainResult {
    pub fn master(&self) -> MasterWeights {
        Layout::new(&self.config.model).unflatten(&self.weights)
    }
}

/// Binary32 little-endian bytes of a flat parameter vector.
pub fn to_bytes(p: &[f32]) -> Vec<u8> {
    p.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The inverse of [`to_bytes`].
pub fn from_bytes(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err("weights file length is not a multiple of 4".into());
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The trained weights from `results/r4-trained-weights.bin`, checked
/// against [`TRAINED_WEIGHTS_SHA256`].
pub fn load_trained(bytes: &[u8]) -> Result<MasterWeights, String> {
    let hash = crate::sha256::hex(bytes);
    if hash != TRAINED_WEIGHTS_SHA256 {
        return Err(format!(
            "trained weights SHA-256 {hash} does not match the pinned {TRAINED_WEIGHTS_SHA256}"
        ));
    }
    let layout = Layout::new(&DecoderConfig::TRAINED);
    let p = from_bytes(bytes)?;
    if p.len() != layout.total {
        return Err("trained weights have the wrong size".into());
    }
    Ok(layout.unflatten(&p))
}

/// Run `f` on every item on the available cores; results keep item order.
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

/// Learning rate of `step` (0-based): linear warm-up to the peak over
/// `warmup` steps, then linear decay to `final_lr_fraction × peak`.
pub fn learning_rate(c: &TrainConfig, step: usize) -> f32 {
    if step < c.warmup {
        return c.peak_lr * ((step + 1) as f32 / c.warmup as f32);
    }
    let span = (c.steps - c.warmup).max(1) as f32;
    let done = (step - c.warmup) as f32 / span;
    c.peak_lr * (1.0 - (1.0 - c.final_lr_fraction) * done)
}

/// Batch loss and gradient: each sequence's gradient on its own, summed in
/// sequence order. Returns (gradient, summed loss, correct targets).
pub fn batch_gradient(
    layout: &Layout,
    p: &[f32],
    rope: &RopeTable<f32>,
    task: &CopyTask,
    batch: &[Vec<u32>],
) -> (Vec<f32>, f64, usize) {
    let pt = layout.transpose(p);
    let net = Net::new(layout, p, &pt, rope);
    let weight = 1.0f32 / (batch.len() * task.len) as f32;
    let parts = parallel(batch, |symbols| {
        let (input, targets) = task.example(symbols);
        let mut g = vec![0.0f32; layout.total];
        let (loss, correct) = net.loss_and_grad(&input, &targets, weight, &mut g);
        (g, loss, correct)
    });
    let mut grad = vec![0.0f32; layout.total];
    let (mut loss, mut correct) = (0.0, 0);
    for (g, l, c) in parts {
        for (a, b) in grad.iter_mut().zip(&g) {
            *a += *b;
        }
        loss += l;
        correct += c;
    }
    (grad, loss, correct)
}

/// Teacher-forced mean loss and accuracy of `p` on `count` held-out sequences.
pub fn evaluate(c: &TrainConfig, layout: &Layout, p: &[f32], rope: &RopeTable<f32>) -> (f64, f64) {
    let batch = c.task.batch(c.eval_seed, 0, c.eval_count);
    let (_, loss, correct) = batch_gradient(layout, p, rope, &c.task, &batch);
    let n = (c.eval_count * c.task.len) as f64;
    (loss / n, correct as f64 / n)
}

/// Train from the deterministic initialization (the R4 master-weight
/// generator with the trained config's seed).
pub fn train(c: &TrainConfig) -> TrainResult {
    let layout = Layout::new(&c.model);
    let rope = RopeTable::<f32>::new(&c.model, 2 * c.task.len + 2);
    let mut p = layout.flatten(&MasterWeights::generate(&c.model));
    let init_sha256 = crate::sha256::hex(&to_bytes(&p));
    let mut m = vec![0.0f32; layout.total];
    let mut v = vec![0.0f32; layout.total];
    let (mut b1t, mut b2t) = (1.0f32, 1.0f32);
    let mut steps = Vec::with_capacity(c.steps);
    for step in 0..c.steps {
        let batch = c.task.batch(c.data_seed, step as u64, c.batch);
        let (mut g, loss, correct) = batch_gradient(&layout, &p, &rope, &c.task, &batch);
        let mut norm2 = 0.0f32;
        for &x in &g {
            norm2 += x * x;
        }
        let norm = norm2.sqrt();
        if norm > c.clip {
            let f = c.clip / norm;
            for x in g.iter_mut() {
                *x *= f;
            }
        }
        let lr = learning_rate(c, step);
        b1t *= c.beta1;
        b2t *= c.beta2;
        let (bc1, bc2) = (1.0 - b1t, 1.0 - b2t);
        let (one_b1, one_b2) = (1.0 - c.beta1, 1.0 - c.beta2);
        for i in 0..layout.total {
            m[i] = c.beta1 * m[i] + one_b1 * g[i];
            v[i] = c.beta2 * v[i] + one_b2 * (g[i] * g[i]);
            let mhat = m[i] / bc1;
            let vhat = v[i] / bc2;
            p[i] -= lr * (mhat / (vhat.sqrt() + c.adam_eps));
        }
        let scored = (c.batch * c.task.len) as f64;
        steps.push(StepRecord {
            step,
            lr,
            loss: loss / scored,
            accuracy: correct as f64 / scored,
            grad_norm: norm,
        });
    }
    let (eval_loss, eval_accuracy) = evaluate(c, &layout, &p, &rope);
    let sha256 = crate::sha256::hex(&to_bytes(&p));
    TrainResult {
        config: c.clone(),
        parameters: layout.total,
        init_sha256,
        weights: p,
        sha256,
        steps,
        eval_loss,
        eval_accuracy,
    }
}

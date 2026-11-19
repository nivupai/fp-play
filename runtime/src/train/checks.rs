//! The trainer's two correctness checks, shared by `tests/train.rs` and the
//! planted-bug tests (`tests/train_planted.rs`).
//!
//! * [`finite_difference_check`]: the hand-written backward pass equals
//!   central finite differences in binary64 for every parameter of a small
//!   model (embedding, norms and every projection of both layers).
//! * [`inference_forward_check`]: the binary32 training forward pass gives
//!   the same logits, up to rounding, as the R4 all-FP32 inference model on
//!   the same weights, so the trained model is the model the R3/R4
//!   experiments run (same RoPE pairing, norms, attention and head).

use super::net::{Layout, Net, RopeTable};
use super::task::CopyTask;
use crate::decoder::config::DecoderConfig;
use crate::precision::model::{MasterWeights, PolicyModel};
use crate::precision::policy::{Level, Policy};

/// A model small enough to check every parameter by finite differences.
pub const SMALL: DecoderConfig = DecoderConfig {
    hidden: 8,
    layers: 2,
    q_heads: 4,
    kv_heads: 2,
    head_dim: 2,
    ffn: 12,
    vocab: 8,
    rope_base: 10_000.0,
    rms_eps: 1e-5,
    seed: 0x0067_7261_6463,
    logit_scale: 1.0,
};

#[cfg(feature = "planted-bugs")]
type Planted = Option<super::net::TrainMutation>;
#[cfg(not(feature = "planted-bugs"))]
type Planted = Option<std::convert::Infallible>;

fn net<'a, T: super::real::Real>(
    layout: &'a Layout,
    p: &'a [T],
    pt: &'a [T],
    rope: &'a RopeTable<T>,
    planted: Planted,
) -> Net<'a, T> {
    #[allow(unused_mut)]
    let mut n = Net::new(layout, p, pt, rope);
    #[cfg(feature = "planted-bugs")]
    {
        n.planted = planted;
    }
    #[cfg(not(feature = "planted-bugs"))]
    let _ = planted;
    n
}

/// Worst relative error between backward and central differences; `Err` if
/// any parameter's gradient is off by more than `1e-6 × |g| + 1e-8` (the
/// difference quotient's rounding noise is about 1e-9).
pub fn finite_difference_check() -> Result<f64, String> {
    fd_check(None)
}

#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
pub fn finite_difference_check_planted(m: super::net::TrainMutation) -> Result<f64, String> {
    fd_check(Some(m))
}

fn fd_check(planted: Planted) -> Result<f64, String> {
    let layout = Layout::new(&SMALL);
    let task = CopyTask { len: 3, vocab: 8 };
    let rope = RopeTable::<f64>::new(&SMALL, 16);
    let p: Vec<f64> = layout
        .flatten(&MasterWeights::generate(&SMALL))
        .into_iter()
        .map(f64::from)
        .collect();
    let loss = |q: &[f64], input: &[u32], targets: &[(usize, u32)]| {
        let qt = layout.transpose(q);
        let mut scratch = vec![0.0; layout.total];
        net(&layout, q, &qt, &rope, planted)
            .loss_and_grad(input, targets, 1.0, &mut scratch)
            .0
    };
    let mut worst: f64 = 0.0;
    for symbols in [vec![2u32, 5, 7], vec![3, 3, 6]] {
        let (input, targets) = task.example(&symbols);
        let pt = layout.transpose(&p);
        let mut grad = vec![0.0; layout.total];
        net(&layout, &p, &pt, &rope, planted).loss_and_grad(&input, &targets, 1.0, &mut grad);
        for i in 0..layout.total {
            let h = 1e-6 * p[i].abs().max(1.0);
            let mut q = p.clone();
            q[i] = p[i] + h;
            let up = loss(&q, &input, &targets);
            q[i] = p[i] - h;
            let down = loss(&q, &input, &targets);
            let numeric = (up - down) / (2.0 * h);
            let diff = (numeric - grad[i]).abs();
            let scale = numeric.abs().max(grad[i].abs());
            worst = worst.max(diff / scale.max(1e-3));
            if diff > 1e-6 * scale + 1e-8 {
                return Err(format!(
                    "parameter {i}: backward {} vs finite difference {numeric}",
                    grad[i]
                ));
            }
        }
    }
    Ok(worst)
}

/// Largest |training logit − inference logit| divided by the largest
/// |inference logit|, over every position of two copy examples; `Err` above 1e-4.
pub fn inference_forward_check() -> Result<f64, String> {
    forward_check(None)
}

#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
pub fn inference_forward_check_planted(m: super::net::TrainMutation) -> Result<f64, String> {
    forward_check(Some(m))
}

fn forward_check(planted: Planted) -> Result<f64, String> {
    // Head dimension 8, so RoPE has four frequencies and pairing matters.
    let c = DecoderConfig {
        hidden: 32,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 8,
        ffn: 48,
        vocab: 16,
        ..SMALL
    };
    let layout = Layout::new(&c);
    let task = CopyTask { len: 6, vocab: 16 };
    let w = MasterWeights::generate(&c);
    let p = layout.flatten(&w);
    let pt = layout.transpose(&p);
    let rope = RopeTable::<f32>::new(&c, 16);
    let mut worst: f64 = 0.0;
    for symbols in [vec![2u32, 9, 4, 15, 4, 7], vec![12, 3, 3, 8, 2, 11]] {
        let (input, _) = task.example(&symbols);
        let train = net(&layout, &p, &pt, &rope, planted).logits(&input);
        let mut pm = PolicyModel::new(c, &w, Policy::uniform(c.layers, Level::Fp32));
        let inference = pm.teacher_forced(&input);
        let top = inference
            .iter()
            .flatten()
            .fold(0.0f64, |m, &z| m.max(f64::from(z).abs()));
        for (t, row) in inference.iter().enumerate() {
            for (v, &z) in row.iter().enumerate() {
                let d = (f64::from(train[t * c.vocab + v]) - f64::from(z)).abs() / top;
                worst = worst.max(d);
            }
        }
    }
    if worst > 1e-4 {
        return Err(format!(
            "training and inference logits differ by {worst:e} of the largest logit"
        ));
    }
    Ok(worst)
}

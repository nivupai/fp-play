//! Drift of a policy's logits from the reference's, in binary64.
//!
//! Softmax and log use [`crate::decoder::mathf::exp`] and
//! [`crate::serving::detmath::ln`] (binary64, a few ulps, deterministic);
//! sums run in index order. These measure; they do not model any format.

use crate::decoder::mathf::exp;
use crate::serving::detmath::ln;

/// Log-softmax of `z` in binary64 (max subtracted first).
pub fn log_softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().map(|&v| f64::from(v)).fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = z.iter().map(|&v| f64::from(v) - m).collect();
    let mut total = 0.0;
    for &s in &shifted {
        total += exp(s.max(-700.0));
    }
    let lt = ln(total);
    shifted.iter().map(|&s| s - lt).collect()
}

/// KL(p ‖ q) for log-probabilities `lp`, `lq`.
pub fn kl(lp: &[f64], lq: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&a, &b) in lp.iter().zip(lq) {
        s += exp(a.max(-700.0)) * (a - b);
    }
    s
}

/// Drift of one token's logits: (max |Δz|, RMS Δz).
pub fn drift(reference: &[f32], policy: &[f32]) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut ss = 0.0;
    for (&a, &b) in reference.iter().zip(policy) {
        let d = f64::from(b) - f64::from(a);
        max = max.max(d.abs());
        ss += d * d;
    }
    (max, (ss / reference.len() as f64).sqrt())
}

/// Aggregate comparison of a policy with the reference over many tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drift {
    pub tokens: usize,
    /// Tokens with finite logits (the means are over these).
    pub scored: usize,
    pub max_abs: f64,
    pub mean_rms: f64,
    pub mean_kl: f64,
    pub top1_agree: usize,
    /// Mean negative log-likelihood of the scored tokens.
    pub nll: f64,
    /// Logits that were not finite.
    pub nonfinite: usize,
}

impl Drift {
    pub fn perplexity(&self) -> f64 {
        exp(self.nll.min(700.0))
    }
}

/// Compare `policy` logits with `reference` logits position by position;
/// `next[t]` is the token scored at position t.
pub fn compare(reference: &[Vec<f32>], policy: &[Vec<f32>], next: &[u32]) -> Drift {
    let mut d = Drift::default();
    let (mut rms, mut kls, mut nll) = (0.0, 0.0, 0.0);
    for ((r, p), &t) in reference.iter().zip(policy).zip(next) {
        d.tokens += 1;
        d.nonfinite += p.iter().filter(|v| !v.is_finite()).count();
        if p.iter().any(|v| !v.is_finite()) {
            continue;
        }
        d.scored += 1;
        let (m, s) = drift(r, p);
        d.max_abs = d.max_abs.max(m);
        rms += s;
        let (lr, lp) = (log_softmax(r), log_softmax(p));
        kls += kl(&lr, &lp);
        nll -= lp[t as usize];
        let top = crate::decoder::forward::Top2::of;
        d.top1_agree += usize::from(top(r).token == top(p).token);
    }
    let n = d.scored.max(1) as f64;
    d.mean_rms = rms / n;
    d.mean_kl = kls / n;
    d.nll = nll / n;
    d
}

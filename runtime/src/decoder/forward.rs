//! The forward pass of one engine step, in binary32 with a fixed order.
//!
//! # Arithmetic
//!
//! Local kernels use native IEEE 754 binary32 `+`, `−`, `×`, `÷` and `sqrt`,
//! each correctly rounded; Rust never contracts `a * b + c` into a fused
//! multiply–add, so every result is bit-identical on every platform. Every
//! sum runs in one written order, starting from `+0`:
//!
//! * a matrix–vector product sums over the input index in increasing order
//!   (one sequential chain per output; many outputs run side by side, which
//!   never reorders a single output's additions);
//! * RMSNorm: `ms = (Σ x²) ÷ d`, `y = (x × (1 ÷ sqrt(ms + ε))) × g`;
//! * attention: per head, scores `(q · k) × d^-½` over positions in order, a
//!   softmax that subtracts the maximum, `exp` from [`super::mathf`], a
//!   sequential denominator and one division per weight, then `Σ p·v` over
//!   positions in order;
//! * SwiGLU: `silu(g) = g ÷ (1 + exp(−g))`, times `up`;
//! * residual adds in binary32; the residual stream is binary32.
//!
//! Weights and the KV cache are BF16 (K after RoPE). Every token's local
//! arithmetic depends only on that token, its own sequence's KV and the
//! weights, never on which other tokens share the step, so local math is
//! batch-invariant by construction. The two row-parallel projections per
//! layer (attention output and MLP down) go through a [`RowReduce`], which
//! sees the whole step's buffer: that is where tensor parallelism and the
//! AllReduce schedule enter.

use std::ops::Range;

use super::config::DecoderConfig;
use super::mathf::{exp, exp_f32, sin_cos};
use super::weights::{bf16_to_f32, f32_to_bf16, Matrix, Weights};
use crate::serving::detmath::ln;
use crate::serving::kv::SeqId;

/// One token of a step: its sequence, position, ID, whether the step must
/// return its logits (the last prompt token of a prefill, and decodes), and
/// whether this sequence computed this position before (a recompute after
/// preemption or a failure; it changes nothing in the arithmetic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepToken {
    pub seq: SeqId,
    pub pos: usize,
    pub token: u32,
    pub logits: bool,
    pub recompute: bool,
}

/// K/V storage the forward pass reads and writes. A slice holds one token's
/// K (or V) for every KV head: `kv_heads × head_dim` BF16 bit patterns.
pub trait KvStore {
    fn write(&mut self, seq: SeqId, layer: usize, pos: usize, k: &[u16], v: &[u16]);
    fn k(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16];
    fn v(&self, seq: SeqId, layer: usize, pos: usize) -> &[u16];
}

/// Which row-parallel projection a reduction belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub layer: usize,
    /// 0: attention output projection; 1: MLP down projection.
    pub index: u8,
}

/// A row-parallel projection over the whole step: `x` is `tokens × m.cols`
/// activations; the result is `tokens × m.rows`, each value exactly
/// representable in binary32 (it is widened from the reduction's output
/// format).
pub trait RowReduce {
    fn project(&mut self, site: Site, m: &Matrix, x: &[f32], tokens: usize) -> Vec<f32>;
}

/// `out[o] = Σ_{k ∈ ks} W[o][k] · x[k]`, each sum sequential in increasing k from `+0`.
pub fn matvec(m: &Matrix, x: &[f32], ks: Range<usize>, out: &mut [f32]) {
    assert_eq!(out.len(), m.rows);
    out.fill(0.0);
    for k in ks {
        let xk = x[k];
        let column = &m.t[k * m.rows..(k + 1) * m.rows];
        for (acc, &w) in out.iter_mut().zip(column) {
            *acc += w * xk;
        }
    }
}

pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let ms = ss / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(gain).map(|(&v, &g)| (v * inv) * g).collect()
}

/// Top-2 summary of one logits row: the greedy token, its logit and the
/// runner-up's logit (binary32). Ties go to the lower token ID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Top2 {
    pub token: u32,
    pub top: f32,
    pub second: f32,
}

impl Top2 {
    pub fn of(logits: &[f32]) -> Top2 {
        assert!(
            logits.len() >= 2 && logits.iter().all(|v| !v.is_nan()),
            "need ≥ 2 non-NaN logits"
        );
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        let second = logits
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != best)
            .map(|(_, &v)| v)
            .fold(f32::NEG_INFINITY, f32::max);
        Top2 {
            token: best as u32,
            top: logits[best],
            second,
        }
    }

    /// `top − second`, exactly (binary64 holds the difference of two binary32 values).
    pub fn margin(&self) -> f64 {
        f64::from(self.top) - f64::from(self.second)
    }
}

/// RoPE cos/sin tables, grown on demand. Angles `pos × θ_i` with
/// `θ_i = base^(−2i/d)` are computed in binary64 with [`super::mathf`] and
/// rounded once to binary32.
#[derive(Clone, Debug)]
pub(crate) struct Rope {
    half: usize,
    theta: Vec<f64>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub(crate) fn new(config: &DecoderConfig) -> Rope {
        let half = config.head_dim / 2;
        let ln_base = ln(config.rope_base);
        let theta = (0..half)
            .map(|i| exp(-(((2 * i) as f64 / config.head_dim as f64) * ln_base)))
            .collect();
        Rope {
            half,
            theta,
            cos: Vec::new(),
            sin: Vec::new(),
        }
    }

    pub(crate) fn ensure(&mut self, positions: usize) {
        while self.cos.len() < positions * self.half {
            let pos = self.cos.len() / self.half;
            for i in 0..self.half {
                let (s, c) = sin_cos(pos as f64 * self.theta[i]);
                self.cos.push(c as f32);
                self.sin.push(s as f32);
            }
        }
    }

    /// Rotate each head of `x` in place (rotate-half pairing: i with i + d/2).
    pub(crate) fn apply(&self, x: &mut [f32], pos: usize, head_dim: usize) {
        let cos = &self.cos[pos * self.half..(pos + 1) * self.half];
        let sin = &self.sin[pos * self.half..(pos + 1) * self.half];
        for head in x.chunks_mut(head_dim) {
            for i in 0..self.half {
                let (a, b) = (head[i], head[i + self.half]);
                head[i] = a * cos[i] - b * sin[i];
                head[i + self.half] = b * cos[i] + a * sin[i];
            }
        }
    }
}

/// The decoder: config, weights and RoPE tables.
#[derive(Clone, Debug)]
pub struct Model {
    pub config: DecoderConfig,
    pub weights: Weights,
    rope: Rope,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<super::DecoderMutation>,
}

/// Logits of one token that asked for them.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogits {
    pub seq: SeqId,
    pub pos: usize,
    pub logits: Vec<f32>,
}

impl Model {
    pub fn new(config: DecoderConfig) -> Model {
        let weights = Weights::generate(&config);
        Model::with_weights(config, weights)
    }

    /// Reuse generated weights with a config that differs only in `logit_scale`.
    pub fn with_weights(config: DecoderConfig, weights: Weights) -> Model {
        config.validate().expect("valid decoder config");
        Model {
            rope: Rope::new(&config),
            config,
            weights,
            #[cfg(feature = "planted-bugs")]
            planted: None,
        }
    }

    /// RoPE position of a token (its sequence position, except under a planted bug).
    fn rope_pos(&self, tok: &StepToken) -> usize {
        #[cfg(feature = "planted-bugs")]
        if tok.recompute && self.planted == Some(super::DecoderMutation::RopeOffByOneOnRecompute) {
            return tok.pos + 1;
        }
        tok.pos
    }

    /// Run one step. Tokens of one sequence must appear in increasing
    /// position order with no gaps after the positions already in `kv`.
    /// Returns the logits of the tokens that asked for them, in step order.
    pub fn step(&mut self, tokens: &[StepToken], kv: &mut dyn KvStore, reduce: &mut dyn RowReduce) -> Vec<TokenLogits> {
        let c = self.config;
        let (h, d, t) = (c.hidden, c.head_dim, tokens.len());
        let kvw = c.kv_width();
        self.rope.ensure(tokens.iter().map(|x| x.pos + 2).max().unwrap_or(0));
        let rope_pos: Vec<usize> = tokens.iter().map(|tok| self.rope_pos(tok)).collect();
        let w = &self.weights;
        // Residual stream: the embedding rows.
        let mut x: Vec<f32> = Vec::with_capacity(t * h);
        for tok in tokens {
            assert!(
                (tok.token as usize) < c.vocab,
                "token {} outside the vocabulary",
                tok.token
            );
            x.extend(w.embed.row(tok.token as usize));
        }
        let scale = 1.0f32 / (d as f32).sqrt();
        for (l, layer) in w.layers.iter().enumerate() {
            // Attention: Q, K, V for every token; write K, V; then attend.
            let mut q_all = vec![0.0f32; t * h];
            let mut k = vec![0.0f32; kvw];
            let mut v = vec![0.0f32; kvw];
            for (i, tok) in tokens.iter().enumerate() {
                let n = rms_norm(&x[i * h..(i + 1) * h], &layer.attn_norm, c.rms_eps);
                let q = &mut q_all[i * h..(i + 1) * h];
                matvec(&layer.wq, &n, 0..h, q);
                matvec(&layer.wk, &n, 0..h, &mut k);
                matvec(&layer.wv, &n, 0..h, &mut v);
                self.rope.apply(q, rope_pos[i], d);
                self.rope.apply(&mut k, rope_pos[i], d);
                let kb: Vec<u16> = k.iter().map(|&a| f32_to_bf16(a)).collect();
                let vb: Vec<u16> = v.iter().map(|&a| f32_to_bf16(a)).collect();
                kv.write(tok.seq, l, tok.pos, &kb, &vb);
            }
            let mut attn = vec![0.0f32; t * h];
            for (i, tok) in tokens.iter().enumerate() {
                // This token's view of its sequence's K and V, widened once.
                let keys: Vec<f32> = (0..=tok.pos)
                    .flat_map(|j| kv.k(tok.seq, l, j).iter().map(|&b| bf16_to_f32(b)))
                    .collect();
                let values: Vec<f32> = (0..=tok.pos)
                    .flat_map(|j| kv.v(tok.seq, l, j).iter().map(|&b| bf16_to_f32(b)))
                    .collect();
                for head in 0..c.q_heads {
                    let kvh = head / c.group();
                    let q = &q_all[i * h + head * d..i * h + (head + 1) * d];
                    let mut scores = Vec::with_capacity(tok.pos + 1);
                    for j in 0..=tok.pos {
                        let kj = &keys[j * kvw + kvh * d..j * kvw + (kvh + 1) * d];
                        let mut s = 0.0f32;
                        for (&a, &b) in q.iter().zip(kj) {
                            s += a * b;
                        }
                        scores.push(s * scale);
                    }
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = exp_f32(*s - m);
                        sum += *s;
                    }
                    let out = &mut attn[i * h + head * d..i * h + (head + 1) * d];
                    for (j, &e) in scores.iter().enumerate() {
                        let p = e / sum;
                        let vj = &values[j * kvw + kvh * d..j * kvw + (kvh + 1) * d];
                        for (o, &b) in out.iter_mut().zip(vj) {
                            *o += p * b;
                        }
                    }
                }
            }
            let y = reduce.project(Site { layer: l, index: 0 }, &layer.wo, &attn, t);
            for (a, b) in x.iter_mut().zip(&y) {
                *a += *b;
            }
            // SwiGLU MLP.
            let mut act = vec![0.0f32; t * c.ffn];
            let mut gate = vec![0.0f32; c.ffn];
            let mut up = vec![0.0f32; c.ffn];
            for i in 0..t {
                let n = rms_norm(&x[i * h..(i + 1) * h], &layer.mlp_norm, c.rms_eps);
                matvec(&layer.w_gate, &n, 0..h, &mut gate);
                matvec(&layer.w_up, &n, 0..h, &mut up);
                for (a, (&g, &u)) in act[i * c.ffn..(i + 1) * c.ffn].iter_mut().zip(gate.iter().zip(&up)) {
                    *a = (g / (1.0 + exp_f32(-g))) * u;
                }
            }
            let y = reduce.project(Site { layer: l, index: 1 }, &layer.w_down, &act, t);
            for (a, b) in x.iter_mut().zip(&y) {
                *a += *b;
            }
        }
        // Output head (tied embedding) for the tokens that need logits.
        let mut out = Vec::new();
        for (i, tok) in tokens.iter().enumerate().filter(|(_, tok)| tok.logits) {
            let n = rms_norm(&x[i * h..(i + 1) * h], &w.final_norm, c.rms_eps);
            let mut logits = vec![0.0f32; c.vocab];
            matvec(&w.embed, &n, 0..h, &mut logits);
            for z in logits.iter_mut() {
                *z *= c.logit_scale;
            }
            out.push(TokenLogits {
                seq: tok.seq,
                pos: tok.pos,
                logits,
            });
        }
        out
    }
}

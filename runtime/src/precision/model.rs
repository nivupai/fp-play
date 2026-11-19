//! The R3 decoder with FP32 master weights and a precision level per site.
//!
//! Every matrix product is a fused chain ([`super::mac`]) in the site's
//! accumulator format, over the input index in increasing order from +0, on
//! weights and input activations rounded to the site's storage format. The
//! attention core (scores and the weighted sum of V) is a fused FP32 chain
//! over the K/V values as stored in the KV cache's format. RMSNorm, RoPE,
//! softmax, SiLU and residual adds use binary32 basic operations exactly as
//! in [`crate::decoder::forward`]; the residual stream is binary32.

use super::mac::{fused_dot, mac, round_value};
use super::narrow::{self, Narrow, ScaledMatrix};
use super::policy::{Level, Policy, Site};
use crate::decoder::config::DecoderConfig;
use crate::decoder::forward::{rms_norm, Rope};
use crate::decoder::mathf::exp_f32;
use crate::format::Format;
use crate::rng::SplitMix64;

/// A row-major matrix of FP32 master weights.
#[derive(Clone, Debug)]
pub struct MasterMatrix {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MasterLayer {
    pub attn_norm: Vec<f32>,
    pub wq: MasterMatrix,
    pub wk: MasterMatrix,
    pub wv: MasterMatrix,
    pub wo: MasterMatrix,
    pub mlp_norm: Vec<f32>,
    pub w_gate: MasterMatrix,
    pub w_up: MasterMatrix,
    pub w_down: MasterMatrix,
}

/// FP32 master weights: the R3 weights' SplitMix streams and Irwin–Hall
/// draws (same seed, tensor indices and scales as
/// [`crate::decoder::weights`]), each rounded once to binary32 instead of BF16.
#[derive(Clone, Debug)]
pub struct MasterWeights {
    pub embed: MasterMatrix,
    pub layers: Vec<MasterLayer>,
    pub final_norm: Vec<f32>,
}

fn draws(seed: u64, index: u64, n: usize, scale: f64, offset: f64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    // `as f32` rounds to nearest, ties to even (IEEE 754 conversion).
    (0..n).map(|_| (offset + scale * rng.next_normalish()) as f32).collect()
}

fn master(seed: u64, index: u64, rows: usize, cols: usize) -> MasterMatrix {
    MasterMatrix {
        rows,
        cols,
        w: draws(seed, index, rows * cols, 1.0 / (cols as f64).sqrt(), 0.0),
    }
}

impl MasterWeights {
    pub fn generate(c: &DecoderConfig) -> MasterWeights {
        c.validate().expect("valid decoder config");
        let (s, h) = (c.seed, c.hidden);
        MasterWeights {
            embed: MasterMatrix {
                rows: c.vocab,
                cols: h,
                w: draws(s, 0, c.vocab * h, 1.0 / (h as f64).sqrt(), 0.0),
            },
            layers: (0..c.layers as u64)
                .map(|l| {
                    let i = 16 + 16 * l;
                    MasterLayer {
                        attn_norm: draws(s, i, h, 0.1, 1.0),
                        wq: master(s, i + 1, c.q_heads * c.head_dim, h),
                        wk: master(s, i + 2, c.kv_width(), h),
                        wv: master(s, i + 3, c.kv_width(), h),
                        wo: master(s, i + 4, h, c.q_heads * c.head_dim),
                        mlp_norm: draws(s, i + 5, h, 0.1, 1.0),
                        w_gate: master(s, i + 6, c.ffn, h),
                        w_up: master(s, i + 7, c.ffn, h),
                        w_down: master(s, i + 8, h, c.ffn),
                    }
                })
                .collect(),
            final_norm: draws(s, 1, h, 0.1, 1.0),
        }
    }
}

/// A matrix rounded to a storage format, column-major (`t[k·rows + o]`).
#[derive(Clone, Debug)]
struct QMatrix {
    rows: usize,
    cols: usize,
    t: Vec<f64>,
    acc: Format,
    storage: Format,
}

impl QMatrix {
    fn new(m: &MasterMatrix, level: Level) -> QMatrix {
        Self::with_acc(m, level, level.accumulator())
    }

    fn with_acc(m: &MasterMatrix, level: Level, acc: Format) -> QMatrix {
        let storage = level.storage();
        let mut t = vec![0.0; m.rows * m.cols];
        for o in 0..m.rows {
            for k in 0..m.cols {
                t[k * m.rows + o] = round_value(storage, f64::from(m.w[o * m.cols + k]), 0.0);
            }
        }
        QMatrix {
            rows: m.rows,
            cols: m.cols,
            t,
            acc,
            storage,
        }
    }

    /// `y[o] = Σ_k W[o][k]·x[k]` as fused chains in the accumulator format,
    /// with `x` first rounded to the storage format. Returns binary32 values.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let mut acc = vec![0.0f64; self.rows];
        for (k, &xk) in x.iter().enumerate() {
            let xq = round_value(self.storage, f64::from(xk), 0.0);
            let column = &self.t[k * self.rows..(k + 1) * self.rows];
            for (a, &w) in acc.iter_mut().zip(column) {
                *a = mac(self.acc, *a, w, xq);
            }
        }
        // Every FP32/BF16/FP16 value (and ±∞) is exactly a binary32.
        acc.into_iter().map(|v| v as f32).collect()
    }
}

/// A site's weights under its level: rounded to a float storage format, or
/// quantized with per-row scales.
#[derive(Clone, Debug)]
enum SiteMatrix {
    Float(QMatrix),
    Scaled(ScaledMatrix),
}

impl SiteMatrix {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        match self {
            SiteMatrix::Float(m) => m.apply(x),
            SiteMatrix::Scaled(m) => m.apply(x),
        }
    }
}

/// One site's matrix product under `level`, as the model computes it:
/// `m` rounded to the level's storage format, `x` likewise, fused chains in
/// its accumulator format; or, for a scaled level, both quantized with
/// their scales ([`ScaledMatrix::apply`]). Exposed for tests.
pub fn site_product(policy: &Policy, m: &MasterMatrix, level: Level, x: &[f32]) -> Vec<f32> {
    qmatrix(policy, m, level).apply(x)
}

fn qmatrix(policy: &Policy, m: &MasterMatrix, level: Level) -> SiteMatrix {
    #[cfg(feature = "planted-bugs")]
    if policy.planted == Some(super::policy::PrecisionMutation::Bf16AccumulatesInFp32) && level == Level::Bf16Acc {
        return SiteMatrix::Float(QMatrix::with_acc(m, level, Format::Fp32));
    }
    let _ = policy;
    match level.narrow() {
        Some(f) => SiteMatrix::Scaled(ScaledMatrix::new(f, m.rows, m.cols, &m.w)),
        None => SiteMatrix::Float(QMatrix::new(m, level)),
    }
}

/// How the KV cache stores K and V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KvFormat {
    /// Each value rounded to a float format.
    Float(Format),
    /// Each KV head's K (or V) of one token quantized with its own scale,
    /// dequantized to binary32 on read ([`narrow::store`]).
    Scaled(Narrow),
}

fn kv_storage(policy: &Policy) -> KvFormat {
    #[cfg(feature = "planted-bugs")]
    if policy.planted == Some(super::policy::PrecisionMutation::SkipKvRounding) {
        return KvFormat::Float(Format::Fp32);
    }
    let level = policy.level(Site::KvCache);
    match level.narrow() {
        Some(f) => KvFormat::Scaled(f),
        None => KvFormat::Float(level.storage()),
    }
}

impl KvFormat {
    /// One token's K or V (all KV heads, `d` values each) as stored.
    fn store(self, x: &[f32], d: usize) -> Vec<f64> {
        match self {
            KvFormat::Float(f) => x.iter().map(|&a| round_value(f, f64::from(a), 0.0)).collect(),
            KvFormat::Scaled(f) => x
                .chunks(d)
                .flat_map(|head| narrow::store(f, head))
                .map(f64::from)
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct QLayer {
    attn_norm: Vec<f32>,
    wq: SiteMatrix,
    wk: SiteMatrix,
    wv: SiteMatrix,
    wo: SiteMatrix,
    mlp_norm: Vec<f32>,
    w_gate: SiteMatrix,
    w_up: SiteMatrix,
    w_down: SiteMatrix,
}

/// The decoder under one precision policy.
#[derive(Clone, Debug)]
pub struct PolicyModel {
    pub config: DecoderConfig,
    pub policy: Policy,
    kv_format: KvFormat,
    /// Embedding rows as the head stores them (row-major): rounded to its
    /// storage format, or dequantized from its scaled format.
    embed_rows: Vec<f32>,
    head: SiteMatrix,
    layers: Vec<QLayer>,
    final_norm: Vec<f32>,
    rope: Rope,
}

/// One sequence's KV cache: per layer, K and V of every position so far
/// (values of the KV cache's storage format).
#[derive(Clone, Debug, Default)]
pub struct Sequence {
    pub len: usize,
    k: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}

impl PolicyModel {
    pub fn new(config: DecoderConfig, w: &MasterWeights, policy: Policy) -> PolicyModel {
        let qm = |m: &MasterMatrix, level: Level| qmatrix(&policy, m, level);
        let head = policy.level(Site::Head);
        let layers = w
            .layers
            .iter()
            .enumerate()
            .map(|(l, m)| {
                let (a, f) = (policy.level(Site::Attention(l)), policy.level(Site::Mlp(l)));
                QLayer {
                    attn_norm: m.attn_norm.clone(),
                    wq: qm(&m.wq, a),
                    wk: qm(&m.wk, a),
                    wv: qm(&m.wv, a),
                    wo: qm(&m.wo, a),
                    mlp_norm: m.mlp_norm.clone(),
                    w_gate: qm(&m.w_gate, f),
                    w_up: qm(&m.w_up, f),
                    w_down: qm(&m.w_down, f),
                }
            })
            .collect();
        let head_matrix = qm(&w.embed, head);
        let embed_rows = match &head_matrix {
            SiteMatrix::Scaled(m) => (0..m.rows).flat_map(|o| m.row(o)).collect(),
            SiteMatrix::Float(_) => w
                .embed
                .w
                .iter()
                .map(|&x| round_value(head.storage(), f64::from(x), 0.0) as f32)
                .collect(),
        };
        let kv_format = kv_storage(&policy);
        PolicyModel {
            kv_format,
            head: head_matrix,
            embed_rows,
            layers,
            final_norm: w.final_norm.clone(),
            rope: Rope::new(&config),
            config,
            policy,
        }
    }

    pub fn sequence(&self) -> Sequence {
        Sequence {
            len: 0,
            k: vec![Vec::new(); self.config.layers],
            v: vec![Vec::new(); self.config.layers],
        }
    }

    /// Process `tokens` at the next positions of `seq`; returns each token's logits.
    pub fn run(&mut self, seq: &mut Sequence, tokens: &[u32]) -> Vec<Vec<f32>> {
        let c = self.config;
        let (h, d, kvw) = (c.hidden, c.head_dim, c.kv_width());
        self.rope.ensure(seq.len + tokens.len());
        let scale = 1.0f32 / (d as f32).sqrt();
        let mut out = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            let pos = seq.len;
            let t = tok as usize;
            assert!(t < c.vocab, "token {tok} outside the vocabulary");
            let mut x: Vec<f32> = self.embed_rows[t * h..(t + 1) * h].to_vec();
            for (l, layer) in self.layers.iter().enumerate() {
                let n = rms_norm(&x, &layer.attn_norm, c.rms_eps);
                let mut q = layer.wq.apply(&n);
                let mut k = layer.wk.apply(&n);
                let v = layer.wv.apply(&n);
                self.rope.apply(&mut q, pos, d);
                self.rope.apply(&mut k, pos, d);
                seq.k[l].extend(self.kv_format.store(&k, d));
                seq.v[l].extend(self.kv_format.store(&v, d));
                let mut attn = vec![0.0f32; h];
                for head in 0..c.q_heads {
                    let kvh = head / c.group();
                    let qh: Vec<f64> = q[head * d..(head + 1) * d].iter().map(|&a| f64::from(a)).collect();
                    let mut scores: Vec<f32> = (0..=pos)
                        .map(|j| {
                            let kj = &seq.k[l][j * kvw + kvh * d..j * kvw + (kvh + 1) * d];
                            (fused_dot(Format::Fp32, &qh, kj) as f32) * scale
                        })
                        .collect();
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = exp_f32(*s - m);
                        sum += *s;
                    }
                    let probs: Vec<f64> = scores.iter().map(|&e| f64::from(e / sum)).collect();
                    for (i, o) in attn[head * d..(head + 1) * d].iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for (j, &p) in probs.iter().enumerate() {
                            acc = mac(Format::Fp32, acc, p, seq.v[l][j * kvw + kvh * d + i]);
                        }
                        *o = acc as f32;
                    }
                }
                for (a, b) in x.iter_mut().zip(layer.wo.apply(&attn)) {
                    *a += b;
                }
                let n = rms_norm(&x, &layer.mlp_norm, c.rms_eps);
                let gate = layer.w_gate.apply(&n);
                let up = layer.w_up.apply(&n);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| (g / (1.0 + exp_f32(-g))) * u)
                    .collect();
                for (a, b) in x.iter_mut().zip(layer.w_down.apply(&act)) {
                    *a += b;
                }
            }
            seq.len += 1;
            let n = rms_norm(&x, &self.final_norm, c.rms_eps);
            let mut logits = self.head.apply(&n);
            for z in logits.iter_mut() {
                *z *= c.logit_scale;
            }
            out.push(logits);
        }
        out
    }

    /// Greedy continuation of `prompt` by `steps` tokens (ties to the lower ID).
    pub fn greedy(&mut self, prompt: &[u32], steps: usize) -> Vec<u32> {
        let mut seq = self.sequence();
        let mut logits = self.run(&mut seq, prompt).pop().expect("non-empty prompt");
        let mut out = Vec::with_capacity(steps);
        for step in 0..steps {
            let t = crate::decoder::forward::Top2::of(&logits).token;
            out.push(t);
            if step + 1 < steps {
                logits = self.run(&mut seq, &[t]).pop().expect("one token");
            }
        }
        out
    }

    /// Logits at every position of `tokens` (teacher forcing).
    pub fn teacher_forced(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut seq = self.sequence();
        self.run(&mut seq, tokens)
    }
}

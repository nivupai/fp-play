//! Deterministic random BF16 weights.
//!
//! Every tensor has its own [`SplitMix64`] stream, seeded from the config
//! seed and the tensor's index, so tensors do not depend on each other's
//! sizes. A value is an Irwin–Hall approximate normal ([`SplitMix64::next_normalish`])
//! times the tensor's scale, rounded once to BF16 ([`ops::from_f64`]). Only
//! basic binary64 operations are used, so the weights are bit-identical on
//! every platform.
//!
//! Scales: projections `1/√fan_in`; the tied embedding `1/√hidden`, so a
//! normalized hidden state (RMS 1) gives logits of standard deviation about
//! 1; RMSNorm gains `1 + 0.1 · normal`.

use super::config::DecoderConfig;
use crate::format::{to_f64, Format};
use crate::ops;
use crate::rng::SplitMix64;

/// A BF16 matrix (`rows` outputs × `cols` inputs), kept as raw bits in
/// row-major order and widened to binary32 in column-major order: element
/// (o, k) is `t[k × rows + o]`, so a matrix–vector product can run the
/// sequential sum over k for many outputs at once without reordering any
/// single output's additions.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub raw: Vec<u16>,
    pub t: Vec<f32>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, raw: Vec<u16>) -> Matrix {
        assert_eq!(raw.len(), rows * cols);
        let mut t = vec![0.0f32; rows * cols];
        for o in 0..rows {
            for k in 0..cols {
                t[k * rows + o] = bf16_to_f32(raw[o * cols + k]);
            }
        }
        Matrix { rows, cols, raw, t }
    }

    /// Row `o` widened to binary32.
    pub fn row(&self, o: usize) -> Vec<f32> {
        self.raw[o * self.cols..(o + 1) * self.cols]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub attn_norm: Vec<f32>,
    /// Query, key and value projections (hidden → q_heads·d, kv_heads·d, kv_heads·d).
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    /// Output projection (hidden → hidden), row-parallel under tensor parallelism.
    pub wo: Matrix,
    pub mlp_norm: Vec<f32>,
    pub w_gate: Matrix,
    pub w_up: Matrix,
    /// Down projection (ffn → hidden), row-parallel.
    pub w_down: Matrix,
}

#[derive(Clone, Debug)]
pub struct Weights {
    /// Tied embedding (vocab × hidden): input rows and output projection.
    pub embed: Matrix,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
}

/// Exact BF16 → binary32 widening.
pub fn bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits(u32::from(raw) << 16)
}

/// binary32 → BF16 with one round-to-nearest-even; NaN becomes BF16's
/// canonical quiet NaN (as [`ops::convert`]). `tests/decoder.rs` checks it
/// against `ops::convert`.
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return Format::Bf16.canonical_nan() as u16;
    }
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

fn tensor(seed: u64, index: u64, n: usize, scale: f64, offset: f64) -> Vec<u16> {
    let mut rng = SplitMix64::new(seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    (0..n)
        .map(|_| ops::from_f64(Format::Bf16, offset + scale * rng.next_normalish()) as u16)
        .collect()
}

fn gains(seed: u64, index: u64, n: usize) -> Vec<f32> {
    tensor(seed, index, n, 0.1, 1.0).into_iter().map(bf16_to_f32).collect()
}

fn matrix(seed: u64, index: u64, rows: usize, cols: usize) -> Matrix {
    let scale = 1.0 / (cols as f64).sqrt();
    Matrix::new(rows, cols, tensor(seed, index, rows * cols, scale, 0.0))
}

impl Weights {
    pub fn generate(config: &DecoderConfig) -> Weights {
        config.validate().expect("valid decoder config");
        let c = config;
        let s = c.seed;
        let h = c.hidden;
        let embed = Matrix::new(c.vocab, h, tensor(s, 0, c.vocab * h, 1.0 / (h as f64).sqrt(), 0.0));
        let layers = (0..c.layers as u64)
            .map(|l| {
                let i = 16 + 16 * l;
                Layer {
                    attn_norm: gains(s, i, h),
                    wq: matrix(s, i + 1, c.q_heads * c.head_dim, h),
                    wk: matrix(s, i + 2, c.kv_width(), h),
                    wv: matrix(s, i + 3, c.kv_width(), h),
                    wo: matrix(s, i + 4, h, c.q_heads * c.head_dim),
                    mlp_norm: gains(s, i + 5, h),
                    w_gate: matrix(s, i + 6, c.ffn, h),
                    w_up: matrix(s, i + 7, c.ffn, h),
                    w_down: matrix(s, i + 8, h, c.ffn),
                }
            })
            .collect();
        Weights {
            embed,
            layers,
            final_norm: gains(s, 1, h),
        }
    }

    /// FP32 master weights (for example the trained model's) rounded once to
    /// BF16 (round to nearest even), norm gains included, as the decoder stores them.
    pub fn from_master(w: &crate::precision::model::MasterWeights) -> Weights {
        let m = |x: &crate::precision::model::MasterMatrix| {
            Matrix::new(x.rows, x.cols, x.w.iter().map(|&v| f32_to_bf16(v)).collect())
        };
        let g = |x: &[f32]| x.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect::<Vec<_>>();
        Weights {
            embed: m(&w.embed),
            layers: w
                .layers
                .iter()
                .map(|l| Layer {
                    attn_norm: g(&l.attn_norm),
                    wq: m(&l.wq),
                    wk: m(&l.wk),
                    wv: m(&l.wv),
                    wo: m(&l.wo),
                    mlp_norm: g(&l.mlp_norm),
                    w_gate: m(&l.w_gate),
                    w_up: m(&l.w_up),
                    w_down: m(&l.w_down),
                })
                .collect(),
            final_norm: g(&w.final_norm),
        }
    }

    /// FNV-1a over every weight's raw bits, in generation order (pinned in tests).
    pub fn checksum(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |v: u16| {
            for b in v.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        let gain_bits = |g: &[f32]| g.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<_>>();
        self.embed.raw.iter().for_each(|&v| eat(v));
        for l in &self.layers {
            gain_bits(&l.attn_norm).into_iter().for_each(&mut eat);
            for m in [&l.wq, &l.wk, &l.wv, &l.wo] {
                m.raw.iter().for_each(|&v| eat(v));
            }
            gain_bits(&l.mlp_norm).into_iter().for_each(&mut eat);
            for m in [&l.w_gate, &l.w_up, &l.w_down] {
                m.raw.iter().for_each(|&v| eat(v));
            }
        }
        gain_bits(&self.final_norm).into_iter().for_each(&mut eat);
        h
    }
}

/// Exact value of a BF16 weight (for tests and reports).
pub fn bf16_value(raw: u16) -> f64 {
    to_f64(Format::Bf16, u32::from(raw))
}

//! The R3 decoder's forward pass for training, and its hand-written
//! backward pass, generic over [`Real`].
//!
//! The forward pass follows [`crate::decoder::forward`] operation for
//! operation (same RMSNorm formula, rotate-half RoPE from the same angle
//! tables, scaled dot-product scores, a max-subtracted softmax with one
//! division per weight, `silu(g) = g ÷ (1 + e^−g)`, tied embedding and
//! head), but with every weight and activation in `T` (no BF16 rounding)
//! and the whole sequence processed causally at once. Every sum runs in one
//! written order from zero, so a binary32 run is bit-identical on every
//! platform. `tests/train.rs` checks the backward pass against central
//! finite differences in binary64, and the binary32 forward pass against
//! the inference model.

use super::real::Real;
use crate::decoder::config::DecoderConfig;
use crate::decoder::mathf::{exp, sin_cos};
use crate::precision::model::{MasterLayer, MasterMatrix, MasterWeights};
use crate::serving::detmath::ln;

/// Offsets of one layer's tensors in the flat parameter vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerOffsets {
    pub attn_norm: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub mlp_norm: usize,
    pub gate: usize,
    pub up: usize,
    pub down: usize,
}

/// Where every tensor lives in the flat parameter vector, in the order the
/// weights file stores them: embedding (vocab × hidden), then per layer
/// attention norm, Q, K, V, O, MLP norm, gate, up, down (each matrix row-major,
/// outputs × inputs), then the final norm.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub config: DecoderConfig,
    pub layers: Vec<LayerOffsets>,
    pub final_norm: usize,
    pub total: usize,
}

/// A matrix's (offset, rows, cols).
#[derive(Clone, Copy, Debug)]
struct Mat {
    at: usize,
    rows: usize,
    cols: usize,
}

impl Layout {
    pub fn new(c: &DecoderConfig) -> Layout {
        c.validate().expect("valid decoder config");
        let (h, qd, kvw, f) = (c.hidden, c.q_heads * c.head_dim, c.kv_width(), c.ffn);
        let mut at = c.vocab * h;
        let mut take = |n: usize| {
            let s = at;
            at += n;
            s
        };
        let layers = (0..c.layers)
            .map(|_| LayerOffsets {
                attn_norm: take(h),
                wq: take(qd * h),
                wk: take(kvw * h),
                wv: take(kvw * h),
                wo: take(h * qd),
                mlp_norm: take(h),
                gate: take(f * h),
                up: take(f * h),
                down: take(h * f),
            })
            .collect();
        let final_norm = take(c.hidden);
        Layout {
            config: *c,
            layers,
            final_norm,
            total: at,
        }
    }

    /// Every matrix, with its shape (the embedding first).
    fn matrices(&self) -> Vec<Mat> {
        let c = &self.config;
        let (h, qd, kvw, f) = (c.hidden, c.q_heads * c.head_dim, c.kv_width(), c.ffn);
        let mut v = vec![Mat {
            at: 0,
            rows: c.vocab,
            cols: h,
        }];
        for l in &self.layers {
            for (at, rows, cols) in [
                (l.wq, qd, h),
                (l.wk, kvw, h),
                (l.wv, kvw, h),
                (l.wo, h, qd),
                (l.gate, f, h),
                (l.up, f, h),
                (l.down, h, f),
            ] {
                v.push(Mat { at, rows, cols });
            }
        }
        v
    }

    /// Flatten FP32 master weights in this layout.
    pub fn flatten(&self, w: &MasterWeights) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.total);
        p.extend_from_slice(&w.embed.w);
        for l in &w.layers {
            p.extend_from_slice(&l.attn_norm);
            for m in [&l.wq, &l.wk, &l.wv, &l.wo] {
                p.extend_from_slice(&m.w);
            }
            p.extend_from_slice(&l.mlp_norm);
            for m in [&l.w_gate, &l.w_up, &l.w_down] {
                p.extend_from_slice(&m.w);
            }
        }
        p.extend_from_slice(&w.final_norm);
        assert_eq!(p.len(), self.total);
        p
    }

    /// FP32 master weights from a flat vector in this layout.
    pub fn unflatten(&self, p: &[f32]) -> MasterWeights {
        assert_eq!(p.len(), self.total);
        let c = &self.config;
        let (h, qd, kvw, f) = (c.hidden, c.q_heads * c.head_dim, c.kv_width(), c.ffn);
        let m = |at: usize, rows: usize, cols: usize| MasterMatrix {
            rows,
            cols,
            w: p[at..at + rows * cols].to_vec(),
        };
        MasterWeights {
            embed: m(0, c.vocab, h),
            layers: self
                .layers
                .iter()
                .map(|l| MasterLayer {
                    attn_norm: p[l.attn_norm..l.attn_norm + h].to_vec(),
                    wq: m(l.wq, qd, h),
                    wk: m(l.wk, kvw, h),
                    wv: m(l.wv, kvw, h),
                    wo: m(l.wo, h, qd),
                    mlp_norm: p[l.mlp_norm..l.mlp_norm + h].to_vec(),
                    w_gate: m(l.gate, f, h),
                    w_up: m(l.up, f, h),
                    w_down: m(l.down, h, f),
                })
                .collect(),
            final_norm: p[self.final_norm..self.final_norm + h].to_vec(),
        }
    }

    /// The parameters with every matrix transposed (inputs × outputs), so a
    /// forward product can run many outputs' sequential sums side by side.
    pub fn transpose<T: Real>(&self, p: &[T]) -> Vec<T> {
        let mut t = p.to_vec();
        for m in self.matrices() {
            for o in 0..m.rows {
                for k in 0..m.cols {
                    t[m.at + k * m.rows + o] = p[m.at + o * m.cols + k];
                }
            }
        }
        t
    }
}

/// RoPE cos/sin tables in `T`, from the same angles as the inference model:
/// `θ_i = e^(−(2i/d)·ln base)` and `sin_cos(pos·θ_i)` in binary64, rounded once.
#[derive(Clone, Debug)]
pub struct RopeTable<T> {
    half: usize,
    cos: Vec<T>,
    sin: Vec<T>,
}

impl<T: Real> RopeTable<T> {
    pub fn new(c: &DecoderConfig, positions: usize) -> RopeTable<T> {
        let half = c.head_dim / 2;
        let ln_base = ln(c.rope_base);
        let theta: Vec<f64> = (0..half)
            .map(|i| exp(-(((2 * i) as f64 / c.head_dim as f64) * ln_base)))
            .collect();
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for pos in 0..positions {
            for &th in &theta {
                let (s, co) = sin_cos(pos as f64 * th);
                cos.push(T::from_f64(co));
                sin.push(T::from_f64(s));
            }
        }
        RopeTable { half, cos, sin }
    }

    fn apply(&self, x: &mut [T], pos: usize, d: usize) {
        let (c, s) = (&self.cos[pos * self.half..], &self.sin[pos * self.half..]);
        for head in x.chunks_mut(d) {
            for i in 0..self.half {
                let (a, b) = (head[i], head[i + self.half]);
                head[i] = a * c[i] - b * s[i];
                head[i + self.half] = b * c[i] + a * s[i];
            }
        }
    }

    /// The transpose of [`Self::apply`] (gradient with respect to the input).
    fn apply_back(&self, g: &mut [T], pos: usize, d: usize) {
        let (c, s) = (&self.cos[pos * self.half..], &self.sin[pos * self.half..]);
        for head in g.chunks_mut(d) {
            for i in 0..self.half {
                let (ga, gb) = (head[i], head[i + self.half]);
                head[i] = ga * c[i] + gb * s[i];
                head[i + self.half] = gb * c[i] - ga * s[i];
            }
        }
    }
}

/// `y[t][o] = Σ_k W[o][k]·x[t][k]`, sequential in k from zero, with `wt` the
/// transposed matrix (`wt[k·rows + o]`).
fn mm_fwd<T: Real>(wt: &[T], rows: usize, cols: usize, x: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; n * rows];
    for t in 0..n {
        let yt = &mut y[t * rows..(t + 1) * rows];
        for k in 0..cols {
            let xk = x[t * cols + k];
            for (acc, &w) in yt.iter_mut().zip(&wt[k * rows..(k + 1) * rows]) {
                *acc += w * xk;
            }
        }
    }
    y
}

/// Backward of [`mm_fwd`] with the row-major `w`: returns `dx[t][k] = Σ_o
/// W[o][k]·dy[t][o]` (sequential in o) and adds `dy[t][o]·x[t][k]` to
/// `dw[o][k]` over t in order.
fn mm_back<T: Real>(w: &[T], rows: usize, cols: usize, x: &[T], dy: &[T], n: usize, dw: &mut [T]) -> Vec<T> {
    let mut dx = vec![T::ZERO; n * cols];
    for t in 0..n {
        let xt = &x[t * cols..(t + 1) * cols];
        let dxt = &mut dx[t * cols..(t + 1) * cols];
        for o in 0..rows {
            let g = dy[t * rows + o];
            let row = &w[o * cols..(o + 1) * cols];
            for (d, &wk) in dxt.iter_mut().zip(row) {
                *d += wk * g;
            }
            for (d, &xk) in dw[o * cols..(o + 1) * cols].iter_mut().zip(xt) {
                *d += g * xk;
            }
        }
    }
    dx
}

/// RMSNorm of each row, as [`crate::decoder::forward::rms_norm`]; returns
/// the outputs and each row's `1/√(ms + ε)`.
fn rms_fwd<T: Real>(x: &[T], gain: &[T], eps: T, n: usize) -> (Vec<T>, Vec<T>) {
    let d = gain.len();
    let mut y = Vec::with_capacity(x.len());
    let mut invs = Vec::with_capacity(n);
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let mut ss = T::ZERO;
        for &v in row {
            ss += v * v;
        }
        let ms = ss / T::from_f64(d as f64);
        let inv = T::ONE / (ms + eps).sqrt();
        y.extend(row.iter().zip(gain).map(|(&v, &g)| (v * inv) * g));
        invs.push(inv);
    }
    (y, invs)
}

/// Backward of [`rms_fwd`]: returns dx, adds to the gain's gradient.
fn rms_back<T: Real>(x: &[T], gain: &[T], invs: &[T], dy: &[T], dgain: &mut [T]) -> Vec<T> {
    let d = gain.len();
    let half = T::from_f64(0.5);
    let two_over_d = T::from_f64(2.0) / T::from_f64(d as f64);
    let mut dx = Vec::with_capacity(x.len());
    for (t, &inv) in invs.iter().enumerate() {
        let (row, dyr) = (&x[t * d..(t + 1) * d], &dy[t * d..(t + 1) * d]);
        let mut s = T::ZERO;
        for i in 0..d {
            s += dyr[i] * gain[i] * row[i];
        }
        let dms = -(s * half * inv * inv * inv);
        for i in 0..d {
            dx.push(dyr[i] * gain[i] * inv + dms * two_over_d * row[i]);
            dgain[i] += dyr[i] * (row[i] * inv);
        }
    }
    dx
}

struct LayerCache<T> {
    xin: Vec<T>,
    n1: Vec<T>,
    inv1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    /// probs[(head·n + i)·n + j] for j ≤ i.
    probs: Vec<T>,
    attn: Vec<T>,
    xmid: Vec<T>,
    n2: Vec<T>,
    inv2: Vec<T>,
    gate: Vec<T>,
    up: Vec<T>,
    sig: Vec<T>,
    silu: Vec<T>,
    act: Vec<T>,
}

/// The parameters for one step: row-major and transposed copies, and RoPE tables.
pub struct Net<'a, T> {
    pub layout: &'a Layout,
    pub p: &'a [T],
    pub pt: &'a [T],
    pub rope: &'a RopeTable<T>,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<TrainMutation>,
}

/// Deliberate bugs for the planted-bug tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainMutation {
    /// The backward pass rotates RoPE gradients forward instead of by the transpose.
    RopeBackwardNotTransposed,
    /// The forward pass pairs adjacent dimensions (2i, 2i + 1) instead of
    /// rotate-half (i, i + d/2), consistently in backward too.
    RopeInterleavedPairs,
}

/// Rotate adjacent pairs (the planted forward-convention bug); `sign` −1 is the transpose.
#[cfg(feature = "planted-bugs")]
fn rope_interleaved<T: Real>(r: &RopeTable<T>, x: &mut [T], pos: usize, d: usize, transpose: bool) {
    let (c, s) = (&r.cos[pos * r.half..], &r.sin[pos * r.half..]);
    for head in x.chunks_mut(d) {
        for i in 0..r.half {
            let (a, b) = (head[2 * i], head[2 * i + 1]);
            let si = if transpose { -s[i] } else { s[i] };
            head[2 * i] = a * c[i] - b * si;
            head[2 * i + 1] = b * c[i] + a * si;
        }
    }
}

/// One sequence's forward pass: logits at every position, and the cache.
struct Forward<T> {
    layers: Vec<LayerCache<T>>,
    xfinal: Vec<T>,
    nf: Vec<T>,
    invf: Vec<T>,
    logits: Vec<T>,
}

impl<'a, T: Real> Net<'a, T> {
    pub fn new(layout: &'a Layout, p: &'a [T], pt: &'a [T], rope: &'a RopeTable<T>) -> Net<'a, T> {
        Net {
            layout,
            p,
            pt,
            rope,
            #[cfg(feature = "planted-bugs")]
            planted: None,
        }
    }

    fn rope_fwd(&self, x: &mut [T], pos: usize, d: usize) {
        #[cfg(feature = "planted-bugs")]
        if self.planted == Some(TrainMutation::RopeInterleavedPairs) {
            return rope_interleaved(self.rope, x, pos, d, false);
        }
        self.rope.apply(x, pos, d)
    }

    fn rope_back(&self, g: &mut [T], pos: usize, d: usize) {
        #[cfg(feature = "planted-bugs")]
        match self.planted {
            Some(TrainMutation::RopeInterleavedPairs) => return rope_interleaved(self.rope, g, pos, d, true),
            Some(TrainMutation::RopeBackwardNotTransposed) => return self.rope.apply(g, pos, d),
            None => {}
        }
        self.rope.apply_back(g, pos, d)
    }

    fn forward(&self, tokens: &[u32]) -> Forward<T> {
        let c = &self.layout.config;
        let (h, d, n) = (c.hidden, c.head_dim, tokens.len());
        let (qd, kvw, f, group) = (c.q_heads * d, c.kv_width(), c.ffn, c.group());
        let (p, pt) = (self.p, self.pt);
        let eps = T::from_f64(f64::from(c.rms_eps));
        let scale = T::ONE / T::from_f64(d as f64).sqrt();
        let mut x: Vec<T> = Vec::with_capacity(n * h);
        for &tok in tokens {
            let t = tok as usize;
            assert!(t < c.vocab, "token outside the vocabulary");
            x.extend_from_slice(&p[t * h..(t + 1) * h]);
        }
        let mut layers = Vec::with_capacity(c.layers);
        for lo in &self.layout.layers {
            let xin = x.clone();
            let (n1, inv1) = rms_fwd(&xin, &p[lo.attn_norm..lo.attn_norm + h], eps, n);
            let mut q = mm_fwd(&pt[lo.wq..], qd, h, &n1, n);
            let mut k = mm_fwd(&pt[lo.wk..], kvw, h, &n1, n);
            let v = mm_fwd(&pt[lo.wv..], kvw, h, &n1, n);
            for t in 0..n {
                self.rope_fwd(&mut q[t * qd..(t + 1) * qd], t, d);
                self.rope_fwd(&mut k[t * kvw..(t + 1) * kvw], t, d);
            }
            let mut probs = vec![T::ZERO; c.q_heads * n * n];
            let mut attn = vec![T::ZERO; n * qd];
            for head in 0..c.q_heads {
                let g = head / group;
                for i in 0..n {
                    let qi = &q[i * qd + head * d..i * qd + (head + 1) * d];
                    let row = &mut probs[(head * n + i) * n..(head * n + i) * n + i + 1];
                    for (j, s) in row.iter_mut().enumerate() {
                        let kj = &k[j * kvw + g * d..j * kvw + (g + 1) * d];
                        let mut acc = T::ZERO;
                        for (&a, &b) in qi.iter().zip(kj) {
                            acc += a * b;
                        }
                        *s = acc * scale;
                    }
                    let m = row.iter().copied().fold(row[0], T::max);
                    let mut sum = T::ZERO;
                    for s in row.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    for s in row.iter_mut() {
                        *s = *s / sum;
                    }
                    let out = &mut attn[i * qd + head * d..i * qd + (head + 1) * d];
                    for (j, &pj) in row.iter().enumerate() {
                        let vj = &v[j * kvw + g * d..j * kvw + (g + 1) * d];
                        for (o, &b) in out.iter_mut().zip(vj) {
                            *o += pj * b;
                        }
                    }
                }
            }
            let y = mm_fwd(&pt[lo.wo..], h, qd, &attn, n);
            for (a, b) in x.iter_mut().zip(&y) {
                *a += *b;
            }
            let xmid = x.clone();
            let (n2, inv2) = rms_fwd(&xmid, &p[lo.mlp_norm..lo.mlp_norm + h], eps, n);
            let gate = mm_fwd(&pt[lo.gate..], f, h, &n2, n);
            let up = mm_fwd(&pt[lo.up..], f, h, &n2, n);
            let mut sig = Vec::with_capacity(n * f);
            let mut silu = Vec::with_capacity(n * f);
            let mut act = Vec::with_capacity(n * f);
            for (&g, &u) in gate.iter().zip(&up) {
                let e = T::ONE + (-g).exp();
                sig.push(T::ONE / e);
                let s = g / e;
                silu.push(s);
                act.push(s * u);
            }
            let y = mm_fwd(&pt[lo.down..], h, f, &act, n);
            for (a, b) in x.iter_mut().zip(&y) {
                *a += *b;
            }
            layers.push(LayerCache {
                xin,
                n1,
                inv1,
                q,
                k,
                v,
                probs,
                attn,
                xmid,
                n2,
                inv2,
                gate,
                up,
                sig,
                silu,
                act,
            });
        }
        let fo = self.layout.final_norm;
        let (nf, invf) = rms_fwd(&x, &p[fo..fo + h], eps, n);
        let logits = mm_fwd(&pt[0..], c.vocab, h, &nf, n);
        Forward {
            layers,
            xfinal: x,
            nf,
            invf,
            logits,
        }
    }

    /// Logits (`tokens.len() × vocab`) of the full causal forward pass.
    pub fn logits(&self, tokens: &[u32]) -> Vec<T> {
        self.forward(tokens).logits
    }

    /// Cross-entropy of `targets` (position, token) given `tokens`, each term
    /// weighted by `weight`. Adds the gradient to `grad`; returns the summed
    /// (unweighted) loss in binary64 and how many targets the argmax got right.
    pub fn loss_and_grad(&self, tokens: &[u32], targets: &[(usize, u32)], weight: T, grad: &mut [T]) -> (f64, usize) {
        let c = &self.layout.config;
        let (h, d, n) = (c.hidden, c.head_dim, tokens.len());
        let (qd, kvw, f, group, vocab) = (c.q_heads * d, c.kv_width(), c.ffn, c.group(), c.vocab);
        let p = self.p;
        let fw = self.forward(tokens);
        // Softmax cross-entropy at the target positions.
        let mut dlogits = vec![T::ZERO; n * vocab];
        let mut loss = 0.0;
        let mut correct = 0;
        for &(pos, target) in targets {
            let z = &fw.logits[pos * vocab..(pos + 1) * vocab];
            let m = z.iter().copied().fold(z[0], T::max);
            let mut best = 0;
            for (i, &v) in z.iter().enumerate() {
                if v > z[best] {
                    best = i;
                }
            }
            correct += usize::from(best == target as usize);
            let mut e: Vec<T> = z.iter().map(|&v| (v - m).exp()).collect();
            let mut sum = T::ZERO;
            for &v in &e {
                sum += v;
            }
            for v in e.iter_mut() {
                *v = *v / sum;
            }
            // Loss in binary64: log Σ e^(z − m) + m − z_target.
            let m64 = m.to_f64();
            let mut s64 = 0.0;
            for &v in z {
                s64 += exp((v.to_f64() - m64).max(-700.0));
            }
            loss += ln(s64) + m64 - z[target as usize].to_f64();
            let dz = &mut dlogits[pos * vocab..(pos + 1) * vocab];
            for (i, (g, &pi)) in dz.iter_mut().zip(&e).enumerate() {
                let y = if i == target as usize { T::ONE } else { T::ZERO };
                *g = (pi - y) * weight;
            }
        }
        // Head (tied embedding) and final norm.
        let dnf = mm_back(&p[0..], vocab, h, &fw.nf, &dlogits, n, &mut grad[0..vocab * h]);
        let fo = self.layout.final_norm;
        let mut dx = rms_back(&fw.xfinal, &p[fo..fo + h], &fw.invf, &dnf, &mut grad[fo..fo + h]);
        let scale = T::ONE / T::from_f64(d as f64).sqrt();
        for (lo, lc) in self.layout.layers.iter().zip(&fw.layers).rev() {
            // MLP: x_out = x_mid + W_down · (silu(gate) ⊙ up).
            let dact = mm_back(
                &p[lo.down..],
                h,
                f,
                &lc.act,
                &dx,
                n,
                &mut grad[lo.down..lo.down + h * f],
            );
            let mut dgate = Vec::with_capacity(n * f);
            let mut dup = Vec::with_capacity(n * f);
            for (i, &da) in dact.iter().enumerate() {
                let (g, s) = (lc.gate[i], lc.sig[i]);
                dup.push(da * lc.silu[i]);
                dgate.push(da * lc.up[i] * (s * (T::ONE + g * (T::ONE - s))));
            }
            let dn2g = mm_back(
                &p[lo.gate..],
                f,
                h,
                &lc.n2,
                &dgate,
                n,
                &mut grad[lo.gate..lo.gate + f * h],
            );
            let dn2u = mm_back(&p[lo.up..], f, h, &lc.n2, &dup, n, &mut grad[lo.up..lo.up + f * h]);
            let dn2: Vec<T> = dn2g.iter().zip(&dn2u).map(|(&a, &b)| a + b).collect();
            let dmid = rms_back(
                &lc.xmid,
                &p[lo.mlp_norm..lo.mlp_norm + h],
                &lc.inv2,
                &dn2,
                &mut grad[lo.mlp_norm..lo.mlp_norm + h],
            );
            for (a, b) in dx.iter_mut().zip(&dmid) {
                *a += *b;
            }
            // Attention: x_mid = x_in + W_o · attn.
            let dattn = mm_back(&p[lo.wo..], h, qd, &lc.attn, &dx, n, &mut grad[lo.wo..lo.wo + h * qd]);
            let mut dq = vec![T::ZERO; n * qd];
            let mut dk = vec![T::ZERO; n * kvw];
            let mut dv = vec![T::ZERO; n * kvw];
            let mut dp = vec![T::ZERO; n];
            for head in 0..c.q_heads {
                let g = head / group;
                for i in 0..n {
                    let probs = &lc.probs[(head * n + i) * n..(head * n + i) * n + i + 1];
                    let dout = &dattn[i * qd + head * d..i * qd + (head + 1) * d];
                    let mut dot = T::ZERO;
                    for (j, &pj) in probs.iter().enumerate() {
                        let vj = &lc.v[j * kvw + g * d..j * kvw + (g + 1) * d];
                        let mut acc = T::ZERO;
                        for (&a, &b) in dout.iter().zip(vj) {
                            acc += a * b;
                        }
                        dp[j] = acc;
                        for (dvv, &o) in dv[j * kvw + g * d..j * kvw + (g + 1) * d].iter_mut().zip(dout) {
                            *dvv += pj * o;
                        }
                        dot += pj * acc;
                    }
                    for (j, &pj) in probs.iter().enumerate() {
                        let ds = pj * (dp[j] - dot) * scale;
                        for e in 0..d {
                            dq[i * qd + head * d + e] += ds * lc.k[j * kvw + g * d + e];
                            dk[j * kvw + g * d + e] += ds * lc.q[i * qd + head * d + e];
                        }
                    }
                }
            }
            for t in 0..n {
                self.rope_back(&mut dq[t * qd..(t + 1) * qd], t, d);
                self.rope_back(&mut dk[t * kvw..(t + 1) * kvw], t, d);
            }
            let a = mm_back(&p[lo.wq..], qd, h, &lc.n1, &dq, n, &mut grad[lo.wq..lo.wq + qd * h]);
            let b = mm_back(&p[lo.wk..], kvw, h, &lc.n1, &dk, n, &mut grad[lo.wk..lo.wk + kvw * h]);
            let cc = mm_back(&p[lo.wv..], kvw, h, &lc.n1, &dv, n, &mut grad[lo.wv..lo.wv + kvw * h]);
            let dn1: Vec<T> = (0..n * h).map(|i| a[i] + b[i] + cc[i]).collect();
            let din = rms_back(
                &lc.xin,
                &p[lo.attn_norm..lo.attn_norm + h],
                &lc.inv1,
                &dn1,
                &mut grad[lo.attn_norm..lo.attn_norm + h],
            );
            for (a, b) in dx.iter_mut().zip(&din) {
                *a += *b;
            }
        }
        // Embedding rows.
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &mut grad[tok as usize * h..(tok as usize + 1) * h];
            for (g, &v) in row.iter_mut().zip(&dx[t * h..(t + 1) * h]) {
                *g += v;
            }
        }
        (loss, correct)
    }
}

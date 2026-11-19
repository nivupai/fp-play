//! FP8 (E4M3, E5M2) and INT8 with explicit scales.
//!
//! # Formats
//!
//! * **E4M3** (OCP FP8): 1 sign, 4 exponent (bias 7), 3 fraction bits.
//!   Subnormals; no infinities; the only NaNs are `S.1111.111` (0x7f, 0xff).
//!   Largest finite 448 = 1.75 × 2⁸, smallest subnormal 2⁻⁹.
//! * **E5M2** (OCP FP8): 1 sign, 5 exponent (bias 15), 2 fraction bits,
//!   IEEE 754-style: subnormals, ±∞ (`S.11111.00`), NaNs (`S.11111.xx`,
//!   xx ≠ 0). Largest finite 57344 = 1.75 × 2¹⁵, smallest subnormal 2⁻¹⁶.
//! * **INT8**: signed integers in the symmetric range −127 … 127 (−128 is
//!   never produced).
//!
//! # Quantization
//!
//! A binary32 tensor slice (a weight row, an activation vector, one KV
//! head's K or V of one token) gets one binary32 scale
//! `s = fl32(absmax ÷ Qmax)` with Qmax = 448, 57344 or 127 (`s = 1` when the
//! slice is all zeros). Each element is `q = Q(w ÷ s)` where the quotient is
//! **exact** (integer division of the significands, no intermediate
//! rounding) and Q rounds once to nearest, ties to even, onto the format's
//! grid and **saturates** to ±Qmax (OCP's "saturating" conversion; INT8
//! clamps to ±127). Saturation can only be reached by rounding: |w ÷ s| is
//! at most slightly above Qmax because s is rounded. A non-finite element
//! makes every output of the product NaN (`apply` reports this as NaN).
//!
//! `tests/narrow.rs` checks all 256 patterns of each FP8 format (decoding
//! against an independent table, encode ∘ decode = identity, ordering, the
//! NaN and infinity counts), every midpoint between adjacent values and
//! its neighbours, and the quantizer against independent Python `Fraction`
//! vectors (`scripts/gen_narrow_vectors.py`).

/// An 8-bit format with an explicit scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Narrow {
    E4M3,
    E5M2,
    Int8,
}

impl Narrow {
    pub const ALL: [Narrow; 3] = [Narrow::E4M3, Narrow::E5M2, Narrow::Int8];

    pub const fn name(self) -> &'static str {
        match self {
            Narrow::E4M3 => "fp8-e4m3",
            Narrow::E5M2 => "fp8-e5m2",
            Narrow::Int8 => "int8",
        }
    }

    /// The largest magnitude (the scale maps a slice's absmax onto it).
    pub const fn qmax(self) -> f32 {
        match self {
            Narrow::E4M3 => 448.0,
            Narrow::E5M2 => 57344.0,
            Narrow::Int8 => 127.0,
        }
    }

    /// (fraction bits, exponent bias) of the FP8 formats.
    const fn fp8(self) -> Option<(i32, i32)> {
        match self {
            Narrow::E4M3 => Some((3, 7)),
            Narrow::E5M2 => Some((2, 15)),
            Narrow::Int8 => None,
        }
    }

    /// Exponent of the least significant bit at magnitude exponent `e`
    /// (`2^e ≤ |v| < 2^(e+1)`).
    fn lsb(self, e: i32) -> i32 {
        match self.fp8() {
            Some((m, bias)) => e.max(1 - bias) - m,
            None => 0,
        }
    }
}

/// The value of an FP8 pattern (NaN as `f64::NAN`, ±∞ for E5M2).
pub fn decode(f: Narrow, raw: u8) -> f64 {
    let (m, bias) = f.fp8().expect("decode: FP8 formats only");
    let e_bits = 7 - m;
    let sign = if raw & 0x80 != 0 { -1.0 } else { 1.0 };
    let field = i32::from((raw >> m) & ((1 << e_bits) - 1));
    let frac = i32::from(raw & ((1 << m) - 1));
    let top = (1 << e_bits) - 1;
    match f {
        Narrow::E4M3 if field == top && frac == (1 << m) - 1 => return f64::NAN,
        Narrow::E5M2 if field == top => return if frac == 0 { sign * f64::INFINITY } else { f64::NAN },
        _ => {}
    }
    let mag = if field == 0 {
        f64::from(frac) * pow2(1 - bias - m)
    } else {
        f64::from((1 << m) + frac) * pow2(field - bias - m)
    };
    sign * mag
}

/// The FP8 pattern of a value of the format (±0 kept; NaN gives 0x7f for
/// E4M3 and 0x7e for E5M2; ±∞ only for E5M2). Panics if `v` is not a value
/// of the format.
pub fn encode(f: Narrow, v: f64) -> u8 {
    let (m, bias) = f.fp8().expect("encode: FP8 formats only");
    if v.is_nan() {
        return if f == Narrow::E4M3 { 0x7f } else { 0x7e };
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let mag = v.abs();
    if mag == 0.0 {
        return sign;
    }
    if mag.is_infinite() {
        assert_eq!(f, Narrow::E5M2, "E4M3 has no infinity");
        return sign | 0x7c;
    }
    let e = exponent(mag);
    let (field, frac) = if e < 1 - bias {
        (0, mag / pow2(1 - bias - m))
    } else {
        (e + bias, mag / pow2(e - m) - f64::from(1 << m))
    };
    assert!(
        frac.fract() == 0.0 && frac >= 0.0 && frac < f64::from(1 << m),
        "{v} is not a {} value",
        f.name()
    );
    let raw = sign | ((field as u8) << m) | frac as u8;
    assert!(decode(f, raw) == v, "{v} is not a {} value", f.name());
    raw
}

fn pow2(k: i32) -> f64 {
    assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// floor(log2 |x|) of a normal binary64.
fn exponent(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// (significand, exponent) with `|x| = M · 2^e`, M < 2^24, of a finite non-zero binary32.
fn parts(x: f32) -> (u128, i32) {
    let b = x.to_bits();
    let field = ((b >> 23) & 0xff) as i32;
    let frac = u128::from(b & 0x7f_ffff);
    if field == 0 {
        (frac, -149)
    } else {
        (frac | (1 << 23), field - 150)
    }
}

/// Quantization options (planted bugs flip these in `planted-bugs` builds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Options {
    ties_away: bool,
    int8_min_128: bool,
}

/// `Q(w ÷ s)`: the exact quotient rounded once to nearest even on the
/// format's grid, saturated to ±Qmax; returned as a binary64 value (an FP8
/// value or an integer). `s` must be finite and positive; NaN or ±∞ `w`
/// gives NaN.
pub fn quantize(f: Narrow, w: f32, s: f32) -> f64 {
    quantize_with(f, w, s, Options::default())
}

/// Deliberate bugs for the planted-bug tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarrowMutation {
    /// Exact ties round away from zero instead of to even.
    TiesAwayFromZero,
    /// INT8 clamps to −128 … 127 instead of the symmetric range.
    Int8ClampsToMinus128,
}

#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
pub fn quantize_planted(f: Narrow, w: f32, s: f32, m: NarrowMutation) -> f64 {
    let o = Options {
        ties_away: m == NarrowMutation::TiesAwayFromZero,
        int8_min_128: m == NarrowMutation::Int8ClampsToMinus128,
    };
    quantize_with(f, w, s, o)
}

fn quantize_with(f: Narrow, w: f32, s: f32, o: Options) -> f64 {
    assert!(s.is_finite() && s > 0.0, "scale must be finite and positive");
    if !w.is_finite() {
        return f64::NAN;
    }
    let negative = w.is_sign_negative();
    let signed = |mag: f64| if negative { -mag } else { mag };
    if w == 0.0 {
        return signed(0.0);
    }
    let qmax = f64::from(f.qmax());
    let (mw, ew) = parts(w);
    let (ms, es) = parts(s);
    // E = floor(log2(w / s)).
    let bits = |m: u128| 127 - m.leading_zeros() as i32;
    let k = bits(mw) - bits(ms);
    let below = if k >= 0 { mw < ms << k } else { mw << -k < ms };
    let e = k - i32::from(below) + (ew - es);
    let lsb = f.lsb(e);
    // Far above the range: saturate; far below half the quantum: zero.
    if e > 20 {
        return signed(saturate(f, qmax, o, negative));
    }
    if e < lsb - 1 {
        return signed(0.0);
    }
    let sh = ew - es - lsb;
    let (num, den) = if sh >= 0 { (mw << sh, ms) } else { (mw, ms << -sh) };
    let (mut n, r) = (num / den, num % den);
    let up = match (2 * r).cmp(&den) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => o.ties_away || n % 2 == 1,
    };
    n += u128::from(up);
    let mag = n as f64 * pow2(lsb);
    if mag > qmax {
        return signed(saturate(f, qmax, o, negative));
    }
    signed(mag)
}

fn saturate(f: Narrow, qmax: f64, o: Options, negative: bool) -> f64 {
    if f == Narrow::Int8 && o.int8_min_128 && negative {
        128.0
    } else {
        qmax
    }
}

/// The scale of a slice: `fl32(absmax ÷ Qmax)`, or 1 for an all-zero slice;
/// `None` if any element is NaN or infinite.
pub fn scale(f: Narrow, x: &[f32]) -> Option<f32> {
    let mut absmax = 0.0f32;
    for &v in x {
        if !v.is_finite() {
            return None;
        }
        absmax = absmax.max(v.abs());
    }
    Some(if absmax == 0.0 { 1.0 } else { absmax / f.qmax() })
}

/// A matrix quantized per output row, column-major values (`q[k·rows + o]`).
#[derive(Clone, Debug)]
pub struct ScaledMatrix {
    pub format: Narrow,
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<f64>,
    pub scales: Vec<f32>,
}

impl ScaledMatrix {
    /// Quantize a row-major binary32 matrix with one scale per row.
    pub fn new(format: Narrow, rows: usize, cols: usize, w: &[f32]) -> ScaledMatrix {
        assert_eq!(w.len(), rows * cols);
        let mut q = vec![0.0; rows * cols];
        let mut scales = Vec::with_capacity(rows);
        for o in 0..rows {
            let row = &w[o * cols..(o + 1) * cols];
            let s = scale(format, row).expect("finite weights");
            for (k, &v) in row.iter().enumerate() {
                q[k * rows + o] = quantize(format, v, s);
            }
            scales.push(s);
        }
        ScaledMatrix {
            format,
            rows,
            cols,
            q,
            scales,
        }
    }

    /// Row `o` dequantized to binary32: `fl32(q · s)`.
    pub fn row(&self, o: usize) -> Vec<f32> {
        (0..self.cols)
            .map(|k| (self.q[k * self.rows + o] as f32) * self.scales[o])
            .collect()
    }

    /// `y[o] = fl32(fl32(acc[o] · s_w[o]) · s_x)` with `x` quantized by its
    /// own scale and `acc[o]` the dot product of the quantized values: for
    /// FP8 a fused FP32 chain (products of FP8 values are exact) in
    /// increasing k from +0, for INT8 an exact integer sum rounded once to
    /// binary32. NaN everywhere if `x` has a non-finite element.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let Some(sx) = scale(self.format, x) else {
            return vec![f32::NAN; self.rows];
        };
        let xq: Vec<f64> = x.iter().map(|&v| quantize(self.format, v, sx)).collect();
        let acc: Vec<f32> = if self.format == Narrow::Int8 {
            let mut acc = vec![0i64; self.rows];
            for (k, &xk) in xq.iter().enumerate() {
                let xi = xk as i64;
                for (a, &w) in acc.iter_mut().zip(&self.q[k * self.rows..(k + 1) * self.rows]) {
                    *a += (w as i64) * xi;
                }
            }
            // `as f32` from i64 rounds to nearest, ties to even.
            acc.into_iter().map(|a| a as f32).collect()
        } else {
            let mut acc = vec![0.0f64; self.rows];
            for (k, &xk) in xq.iter().enumerate() {
                for (a, &w) in acc.iter_mut().zip(&self.q[k * self.rows..(k + 1) * self.rows]) {
                    *a = super::mac::mac(crate::format::Format::Fp32, *a, w, xk);
                }
            }
            acc.into_iter().map(|a| a as f32).collect()
        };
        acc.iter().zip(&self.scales).map(|(&a, &sw)| (a * sw) * sx).collect()
    }
}

/// A vector stored quantized with its own scale, dequantized to binary32
/// (`fl32(q · s)`): how the KV cache stores one KV head's K or V of one token.
pub fn store(f: Narrow, x: &[f32]) -> Vec<f32> {
    match scale(f, x) {
        None => vec![f32::NAN; x.len()],
        Some(s) => x.iter().map(|&v| (quantize(f, v, s) as f32) * s).collect(),
    }
}

/// Every adjacent pair of non-negative finite values of an FP8 format, in
/// increasing order (for the exhaustive checks).
pub fn positive_values(f: Narrow) -> Vec<f64> {
    let mut v: Vec<f64> = (0u8..0x80).map(|r| decode(f, r)).filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v
}

/// The exhaustive midpoint check: for every pair of adjacent non-negative
/// values (and the gap above the largest), the exact midpoint goes to the
/// even neighbour (the larger one at the top, by saturation), the binary32
/// values just below and above it go down and up, each in both signs, with
/// scale 1 and with scale 2⁻³. INT8: the same for every integer 0…127.
/// `quantizer` is [`quantize`] (or a planted variant).
pub fn midpoint_check(quantizer: impl Fn(Narrow, f32, f32) -> f64) -> Result<usize, String> {
    let mut checked = 0;
    for f in Narrow::ALL {
        let grid: Vec<f64> = match f {
            Narrow::Int8 => (0..=127).map(f64::from).collect(),
            _ => positive_values(f),
        };
        let even = |v: f64| match f {
            Narrow::Int8 => v as i64 % 2 == 0,
            _ => encode(f, v) & 1 == 0,
        };
        for pair in grid.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let mid = (lo + hi) / 2.0;
            let tie = if even(lo) { lo } else { hi };
            for s in [1.0f32, 0.125] {
                let m = (mid * f64::from(s)) as f32;
                assert_eq!(f64::from(m), mid * f64::from(s), "midpoints are binary32 values");
                let cases = [
                    (m, tie),
                    (f32::from_bits(m.to_bits() - 1), lo),
                    (f32::from_bits(m.to_bits() + 1), hi),
                ];
                for (w, want) in cases {
                    for sign in [1.0f32, -1.0] {
                        let got = quantizer(f, sign * w, s);
                        let want = f64::from(sign) * want;
                        if got != want || got.is_sign_negative() != want.is_sign_negative() {
                            return Err(format!("{}: Q({} / {s}) = {got}, expected {want}", f.name(), sign * w));
                        }
                        checked += 1;
                    }
                }
            }
        }
        // Above the largest value: saturate.
        let top = *grid.last().expect("non-empty");
        for w in [top * 1.5, top * 4.0, 1e30] {
            for sign in [1.0, -1.0] {
                let got = quantizer(f, (sign * w) as f32, 1.0);
                if got != sign * top {
                    return Err(format!(
                        "{}: Q({}) = {got}, expected {}",
                        f.name(),
                        sign * w,
                        sign * top
                    ));
                }
                checked += 1;
            }
        }
    }
    Ok(checked)
}

/// Compare `quantizer` with the Python `Fraction` vectors
/// (`tests/vectors/narrow.txt`, lines `q <format> <w bits> <s bits> <expected>`
/// where expected is an FP8 pattern in hex or an INT8 integer). Returns the
/// number of lines checked.
pub fn vector_check(text: &str, quantizer: impl Fn(Narrow, f32, f32) -> f64) -> Result<usize, String> {
    let mut n = 0;
    for line in text.lines().filter(|l| l.starts_with("q ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let format = Narrow::ALL
            .into_iter()
            .find(|x| x.name() == f[1])
            .ok_or(format!("unknown format in `{line}`"))?;
        let w = f32::from_bits(u32::from_str_radix(f[2], 16).map_err(|e| e.to_string())?);
        let s = f32::from_bits(u32::from_str_radix(f[3], 16).map_err(|e| e.to_string())?);
        let got = quantizer(format, w, s);
        let ok = match format {
            Narrow::Int8 => {
                let want: i64 = f[4].parse().map_err(|_| format!("bad integer in `{line}`"))?;
                got == want as f64
            }
            _ => {
                let want = u8::from_str_radix(f[4], 16).map_err(|e| e.to_string())?;
                encode(format, got) == want
            }
        };
        if !ok {
            return Err(format!("`{line}`: got {got}"));
        }
        n += 1;
    }
    Ok(n)
}

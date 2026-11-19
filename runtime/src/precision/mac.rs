//! A fast fused multiply–accumulate in FP32, BF16 or FP16.
//!
//! `mac(F, acc, a, b) = round_F(acc + a·b)`: the product is exact and the sum
//! is rounded once to F, round to nearest, ties to even — the semantics of
//! [`crate::ops::fused_chain_dot`] (exact product, one rounding per term) and
//! of FP Play's FMA. Values travel as binary64 holding a value of their
//! format.
//!
//! Why it is exact: inputs are FP32/BF16/FP16 values (at most 24
//! significant bits), so `a·b` is exact in binary64 (at most 48 bits). The
//! sum `s = fl64(acc + a·b)` may round, but TwoSum gives its exact error
//! `e`, so `s + e` is the exact sum. Every midpoint between two neighbours
//! of F is a binary64 number (F has at most 24 bits), and `|e|` is at most
//! half a binary64 ulp of `s`, so the exact sum lies on the same side of
//! every F midpoint as `s`, except when `s` *is* a midpoint: then the sign of
//! `e` decides, and `e = 0` is a true tie. [`round_value`] implements exactly
//! that. `tests/precision.rs` checks it bit for bit against
//! `ops::fused_chain_dot` (the exact superaccumulator) and against
//! independent Python `Fraction` vectors (`scripts/gen_chain_vectors.py`).
//!
//! Special values: NaN in, or ∞ − ∞, gives NaN (encoded as the format's
//! canonical NaN); an exact zero sum is −0 only when both addends are −0.

use crate::format::Format;

fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Round the exact value `s + e` to `format` and return it as a binary64.
/// Requires `s = fl64(s + e)` (as TwoSum produces); with `e = 0` this rounds
/// `s` itself. Overflow gives ±∞.
///
/// When the result is in the format's normal range this rounds the binary64
/// bit pattern directly (round to nearest even at the format's last kept
/// bit, the carry running into the exponent); otherwise it uses
/// [`round_value_reference`]. `tests/precision.rs` checks the two agree.
#[inline]
pub fn round_value(format: Format, s: f64, e: f64) -> f64 {
    const SIGN: u64 = 1 << 63;
    let bits = s.to_bits();
    let mag = bits & !SIGN;
    let field = (mag >> 52) as i32;
    if field == 0x7ff || mag == 0 || field - 1023 < format.emin() {
        return round_value_reference(format, s, e);
    }
    let drop = 53 - format.precision();
    let half = 1u64 << (drop - 1);
    let low = mag & ((1u64 << drop) - 1);
    let mut r = mag - low;
    let up = if low != half {
        low > half
    } else if e != 0.0 {
        (e > 0.0) == (s > 0.0)
    } else {
        (mag >> drop) & 1 == 1
    };
    if up {
        r += 1u64 << drop;
    }
    // Overflow: at or above 2^(emax + 1).
    if ((r >> 52) as i32) - 1023 > format.bias() {
        r = 0x7ff0_0000_0000_0000;
    }
    f64::from_bits(r | (bits & SIGN))
}

/// The same rounding by scaling and `floor` (every range, including
/// subnormal results): the reference for [`round_value`]'s fast path.
pub fn round_value_reference(format: Format, s: f64, e: f64) -> f64 {
    if !s.is_finite() || s == 0.0 {
        return s;
    }
    let p = format.precision() as i32;
    let emin = format.emin();
    let emax = format.bias();
    let mag = s.abs();
    let k = ((mag.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    assert!(
        k > -1023,
        "binary64 subnormal sum: outside every modeled format's range"
    );
    let q = k.max(emin) - (p - 1);
    let scaled = mag * pow2(-q);
    let n = scaled.floor();
    let frac = scaled - n;
    let up = if frac > 0.5 {
        true
    } else if frac < 0.5 {
        false
    } else if e != 0.0 {
        (e > 0.0) == (s > 0.0)
    } else {
        n % 2.0 == 1.0
    };
    let m = if up { n + 1.0 } else { n };
    let v = m * pow2(q);
    let v = if v >= pow2(emax + 1) { f64::INFINITY } else { v };
    if s < 0.0 {
        -v
    } else {
        v
    }
}

/// `round_F(acc + a·b)` with one rounding (see the module documentation).
#[inline]
pub fn mac(format: Format, acc: f64, a: f64, b: f64) -> f64 {
    let p = a * b;
    let s = acc + p;
    if !s.is_finite() {
        return s;
    }
    // TwoSum (Knuth): s + e = acc + p exactly.
    let bv = s - acc;
    let av = s - bv;
    let e = (acc - av) + (p - bv);
    round_value(format, s, e)
}

/// Encode a value of `format` (or ±∞, NaN) as its raw pattern; NaN becomes
/// the canonical NaN.
pub fn encode(format: Format, v: f64) -> u32 {
    crate::ops::from_f64(format, v)
}

/// A sequential fused chain from +0: `acc = mac(F, acc, a[k], b[k])` for k in order.
pub fn fused_dot(format: Format, a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        acc = mac(format, acc, x, y);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_break_on_the_twosum_error() {
        // BF16: 1 + 2^-8 is a tie between 1 and 1 + 2^-7.
        let tie = 1.0 + 2f64.powi(-8);
        assert_eq!(round_value(Format::Bf16, tie, 0.0), 1.0);
        assert_eq!(round_value(Format::Bf16, tie, 1e-30), 1.0 + 2f64.powi(-7));
        assert_eq!(round_value(Format::Bf16, tie, -1e-30), 1.0);
        assert_eq!(round_value(Format::Fp16, 70000.0, 0.0), f64::INFINITY);
        assert_eq!(round_value(Format::Fp16, 65519.0, 0.0), 65504.0);
        assert_eq!(round_value(Format::Fp16, 65520.0, 0.0), f64::INFINITY);
        assert_eq!(round_value(Format::Fp16, -2f64.powi(-25), 0.0), -0.0);
        assert_eq!(round_value(Format::Fp16, 3.0 * 2f64.powi(-26), 0.0), 2f64.powi(-24));
    }
}

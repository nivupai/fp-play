//! Correctly rounded operations.
//!
//! Every function returns the correctly rounded result (round to nearest,
//! ties to even, canonical NaN). Most compute the exact result with
//! [`ExactSum`] and round once. Binary32 and BF16 [`add`] and the
//! binary32 ↔ BF16 [`convert`] use the host's IEEE 754 binary32 basic
//! operations instead (a native fast path); `tests/ops_fast_path.rs` tests
//! them equal to the exact path ([`add_exact`], [`convert_exact`]) on
//! subnormals, ties, cancellation, overflow and NaNs as well as random
//! inputs, and checks the exact path against the host independently.

use crate::exact::ExactSum;
use crate::format::{decode, decode_f64, Dyadic, Format, Value};

/// `round(a + b)` in `format`.
///
/// For binary32 and BF16 this uses the host's IEEE 754 binary32 addition,
/// which is correctly rounded (Rust does not flush subnormals). A BF16 sum
/// is added in binary32 and then rounded to BF16: binary32 has at least
/// 2·8 + 2 significant bits, so that double rounding equals one rounding
/// of the exact sum. NaN results are the canonical NaN, as on the exact
/// path. `tests/ops_fast_path.rs` checks both against [`add_exact`]; the
/// fast path exists because the R3 decoder runs millions of collective
/// additions.
pub fn add(format: Format, a: u32, b: u32) -> u32 {
    match format {
        Format::Fp32 => {
            let sum = f32::from_bits(a) + f32::from_bits(b);
            if sum.is_nan() {
                format.canonical_nan()
            } else {
                sum.to_bits()
            }
        }
        Format::Bf16 => {
            format.assert_raw(a);
            format.assert_raw(b);
            let sum = f32::from_bits(a << 16) + f32::from_bits(b << 16);
            fp32_to_bf16(sum.to_bits())
        }
        Format::Fp16 => add_exact(format, a, b),
    }
}

/// `round(a + b)` through the exact superaccumulator (the reference for [`add`]).
pub fn add_exact(format: Format, a: u32, b: u32) -> u32 {
    let mut sum = ExactSum::new();
    sum.add_raw(format, a);
    sum.add_raw(format, b);
    sum.round(format)
}

/// binary32 → BF16, round to nearest, ties to even, by bit arithmetic
/// (canonical NaN for NaN). Overflow carries into the exponent and gives ∞.
fn fp32_to_bf16(raw: u32) -> u32 {
    if f32::from_bits(raw).is_nan() {
        return Format::Bf16.canonical_nan();
    }
    raw.wrapping_add(0x7fff + ((raw >> 16) & 1)) >> 16
}

/// Round a value from one format to another (NaN becomes the target's canonical NaN).
///
/// binary32 → BF16 and BF16 → binary32 use bit arithmetic (checked against
/// [`convert_exact`] in `tests/ops_fast_path.rs`); other pairs use the exact path.
pub fn convert(from: Format, to: Format, raw: u32) -> u32 {
    match (from, to) {
        (Format::Fp32, Format::Bf16) => {
            from.assert_raw(raw);
            fp32_to_bf16(raw)
        }
        (Format::Bf16, Format::Fp32) => {
            from.assert_raw(raw);
            let wide = f32::from_bits(raw << 16);
            if wide.is_nan() {
                Format::Fp32.canonical_nan()
            } else {
                wide.to_bits()
            }
        }
        _ => convert_exact(from, to, raw),
    }
}

/// [`convert`] through the exact superaccumulator.
pub fn convert_exact(from: Format, to: Format, raw: u32) -> u32 {
    let mut sum = ExactSum::new();
    sum.add_raw(from, raw);
    sum.round(to)
}

/// Quantize a binary64 value to `format` with one nearest-even rounding.
pub fn from_f64(format: Format, x: f64) -> u32 {
    let value = decode_f64(x);
    if let Value::Finite(d) = value {
        // Far outside every modeled format's range: decide without entering the
        // exact grid. 2^-300 is below half of the smallest FP32 subnormal
        // (2^-150); 2^200 is above every finite FP32 value.
        let lead = d.exponent + 63 - d.significand.leading_zeros() as i32;
        if lead < -300 {
            return format.zero(d.negative);
        }
        if lead > 200 {
            return format.infinity(d.negative);
        }
    }
    let mut sum = ExactSum::new();
    sum.add_value(value);
    sum.round(format)
}

/// The exact product of two encodings as a [`Value`] (never rounded).
pub fn exact_product(format_a: Format, a: u32, format_b: Format, b: u32) -> Value {
    match (decode(format_a, a), decode(format_b, b)) {
        (Value::NaN, _) | (_, Value::NaN) => Value::NaN,
        (Value::Infinity { .. }, Value::Zero { .. }) | (Value::Zero { .. }, Value::Infinity { .. }) => Value::NaN,
        (x, y) => {
            let negative = sign_of(x) != sign_of(y);
            match (x, y) {
                (Value::Infinity { .. }, _) | (_, Value::Infinity { .. }) => Value::Infinity { negative },
                (Value::Zero { .. }, _) | (_, Value::Zero { .. }) => Value::Zero { negative },
                (Value::Finite(p), Value::Finite(q)) => Value::Finite(Dyadic {
                    negative,
                    significand: p.significand * q.significand,
                    exponent: p.exponent + q.exponent,
                }),
                _ => unreachable!(),
            }
        }
    }
}

fn sign_of(v: Value) -> bool {
    match v {
        Value::Zero { negative } | Value::Infinity { negative } => negative,
        Value::Finite(d) => d.negative,
        Value::NaN => false,
    }
}

/// Sequential dot product: `acc = round_acc(acc + a[k]·b[k])` for k in order,
/// starting from `+0`. Each product is exact (fused), so there is exactly one
/// rounding per term. This is one defined evaluation order, not a model of a
/// particular GEMM kernel's blocking.
pub fn fused_chain_dot(accumulator: Format, input: Format, a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len(), "dot product operands differ in length");
    let mut acc = accumulator.zero(false);
    for (&x, &y) in a.iter().zip(b) {
        let mut sum = ExactSum::new();
        sum.add_raw(accumulator, acc);
        sum.add_value(exact_product(input, x, input, y));
        acc = sum.round(accumulator);
    }
    acc
}

/// The exact dot product rounded once to `output`.
pub fn exact_dot(output: Format, input: Format, a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len(), "dot product operands differ in length");
    let mut sum = ExactSum::new();
    for (&x, &y) in a.iter().zip(b) {
        sum.add_value(exact_product(input, x, input, y));
    }
    sum.round(output)
}

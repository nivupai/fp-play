//! Correctly rounded operations built on [`ExactSum`].
//!
//! Each function computes its mathematical result exactly and rounds once.
//! No host floating-point arithmetic is used to model a format's arithmetic.

use crate::exact::ExactSum;
use crate::format::{decode, decode_f64, Dyadic, Format, Value};

/// `round(a + b)` in `format`.
pub fn add(format: Format, a: u32, b: u32) -> u32 {
    let mut sum = ExactSum::new();
    sum.add_raw(format, a);
    sum.add_raw(format, b);
    sum.round(format)
}

/// Round a value from one format to another (NaN becomes the target's canonical NaN).
pub fn convert(from: Format, to: Format, raw: u32) -> u32 {
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

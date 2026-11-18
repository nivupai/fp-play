//! A natural logarithm computed only from IEEE-specified binary64 operations.
//!
//! The host `f64::ln` comes from the platform's math library, which is not
//! required to be correctly rounded and differs between platforms. Poisson
//! arrival times need a logarithm, and the recorded serving results must be
//! byte-identical on macOS and Linux, so this one uses only exponent
//! extraction, `+`, `−`, `×` and `÷`, each correctly rounded by IEEE 754. It is
//! accurate to a few ulps (tested against the host), which is ample: only
//! determinism matters for the simulator, not the last bit.

const LN_2: f64 = std::f64::consts::LN_2;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Terms of the atanh series. With |s| ≤ 0.1716 (s² ≤ 0.0295) the omitted
/// tail is below 2^-60 relative to the result.
const SERIES_TERMS: u32 = 14;

/// ln(x) for finite x > 0. Panics on zero, negative, infinite or NaN input
/// (the simulator never produces them; failing loudly beats a silent value).
pub fn ln(x: f64) -> f64 {
    assert!(x.is_finite() && x > 0.0, "ln needs a finite positive input, got {x}");
    // Normalize subnormals exactly: ×2^54 is exact for any subnormal.
    let (mut x, mut exponent) = if x < f64::MIN_POSITIVE {
        (x * (1u64 << 54) as f64, -54i64)
    } else {
        (x, 0)
    };
    // x = m × 2^e with m in [1, 2), from the bit pattern (exact).
    let bits = x.to_bits();
    exponent += ((bits >> 52) & 0x7ff) as i64 - 1023;
    x = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    // Move m into [√½, √2) so |s| below stays small (exact: ÷2 of a normal).
    if x >= SQRT_2 {
        x /= 2.0;
        exponent += 1;
    }
    // ln m = 2 atanh(s), s = (m − 1)/(m + 1); atanh s = Σ s^(2k+1)/(2k+1).
    let s = (x - 1.0) / (x + 1.0);
    let s2 = s * s;
    // Horner from the highest term down, a fixed order.
    let mut poly = 0.0;
    for k in (0..SERIES_TERMS).rev() {
        poly = poly * s2 + 1.0 / (2 * k + 1) as f64;
    }
    exponent as f64 * LN_2 + 2.0 * s * poly
}

#[cfg(test)]
mod tests {
    use super::ln;

    #[test]
    fn exact_points() {
        assert_eq!(ln(1.0), 0.0);
        assert!((ln(2.0) - std::f64::consts::LN_2).abs() <= f64::EPSILON);
    }
}

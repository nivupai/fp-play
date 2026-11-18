//! `exp`, `sin` and `cos` from IEEE-specified binary64 operations only.
//!
//! The host math library is not required to be correctly rounded and differs
//! between platforms, so the decoder's softmax, SiLU and RoPE tables use these
//! instead (as [`crate::serving::detmath::ln`] does for arrivals). Each uses
//! exponent manipulation, `+`, `−`, `×`, `÷` and `floor` only, in a fixed
//! order, so the result is bit-identical on every platform. They are accurate
//! to a few binary64 ulps (tested against the host), not correctly rounded.
//! `scripts/gen_decoder_vectors.py` mirrors them operation for operation;
//! they are the one part of the Python reference that is a mirror rather
//! than an independent implementation.

/// log2(e), and ln 2 split for Cody–Waite reduction: `LN2_HI` has trailing
/// zero bits, so `k × LN2_HI` is exact for |k| < 2^11.
const LOG2_E_BITS: u64 = 0x3ff7_1547_652b_82fe;
const LN2_HI_BITS: u64 = 0x3fe6_2e42_fee0_0000;
const LN2_LO_BITS: u64 = 0x3dea_39ef_3579_3c76;

/// 2/π, and π/2 in three parts (31, 32 and 53 significant bits), so `n × PIO2_1` and
/// `n × PIO2_2` are exact for n < 2^20.
const TWO_OVER_PI_BITS: u64 = 0x3fe4_5f30_6dc9_c883;
const PIO2_1_BITS: u64 = 0x3ff9_21fb_5440_0000;
const PIO2_2_BITS: u64 = 0x3dd0_b461_1a60_0000;
const PIO2_3_BITS: u64 = 0x3ba3_198a_2e03_7073;

/// Taylor terms. exp: |r| ≤ 0.35, so r^14/14! < 2^-60. sin/cos: |r| ≤ π/4,
/// so the first omitted terms are below 2^-60.
const EXP_TERMS: u32 = 14;
const TRIG_TERMS: u32 = 10;

/// Binary64 from bits (the constants are bit patterns so they stay `const`
/// under the crate's minimum Rust version).
fn f(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn inverse_factorial(n: u32) -> f64 {
    // n! is exact in binary64 for n ≤ 18, so this is one correctly rounded division.
    let factorial = (1..=n).fold(1u64, |a, k| a * u64::from(k));
    1.0 / factorial as f64
}

/// e^x for |x| ≤ 700. Panics outside that range or on NaN.
pub fn exp(x: f64) -> f64 {
    assert!(x.abs() <= 700.0, "exp needs |x| ≤ 700, got {x}");
    let k = (x * f(LOG2_E_BITS) + 0.5).floor();
    let r = (x - k * f(LN2_HI_BITS)) - k * f(LN2_LO_BITS);
    let mut p = inverse_factorial(EXP_TERMS - 1);
    for n in (0..EXP_TERMS - 1).rev() {
        p = p * r + inverse_factorial(n);
    }
    // 2^k with k in [-1010, 1010] is a normal binary64, built from bits.
    let scale = f64::from_bits(((k as i64 + 1023) as u64) << 52);
    p * scale
}

/// (sin a, cos a) for 0 ≤ a ≤ 2^19 (RoPE angles are non-negative).
pub fn sin_cos(a: f64) -> (f64, f64) {
    assert!((0.0..=524_288.0).contains(&a), "sin_cos needs 0 ≤ a ≤ 2^19, got {a}");
    let n = (a * f(TWO_OVER_PI_BITS) + 0.5).floor();
    let r = ((a - n * f(PIO2_1_BITS)) - n * f(PIO2_2_BITS)) - n * f(PIO2_3_BITS);
    let r2 = r * r;
    // sin r = r Σ (−1)^k r^2k/(2k+1)!, cos r = Σ (−1)^k r^2k/(2k)!, Horner from the top.
    let (mut s, mut c) = (0.0, 0.0);
    for k in (0..TRIG_TERMS).rev() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        s = s * r2 + sign * inverse_factorial(2 * k + 1);
        c = c * r2 + sign * inverse_factorial(2 * k);
    }
    let (s, c) = (r * s, c);
    match (n as u64) % 4 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// e^x rounded to binary32: binary64 [`exp`] then one round-to-nearest-even
/// conversion. `+∞` above 100 and `+0` below −110 (beyond binary32's range).
pub fn exp_f32(x: f32) -> f32 {
    assert!(!x.is_nan(), "exp_f32 of NaN");
    if x > 100.0 {
        f32::INFINITY
    } else if x < -110.0 {
        0.0
    } else {
        exp(f64::from(x)) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f64, b: f64) -> u64 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
    }

    #[test]
    fn exp_matches_the_host_within_a_few_ulps() {
        assert_eq!(exp(0.0), 1.0);
        let mut worst = 0;
        let mut x = -700.0;
        while x <= 700.0 {
            worst = worst.max(ulps(exp(x), x.exp()));
            x += 0.0137;
        }
        assert!(worst <= 4, "worst {worst} ulps");
    }

    #[test]
    fn sin_cos_match_the_host_within_a_few_ulps() {
        let mut worst = 0u64;
        let mut a = 0.0;
        while a <= 4096.0 {
            let (s, c) = sin_cos(a);
            // Compare absolute error in units of 2^-52 (relative ulps blow up near zeros).
            let err = ((s - a.sin()).abs().max((c - a.cos()).abs()) / f64::EPSILON) as u64;
            worst = worst.max(err);
            a += 0.00731;
        }
        assert!(worst <= 8, "worst {worst} × 2^-52");
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
    }
}

//! Numerics checked against oracles that share no code with the crate:
//! the host's IEEE binary32 hardware, the standard f32→bf16 bit-rounding
//! identity, and Python `Fraction` vectors (scripts/gen_sum_vectors.py).

use fp_play_runtime::exact::ExactSum;
use fp_play_runtime::format::{decode, is_nan, to_f64, ulp_distance, Format, Value};
use fp_play_runtime::ops::{add, convert, from_f64, fused_chain_dot};
use fp_play_runtime::rng::SplitMix64;

#[test]
fn every_fp16_and_bf16_pattern_round_trips() {
    for format in [Format::Fp16, Format::Bf16] {
        for raw in 0..=format.max_raw() {
            if is_nan(format, raw) {
                assert_eq!(convert(format, format, raw), format.canonical_nan());
                assert!(to_f64(format, raw).is_nan());
                continue;
            }
            assert_eq!(convert(format, format, raw), raw, "{} {raw:#x}", format.name());
            // binary64 path: to_f64 builds the value from bits, from_f64 re-rounds it.
            assert_eq!(from_f64(format, to_f64(format, raw)), raw, "{} {raw:#x}", format.name());
        }
    }
}

#[test]
fn decoding_known_patterns() {
    assert_eq!(to_f64(Format::Fp16, 0x3c00), 1.0);
    assert_eq!(to_f64(Format::Fp16, 0x0001), 2f64.powi(-24));
    assert_eq!(to_f64(Format::Fp16, 0x7bff), 65504.0);
    assert_eq!(to_f64(Format::Bf16, 0x3f80), 1.0);
    assert_eq!(to_f64(Format::Bf16, 0x0001), 2f64.powi(-133));
    assert_eq!(to_f64(Format::Fp32, 0x0000_0001), 2f64.powi(-149));
    assert_eq!(decode(Format::Fp16, 0x8000), Value::Zero { negative: true });
    assert_eq!(decode(Format::Bf16, 0xff80), Value::Infinity { negative: true });
}

#[test]
#[should_panic(expected = "does not fit fp16")]
fn oversized_raw_patterns_are_rejected() {
    decode(Format::Fp16, 0x1_0000);
}

#[test]
fn ulp_distance_semantics() {
    assert_eq!(ulp_distance(Format::Fp16, 0x0000, 0x8000), Some(0));
    assert_eq!(ulp_distance(Format::Fp16, 0x0001, 0x8001), Some(2));
    assert_eq!(ulp_distance(Format::Fp16, 0x3c00, 0x3c01), Some(1));
    assert_eq!(ulp_distance(Format::Fp16, 0x7bff, 0x7c00), Some(1));
    assert_eq!(ulp_distance(Format::Fp16, 0x7e00, 0x3c00), None);
}

/// Random raw FP32 patterns biased toward subnormals, ties and cancellation.
fn styled_fp32(rng: &mut SplitMix64, style: u64) -> u32 {
    let sign = (rng.next_u64() & 1) as u32;
    let field = match style {
        0 => rng.below(256) as u32,
        1 => 0,
        2 => 127 + rng.below(7) as u32 - 3,
        _ => 1 + rng.below(30) as u32,
    };
    let mut fraction = (rng.next_u64() as u32) & 0x7f_ffff;
    if rng.below(3) == 0 {
        fraction &= !0xffff; // few significant bits: frequent exact ties
    }
    (sign << 31) | (field.min(255) << 23) | fraction
}

#[test]
fn fp32_quantization_matches_host_conversion() {
    let mut rng = SplitMix64::new(7);
    for i in 0..400_000u32 {
        let x = match i % 4 {
            0 => f64::from_bits(rng.next_u64()),
            // an FP32 value plus a tail: exercises halfway and near-halfway cases
            1 => {
                let style = rng.below(4);
                let base = f32::from_bits(styled_fp32(&mut rng, style) & 0x7f7f_ffff) as f64;
                let tail = (rng.next_u64() >> 40) as f64 * 2f64.powi(-24 - 24);
                base * (1.0 + tail)
            }
            2 => f32::from_bits(styled_fp32(&mut rng, 1)) as f64 * (1.5 + rng.next_unit() * 2f64.powi(-20)),
            _ => (rng.next_unit() - 0.5) * 2f64.powi(rng.below(300) as i32 - 150),
        };
        if x.is_nan() {
            continue;
        }
        assert_eq!(
            from_f64(Format::Fp32, x),
            (x as f32).to_bits(),
            "x = {x:e} ({:#x})",
            x.to_bits()
        );
    }
}

#[test]
fn fp32_addition_matches_host_hardware() {
    let mut rng = SplitMix64::new(11);
    for i in 0..400_000u64 {
        let a = styled_fp32(&mut rng, i % 4);
        let b = if i % 5 == 0 {
            a ^ 0x8000_0000 // exact cancellation
        } else {
            styled_fp32(&mut rng, (i / 4) % 4)
        };
        let expected = (f32::from_bits(a) + f32::from_bits(b)).to_bits();
        let got = add(Format::Fp32, a, b);
        if f32::from_bits(expected).is_nan() {
            assert!(is_nan(Format::Fp32, got));
        } else {
            assert_eq!(got, expected, "{a:#010x} + {b:#010x}");
        }
    }
}

/// Standard f32 → bf16 nearest-even rounding by integer arithmetic on the bits.
fn bf16_from_f32_bits(bits: u32) -> u32 {
    (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
}

#[test]
fn bf16_addition_matches_independent_oracle() {
    // A binary32 sum of two bf16 values is correctly rounded to 24 bits; since
    // 24 ≥ 2·8 + 2, rounding it again to bf16 gives the correctly rounded bf16 sum.
    for a in (0..=0xffffu32).step_by(7) {
        for b in (0..=0xffffu32).step_by(331) {
            if is_nan(Format::Bf16, a) || is_nan(Format::Bf16, b) {
                continue;
            }
            let sum = f32::from_bits(a << 16) + f32::from_bits(b << 16);
            let got = add(Format::Bf16, a, b);
            if sum.is_nan() {
                assert!(is_nan(Format::Bf16, got));
            } else {
                assert_eq!(got, bf16_from_f32_bits(sum.to_bits()), "{a:#06x} + {b:#06x}");
            }
        }
    }
}

#[test]
fn fp32_to_bf16_conversion_matches_bit_rounding() {
    let mut rng = SplitMix64::new(3);
    for i in 0..300_000u64 {
        let raw = styled_fp32(&mut rng, i % 4);
        if is_nan(Format::Fp32, raw) {
            continue;
        }
        assert_eq!(
            convert(Format::Fp32, Format::Bf16, raw),
            bf16_from_f32_bits(raw),
            "{raw:#010x}"
        );
    }
}

#[test]
fn fused_chain_matches_host_fp32_when_products_are_exact() {
    // bf16 × bf16 has at most 16 significant bits, so in normal range the host's
    // binary32 product is exact and the host chain equals the fused chain.
    let mut rng = SplitMix64::new(5);
    for _ in 0..2_000 {
        let len = 1 + rng.below(96) as usize;
        let a: Vec<u32> = (0..len).map(|_| from_f64(Format::Bf16, rng.next_normalish())).collect();
        let b: Vec<u32> = (0..len)
            .map(|_| from_f64(Format::Bf16, rng.next_normalish() * 0.1))
            .collect();
        let mut host = 0.0f32;
        for (&x, &y) in a.iter().zip(&b) {
            host += f32::from_bits(x << 16) * f32::from_bits(y << 16);
        }
        assert_eq!(fused_chain_dot(Format::Fp32, Format::Bf16, &a, &b), host.to_bits());
    }
}

fn load_vectors() -> Vec<(Format, Vec<u32>, u32)> {
    let text = include_str!("vectors/exact-sums.txt");
    text.lines()
        .map(|line| {
            let mut parts = line.split(' ');
            let format = Format::parse(parts.next().unwrap()).unwrap();
            let raws = parts
                .next()
                .unwrap()
                .split(',')
                .map(|h| u32::from_str_radix(h, 16).unwrap())
                .collect();
            let expected = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            (format, raws, expected)
        })
        .collect()
}

#[test]
fn exact_sums_match_python_fraction_vectors() {
    let vectors = load_vectors();
    assert_eq!(vectors.len(), 18_000, "vector file changed size");
    for (format, raws, expected) in &vectors {
        let mut sum = ExactSum::new();
        for &raw in raws {
            sum.add_raw(*format, raw);
        }
        assert_eq!(sum.round(*format), *expected, "{} {raws:x?}", format.name());
    }
}

#[test]
fn exact_sums_are_order_independent() {
    let mut rng = SplitMix64::new(9);
    for (format, raws, expected) in load_vectors().into_iter().step_by(3) {
        let mut shuffled = raws.clone();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut sum = ExactSum::new();
        for raw in shuffled {
            sum.add_raw(format, raw);
        }
        assert_eq!(sum.round(format), expected);
    }
}

#[test]
fn special_values_and_zero_signs() {
    let f = Format::Fp16;
    assert_eq!(add(f, 0x8000, 0x8000), 0x8000); // -0 + -0
    assert_eq!(add(f, 0x8000, 0x0000), 0x0000); // -0 + +0
    assert_eq!(add(f, 0x3c00, 0xbc00), 0x0000); // 1 + -1 → +0
    assert_eq!(add(f, 0x7c00, 0xfc00), f.canonical_nan()); // ∞ − ∞
    assert_eq!(add(f, 0x7c00, 0x3c00), 0x7c00);
    assert_eq!(add(f, 0x7bff, 0x7bff), 0x7c00); // overflow
    assert_eq!(add(f, 0x7bff, 0x4800), 0x7bff); // 65504 + 8: below the halfway point to 65536
    assert_eq!(add(f, 0x7bff, 0x4c00), 0x7c00); // 65504 + 16: halfway; 65504 is odd, so ties-to-even → ∞
    assert_eq!(add(f, 0xfbff, 0xcc00), 0xfc00); // the same tie, negated
    assert_eq!(add(f, 0x7e01, 0x3c00), f.canonical_nan()); // NaN payload canonicalized (documented)
    assert_eq!(from_f64(Format::Fp16, 1e-300), 0x0000);
    assert_eq!(from_f64(Format::Fp16, -1e300), 0xfc00);
}

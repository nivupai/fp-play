//! The native fast paths of `ops::add` and `ops::convert` against the exact
//! superaccumulator: special values, rounding ties, overflow and
//! subnormal boundaries, and random patterns across every exponent.

use fp_play_runtime::format::Format;
use fp_play_runtime::ops::{add, add_exact, convert, convert_exact};
use fp_play_runtime::rng::SplitMix64;

/// Interesting raw patterns of `format`: zeros, subnormal edges, normal
/// edges, the largest finite values, infinities and NaNs, both signs.
fn specials(format: Format) -> Vec<u32> {
    let f = format.fraction_bits();
    let top = format.exponent_field_max();
    let mut v = Vec::new();
    for exp in [0, 1, 2, top / 2, top / 2 + 1, top - 2, top - 1, top] {
        for frac in [
            0,
            1,
            2,
            (1 << (f - 1)) - 1,
            1 << (f - 1),
            (1 << (f - 1)) + 1,
            (1 << f) - 2,
            (1 << f) - 1,
        ] {
            let raw = exp << f | frac;
            v.push(raw);
            v.push(raw | format.sign_mask());
        }
    }
    v
}

/// A random pattern whose exponent field is uniform, so sums across every
/// binade (and near-cancellations) are exercised.
fn random(format: Format, rng: &mut SplitMix64) -> u32 {
    (rng.next_u64() as u32) & format.max_raw()
}

#[test]
fn add_matches_the_exact_path() {
    for format in [Format::Fp32, Format::Bf16] {
        let s = specials(format);
        for &a in &s {
            for &b in &s {
                assert_eq!(add(format, a, b), add_exact(format, a, b), "{format:?} {a:#x} + {b:#x}");
            }
        }
        let mut rng = SplitMix64::new(0xadd ^ format.width() as u64);
        for i in 0..2_000_000 {
            let a = random(format, &mut rng);
            // Half the pairs share or nearly share a's exponent: ties, carries, cancellation.
            let b = if i % 2 == 0 {
                random(format, &mut rng)
            } else {
                let flip = if rng.below(2) == 0 { 0 } else { format.sign_mask() };
                let low = rng.below(1 << 12) as u32;
                (((a ^ flip) & !0xfff) | low) & format.max_raw()
            };
            assert_eq!(add(format, a, b), add_exact(format, a, b), "{format:?} {a:#x} + {b:#x}");
        }
    }
}

#[test]
fn bf16_add_ties_round_to_even_once() {
    // 1 + 2^-8 is exactly halfway between 1 and 1 + 2^-7 in BF16: even is 1.
    let one = 0x3f80;
    let half_ulp = 0x3b80; // 2^-8
    assert_eq!(add(Format::Bf16, one, half_ulp), one);
    assert_eq!(add_exact(Format::Bf16, one, half_ulp), one);
    // The largest finite BF16 (odd significand) plus half an ulp of its
    // binade (2^119) is an exact tie: even rounds up to ∞. Just below the tie
    // stays finite; above it overflows.
    let (max, inf) = (0x7f7f, 0x7f80);
    for (b, want) in [(0x7b00, inf), (0x7a80, max), (0x7b80, inf)] {
        assert_eq!(add(Format::Bf16, max, b), want, "{b:#x}");
        assert_eq!(add_exact(Format::Bf16, max, b), want, "{b:#x}");
    }
}

#[test]
fn convert_matches_the_exact_path() {
    for high in 0..=0xffffu32 {
        for low in [0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff] {
            let raw = high << 16 | low;
            assert_eq!(
                convert(Format::Fp32, Format::Bf16, raw),
                convert_exact(Format::Fp32, Format::Bf16, raw),
                "{raw:#x}"
            );
        }
        assert_eq!(
            convert(Format::Bf16, Format::Fp32, high),
            convert_exact(Format::Bf16, Format::Fp32, high),
            "{high:#x}"
        );
    }
    let mut rng = SplitMix64::new(0xc0de);
    for _ in 0..2_000_000 {
        let raw = rng.next_u64() as u32;
        assert_eq!(
            convert(Format::Fp32, Format::Bf16, raw),
            convert_exact(Format::Fp32, Format::Bf16, raw),
            "{raw:#x}"
        );
    }
}

#[test]
#[should_panic(expected = "does not fit")]
fn add_still_validates_raw_patterns() {
    add(Format::Bf16, 0x1_0000, 0);
}

/// Directed cases with hand-derived expected bits: both paths must give them.
fn both(format: Format, a: u32, b: u32, want: u32) {
    assert_eq!(add(format, a, b), want, "fast {format:?} {a:#x} + {b:#x}");
    assert_eq!(add_exact(format, a, b), want, "exact {format:?} {a:#x} + {b:#x}");
}

#[test]
fn fp32_directed_cases() {
    let f = Format::Fp32;
    // Subnormals: sums on the 2^-149 grid, crossing into the normal range and back.
    both(f, 0x0000_0001, 0x0000_0001, 0x0000_0002);
    both(f, 0x007f_ffff, 0x0000_0001, 0x0080_0000);
    both(f, 0x0080_0000, 0x8000_0001, 0x007f_ffff);
    both(f, 0x0080_0001, 0x8080_0000, 0x0000_0001);
    // Ties to even: 1 + 2^-24 is halfway; 1 + 2^-23 + 2^-24 rounds up to even.
    both(f, 0x3f80_0000, 0x3380_0000, 0x3f80_0000);
    both(f, 0x3f80_0001, 0x3380_0000, 0x3f80_0002);
    // Cancellation and signed zeros.
    both(f, 0x3f80_0000, 0xbf80_0000, 0x0000_0000);
    both(f, 0x8000_0000, 0x8000_0000, 0x8000_0000);
    both(f, 0x0000_0000, 0x8000_0000, 0x0000_0000);
    both(f, 0x3f80_0001, 0xbf80_0000, 0x3400_0000);
    // Overflow: max + max; max + half an ulp of its binade (2^103) is a tie
    // and the odd significand rounds up to ∞; max + 2^102 stays finite.
    both(f, 0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000);
    both(f, 0x7f7f_ffff, 0x7300_0000, 0x7f80_0000);
    both(f, 0x7f7f_ffff, 0x7280_0000, 0x7f7f_ffff);
    both(f, 0xff7f_ffff, 0xff7f_ffff, 0xff80_0000);
    // NaN: payloads, signalling and negative NaNs, and ∞ − ∞ all give the canonical NaN.
    both(f, 0x7fc0_0001, 0x3f80_0000, 0x7fc0_0000);
    both(f, 0x7f80_0001, 0x3f80_0000, 0x7fc0_0000);
    both(f, 0x3f80_0000, 0xffc0_0000, 0x7fc0_0000);
    both(f, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000);
    both(f, 0x7f80_0000, 0x3f80_0000, 0x7f80_0000);
}

#[test]
fn bf16_directed_cases() {
    let f = Format::Bf16;
    // Subnormals (2^-133 grid) and the crossing into normals.
    both(f, 0x0001, 0x0001, 0x0002);
    both(f, 0x007f, 0x0001, 0x0080);
    both(f, 0x0080, 0x8001, 0x007f);
    // Ties to even.
    both(f, 0x3f80, 0x3b80, 0x3f80);
    both(f, 0x3f81, 0x3b80, 0x3f82);
    // Just above a tie, and far below binary32 precision (the double-rounding cases).
    both(f, 0x3f80, 0x3b81, 0x3f81);
    both(f, 0x3f80, 0x3081, 0x3f80);
    // Cancellation and signed zeros.
    both(f, 0x3f80, 0xbf80, 0x0000);
    both(f, 0x8000, 0x8000, 0x8000);
    both(f, 0x3f81, 0xbf80, 0x3c00);
    // Overflow.
    both(f, 0x7f7f, 0x7f7f, 0x7f80);
    both(f, 0xff7f, 0xfb00, 0xff80);
    // NaN.
    both(f, 0x7fc1, 0x3f80, 0x7fc0);
    both(f, 0x7f81, 0x3f80, 0x7fc0);
    both(f, 0xff81, 0x3f80, 0x7fc0);
    both(f, 0x7f80, 0xff80, 0x7fc0);
}

#[test]
fn convert_directed_cases() {
    let to_bf16 = |raw: u32, want: u32| {
        assert_eq!(convert(Format::Fp32, Format::Bf16, raw), want, "fast {raw:#x}");
        assert_eq!(convert_exact(Format::Fp32, Format::Bf16, raw), want, "exact {raw:#x}");
    };
    to_bf16(0x0000_8000, 0x0000); // subnormal tie, even stays
    to_bf16(0x0001_8000, 0x0002); // subnormal tie, odd rounds up
    to_bf16(0x007f_ffff, 0x0080); // largest subnormal rounds into the normals
    to_bf16(0x3f80_8000, 0x3f80); // normal tie
    to_bf16(0x7f7f_ffff, 0x7f80); // overflow to ∞
    to_bf16(0x7f7f_7fff, 0x7f7f);
    to_bf16(0x8000_0000, 0x8000); // −0
    to_bf16(0x7f80_0001, 0x7fc0); // a NaN whose payload would truncate to ∞
    to_bf16(0xffff_ffff, 0x7fc0);
    let to_fp32 = |raw: u32, want: u32| {
        assert_eq!(convert(Format::Bf16, Format::Fp32, raw), want, "fast {raw:#x}");
        assert_eq!(convert_exact(Format::Bf16, Format::Fp32, raw), want, "exact {raw:#x}");
    };
    to_fp32(0x0001, 0x0001_0000);
    to_fp32(0x8000, 0x8000_0000);
    to_fp32(0x7f81, 0x7fc0_0000);
    to_fp32(0xff80, 0xff80_0000);
}

/// The exact path keeps an oracle that does not involve the fast path's
/// code: binary32 additions against the host's IEEE hardware.
#[test]
fn exact_fp32_addition_matches_host_hardware() {
    let mut rng = SplitMix64::new(0x4057);
    for _ in 0..400_000 {
        let (a, b) = (rng.next_u64() as u32, rng.next_u64() as u32);
        let host = f32::from_bits(a) + f32::from_bits(b);
        let want = if host.is_nan() {
            Format::Fp32.canonical_nan()
        } else {
            host.to_bits()
        };
        assert_eq!(add_exact(Format::Fp32, a, b), want, "{a:#x} + {b:#x}");
    }
}

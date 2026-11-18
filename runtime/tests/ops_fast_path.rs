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

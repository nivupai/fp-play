//! FP8 (E4M3, E5M2) and INT8 with scales: exhaustive pattern checks, every
//! midpoint, and the independent Python `Fraction` vectors.

use fp_play_runtime::precision::narrow::{
    decode, encode, midpoint_check, positive_values, quantize, scale, store, vector_check, Narrow, ScaledMatrix,
};

const VECTORS: &str = include_str!("vectors/narrow.txt");

fn fraction(text: &str) -> f64 {
    match text {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        "-0" => -0.0,
        _ => match text.split_once('/') {
            // Every FP8 value is a dyadic rational exactly representable in binary64.
            Some((n, d)) => n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap(),
            None => text.parse().unwrap(),
        },
    }
}

#[test]
fn all_256_patterns_decode_as_the_independent_table() {
    let mut seen = 0;
    for line in VECTORS.lines().filter(|l| l.starts_with("d ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let format = if f[1] == "fp8-e4m3" { Narrow::E4M3 } else { Narrow::E5M2 };
        let raw = u8::from_str_radix(f[2], 16).unwrap();
        let (got, want) = (decode(format, raw), fraction(f[3]));
        assert!(
            (got.is_nan() && want.is_nan()) || (got == want && got.is_sign_negative() == want.is_sign_negative()),
            "{line}: got {got}"
        );
        seen += 1;
    }
    assert_eq!(seen, 512);
}

#[test]
fn every_pattern_round_trips_and_specials_are_counted() {
    for (f, nans, infs, max, min_sub) in [
        (Narrow::E4M3, 2, 0, 448.0, 2f64.powi(-9)),
        (Narrow::E5M2, 6, 2, 57344.0, 2f64.powi(-16)),
    ] {
        let (mut n, mut i) = (0, 0);
        for raw in 0..=255u8 {
            let v = decode(f, raw);
            if v.is_nan() {
                n += 1;
                continue;
            }
            i += usize::from(v.is_infinite());
            assert_eq!(encode(f, v), raw, "{} {raw:#04x}", f.name());
            // Sign symmetry.
            assert_eq!(decode(f, raw ^ 0x80).to_bits(), (-v).to_bits());
        }
        assert_eq!((n, i), (nans, infs), "{}", f.name());
        let pos = positive_values(f);
        assert_eq!(pos.len(), 128 - nans / 2 - infs / 2);
        assert!(pos.windows(2).all(|w| w[0] < w[1]));
        // Increasing raw patterns give increasing values (sign-magnitude order).
        let raw_order: Vec<f64> = (0..0x80u8).map(|r| decode(f, r)).filter(|v| v.is_finite()).collect();
        assert_eq!(raw_order, pos);
        assert_eq!((pos[1], *pos.last().unwrap()), (min_sub, max));
        assert_eq!(f.qmax() as f64, max);
    }
}

#[test]
fn every_midpoint_and_its_neighbours_round_correctly() {
    let n = midpoint_check(quantize).unwrap();
    assert!(n > 4000, "{n}");
}

#[test]
fn quantizer_matches_the_python_fraction_vectors() {
    assert_eq!(vector_check(VECTORS, quantize).unwrap(), 15_300);
}

#[test]
fn scaled_products_match_the_python_fraction_vectors() {
    let mut n = 0;
    for line in VECTORS.lines().filter(|l| l.starts_with("dot ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let format = Narrow::ALL.into_iter().find(|x| x.name() == f[1]).unwrap();
        let k: usize = f[2].parse().unwrap();
        let hex = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
        let w: Vec<f32> = f[3..3 + k].iter().map(|s| hex(s)).collect();
        let x: Vec<f32> = f[3 + k..3 + 2 * k].iter().map(|s| hex(s)).collect();
        let want = u32::from_str_radix(f[3 + 2 * k], 16).unwrap();
        let y = ScaledMatrix::new(format, 1, k, &w).apply(&x)[0];
        assert_eq!(y.to_bits(), want, "{line}");
        n += 1;
    }
    assert_eq!(n, 480);
}

#[test]
fn scales_specials_and_storage() {
    for f in Narrow::ALL {
        assert_eq!(scale(f, &[0.0, -0.0]), Some(1.0));
        assert_eq!(scale(f, &[1.0, f32::NAN]), None);
        assert_eq!(scale(f, &[f32::INFINITY]), None);
        assert_eq!(scale(f, &[-2.0, 1.0]), Some(2.0 / f.qmax()));
        assert!(quantize(f, f32::NAN, 1.0).is_nan());
        assert_eq!(quantize(f, -0.0, 3.0).to_bits(), (-0.0f64).to_bits());
        // The absmax element maps to ±Qmax (s is rounded; the quotient rounds back).
        let x = [0.3f32, -1.7, 0.01];
        let s = scale(f, &x).unwrap();
        assert_eq!(quantize(f, -1.7, s), -f64::from(f.qmax()));
        let m = ScaledMatrix::new(f, 2, 3, &[0.3, -1.7, 0.01, 0.0, 0.0, 0.0]);
        assert!(m.apply(&[1.0, f32::NAN, 0.0]).iter().all(|v| v.is_nan()));
        assert_eq!(m.row(1), vec![0.0; 3]);
        let stored = store(f, &x);
        assert_eq!(stored[1], -1.7f32 / f.qmax() * f.qmax());
    }
    // INT8 is symmetric: −128 is never produced.
    assert_eq!(quantize(Narrow::Int8, -1000.0, 1.0), -127.0);
    assert_eq!(quantize(Narrow::Int8, 2.5, 1.0), 2.0);
    assert_eq!(quantize(Narrow::Int8, -3.5, 1.0), -4.0);
    // E4M3 saturates instead of producing NaN; E5M2 saturates instead of ∞.
    assert_eq!(quantize(Narrow::E4M3, 1e6, 1.0), 448.0);
    assert_eq!(quantize(Narrow::E5M2, -1e9, 1.0), -57344.0);
}

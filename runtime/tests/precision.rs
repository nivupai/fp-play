//! R4a: the fast fused multiply–accumulate against the exact path and
//! independent Python vectors; precision policies; drift metrics.

use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::decoder::decode::{fnv1a, prompts, FNV_OFFSET};
use fp_play_runtime::exact::ExactSum;
use fp_play_runtime::format::{to_f64, Format};
use fp_play_runtime::ops::{exact_product, fused_chain_dot};
use fp_play_runtime::precision::mac::{encode, fused_dot, mac, round_value, round_value_reference};
use fp_play_runtime::precision::metrics::{compare, drift, kl, log_softmax};
use fp_play_runtime::precision::model::{site_product, MasterWeights, PolicyModel};
use fp_play_runtime::precision::narrow::ScaledMatrix;
use fp_play_runtime::precision::policy::{Level, Policy, Site};
use fp_play_runtime::rng::SplitMix64;

const FORMATS: [Format; 3] = [Format::Fp32, Format::Bf16, Format::Fp16];

fn value(format: Format, raw: u32) -> f64 {
    to_f64(format, raw)
}

/// A random raw pattern of `format`, biased toward interesting fields.
fn raw(format: Format, rng: &mut SplitMix64) -> u32 {
    let f = format.fraction_bits();
    let top = format.exponent_field_max();
    let field = match rng.below(10) {
        0 => 0,
        1 => top - 1,
        2 => top,
        _ => (format.bias() as u32)
            .saturating_add(rng.below(13) as u32)
            .saturating_sub(6)
            .min(top - 1),
    };
    let frac = if rng.below(8) == 0 {
        0
    } else {
        rng.next_u64() as u32 & format.fraction_mask()
    };
    let sign = if rng.below(2) == 0 { 0 } else { format.sign_mask() };
    sign | field << f | frac
}

#[test]
fn bit_level_rounding_equals_the_reference_rounding() {
    let mut rng = SplitMix64::new(0x0b17);
    for format in FORMATS {
        for _ in 0..300_000 {
            // A random binary64 near the format's range, sometimes exactly a midpoint.
            let e = rng.below(60) as i32 - 30 + if rng.below(4) == 0 { format.emin() } else { 0 };
            let mant = rng.next_u64() >> 11 | 1 << 52;
            let mant = if rng.below(4) == 0 {
                let drop = 53 - format.precision();
                (mant & !((1u64 << drop) - 1)) | 1 << (drop - 1)
            } else {
                mant
            };
            let s = (mant as f64) * 2f64.powi(e - 52) * if rng.below(2) == 0 { 1.0 } else { -1.0 };
            let tail = [0.0, 1e-300, -1e-300][rng.below(3) as usize] * s.abs().max(1e-300);
            assert_eq!(
                round_value(format, s, tail).to_bits(),
                round_value_reference(format, s, tail).to_bits(),
                "{format:?} {s:e} {tail:e}"
            );
        }
        // Rounding with no tail equals the crate's exact conversion.
        for _ in 0..100_000 {
            let x = f64::from_bits(rng.next_u64() & 0x80ff_ffff_ffff_ffff | 0x3700_0000_0000_0000);
            assert_eq!(
                encode(format, round_value(format, x, 0.0)),
                fp_play_runtime::ops::from_f64(format, x),
                "{x:e}"
            );
        }
    }
}

#[test]
fn fused_chains_equal_the_exact_superaccumulator() {
    let mut rng = SplitMix64::new(0xc4a1);
    for acc in FORMATS {
        for input in FORMATS {
            for _ in 0..3_000 {
                let n = 1 + rng.below(24) as usize;
                let a: Vec<u32> = (0..n).map(|_| raw(input, &mut rng)).collect();
                let b: Vec<u32> = (0..n).map(|_| raw(input, &mut rng)).collect();
                let av: Vec<f64> = a.iter().map(|&r| value(input, r)).collect();
                let bv: Vec<f64> = b.iter().map(|&r| value(input, r)).collect();
                let got = encode(acc, fused_dot(acc, &av, &bv));
                assert_eq!(
                    got,
                    fused_chain_dot(acc, input, &a, &b),
                    "{acc:?} {input:?} {a:x?} {b:x?}"
                );
            }
        }
    }
}

/// The attention core mixes formats (binary32 queries, KV-format keys).
#[test]
fn mixed_format_products_equal_the_exact_path() {
    let mut rng = SplitMix64::new(0x31c5);
    for (fa, fb) in [
        (Format::Fp32, Format::Bf16),
        (Format::Fp32, Format::Fp16),
        (Format::Fp32, Format::Fp32),
    ] {
        for _ in 0..20_000 {
            let (acc, a, b) = (raw(Format::Fp32, &mut rng), raw(fa, &mut rng), raw(fb, &mut rng));
            let mut sum = ExactSum::new();
            sum.add_raw(Format::Fp32, acc);
            sum.add_value(exact_product(fa, a, fb, b));
            let want = sum.round(Format::Fp32);
            let got = encode(
                Format::Fp32,
                mac(Format::Fp32, value(Format::Fp32, acc), value(fa, a), value(fb, b)),
            );
            assert_eq!(got, want, "{acc:#x} + {a:#x}·{b:#x}");
        }
    }
}

#[test]
fn fused_chains_match_the_python_fraction_vectors() {
    // tests/vectors/chain-dots.txt comes from scripts/gen_chain_vectors.py
    // (Python Fraction arithmetic, no code shared with the crate); CI
    // regenerates it and compares.
    let parse = |s: &str| Format::parse(s).unwrap();
    let hex = |s: &str| -> Vec<u32> { s.split(',').map(|v| u32::from_str_radix(v, 16).unwrap()).collect() };
    let mut n = 0;
    for line in include_str!("vectors/chain-dots.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (acc, input) = (parse(f[0]), parse(f[1]));
        let (a, b) = (hex(f[2]), hex(f[3]));
        let want = u32::from_str_radix(f[4], 16).unwrap();
        let av: Vec<f64> = a.iter().map(|&r| value(input, r)).collect();
        let bv: Vec<f64> = b.iter().map(|&r| value(input, r)).collect();
        assert_eq!(encode(acc, fused_dot(acc, &av, &bv)), want, "{line}");
        n += 1;
    }
    assert_eq!(n, 3600);
}

#[test]
fn every_level_computes_its_site_as_an_exact_chain() {
    // The model's matrix product for each level equals the exact fused
    // chain on the stored (rounded) weights and activations.
    let c = DecoderConfig::REFERENCE;
    let w = MasterWeights::generate(&c);
    let m = &w.layers[0].w_up;
    let mut rng = SplitMix64::new(0x51e);
    let x: Vec<f32> = (0..m.cols).map(|_| rng.next_normalish() as f32).collect();
    for level in Level::ALL {
        let got = site_product(&Policy::uniform(c.layers, level), m, level, &x);
        let st = level.storage();
        let xs: Vec<u32> = x
            .iter()
            .map(|&v| fp_play_runtime::ops::from_f64(st, f64::from(v)))
            .collect();
        for (o, &g) in got.iter().enumerate() {
            let row: Vec<u32> = m.w[o * m.cols..(o + 1) * m.cols]
                .iter()
                .map(|&v| fp_play_runtime::ops::from_f64(st, f64::from(v)))
                .collect();
            let want = fused_chain_dot(level.accumulator(), st, &row, &xs);
            assert_eq!(
                encode(level.accumulator(), f64::from(g)),
                want,
                "{} row {o}",
                level.name()
            );
        }
    }
}

fn tiny() -> (DecoderConfig, MasterWeights) {
    let c = DecoderConfig::TINY;
    let w = MasterWeights::generate(&c);
    (c, w)
}

#[test]
fn fp32_policy_is_pinned_and_explicit_fp32_sites_change_nothing() {
    let (c, w) = tiny();
    let ps = prompts(0x9e4, 2, 8, c.vocab);
    let mut reference = PolicyModel::new(c, &w, Policy::uniform(c.layers, Level::Fp32));
    let tokens: Vec<Vec<u32>> = ps.iter().map(|p| reference.greedy(p, 6)).collect();
    let hash = ps
        .iter()
        .zip(&tokens)
        .flat_map(|(p, t)| reference.teacher_forced(&[p.clone(), t.clone()].concat()))
        .fold(FNV_OFFSET, |h, row| fnv1a(h, &row));
    assert_eq!(hash, 15233162855854555290, "pinned FP32-policy logits hash");
    for site in Site::all(c.layers) {
        let mut m = PolicyModel::new(c, &w, Policy::only(c.layers, site, Level::Fp32));
        assert_eq!(m.greedy(&ps[0], 6), tokens[0], "{}", site.name());
    }
}

/// Lowering any single site changes some logit: every site is wired to its level.
#[test]
fn every_site_affects_the_logits() {
    let (c, w) = tiny();
    let p = &prompts(0x51735, 1, 12, c.vocab)[0];
    let mut reference = PolicyModel::new(c, &w, Policy::uniform(c.layers, Level::Fp32));
    let base = reference.teacher_forced(p);
    for site in Site::all(c.layers) {
        for level in Level::LOW.into_iter().chain(Level::SCALED) {
            let mut m = PolicyModel::new(c, &w, Policy::only(c.layers, site, level));
            assert_ne!(m.teacher_forced(p), base, "{} at {}", site.name(), level.name());
        }
    }
}

/// A scaled level's site product is the scaled product of `narrow`
/// (itself checked against the Python vectors) on the site's master weights.
#[test]
fn scaled_levels_compute_their_site_as_the_scaled_product() {
    let (c, w) = tiny();
    let m = &w.layers[1].w_up;
    let mut rng = SplitMix64::new(0x5ca1e);
    let x: Vec<f32> = (0..m.cols).map(|_| rng.next_normalish() as f32).collect();
    for level in Level::SCALED {
        let f = level.narrow().unwrap();
        let got = site_product(&Policy::uniform(c.layers, level), m, level, &x);
        let want = ScaledMatrix::new(f, m.rows, m.cols, &m.w).apply(&x);
        assert_eq!(got, want, "{}", level.name());
        assert_eq!(level.accumulator(), Format::Fp32);
    }
    assert_eq!(Level::EVERY.len(), Level::ALL.len() + Level::SCALED.len());
}

#[test]
fn drift_metrics_on_hand_cases() {
    let uniform = log_softmax(&[0.0; 4]);
    assert!(uniform.iter().all(|&v| (v + 4f64.ln()).abs() < 1e-12));
    assert_eq!(kl(&uniform, &uniform), 0.0);
    assert_eq!(drift(&[1.0, 2.0], &[1.5, 2.0]), (0.5, (0.125f64).sqrt()));
    let r = vec![vec![2.0f32, 0.0, 0.0], vec![0.0, 3.0, 0.0]];
    let d = compare(&r, &r, &[0, 1]);
    assert_eq!(
        (d.tokens, d.scored, d.top1_agree, d.max_abs, d.mean_kl),
        (2, 2, 2, 0.0, 0.0)
    );
    let worse = vec![vec![0.0f32, 2.0, 0.0], vec![0.0, 3.0, f32::INFINITY]];
    let d = compare(&r, &worse, &[0, 1]);
    assert_eq!((d.scored, d.nonfinite, d.top1_agree), (1, 1, 0));
    assert!(d.mean_kl > 0.0 && d.perplexity() > 1.0);
}

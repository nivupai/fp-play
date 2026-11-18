//! R3a: the decoder against the independent Python reference, pinned
//! cross-platform hashes, conversions, and invariances at P = 1.

use fp_play_runtime::collectives::WirePolicy;
use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::decoder::decode::{generate_static, prompts, Generation};
use fp_play_runtime::decoder::forward::Model;
use fp_play_runtime::decoder::parallel::Unsharded;
use fp_play_runtime::decoder::weights::{f32_to_bf16, Weights};
use fp_play_runtime::format::{ulp_distance, Format};
use fp_play_runtime::ops;
use fp_play_runtime::rng::SplitMix64;

const POLICIES: [WirePolicy; 2] = [WirePolicy::uniform(Format::Bf16), WirePolicy::FP32_RS_BF16_AG];

/// A run name, a sequence, its tokens and its logits rows (raw binary32 bits).
type Run = (String, usize, Vec<u32>, Vec<Vec<u32>>);

struct Vectors {
    weights: u64,
    prompts: Vec<Vec<u32>>,
    steps: usize,
    runs: Vec<Run>,
}

fn vectors() -> Vectors {
    // tests/vectors/decoder.txt comes from scripts/gen_decoder_vectors.py,
    // a stdlib-only reimplementation that shares no code with the crate;
    // CI regenerates it and compares.
    let text = include_str!("vectors/decoder.txt");
    let mut v = Vectors {
        weights: 0,
        prompts: Vec::new(),
        steps: 0,
        runs: Vec::new(),
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "weights" => v.weights = u64::from_str_radix(f[1], 16).unwrap(),
            "prompt" => v.prompts.push(f[2..].iter().map(|t| t.parse().unwrap()).collect()),
            "run" => {
                let (name, seq) = (f[1].to_string(), f[3].parse::<usize>().unwrap());
                if !v.runs.iter().any(|r| r.0 == name && r.1 == seq) {
                    v.runs.push((name.clone(), seq, Vec::new(), Vec::new()));
                }
                let run = v.runs.iter_mut().find(|r| r.0 == name && r.1 == seq).unwrap();
                if f[4] == "tokens" {
                    run.2 = f[5..].iter().map(|t| t.parse().unwrap()).collect();
                    v.steps = run.2.len();
                } else {
                    assert_eq!(f[6], "logits");
                    run.3
                        .push(f[7..].iter().map(|z| u32::from_str_radix(z, 16).unwrap()).collect());
                }
            }
            other => panic!("unknown line kind {other}"),
        }
    }
    v
}

fn bits(g: &Generation) -> Vec<Vec<u32>> {
    g.logits
        .iter()
        .map(|row| row.iter().map(|z| z.to_bits()).collect())
        .collect()
}

#[test]
fn reference_model_matches_the_python_vectors_bit_for_bit() {
    let v = vectors();
    let config = DecoderConfig::REFERENCE;
    let weights = Weights::generate(&config);
    assert_eq!(
        weights.checksum(),
        v.weights,
        "weights differ from the Python reference"
    );
    assert_eq!(prompts(0x5eed_0003, 4, 5, config.vocab), v.prompts);
    let expected: Vec<_> = v.runs.iter().filter(|r| r.0 == "p1").collect();
    assert_eq!(expected.len(), v.prompts.len());
    // At P = 1 both policies round the binary32 sum once to BF16.
    for policy in POLICIES {
        let mut model = Model::with_weights(config, weights.clone());
        let got = generate_static(&mut model, &mut Unsharded { policy }, &v.prompts, v.steps, true);
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(g.tokens, e.2, "seq {} tokens ({})", e.1, policy.name());
            assert_eq!(bits(g), e.3, "seq {} logits ({})", e.1, policy.name());
        }
    }
}

#[test]
fn f32_to_bf16_matches_the_exact_conversion() {
    let check = |b: u32| {
        assert_eq!(
            u32::from(f32_to_bf16(f32::from_bits(b))),
            ops::convert(Format::Fp32, Format::Bf16, b),
            "{b:#010x}"
        );
    };
    // Every high half with the low halves around each rounding boundary.
    for high in 0..=0xffffu32 {
        for low in [0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff] {
            check(high << 16 | low);
        }
    }
    let mut rng = SplitMix64::new(0xbf16);
    for _ in 0..1_000_000 {
        check(rng.next_u64() as u32);
    }
}

/// Pins the TINY model across platforms (CI runs on Linux; the hashes were
/// recorded on macOS). Any change to weights or arithmetic changes them.
#[test]
fn tiny_model_is_pinned() {
    let config = DecoderConfig::TINY;
    let mut model = Model::new(config);
    assert_eq!(model.weights.checksum(), 12069284499870111771, "weights checksum");
    let ps = prompts(0x0b5e_55ed, 4, 8, config.vocab);
    let g = generate_static(&mut model, &mut Unsharded { policy: POLICIES[0] }, &ps, 8, false);
    let hashes: Vec<u64> = g.iter().map(|x| x.logits_hash).collect();
    assert_eq!(
        hashes,
        vec![
            12387495934403403678,
            13469239874206191442,
            5305117109259471672,
            8135583199819314031
        ]
    );
}

/// A power-of-two logit scale multiplies every binary32 logit exactly, so
/// greedy tokens, and every margin measured in ulps, are unchanged.
#[test]
fn power_of_two_logit_scale_changes_no_token() {
    let base = DecoderConfig::TINY;
    let weights = Weights::generate(&base);
    let ps = prompts(0x5ca1e, 6, 8, base.vocab);
    let run = |scale: f32| {
        let mut model = Model::with_weights(
            DecoderConfig {
                logit_scale: scale,
                ..base
            },
            weights.clone(),
        );
        generate_static(&mut model, &mut Unsharded { policy: POLICIES[0] }, &ps, 12, true)
    };
    let (one, eight) = (run(1.0), run(8.0));
    for (a, b) in one.iter().zip(&eight) {
        assert_eq!(a.tokens, b.tokens);
        for (ra, rb) in a.logits.iter().zip(&b.logits) {
            assert!(ra
                .iter()
                .zip(rb)
                .all(|(x, y)| *y == *x * 8.0 && f64::from(*y) == 8.0 * f64::from(*x)));
        }
        for (ta, tb) in a.top.iter().zip(&b.top) {
            let u = |t: &fp_play_runtime::decoder::forward::Top2| {
                ulp_distance(Format::Fp32, t.top.to_bits(), t.second.to_bits())
            };
            assert_eq!(u(ta), u(tb));
            assert_eq!(tb.margin(), 8.0 * ta.margin());
        }
    }
}

/// Without a collective, a sequence's logits do not depend on which other
/// sequences share its steps.
#[test]
fn single_device_decoding_is_batch_invariant() {
    let config = DecoderConfig::TINY;
    let weights = Weights::generate(&config);
    let ps = prompts(0xba7c4, 5, 7, config.vocab);
    let mut model = Model::with_weights(config, weights);
    let reduce = &mut Unsharded { policy: POLICIES[1] };
    let together = generate_static(&mut model, reduce, &ps, 10, false);
    for (i, p) in ps.iter().enumerate() {
        let alone = generate_static(&mut model, reduce, std::slice::from_ref(p), 10, false);
        assert_eq!(alone[0], together[i], "prompt {i}");
    }
}

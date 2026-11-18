//! R2: model constants, the deterministic logarithm and the workload generator.

use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::detmath;
use fp_play_runtime::serving::model::{kv_capacity_blocks, DeviceConfig, ModelConfig, ServingError};
use fp_play_runtime::serving::workload::{fingerprint, LengthDist, WorkloadConfig};

const MODEL: ModelConfig = ModelConfig::EIGHT_B_CLASS;

#[test]
fn eight_b_class_parameter_count() {
    // Hand count: per layer 2·4096·4096 (Q, O) + 2·4096·1024 (K, V)
    // + 3·4096·14336 (MLP) + 2·4096 (norms) = 218,112,000; ×32 layers
    // = 6,979,584,000; + 2·128,000·4096 (embedding, head) + 4096 (final norm).
    assert_eq!(MODEL.layer_params(), 218_112_000);
    assert_eq!(MODEL.params(), 8_028_164_096);
    // 32 layers × 8 heads × 128 × (K, V) × 2 bytes.
    assert_eq!(MODEL.kv_bytes_per_token_per_device(1), 131_072);
    assert_eq!(MODEL.kv_bytes_per_token_per_device(8), 16_384);
}

#[test]
fn default_device_capacity() {
    let device = DeviceConfig::default();
    // (80e9 − 16,056,328,192 − 4e9) ÷ (16 × 131,072) = 28,583.7.
    assert_eq!(kv_capacity_blocks(&MODEL, &device, 1, 16).unwrap(), 28_583);
    // About 450k tokens at P = 1.
    assert_eq!(28_583 * 16, 457_328);
    // (80e9 − 8,028,164,096 − 4e9) ÷ (16 × 65,536) = 64,822.9.
    assert_eq!(kv_capacity_blocks(&MODEL, &device, 2, 16).unwrap(), 64_822);
    // Weights alone overflow a tiny device.
    let tiny = DeviceConfig {
        memory_bytes: 10_000_000_000,
        ..device
    };
    assert!(matches!(
        kv_capacity_blocks(&MODEL, &tiny, 1, 16),
        Err(ServingError::NoKvMemory { .. })
    ));
}

#[test]
fn world_must_divide_kv_heads() {
    let device = DeviceConfig::default();
    for p in [1, 2, 4, 8] {
        assert!(kv_capacity_blocks(&MODEL, &device, p, 16).is_ok(), "P={p}");
    }
    for p in [0, 3, 5, 6, 7, 12, 16, 32] {
        let err = kv_capacity_blocks(&MODEL, &device, p, 16).unwrap_err();
        assert_eq!(err, ServingError::WorldDoesNotDivideHeads { world: p, kv_heads: 8 });
        assert!(err.to_string().contains("must divide the 8 KV heads"), "{err}");
    }
}

#[test]
fn deterministic_ln_matches_the_host() {
    let mut rng = SplitMix64::new(0x1e);
    let mut worst = 0.0f64;
    let mut check = |x: f64| {
        let (ours, host) = (detmath::ln(x), x.ln());
        // Relative to max(|ln x|, 1): near x = 1 the absolute error is what matters.
        let err = (ours - host).abs() / host.abs().max(1.0);
        worst = worst.max(err);
    };
    for _ in 0..200_000 {
        check(1.0 - rng.next_unit());
    }
    for e in -1074..=1023 {
        // 2^e exactly, from its bit pattern (powi underflows to 0 below 2^-1022).
        let x = if e >= -1022 {
            f64::from_bits(((e + 1023) as u64) << 52)
        } else {
            f64::from_bits(1u64 << (e + 1074))
        };
        check(x);
        check(x * 1.3);
    }
    for x in [
        f64::MIN_POSITIVE,
        5e-324,
        f64::MAX,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
    ] {
        check(x);
    }
    assert!(worst < 1e-15, "worst relative error {worst:e}");
}

#[test]
#[should_panic(expected = "finite positive")]
fn ln_rejects_zero() {
    detmath::ln(0.0);
}

/// The default workload's fingerprint. Generated on macOS arm64; CI checks it
/// on Linux x86-64, so this also pins cross-platform determinism.
const DEFAULT_FINGERPRINT: u64 = 0x6dc5_be01_7545_1b62;

#[test]
fn workload_is_deterministic_and_pinned() {
    let config = WorkloadConfig::default();
    let a = config.generate().unwrap();
    let b = config.generate().unwrap();
    assert_eq!(a, b);
    assert_eq!(
        fingerprint(&a),
        DEFAULT_FINGERPRINT,
        "fingerprint {:#018x}",
        fingerprint(&a)
    );
    let other = WorkloadConfig {
        seed: config.seed + 1,
        ..config
    };
    assert_ne!(fingerprint(&other.generate().unwrap()), DEFAULT_FINGERPRINT);
}

#[test]
fn workload_shape() {
    let config = WorkloadConfig {
        requests: 4000,
        rate_per_s: 50.0,
        prompt: LengthDist::Table(vec![(3, 100), (1, 300)]),
        output: LengthDist::Uniform { lo: 5, hi: 9 },
        prefix_groups: 3,
        prefix_len: 64,
        prefix_permille: 500,
        samples: vec![(2, 1), (1, 2), (1, 4)],
        priorities: vec![(9, 0), (1, 1)],
        ..WorkloadConfig::default()
    };
    let reqs = config.generate().unwrap();
    assert_eq!(reqs.len(), 4000);
    assert!(reqs.windows(2).all(|w| w[0].arrival_ns <= w[1].arrival_ns));
    for (i, r) in reqs.iter().enumerate() {
        assert_eq!(r.id as usize, i);
        let own = r.prompt_len - r.prefix_len;
        assert!(own == 100 || own == 300);
        assert!((5..=9).contains(&r.output_len));
        assert!([1, 2, 4].contains(&r.samples));
        match r.prefix_group {
            Some(g) => assert!(g < 3 && r.prefix_len == 64),
            None => assert_eq!(r.prefix_len, 0),
        }
    }
    // Mean inter-arrival time near 1/rate = 20 ms (4000 samples: within 5%).
    let mean_gap = reqs.last().unwrap().arrival_ns as f64 / 4000.0;
    assert!((mean_gap / 20e6 - 1.0).abs() < 0.05, "{mean_gap}");
    let shared = reqs.iter().filter(|r| r.prefix_group.is_some()).count();
    assert!((1800..2200).contains(&shared), "{shared}");
    let high = reqs.iter().filter(|r| r.priority == 1).count();
    assert!((300..500).contains(&high), "{high}");
}

#[test]
fn tokens_are_shared_exactly_where_intended() {
    let config = WorkloadConfig {
        requests: 50,
        prefix_groups: 2,
        prefix_len: 32,
        prefix_permille: 1000,
        samples: vec![(1, 4)],
        ..WorkloadConfig::default()
    };
    let reqs = config.generate().unwrap();
    let (a, b) = reqs
        .iter()
        .flat_map(|a| reqs.iter().map(move |b| (a, b)))
        .find(|(a, b)| a.id != b.id && a.prefix_group == b.prefix_group)
        .unwrap();
    for p in 0..32 {
        assert_eq!(a.prompt_token(p), b.prompt_token(p));
    }
    assert_ne!(a.prompt_token(32), b.prompt_token(32));
    // Samples share the prompt and differ in their outputs.
    let n = a.prompt_len;
    assert_eq!(a.sequence_token(0, n - 1), a.sequence_token(3, n - 1));
    assert_ne!(a.sequence_token(0, n), a.sequence_token(1, n));
}

#[test]
fn invalid_workloads_are_rejected() {
    let base = WorkloadConfig::default();
    let bad = [
        WorkloadConfig {
            rate_per_s: 0.0,
            ..base.clone()
        },
        WorkloadConfig {
            rate_per_s: f64::NAN,
            ..base.clone()
        },
        WorkloadConfig {
            prompt: LengthDist::Uniform { lo: 9, hi: 3 },
            ..base.clone()
        },
        WorkloadConfig {
            output: LengthDist::Fixed(0),
            ..base.clone()
        },
        WorkloadConfig {
            samples: vec![(1, 3)],
            ..base.clone()
        },
        WorkloadConfig {
            prefix_permille: 10,
            ..base.clone()
        },
        WorkloadConfig {
            priorities: vec![],
            ..base.clone()
        },
    ];
    for config in bad {
        assert!(config.generate().is_err(), "{config:?}");
    }
}

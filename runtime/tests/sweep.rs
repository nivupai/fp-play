//! The sensitivity sweep (results/r1.1-sweep.*).

use fp_play_runtime::collectives::{Algorithm, WirePolicy};
use fp_play_runtime::report;
use fp_play_runtime::sweep::{run, stats, Schedule, Stats, SweepConfig, POLICIES};

#[test]
fn order_statistics_are_exact() {
    let s = |v: &[f64]| stats(v);
    assert_eq!(
        s(&[3.0]),
        Stats {
            min: 3.0,
            median: 3.0,
            max: 3.0
        }
    );
    assert_eq!(
        s(&[5.0, 1.0, 3.0]),
        Stats {
            min: 1.0,
            median: 3.0,
            max: 5.0
        }
    );
    // Even count: mean of the two middle values (8 seeds in the recorded sweep).
    assert_eq!(
        s(&[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]),
        Stats {
            min: 1.0,
            median: 4.5,
            max: 8.0
        }
    );
    assert_eq!(
        s(&[-1.0, 0.0]),
        Stats {
            min: -1.0,
            median: -0.5,
            max: 0.0
        }
    );
    assert_eq!(s(&[2.0, 2.0, 2.0, 9.0]).median, 2.0);
}

fn small() -> SweepConfig {
    SweepConfig {
        tokens: 2,
        outputs: 8,
        hiddens: vec![32, 48],
        world_sizes: vec![2, 3, 4, 8],
        seeds: vec![1, 2, 3],
        shards: 8,
    }
}

#[test]
fn the_grid_covers_every_applicable_configuration() {
    let config = small();
    let result = run(&config).unwrap();
    assert!(result.shard_tree_p_invariant);
    // Per (seed, hidden, shard-capable policy): every swept P after the first.
    assert_eq!(result.shard_tree_invariance_checks, 3 * 2 * 3 * (4 - 1));
    let mut expected_rows = 0;
    for &world in &config.world_sizes {
        for policy in POLICIES {
            for schedule in Schedule::ALL {
                let applies = match schedule {
                    Schedule::Collective(a) => a.supports(world),
                    Schedule::ShardTree => policy != WirePolicy::BF16_WIRE_FP32_ADD,
                };
                assert_eq!(schedule.applies(world, policy, config.shards), applies);
                expected_rows += usize::from(applies);
            }
        }
    }
    assert_eq!(result.rows.len(), expected_rows * config.hiddens.len());
    // The listed top-1 changes are exactly the runs behind each row's top-1 statistics.
    for r in &result.rows {
        let listed: Vec<f64> = result
            .top1_changes
            .iter()
            .filter(|t| t.hidden == r.hidden && t.world == r.world && t.policy == r.policy && t.schedule == r.schedule)
            .map(|t| t.tokens.len() as f64)
            .collect();
        assert_eq!(listed.iter().copied().fold(0.0, f64::max), r.top1_changes.max);
        assert_eq!(listed.is_empty(), r.top1_changes.max == 0.0);
        assert!(listed.len() <= r.runs);
    }
    for r in &result.rows {
        assert_eq!(r.runs, config.seeds.len());
        for s in [
            r.mismatched_fraction,
            r.relative_rms_error,
            r.max_ulp,
            r.vs_ring_fraction,
            r.top1_changes,
        ] {
            assert!(s.min <= s.median && s.median <= s.max);
        }
        if r.schedule == Schedule::Collective(Algorithm::Ring) {
            assert_eq!(r.vs_ring_fraction.max, 0.0);
        }
        if r.world == 2 && r.schedule != Schedule::ShardTree && r.policy.reduce_wire.contains(r.policy.accumulator) {
            // With two partials and a lossless wire every collective computes the
            // same sum. (Under bf16-wire/fp32-add the schedules differ in which
            // rank's partial is rounded onto the wire.)
            assert_eq!(r.vs_ring_fraction.max, 0.0, "{} {}", r.policy.name(), r.schedule.name());
        }
    }
}

#[test]
fn invalid_sweeps_are_rejected() {
    assert!(run(&SweepConfig {
        seeds: vec![],
        ..small()
    })
    .is_err());
    assert!(run(&SweepConfig {
        hiddens: vec![4],
        ..small()
    })
    .is_err());
    assert!(run(&SweepConfig {
        world_sizes: vec![],
        ..small()
    })
    .is_err());
}

#[test]
fn recorded_artifact_is_current() {
    let command = "fp-play-runtime experiment --out results";
    let result = run(&SweepConfig::default()).unwrap();
    assert!(
        result.shard_tree_p_invariant,
        "shard-tree P-invariance failed in the sweep"
    );
    let json = report::sweep_to_json(&result, command).to_pretty();
    let markdown = report::sweep_to_markdown(&result, command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r1.1-sweep.json"))
        .expect("results/r1.1-sweep.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r1.1-sweep.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r1.1-sweep.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r1.1-sweep.md is stale; re-run the documented command"
    );
}

#[test]
fn top1_lead_counts_representable_values_to_the_runner_up() {
    use fp_play_runtime::format::Format;
    use fp_play_runtime::tensor_parallel::{argmax, changed_top1, top1_lead_ulps};
    let bf16 = Format::Bf16;
    // BF16 1.0 = 0x3f80; 0x3f83 is three ulps above it.
    assert_eq!(top1_lead_ulps(bf16, &[0x3f80, 0x3f83, 0x3f00]), Some(3));
    assert_eq!(argmax(bf16, &[0x3f80, 0x3f83, 0x3f00]), 1);
    // An exact tie: lead 0, and the lowest index wins.
    assert_eq!(top1_lead_ulps(bf16, &[0x3f83, 0x3f00, 0x3f83]), Some(0));
    assert_eq!(argmax(bf16, &[0x3f83, 0x3f00, 0x3f83]), 0);
    // All negative: -1.0 (0xbf80) leads -1.0 - 1 ulp (0xbf81) by one.
    assert_eq!(top1_lead_ulps(bf16, &[0xbf81, 0xbf80]), Some(1));
    // Across zero: +1 smallest subnormal vs -1 smallest subnormal is two ulps apart.
    assert_eq!(top1_lead_ulps(bf16, &[0x8001, 0x0001]), Some(2));
    assert_eq!(top1_lead_ulps(bf16, &[0x3f80]), None);
    // Rows are tokens: only token 1's argmax moves.
    assert_eq!(
        changed_top1(
            bf16,
            2,
            &[0x3f80, 0x3f00, 0x3f00, 0x3f80],
            &[0x3f80, 0x3f00, 0x3f80, 0x3f00]
        ),
        vec![1]
    );
}

#[test]
fn recorded_leads_are_consistent_with_the_top1_list() {
    let config = small();
    let result = run(&config).unwrap();
    let per_format = config.tokens * config.seeds.len() * config.hiddens.len();
    for summary in &result.lead_summary {
        assert_eq!(summary.tokens, per_format);
        let leads: Vec<f64> = result
            .reference_leads
            .iter()
            .filter(|p| p.format == summary.format)
            .flat_map(|p| p.leads.iter().map(|&l| l as f64))
            .collect();
        assert_eq!(stats(&leads), summary.leads);
    }
    for t in &result.top1_changes {
        assert!(!t.tokens.is_empty());
        let problem = result
            .reference_leads
            .iter()
            .find(|p| p.seed == t.seed && p.hidden == t.hidden && p.format == t.policy.output)
            .unwrap();
        for l in &t.tokens {
            assert_eq!(problem.leads[l.token], l.lead_ulps);
        }
    }
}

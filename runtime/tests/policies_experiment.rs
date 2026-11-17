//! The R1.1 policies experiment (results/r1.1-policies.*).

use fp_play_runtime::collectives::{Algorithm, WirePolicy};
use fp_play_runtime::format::Format;
use fp_play_runtime::policies::{self, POLICIES, SPLIT_ALGORITHMS};
use fp_play_runtime::report;
use fp_play_runtime::tensor_parallel::ExperimentConfig;

fn small() -> ExperimentConfig {
    ExperimentConfig {
        tokens: 3,
        hidden: 96,
        outputs: 17,
        world_sizes: vec![2, 3, 4, 8],
        ..Default::default()
    }
}

#[test]
fn reruns_are_byte_identical() {
    let a = report::policies_to_json(&policies::run(&small()).unwrap(), "x").to_pretty();
    let b = report::policies_to_json(&policies::run(&small()).unwrap(), "x").to_pretty();
    assert_eq!(a, b);
}

#[test]
fn structural_invariants_hold() {
    let result = policies::run(&small()).unwrap();
    for w in &result.worlds {
        let p = w.world as u32;
        let supported = Algorithm::ALL.into_iter().filter(|a| a.supports(w.world)).count();
        assert_eq!(w.runs.len(), POLICIES.len() * supported);
        for r in &w.runs {
            let what = format!("{} {} P={p}", r.policy.name(), r.algorithm);
            assert!(r.consistent_across_ranks, "{what}");
            assert_eq!(r.output.len(), result.elements);
            // Rounding counts per element are those derived in tests/policies.rs.
            let exact_gather = r.algorithm == Algorithm::ExactGather;
            let expected = match (r.policy, exact_gather) {
                (_, true) if r.policy.is_uniform() => (p, 1),
                (_, false) if r.policy.is_uniform() => (p, p - 1),
                (WirePolicy::FP32_RS_BF16_AG, true) => (0, 1),
                (WirePolicy::FP32_RS_BF16_AG, false) => (0, p),
                (WirePolicy::BF16_WIRE_FP32_ADD, true) => (0, p + 1),
                (WirePolicy::BF16_WIRE_FP32_ADD, false) => (0, 2 * p - 1),
                _ => unreachable!(),
            };
            assert_eq!(
                (r.input_roundings, r.collective_roundings),
                (expected.0, (expected.1, expected.1)),
                "{what}"
            );
        }
        // Exact-gather of FP32 partials rounded once is the ideal by construction.
        let ideal = w.run(WirePolicy::FP32_RS_BF16_AG, Algorithm::ExactGather).unwrap();
        assert_eq!(ideal.vs_ideal.mismatched, 0);
        assert_eq!(ideal.vs_exact.mismatched, w.ideal_vs_exact.mismatched);
        // Exact-gather quantizes every input to the wire, so bf16-wire/fp32-add
        // exact-gather is bf16 exact-gather bit for bit.
        assert_eq!(
            w.run(WirePolicy::BF16_WIRE_FP32_ADD, Algorithm::ExactGather)
                .unwrap()
                .output,
            w.run(POLICIES[0], Algorithm::ExactGather).unwrap().output
        );
        // fp32-rs/bf16-ag output is exactly the FP32 collective rounded once:
        // with two ranks every schedule then computes round(x0 + x1) = ideal.
        if w.world == 2 {
            for r in w.runs.iter().filter(|r| r.policy == WirePolicy::FP32_RS_BF16_AG) {
                assert_eq!(r.vs_ideal.mismatched, 0, "{}", r.algorithm);
            }
        }
        let split = SPLIT_ALGORITHMS.into_iter().filter(|a| a.supports(w.world)).count();
        assert_eq!(w.split.len(), POLICIES.len() * split);
        for s in &w.split {
            assert!(s.bits_match_all_reduce, "{} {} P={p}", s.policy.name(), s.algorithm);
            assert!(s.reduce_scatter.rounds < s.all_reduce.rounds);
            assert_eq!(s.reduce_scatter.max_gather_bytes_per_rank, 0);
            assert_eq!(
                s.reduce_scatter.max_reduce_bytes_per_rank,
                s.all_reduce.max_reduce_bytes_per_rank
            );
        }
        // Recovery rows exist for every reduction schedule and both mixed policies.
        assert_eq!(w.recovery.len(), 2 * (supported - 1));
    }
    assert!(result
        .worlds
        .iter()
        .all(|w| w.recovery.iter().all(|r| r.policy.accumulator == Format::Fp32)));
}

#[test]
fn recorded_artifact_is_current() {
    // The committed artifact must equal a fresh run of the documented command.
    let config = ExperimentConfig::default();
    let command = "fp-play-runtime experiment --out results";
    let result = policies::run(&config).unwrap();
    let json = report::policies_to_json(&result, command).to_pretty();
    let markdown = report::policies_to_markdown(&result, command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r1.1-policies.json"))
        .expect("results/r1.1-policies.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r1.1-policies.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r1.1-policies.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r1.1-policies.md is stale; re-run the documented command"
    );
}

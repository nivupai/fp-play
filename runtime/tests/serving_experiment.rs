//! The recorded serving scenarios (results/r2-serving.*).

use std::sync::OnceLock;

use fp_play_runtime::serving::experiment::{run_all, ServingConfig, ServingResult};
use fp_play_runtime::serving::report;

/// One run of every scenario, shared by the tests in this file.
fn result() -> &'static ServingResult {
    static RESULT: OnceLock<ServingResult> = OnceLock::new();
    RESULT.get_or_init(|| run_all(&ServingConfig::default()).unwrap())
}

#[test]
fn recorded_artifact_is_current() {
    let command = "fp-play-runtime experiment --out results";
    let json = report::to_json(result(), command).to_pretty();
    let markdown = report::to_markdown(result(), command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r2-serving.json"))
        .expect("results/r2-serving.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r2-serving.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r2-serving.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r2-serving.md is stale; re-run the documented command"
    );
}

#[test]
fn memory_pressure_scenario_preempts_in_both_modes() {
    let pressure = &result().pressure;
    assert_eq!(pressure.len(), 2);
    let (recompute, swap) = (&pressure[0].metrics, &pressure[1].metrics);
    assert!(recompute.preemptions_recompute > 0, "the recompute run never preempted");
    assert!(recompute.recomputed_tokens > 0);
    assert!(swap.preemptions_swap > 0, "the swap run never swapped");
    assert!(swap.swapped_blocks > 0);
}

#[test]
fn every_recorded_run_completes() {
    let r = result();
    for run in r.load.iter().chain(&r.pressure).chain(&r.prefix).chain(&r.collectives) {
        let m = &run.metrics;
        assert_eq!(m.finished, m.requests, "{}", run.label);
        assert_eq!(m.rejected, 0, "{}", run.label);
        assert_eq!(m.stalls, 0, "{}", run.label);
        assert!(m.tpot.is_some() && m.ttft.is_some(), "{}", run.label);
    }
    // The prefix cache only hits when it is on.
    assert!(r.prefix[0].metrics.prefix_hit_blocks == 0 && r.prefix[1].metrics.prefix_hit_blocks > 0);
}

#[test]
fn halving_doubling_communicates_less_than_ring_at_p8() {
    // Same bytes per rank, 6 rounds instead of 14: a property of the schedules.
    let r = result();
    let comm = |name: &str, policy: &str| {
        r.collectives
            .iter()
            .find(|x| x.world == 8 && x.collective.name() == name && x.policy.name() == policy)
            .unwrap()
            .metrics
            .comm_ns
    };
    for policy in ["bf16", "fp32-rs/bf16-ag"] {
        assert!(comm("halving-doubling", policy) < comm("ring", policy), "{policy}");
    }
}

#[test]
fn failure_scenario_outcomes() {
    let f = &result().failures;
    assert_eq!(f.len(), 4);
    assert!(f[0].fault_policy.is_none() && f[0].fault_outcomes.is_empty());
    for run in &f[1..] {
        assert_eq!(run.fault_outcomes.len(), 1, "{}: the failure must fire once", run.label);
        assert_eq!(run.rerun_identical, Some(true), "{}: G5", run.label);
        assert_eq!(run.metrics.fault_steps, 1);
        let m = &run.metrics;
        assert_eq!(m.finished + m.failed + m.rejected, m.requests, "{}: G4", run.label);
    }
    let (abort, retry, drain) = (&f[1].metrics, &f[2].metrics, &f[3].metrics);
    assert!(abort.failed > 0 && abort.recomputed_tokens == 0);
    assert!(retry.failed == 0 && retry.finished == retry.requests && retry.recomputed_tokens > 0);
    assert!(drain.rejected > 0 && drain.failed == 0);
}

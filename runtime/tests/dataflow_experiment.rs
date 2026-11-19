//! The recorded R5 experiment (results/r5-dataflow.*): stale check and the
//! properties its text relies on.

use fp_play_runtime::dataflow::experiment::{run, DataflowConfig};
use fp_play_runtime::dataflow::report;

#[test]
fn recorded_artifact_is_current() {
    let r = run(&DataflowConfig::default());
    let command = "fp-play-runtime experiment --out results";
    let json = report::to_json(&r, command).to_pretty();
    let markdown = report::to_markdown(&r, command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r5-dataflow.json"))
        .expect("results/r5-dataflow.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r5-dataflow.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r5-dataflow.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r5-dataflow.md is stale; re-run the documented command"
    );
}

#[test]
fn every_run_is_at_or_above_its_roofline_bound_and_shares_add_up() {
    let r = run(&DataflowConfig::default());
    for x in &r.runs {
        let m = &r.config.machines[x.machine];
        assert!(x.timing.cycles() >= x.roofline_cycles(m), "{}", x.phase.name());
        let b = x.breakdown(m);
        let sum = b.useful + b.waste + b.vector + b.memory_stall + b.fill;
        assert!(
            (sum - x.timing.cycles() as f64).abs() < 1e-6 * sum,
            "{}",
            x.phase.name()
        );
    }
}

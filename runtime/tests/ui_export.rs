//! R6a: the `#collectives` workspace data (results/r6-collectives.*).

use fp_play_runtime::tensor_parallel::{run, ExperimentConfig};
use fp_play_runtime::ui_export;

#[test]
fn recorded_artifact_is_current() {
    let command = "fp-play-runtime experiment --out results";
    let tp = run(&ExperimentConfig::default()).unwrap();
    let json = ui_export::to_json(&tp, command).unwrap().to_pretty();
    let markdown = ui_export::to_markdown(command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r6-collectives.json"))
        .expect("results/r6-collectives.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r6-collectives.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r6-collectives.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r6-collectives.md is stale; re-run the documented command"
    );
    // Every schedule ends with all ranks agreeing on every association.
    assert!(!json.contains("\"ranks_agree\": false"));
}

//! Slow: retrains the model; run by the trained-model workflow, not the
//! normal runtime job (the weights' SHA-256 is still checked there by
//! `tests/train.rs` and by `hashlib`).
//!
//! The recorded training run (results/r4-training.* and
//! results/r4-trained-weights.bin): retrains with the default settings and
//! requires every byte to match, and the weights' SHA-256 to be the pinned one.

use std::sync::OnceLock;

use fp_play_runtime::train::report::{run, to_json, to_markdown, TrainingRecord, GREEDY_PROMPTS};
use fp_play_runtime::train::{to_bytes, TrainConfig, TRAINED_WEIGHTS_SHA256};

fn record() -> &'static TrainingRecord {
    static R: OnceLock<TrainingRecord> = OnceLock::new();
    R.get_or_init(|| run(&TrainConfig::default()))
}

#[test]
#[ignore = "slow: runs in .github/workflows/runtime-trained.yml (cargo test -- --include-ignored)"]
fn recorded_training_is_reproduced_bit_for_bit() {
    let r = record();
    assert_eq!(
        r.result.sha256, TRAINED_WEIGHTS_SHA256,
        "retrained weights differ from the pinned SHA-256"
    );
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let bin = std::fs::read(format!("{dir}/r4-trained-weights.bin")).expect("weights file missing");
    assert!(
        bin == to_bytes(&r.result.weights),
        "results/r4-trained-weights.bin is stale"
    );
    let command = "fp-play-runtime experiment --out results";
    let json = std::fs::read_to_string(format!("{dir}/r4-training.json")).unwrap();
    let md = std::fs::read_to_string(format!("{dir}/r4-training.md")).unwrap();
    assert!(
        to_json(r, command).to_pretty() == json,
        "results/r4-training.json is stale"
    );
    assert!(to_markdown(r, command) == md, "results/r4-training.md is stale");
}

#[test]
#[ignore = "slow: runs in .github/workflows/runtime-trained.yml (cargo test -- --include-ignored)"]
fn the_trained_model_solves_the_task() {
    let r = record();
    let last = r.result.steps.last().unwrap();
    assert!(last.loss < r.result.steps[0].loss / 1000.0);
    assert_eq!(r.result.eval_accuracy, 1.0);
    for g in &r.greedy {
        assert_eq!(g.sequences, GREEDY_PROMPTS);
        assert_eq!(g.correct_sequences, g.sequences, "{}", g.model);
    }
}

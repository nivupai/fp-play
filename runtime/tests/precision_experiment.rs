//! The recorded R4 experiment (results/r4-precision.*): stale check and
//! the properties its text relies on.

use std::sync::OnceLock;

use fp_play_runtime::precision::experiment::{run, Kind, PrecisionConfig, PrecisionResult};
use fp_play_runtime::precision::policy::{Level, Site};
use fp_play_runtime::precision::report;

fn result() -> &'static PrecisionResult {
    static RESULT: OnceLock<PrecisionResult> = OnceLock::new();
    RESULT.get_or_init(|| run(&PrecisionConfig::default()).unwrap())
}

#[test]
fn recorded_artifact_is_current() {
    let command = "fp-play-runtime experiment --out results";
    let json = report::to_json(result(), command).to_pretty();
    let markdown = report::to_markdown(result(), command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r4-precision.json"))
        .expect("results/r4-precision.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r4-precision.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r4-precision.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r4-precision.md is stale; re-run the documented command"
    );
}

#[test]
fn the_fp32_policy_reproduces_the_reference_exactly() {
    let r = result();
    let fp32 = r.runs.iter().find(|x| x.kind == Kind::Uniform(Level::Fp32)).unwrap();
    assert_eq!(fp32.drift.max_abs, 0.0);
    assert_eq!(fp32.drift.top1_agree, fp32.drift.scored);
    assert_eq!(fp32.diverged(), 0);
    assert_eq!(fp32.drift.perplexity(), r.reference_perplexity);
}

#[test]
fn every_run_scores_every_continuation_token() {
    let r = result();
    let want = r.config.prompts * r.config.continuation;
    assert!(r.runs.iter().all(|x| x.drift.tokens == want));
    // The KV cache appears only at its storage levels.
    let kv = r
        .runs
        .iter()
        .filter(|x| matches!(x.kind, Kind::OneSite(Site::KvCache, _)))
        .count();
    assert_eq!(kv, 2);
}

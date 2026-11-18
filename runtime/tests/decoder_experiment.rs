//! The recorded R3 experiment (results/r3-decode.*): stale check and the
//! properties its text states.

use std::sync::OnceLock;

use fp_play_runtime::decoder::experiment::{run, DecodeConfig, DecodeResult};
use fp_play_runtime::decoder::parallel::Schedule;
use fp_play_runtime::decoder::report;

fn result() -> &'static DecodeResult {
    static RESULT: OnceLock<DecodeResult> = OnceLock::new();
    RESULT.get_or_init(|| run(&DecodeConfig::default()).unwrap())
}

#[test]
fn recorded_artifact_is_current() {
    let command = "fp-play-runtime experiment --out results";
    let json = report::to_json(result(), command).to_pretty();
    let markdown = report::to_markdown(result(), command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r3-decode.json"))
        .expect("results/r3-decode.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r3-decode.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r3-decode.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r3-decode.md is stale; re-run the documented command"
    );
}

#[test]
fn the_batch_invariance_prediction_held() {
    assert!(result().prediction_held());
}

#[test]
fn shard_tree_tokens_and_divergence_do_not_depend_on_p() {
    let r = result();
    for &policy in &r.config.policies {
        for &scale in &r.config.scales {
            let trees: Vec<_> = r
                .grid
                .iter()
                .filter(|g| g.policy == policy && g.scale == scale && matches!(g.schedule, Schedule::ShardTree { .. }))
                .collect();
            assert_eq!(trees.len(), 4);
            assert!(trees
                .iter()
                .all(|g| g.tokens == trees[0].tokens && g.first_divergence == trees[0].first_divergence));
        }
    }
}

#[test]
fn logit_scale_changes_no_token_and_every_flip_is_reachable() {
    let r = result();
    let s0 = r.config.scales[0];
    for g in r.grid.iter().filter(|g| g.scale != s0) {
        let h = r
            .grid
            .iter()
            .find(|h| h.scale == s0 && h.world == g.world && h.schedule == g.schedule && h.policy == g.policy)
            .unwrap();
        assert_eq!(g.tokens, h.tokens);
        assert_eq!(g.first_divergence, h.first_divergence);
    }
    for g in &r.grid {
        for t in g.at_risk.iter().filter(|t| t.flipped) {
            assert!(t.margin_ulps <= 2.0 * t.perturbation_ulps, "{t:?}");
        }
        assert_eq!(g.at_risk.iter().filter(|t| t.flipped).count(), g.diverged());
    }
}

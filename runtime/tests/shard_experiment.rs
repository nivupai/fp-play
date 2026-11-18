//! The shard-tree experiment (results/r1.1-shard-tree.*).

use fp_play_runtime::collectives::WirePolicy;
use fp_play_runtime::report;
use fp_play_runtime::shard_experiment::{run_prepared, POLICIES};
use fp_play_runtime::shard_tree::DEFAULT_SHARDS;
use fp_play_runtime::tensor_parallel::{prepare, ExperimentConfig};

fn small() -> ExperimentConfig {
    ExperimentConfig {
        tokens: 3,
        hidden: 96,
        outputs: 17,
        world_sizes: vec![2, 3, 4],
        ..Default::default()
    }
}

#[test]
fn invariants_hold_for_two_shard_counts() {
    let config = small();
    let prepared = prepare(&config).unwrap();
    for shards in [16, 12] {
        let result = run_prepared(&config, &prepared, shards).unwrap();
        assert_eq!(result.shards, shards);
        assert_eq!(result.policies.len(), POLICIES.len());
        for p in &result.policies {
            let what = format!("{} S={shards}", p.policy.name());
            assert!(p.p_invariant && p.consistent_across_ranks, "{what}");
            assert_eq!(p.per_world.len(), shards, "{what}: every P from 1 to S");
            assert!(p.per_world.iter().all(|w| w.serialization_extra_rounds == 0), "{what}");
            let fused = p.policy == WirePolicy::FP32_RS_BF16_AG;
            assert_eq!(p.input_roundings, if fused { 0 } else { shards as u32 }, "{what}");
            let expected = shards as u32 - 1 + u32::from(fused);
            assert_eq!(p.collective_roundings, (expected, expected), "{what}");
        }
        assert_eq!(result.baselines.len(), config.world_sizes.len() * POLICIES.len() * 2);
    }
    // Reruns are byte-identical. (The invariance is across P for a fixed S; a
    // different S gives different shard partials and a different tree.)
    let reruns = |s| report::shard_tree_to_json(&run_prepared(&config, &prepared, s).unwrap(), "x").to_pretty();
    assert_eq!(reruns(16), reruns(16));
}

#[test]
fn invalid_shard_counts_are_rejected() {
    let config = small();
    let prepared = prepare(&config).unwrap();
    assert!(run_prepared(&config, &prepared, 0).is_err());
    assert!(run_prepared(&config, &prepared, 97).is_err());
    // A baseline world larger than S cannot own a shard per device.
    assert!(run_prepared(&config, &prepared, 3).is_err());
}

#[test]
fn recorded_artifact_is_current() {
    let config = ExperimentConfig::default();
    let command = "fp-play-runtime experiment --out results";
    let result = run_prepared(&config, &prepare(&config).unwrap(), DEFAULT_SHARDS).unwrap();
    assert!(
        result.policies.iter().all(|p| p.p_invariant),
        "P-invariance failed on the recorded problem"
    );
    let json = report::shard_tree_to_json(&result, command).to_pretty();
    let markdown = report::shard_tree_to_markdown(&result, command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/r1.1-shard-tree.json"))
        .expect("results/r1.1-shard-tree.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/r1.1-shard-tree.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/r1.1-shard-tree.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/r1.1-shard-tree.md is stale; re-run the documented command"
    );
}

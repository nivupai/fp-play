use fp_play_runtime::collectives::Algorithm;
use fp_play_runtime::format::Format;
use fp_play_runtime::report;
use fp_play_runtime::tensor_parallel::{run, ExperimentConfig};

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
fn reruns_are_byte_identical() {
    let a = report::to_json(&run(&small()).unwrap(), "x").to_pretty();
    let b = report::to_json(&run(&small()).unwrap(), "x").to_pretty();
    assert_eq!(a, b);
}

#[test]
fn structural_invariants_hold() {
    let result = run(&small()).unwrap();
    for w in &result.worlds {
        let names: Vec<_> = w.algorithms.iter().map(|a| a.algorithm).collect();
        assert_eq!(names.contains(&Algorithm::HalvingDoubling), w.world.is_power_of_two());
        for a in &w.algorithms {
            assert!(a.consistent_across_ranks, "{} P={}", a.algorithm, w.world);
            assert_eq!(a.output.len(), result.elements);
        }
        if w.world == 2 {
            // With two partials every schedule computes round(x0 + x1).
            assert!(w.pairwise_mismatches.iter().all(|(_, _, d)| *d == 0));
        }
    }
}

#[test]
fn invalid_configurations_are_rejected() {
    assert!(run(&ExperimentConfig {
        world_sizes: vec![1],
        ..small()
    })
    .is_err());
    assert!(run(&ExperimentConfig {
        world_sizes: vec![200],
        ..small()
    })
    .is_err());
    assert!(run(&ExperimentConfig {
        wire_formats: vec![],
        ..small()
    })
    .is_err());
    assert!(run(&ExperimentConfig { tokens: 0, ..small() }).is_err());
}

#[test]
fn recorded_artifact_is_current() {
    // The committed artifact must equal a fresh run of the documented command.
    let config = ExperimentConfig::default();
    assert_eq!(config.wire_formats, vec![Format::Fp32, Format::Bf16]);
    let command = "fp-play-runtime experiment --out results";
    let result = run(&config).unwrap();
    let json = report::to_json(&result, command).to_pretty();
    let markdown = report::to_markdown(&result, command);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let recorded_json = std::fs::read_to_string(format!("{dir}/tp-allreduce.json"))
        .expect("results/tp-allreduce.json missing: run `cargo run --release -- experiment --out results`");
    let recorded_md = std::fs::read_to_string(format!("{dir}/tp-allreduce.md")).unwrap();
    assert!(
        json == recorded_json,
        "results/tp-allreduce.json is stale; re-run the documented command"
    );
    assert!(
        markdown == recorded_md,
        "results/tp-allreduce.md is stale; re-run the documented command"
    );
}

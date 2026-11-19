//! The reruns on the trained model (results/r3-trained.*, results/r4-trained.*):
//! stale checks on the committed weights, and the properties their text relies on.

use std::sync::OnceLock;

use fp_play_runtime::precision::experiment::Kind;
use fp_play_runtime::precision::model::MasterWeights;
use fp_play_runtime::precision::policy::Level;
use fp_play_runtime::train::load_trained;
use fp_play_runtime::train::rerun::{copy_workload_prompts, run_r3, run_r4, R3Rerun, R4Rerun, COPY_WORKLOAD};
use fp_play_runtime::train::rerun_report::{
    invariance, pooled, r3_to_json, r3_to_markdown, r4_to_json, r4_to_markdown,
};

const COMMAND: &str = "fp-play-runtime experiment --out results";

fn weights() -> &'static MasterWeights {
    static W: OnceLock<MasterWeights> = OnceLock::new();
    W.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/results/r4-trained-weights.bin");
        load_trained(&std::fs::read(path).expect("weights file")).unwrap()
    })
}

fn r3() -> &'static R3Rerun {
    static R: OnceLock<R3Rerun> = OnceLock::new();
    R.get_or_init(|| run_r3(weights()).unwrap())
}

fn r4() -> &'static R4Rerun {
    static R: OnceLock<R4Rerun> = OnceLock::new();
    R.get_or_init(|| run_r4(weights()).unwrap())
}

fn recorded(name: &str) -> String {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    std::fs::read_to_string(format!("{dir}/{name}"))
        .unwrap_or_else(|_| panic!("results/{name} missing: run `cargo run --release -- experiment --out results`"))
}

#[test]
fn recorded_r3_rerun_is_current() {
    assert!(
        r3_to_json(r3(), COMMAND).to_pretty() == recorded("r3-trained.json"),
        "results/r3-trained.json is stale"
    );
    assert!(
        r3_to_markdown(r3(), COMMAND) == recorded("r3-trained.md"),
        "results/r3-trained.md is stale"
    );
}

#[test]
fn recorded_r4_rerun_is_current() {
    assert!(
        r4_to_json(r4(), COMMAND).to_pretty() == recorded("r4-trained.json"),
        "results/r4-trained.json is stale"
    );
    assert!(
        r4_to_markdown(r4(), COMMAND) == recorded("r4-trained.md"),
        "results/r4-trained.md is stale"
    );
}

#[test]
fn copy_prompts_share_group_prefixes() {
    let (prompts, symbols) = copy_workload_prompts();
    assert_eq!(prompts.len(), COPY_WORKLOAD.count());
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(p.len(), COPY_WORKLOAD.shared + COPY_WORKLOAD.own);
        assert_eq!(&p[1..25], &symbols[i][..]);
        let g = i % COPY_WORKLOAD.groups;
        assert_eq!(p[..COPY_WORKLOAD.shared], prompts[g][..COPY_WORKLOAD.shared]);
    }
    assert_ne!(prompts[0][..8], prompts[1][..8]);
}

/// The batch-invariance prediction holds for both models, and the facts the
/// R3 report states (flips imply being within reach) hold by construction.
#[test]
fn r3_rerun_properties() {
    let r = r3();
    for d in [&r.random, &r.trained] {
        assert!(d.prediction_held());
        assert_eq!(invariance(d).predicted_rows_differing, 0);
        for &p in &d.config.policies {
            let x = pooled(d, p);
            assert!(x.flips <= x.within_reach);
            assert_eq!(x.flips, x.prompts_diverged);
        }
    }
}

#[test]
fn r4_rerun_properties() {
    let r = r4();
    for p in [&r.random, &r.trained] {
        let fp32 = p.runs.iter().find(|x| x.kind == Kind::Uniform(Level::Fp32)).unwrap();
        assert_eq!(fp32.drift.max_abs, 0.0);
        assert_eq!(fp32.diverged(), 0);
        for level in Level::EVERY {
            assert!(
                p.runs.iter().any(|x| x.kind == Kind::Uniform(level)),
                "{}",
                level.name()
            );
        }
    }
    assert_eq!(
        r.reference_copy_correct.1,
        r.trained.config.prompts * r.trained.config.continuation
    );
}

//! R3 planted bugs (compiled only with `--features planted-bugs`). Each must
//! be caught by the check named in its test.
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::collectives::WirePolicy;
use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::decoder::decode::prompts;
use fp_play_runtime::decoder::forward::{Model, RowReduce};
use fp_play_runtime::decoder::parallel::{Schedule, TensorParallel, Unsharded};
use fp_play_runtime::decoder::serve::ModelExecutor;
use fp_play_runtime::decoder::variants::{run_variant, Variant, Workload, TIMER};
use fp_play_runtime::decoder::DecoderMutation;
use fp_play_runtime::format::Format;
use fp_play_runtime::serving::engine::{run_model, EngineConfig, EngineError, FaultConfig};
use fp_play_runtime::serving::kv::KvConfig;
use fp_play_runtime::serving::scheduler::{PreemptionMode, SchedulerConfig};
use fp_play_runtime::serving::workload::Request;

const BF16: WirePolicy = WirePolicy::uniform(Format::Bf16);
const SMALL: Workload = Workload {
    seed: 0x5ea1,
    groups: 2,
    per_group: 4,
    shared: 8,
    own: 8,
    output_len: 12,
    block_size: 4,
};

/// Prompts whose logits differ from the `Alone` run, per variant.
fn differing(m: &mut Model, make: &mut dyn FnMut() -> Box<dyn RowReduce>, variants: &[Variant]) -> Vec<usize> {
    let ps = SMALL.prompts(m.config.vocab);
    let alone = run_variant(m, make, &SMALL, &ps, Variant::Alone).unwrap();
    variants
        .iter()
        .map(|&v| {
            let r = run_variant(m, make, &SMALL, &ps, v).unwrap();
            r.generations
                .iter()
                .zip(&alone.generations)
                .filter(|(a, b)| a.logits_hash != b.logits_hash)
                .count()
        })
        .collect()
}

/// Caught by the tensor invariant (`TensorStore::verify`, after every step):
/// the copied block lacks the K/V of the shared positions.
#[test]
fn skipped_kv_tensor_copy_on_write_is_caught_by_the_tensor_invariant() {
    let mut m = Model::new(DecoderConfig::TINY);
    let ps = prompts(0xc0, 6, 7, m.config.vocab);
    let requests: Vec<Request> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| Request {
            id: i as u32,
            arrival_ns: 0,
            priority: 0,
            prefix_group: None,
            prefix_len: 0,
            prompt_len: p.len() as u32,
            output_len: 10,
            samples: 2,
        })
        .collect();
    let config = EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: 64,
            max_seqs: 16,
            watermark_blocks: 0,
            preemption: PreemptionMode::Recompute,
        },
        kv: KvConfig {
            block_size: 4,
            device_blocks: 200,
            swap_blocks: 0,
            prefix_caching: false,
        },
        faults: FaultConfig::none(),
        kv_check_interval: 1,
    };
    let mut reduce = Unsharded { policy: BF16 };
    let mut executor = ModelExecutor::new(
        &mut m,
        &mut reduce,
        &config.kv,
        ps.iter().map(Vec::len).collect(),
        false,
    );
    executor.store.planted = Some(DecoderMutation::SkipKvTensorCow);
    match run_model(&requests, &ps, &config, &TIMER, &mut executor) {
        Err(EngineError::Invariant { what, step }) => {
            eprintln!("SkipKvTensorCow: caught at step {step}: {what}");
            assert!(what.starts_with("model: tensor:"), "{what}");
        }
        other => panic!("SkipKvTensorCow was not caught: {:?}", other.map(|r| r.steps.len())),
    }
}

/// Caught by the single-device control: at P = 1 every variant must be
/// bit-identical to the prompt served alone, and the recompute variants
/// are not.
#[test]
fn rope_off_by_one_on_recompute_is_caught_by_the_p1_control() {
    let mut m = Model::new(DecoderConfig::TINY);
    m.planted = Some(DecoderMutation::RopeOffByOneOnRecompute);
    let mut make = || Box::new(Unsharded { policy: BF16 }) as Box<dyn RowReduce>;
    let d = differing(
        &mut m,
        &mut make,
        &[Variant::Together, Variant::Pressure, Variant::FailureRetry],
    );
    eprintln!("RopeOffByOneOnRecompute: prompts differing (together, pressure, failure) = {d:?}");
    assert_eq!(d[0], 0, "no recompute in the together variant");
    assert!(d[1] > 0 && d[2] > 0, "not caught: {d:?}");
}

/// Caught by the batch-invariance check for the shard tree (a schedule the
/// prediction says is batch-invariant).
#[test]
fn batch_dependent_shard_order_is_caught_by_the_invariance_check() {
    let mut m = Model::new(DecoderConfig::TINY);
    let mut make = || {
        let mut tp = TensorParallel::new(4, Schedule::ShardTree { shards: 64 }, BF16).unwrap();
        tp.planted = Some(DecoderMutation::BatchDependentShardOrder);
        Box::new(tp) as Box<dyn RowReduce>
    };
    let d = differing(&mut m, &mut make, &[Variant::Together, Variant::Chunk8]);
    eprintln!("BatchDependentShardOrder: prompts differing (together, chunk 8) = {d:?}");
    assert!(d.iter().any(|&n| n > 0), "not caught");
    // Without the bug the same check passes.
    let mut clean =
        || Box::new(TensorParallel::new(4, Schedule::ShardTree { shards: 64 }, BF16).unwrap()) as Box<dyn RowReduce>;
    assert_eq!(
        differing(&mut m, &mut clean, &[Variant::Together, Variant::Chunk8]),
        vec![0, 0]
    );
}

//! R3b: tensor parallelism, the engine integration, the K/V tensor store and
//! the batch-invariance prediction.
//!
//! The prediction, derived from the schedules before measuring: a value's
//! bits can depend on batch composition only through a reduction whose
//! expression for an element (up to the commutativity of each single
//! addition) depends on the element's index in the step buffer. Local math
//! is per token, so it is batch-invariant by construction. Sequential and
//! binomial-tree reduce the whole buffer the same way; exact-gather sums
//! every element exactly; the shard tree's tree depends only on S; and
//! halving-doubling at a power-of-two P combines (x0 + x2) + (x1 + x3) (up to
//! commutativity) for every element. Only the ring folds each chunk starting
//! from a different rank, so only the ring can change with batch
//! composition. At P = 2 sequential, ring, binomial tree and
//! halving-doubling all reduce an element with one (commutative)
//! accumulator addition, so they agree. Exact-gather rounds the exact sum
//! once to the output format: under the BF16 policy its inputs are BF16 and
//! that equals one BF16 addition, but under fp32-rs/bf16-ag the others round
//! the binary32 partials' sum to binary32 first, and that double rounding
//! can differ.

use fp_play_runtime::collectives::{Algorithm, WirePolicy};
use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::decoder::decode::{generate_static, prompts, Generation};
use fp_play_runtime::decoder::forward::{Model, RowReduce};
use fp_play_runtime::decoder::parallel::{Schedule, TensorParallel, Unsharded};
use fp_play_runtime::decoder::serve::{serve, ModelExecutor};
use fp_play_runtime::decoder::variants::{run_variant, specs, Variant, VariantRun, Workload, TIMER};
use fp_play_runtime::decoder::weights::Weights;
use fp_play_runtime::format::Format;
use fp_play_runtime::serving::engine::{run_model, EngineConfig, FaultConfig};
use fp_play_runtime::serving::kv::KvConfig;
use fp_play_runtime::serving::scheduler::{PreemptionMode, SchedulerConfig};
use fp_play_runtime::serving::workload::Request;

const BF16: WirePolicy = WirePolicy::uniform(Format::Bf16);
const POLICIES: [WirePolicy; 2] = [BF16, WirePolicy::FP32_RS_BF16_AG];

/// A small workload with the R3 structure: 8 prompts, 12 greedy tokens.
const SMALL: Workload = Workload {
    seed: 0x5ea1,
    groups: 2,
    per_group: 4,
    shared: 8,
    own: 8,
    output_len: 12,
    block_size: 4,
};

fn model() -> Model {
    Model::new(DecoderConfig::TINY)
}

fn tp(world: usize, schedule: Schedule, policy: WirePolicy) -> TensorParallel {
    TensorParallel::new(world, schedule, policy).unwrap()
}

#[test]
fn shard_tree_logits_do_not_depend_on_p() {
    let mut m = model();
    let ps = prompts(0x7ee, 6, 10, m.config.vocab);
    for policy in POLICIES {
        let run = |m: &mut Model, world| {
            generate_static(
                m,
                &mut tp(world, Schedule::ShardTree { shards: 64 }, policy),
                &ps,
                10,
                false,
            )
        };
        let one = run(&mut m, 1);
        for world in [2, 4, 8] {
            assert_eq!(run(&mut m, world), one, "{} P = {world}", policy.name());
        }
    }
}

#[test]
fn collectives_agree_at_p2_and_differ_from_p1_somewhere() {
    let mut m = model();
    let ps = prompts(0x2a, 6, 10, m.config.vocab);
    for policy in POLICIES {
        let runs: Vec<Vec<Generation>> = Algorithm::ALL
            .into_iter()
            .filter(|&a| a != Algorithm::ExactGather || policy == BF16)
            .map(|a| generate_static(&mut m, &mut tp(2, Schedule::Collective(a), policy), &ps, 10, false))
            .collect();
        assert!(runs.windows(2).all(|w| w[0] == w[1]), "{}", policy.name());
        // P = 1 through the tensor-parallel path equals the unsharded path.
        let p1 = generate_static(
            &mut m,
            &mut tp(1, Schedule::Collective(Algorithm::Ring), policy),
            &ps,
            10,
            false,
        );
        assert_eq!(p1, generate_static(&mut m, &mut Unsharded { policy }, &ps, 10, false));
        if policy == BF16 {
            assert_ne!(runs[0], p1, "P = 2 should change some logit under a BF16 wire");
        }
    }
}

fn matrix_row(m: &mut Model, world: usize, schedule: Schedule, policy: WirePolicy) -> Vec<VariantRun> {
    let ps = SMALL.prompts(m.config.vocab);
    Variant::ALL
        .iter()
        .map(|&v| {
            let mut make = || Box::new(tp(world, schedule, policy)) as Box<dyn RowReduce>;
            run_variant(m, &mut make, &SMALL, &ps, v).unwrap_or_else(|e| panic!("{}: {e}", v.name()))
        })
        .collect()
}

/// Prompts whose logits (at any step) differ from the `Alone` reference.
fn differing(runs: &[VariantRun]) -> Vec<(Variant, usize)> {
    runs[1..]
        .iter()
        .map(|r| {
            let n = r
                .generations
                .iter()
                .zip(&runs[0].generations)
                .filter(|(a, b)| a.logits_hash != b.logits_hash)
                .count();
            (r.variant, n)
        })
        .collect()
}

#[test]
fn batch_invariance_prediction_holds_at_p4() {
    let mut m = model();
    let mut schedules: Vec<Schedule> = Algorithm::ALL.into_iter().map(Schedule::Collective).collect();
    schedules.push(Schedule::ShardTree { shards: 64 });
    let mut ring_varies = false;
    for policy in POLICIES {
        for &schedule in &schedules {
            let runs = matrix_row(&mut m, 4, schedule, policy);
            let diff = differing(&runs);
            if schedule == Schedule::Collective(Algorithm::Ring) {
                ring_varies |= diff.iter().any(|&(_, n)| n > 0);
            } else {
                assert!(
                    diff.iter().all(|&(_, n)| n == 0),
                    "{} {}: {diff:?}",
                    schedule.name(),
                    policy.name()
                );
            }
        }
    }
    assert!(ring_varies, "the ring never changed with batch composition");
}

#[test]
fn single_device_control_is_bit_identical_in_every_variant() {
    let mut m = model();
    let runs = matrix_row(&mut m, 1, Schedule::Collective(Algorithm::Ring), BF16);
    assert!(differing(&runs).iter().all(|&(_, n)| n == 0), "{:?}", differing(&runs));
    // Each variant exercised its mechanism.
    let by = |v: Variant| runs.iter().find(|r| r.variant == v).unwrap();
    assert!(by(Variant::PrefixCache).prefix_hit_blocks > 0);
    assert!(by(Variant::Pressure).preemptions > 0 && by(Variant::Pressure).exec.recomputed_tokens > 0);
    assert_eq!(by(Variant::FailureRetry).failed_steps, 1);
    assert!(by(Variant::FailureRetry).exec.recomputed_tokens > 0);
    assert!(by(Variant::Chunk8).engine_steps > by(Variant::Together).engine_steps);
}

/// Greedy decoding with fixed lengths makes the scheduling independent of
/// the token values, so every numerical configuration sees the same steps.
#[test]
fn scheduling_does_not_depend_on_the_numerics() {
    let mut m = model();
    let ps = SMALL.prompts(m.config.vocab);
    for v in [Variant::PrefixCache, Variant::Pressure, Variant::FailureRetry] {
        let spec = &specs(&SMALL, &ps, v, m.config.layers)[0];
        let a = serve(&mut m, &mut Unsharded { policy: BF16 }, spec, &TIMER, false).unwrap();
        let mut ring = tp(8, Schedule::Collective(Algorithm::Ring), BF16);
        let b = serve(&mut m, &mut ring, spec, &TIMER, false).unwrap();
        assert_eq!(a.run.steps, b.run.steps, "{}", v.name());
        assert_ne!(
            a.generations.iter().map(|g| g.logits_hash).collect::<Vec<_>>(),
            b.generations.iter().map(|g| g.logits_hash).collect::<Vec<_>>()
        );
    }
}

/// Forked samples share prompt blocks, so appending copies on write; swap
/// preemption copies blocks to the host and back. The tensor invariant is
/// checked after every step, and every sample must decode exactly as the
/// prompt does alone (greedy siblings are identical, P = 1).
#[test]
fn tensor_store_follows_copy_on_write_and_swap() {
    let mut m = model();
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
            preemption: PreemptionMode::Swap,
        },
        kv: KvConfig {
            block_size: 4,
            device_blocks: 30,
            swap_blocks: 200,
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
    let run = run_model(&requests, &ps, &config, &TIMER, &mut executor).unwrap();
    assert!(run.kv_stats.cow_copies > 0, "no copy-on-write happened");
    assert!(run.kv_stats.swapped_out_blocks > 0, "no swap happened");
    assert!(executor.stats.block_copies > 0);
    let generations = executor.generations.clone();
    let alone = generate_static(&mut m, &mut Unsharded { policy: BF16 }, &ps, 10, false);
    for (seq, g) in generations {
        assert_eq!(g, alone[seq.request as usize], "{seq}");
    }
}

#[test]
fn the_vector_workload_is_what_the_docs_say() {
    let w = Workload::R3;
    let ps = w.prompts(DecoderConfig::TINY.vocab);
    assert_eq!(ps.len(), 32);
    assert!(ps.iter().all(|p| p.len() == 16));
    assert_eq!(ps[0][..8], ps[4][..8]);
    assert_ne!(ps[0][..8], ps[1][..8]);
    assert_eq!(w.blocks_needed(), 512);
    assert_eq!(w.pressure_blocks(), 320);
    assert_eq!(w.failure_step(), 12);
    // Weights are shared, so tensor-parallel runs reuse one generation.
    assert_eq!(
        Weights::generate(&DecoderConfig::TINY).checksum(),
        model().weights.checksum()
    );
}

#[test]
fn bad_parallel_configs_are_rejected() {
    assert!(TensorParallel::new(3, Schedule::Collective(Algorithm::HalvingDoubling), BF16).is_err());
    assert!(TensorParallel::new(65, Schedule::ShardTree { shards: 64 }, BF16).is_err());
    assert!(TensorParallel::new(2, Schedule::ShardTree { shards: 64 }, WirePolicy::BF16_WIRE_FP32_ADD).is_err());
    assert!(TensorParallel::new(0, Schedule::Collective(Algorithm::Ring), BF16).is_err());
}

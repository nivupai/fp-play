//! Dry-run schedules equal the rounds of executed collectives.
//!
//! The serving simulator costs very large AllReduces from their schedule
//! alone ([`schedule_rounds`]), so the dry run must record exactly the rounds
//! the real simulator records: same phases, kinds, sources, destinations,
//! element counts and wire bytes. Checked for every algorithm, every uniform
//! format and both mixed wire policies, P = 1…16 and several lengths, and for
//! the shard tree over every P ≤ S, including the rejected lossy policy.

use fp_play_runtime::collectives::{all_reduce_with, schedule_rounds, Algorithm, WirePolicy};
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::from_f64;
use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::shard_tree;

/// Lengths: empty, tiny, uneven chunks and the R1/R1.1 buffer sizes
/// (sweep: 4 tokens × 256 outputs; R1: 8 × 512).
const LENS: [usize; 7] = [0, 1, 5, 64, 257, 1024, 4096];

fn policies() -> Vec<WirePolicy> {
    Format::ALL
        .into_iter()
        .map(WirePolicy::uniform)
        .chain([WirePolicy::FP32_RS_BF16_AG, WirePolicy::BF16_WIRE_FP32_ADD])
        .collect()
}

fn inputs(format: Format, world: usize, len: usize, rng: &mut SplitMix64) -> Vec<Vec<u32>> {
    (0..world)
        .map(|_| (0..len).map(|_| from_f64(format, rng.next_normalish())).collect())
        .collect()
}

#[test]
fn collective_dry_run_equals_executed_rounds() {
    let mut rng = SplitMix64::new(0xd27);
    let mut compared = 0;
    for algorithm in Algorithm::ALL {
        for policy in policies() {
            for world in 1..=16 {
                for len in LENS {
                    if !algorithm.supports(world) {
                        assert!(schedule_rounds(algorithm, policy, world, len).is_err());
                        continue;
                    }
                    // Only the largest length at a few world sizes, to keep the test quick.
                    if len == 4096 && ![2, 3, 8, 16].contains(&world) {
                        continue;
                    }
                    let data = inputs(policy.accumulator, world, len, &mut rng);
                    let executed = all_reduce_with(algorithm, policy, &data).unwrap();
                    let dry = schedule_rounds(algorithm, policy, world, len).unwrap();
                    assert_eq!(
                        dry,
                        executed.rounds,
                        "{algorithm} {} P={world} len={len}",
                        policy.name()
                    );
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 2000, "{compared}");
    assert!(schedule_rounds(Algorithm::Ring, WirePolicy::uniform(Format::Bf16), 0, 8).is_err());
}

#[test]
fn shard_tree_dry_run_equals_executed_rounds() {
    let mut rng = SplitMix64::new(0x5d27);
    for shards in [shard_tree::DEFAULT_SHARDS, 16, 5, 1] {
        for policy in policies() {
            for world in 1..=shards {
                for len in [0, 1, 7, 96] {
                    let data = inputs(policy.accumulator, shards, len, &mut rng);
                    let executed = shard_tree::all_reduce(policy, world, &data);
                    let dry = shard_tree::schedule_rounds(policy, world, shards, len);
                    match (executed, dry) {
                        (Ok(e), Ok(d)) => {
                            let label = format!("{} S={shards} P={world} len={len}", policy.name());
                            assert_eq!(d.rounds, e.outcome.rounds, "{label}");
                            assert_eq!(d.levels, e.levels, "{label}");
                            assert_eq!(d.levels_with_messages, e.levels_with_messages, "{label}");
                            assert_eq!(d.reduce_rounds, e.reduce_rounds, "{label}");
                            assert_eq!(d.broadcast_rounds, e.broadcast_rounds, "{label}");
                        }
                        // The lossy bf16-wire/fp32-add policy (and FP16 → BF16
                        // style mismatches) are rejected the same way by both.
                        (Err(e), Err(d)) => assert_eq!(e, d),
                        (e, d) => panic!("{} S={shards} P={world}: {e:?} vs {d:?}", policy.name()),
                    }
                }
            }
        }
    }
    assert!(shard_tree::schedule_rounds(WirePolicy::BF16_WIRE_FP32_ADD, 4, 64, 8).is_err());
    assert!(shard_tree::schedule_rounds(WirePolicy::FP32_RS_BF16_AG, 65, 64, 8).is_err());
    assert!(shard_tree::schedule_rounds(WirePolicy::FP32_RS_BF16_AG, 8, 64, 8).is_ok());
}

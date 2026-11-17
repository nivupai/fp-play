//! Reduce-scatter and all-gather as separate collectives.

use fp_play_runtime::collectives::{
    all_gather, all_reduce_with, owned_ranges, reduce_scatter, Algorithm, CollectiveError, TransferKind, WirePolicy,
};
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::from_f64;
use fp_play_runtime::rng::SplitMix64;

const SPLIT: [Algorithm; 2] = [Algorithm::Ring, Algorithm::HalvingDoubling];
const LENGTHS: [usize; 7] = [0, 1, 2, 5, 9, 16, 37];

fn policies() -> Vec<WirePolicy> {
    Format::ALL
        .into_iter()
        .map(WirePolicy::uniform)
        .chain([WirePolicy::FP32_RS_BF16_AG, WirePolicy::BF16_WIRE_FP32_ADD])
        .collect()
}

/// Random inputs in the policy's accumulator format: raw patterns over the
/// whole encoding space for uniform policies, approximate normals for mixed.
fn inputs(policy: WirePolicy, world: usize, len: usize, rng: &mut SplitMix64) -> Vec<Vec<u32>> {
    let format = policy.accumulator;
    (0..world)
        .map(|_| {
            (0..len)
                .map(|_| {
                    if policy.is_uniform() {
                        (rng.next_u64() as u32) & format.max_raw()
                    } else {
                        from_f64(format, rng.next_normalish())
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn all_gather_after_reduce_scatter_is_the_all_reduce() {
    let mut rng = SplitMix64::new(42);
    for policy in policies() {
        for world in 1..=16 {
            for len in LENGTHS {
                let data = inputs(policy, world, len, &mut rng);
                for algorithm in SPLIT.into_iter().filter(|a| a.supports(world)) {
                    let what = format!("{} {algorithm} P={world} len={len}", policy.name());
                    let all = all_reduce_with(algorithm, policy, &data).unwrap();
                    let rs = reduce_scatter(algorithm, policy, &data).unwrap();
                    let ag = all_gather(algorithm, policy.output, len, &rs.shards).unwrap();
                    assert_eq!(ag.results, all.results, "{what}: bits");
                    // The two phases are exactly the all-reduce's rounds, in order.
                    let phases: Vec<_> = rs.rounds.iter().chain(&ag.rounds).collect();
                    assert_eq!(phases, all.rounds.iter().collect::<Vec<_>>(), "{what}: rounds");
                    assert_eq!(rs.additions, all.additions, "{what}");
                    assert_eq!(ag.additions, 0, "{what}");
                    // Each owned value already carries every rounding the all-reduce output has.
                    for (r, range) in rs.owned.iter().enumerate() {
                        assert_eq!(rs.roundings[r], all.roundings[r][range.clone()], "{what}: roundings");
                        assert_eq!(rs.shards[r], all.results[r][range.clone()], "{what}: shard bits");
                    }
                    assert_eq!(rs.assemble(len), all.results[0], "{what}");
                }
            }
        }
    }
}

#[test]
fn owned_ranges_partition_the_buffer() {
    for world in 1..=16 {
        for len in 0..40 {
            for algorithm in SPLIT.into_iter().filter(|a| a.supports(world)) {
                let mut owned = owned_ranges(algorithm, len, world).unwrap();
                assert_eq!(owned.len(), world);
                owned.sort_by_key(|r| r.start);
                let mut next = 0;
                for range in owned {
                    assert_eq!(range.start, next, "{algorithm} P={world} len={len}");
                    next = range.end;
                }
                assert_eq!(next, len, "{algorithm} P={world} len={len}");
            }
        }
    }
}

#[test]
fn each_rank_sends_everything_except_what_it_owns() {
    // In both reduce-scatters a rank gives away every element it does not
    // end up owning, exactly once; and both phases obey the single-port model.
    let policy = WirePolicy::FP32_RS_BF16_AG;
    let mut rng = SplitMix64::new(9);
    for world in 2..=16 {
        let len = 1000 + world;
        let data = inputs(policy, world, len, &mut rng);
        for algorithm in SPLIT.into_iter().filter(|a| a.supports(world)) {
            let rs = reduce_scatter(algorithm, policy, &data).unwrap();
            let mut sent = vec![0usize; world];
            for round in &rs.rounds {
                assert_eq!(round.kind, TransferKind::Reduce);
                assert!(round.respects_single_port(world), "{algorithm} P={world}");
                for t in &round.transfers {
                    assert_eq!(t.bytes_per_element, 4);
                    sent[t.src] += t.elements;
                }
            }
            for (r, (sent, owned)) in sent.iter().zip(&rs.owned).enumerate() {
                assert_eq!(*sent, len - owned.len(), "{algorithm} P={world} rank {r}");
            }
            let ag = all_gather(algorithm, Format::Bf16, len, &rs.shards).unwrap();
            for round in &ag.rounds {
                assert_eq!(round.kind, TransferKind::Gather);
                assert!(round.respects_single_port(world), "{algorithm} P={world}");
                assert!(round.transfers.iter().all(|t| t.bytes_per_element == 2));
            }
        }
    }
}

#[test]
fn all_gather_copies_raw_bits_unchanged() {
    // Shards with arbitrary patterns, NaN payloads included, arrive bit-exact everywhere.
    let mut rng = SplitMix64::new(5);
    for world in 1..=16 {
        for algorithm in SPLIT.into_iter().filter(|a| a.supports(world)) {
            let owned = owned_ranges(algorithm, 37, world).unwrap();
            let shards: Vec<Vec<u32>> = owned
                .iter()
                .map(|r| (0..r.len()).map(|_| (rng.next_u64() as u32) & 0xffff).collect())
                .collect();
            let out = all_gather(algorithm, Format::Bf16, 37, &shards).unwrap();
            for result in &out.results {
                for (range, shard) in owned.iter().zip(&shards) {
                    assert_eq!(&result[range.clone()], shard.as_slice(), "{algorithm} P={world}");
                }
            }
            assert_eq!(out.roundings_range().map(|(_, max)| max), Some(0));
        }
    }
}

#[test]
fn invalid_split_requests_are_rejected() {
    let data = vec![vec![0u32; 4]; 3];
    for algorithm in [Algorithm::Sequential, Algorithm::BinomialTree, Algorithm::ExactGather] {
        assert_eq!(
            reduce_scatter(algorithm, WirePolicy::uniform(Format::Bf16), &data).unwrap_err(),
            CollectiveError::NoSeparatePhases { algorithm }
        );
        assert!(owned_ranges(algorithm, 4, 3).is_err());
    }
    assert_eq!(
        reduce_scatter(Algorithm::HalvingDoubling, WirePolicy::uniform(Format::Bf16), &data).unwrap_err(),
        CollectiveError::UnsupportedWorld {
            algorithm: Algorithm::HalvingDoubling,
            world: 3
        }
    );
    // Ring over 3 ranks and 4 elements: rank 0 owns chunk 1 (one element), not two.
    assert_eq!(
        all_gather(Algorithm::Ring, Format::Bf16, 4, &[vec![0, 0], vec![0], vec![0]]).unwrap_err(),
        CollectiveError::ShardLength {
            rank: 0,
            expected: 1,
            got: 2
        }
    );
    assert_eq!(
        all_gather(Algorithm::Ring, Format::Bf16, 3, &[vec![0x1_0000], vec![0], vec![0]]).unwrap_err(),
        CollectiveError::InvalidRaw {
            rank: 0,
            index: 1,
            raw: 0x1_0000
        }
    );
}

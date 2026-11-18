//! The fixed shard tree: structure, P-invariance, single-port rounds, and the
//! independent Python tree-sum vectors.

use fp_play_runtime::collectives::{TransferKind, WirePolicy};
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::from_f64;
use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::shard_tree::{
    all_reduce, owned_shards, serialize, shard_owner, tree, ShardTreeError, DEFAULT_SHARDS,
};

/// Shard counts tested: the default, a power of two below it, and non-powers of two.
const SHARD_COUNTS: [usize; 5] = [DEFAULT_SHARDS, 16, 48, 5, 1];

fn policies() -> Vec<WirePolicy> {
    Format::ALL
        .into_iter()
        .map(WirePolicy::uniform)
        .chain([WirePolicy::FP32_RS_BF16_AG])
        .collect()
}

fn shard_inputs(policy: WirePolicy, shards: usize, len: usize, rng: &mut SplitMix64) -> Vec<Vec<u32>> {
    let format = policy.accumulator;
    (0..shards)
        .map(|_| {
            (0..len)
                .map(|_| {
                    if policy.is_uniform() {
                        // Raw patterns over the whole space: NaN payloads, infinities, -0.
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
fn the_tree_depends_only_on_s_and_absorbs_every_node_once() {
    for s in 1..=130usize {
        let combines = tree(s);
        assert_eq!(combines.len(), s - 1, "S={s}");
        let levels = combines.last().map_or(0, |c| c.level + 1);
        assert_eq!(levels, s.next_power_of_two().trailing_zeros(), "S={s}");
        let mut absorbed_at = vec![None; s];
        for c in &combines {
            assert!(c.left < c.right && c.right < s);
            assert_eq!(c.right - c.left, 1 << c.level);
            assert_eq!(c.left % (2 << c.level), 0);
            // A node is absorbed once, and is never used after that.
            assert_eq!(absorbed_at[c.right], None, "S={s}");
            assert_eq!(absorbed_at[c.left], None, "S={s}");
            absorbed_at[c.right] = Some(c.level);
        }
        assert_eq!(absorbed_at[0], None);
        assert!(absorbed_at[1..].iter().all(Option::is_some), "S={s}");
    }
}

#[test]
fn ownership_is_contiguous_and_covers_every_shard() {
    for s in 1..=70 {
        for p in 1..=s {
            let mut next = 0;
            for r in 0..p {
                let owned = owned_shards(s, p, r);
                assert_eq!(owned.start, next);
                assert!(!owned.is_empty(), "S={s} P={p}: rank {r} owns nothing");
                for shard in owned.clone() {
                    assert_eq!(shard_owner(s, p, shard), r);
                }
                next = owned.end;
            }
            assert_eq!(next, s);
        }
    }
}

#[test]
fn every_p_from_1_to_s_gives_the_same_bits() {
    // The invariant: a node is computed the same way whether its children are on
    // one rank or on different ranks. P = 1 computes every node locally, and
    // P = S sends a message for every node, so equal bits for every P in
    // between are a direct check of the invariant.
    let mut rng = SplitMix64::new(0x5eed);
    for s in SHARD_COUNTS {
        for policy in policies() {
            for len in [1, 37] {
                let inputs = shard_inputs(policy, s, len, &mut rng);
                let local = all_reduce(policy, 1, &inputs).unwrap();
                assert!(local.outcome.rounds.is_empty(), "P = 1 needs no messages");
                for p in 1..=s {
                    let what = format!("{} S={s} P={p} len={len}", policy.name());
                    let got = all_reduce(policy, p, &inputs).unwrap();
                    assert!(got.outcome.consistent(), "{what}");
                    assert_eq!(got.outcome.results[0], local.outcome.results[0], "{what}");
                    assert_eq!(
                        got.outcome.roundings,
                        vec![local.outcome.roundings[0].clone(); p],
                        "{what}"
                    );
                    assert_eq!(got.outcome.additions, (s - 1) * len, "{what}");
                    // One message for each node whose children are on different ranks.
                    let crossing = tree(s)
                        .iter()
                        .filter(|c| shard_owner(s, p, c.left) != shard_owner(s, p, c.right))
                        .count();
                    let sent: usize = got
                        .outcome
                        .rounds
                        .iter()
                        .filter(|r| r.kind == TransferKind::Reduce)
                        .map(|r| r.transfers.len())
                        .sum();
                    assert_eq!(sent, crossing, "{what}");
                    if p == s {
                        assert_eq!(crossing, s - 1, "{what}: every node crosses ranks");
                    }
                }
            }
        }
    }
}

#[test]
fn rounds_are_single_port_and_counted_as_scheduled() {
    let policy = WirePolicy::FP32_RS_BF16_AG;
    let mut rng = SplitMix64::new(3);
    for s in SHARD_COUNTS.into_iter().chain([7, 31, 33, 63, 100]) {
        let inputs = shard_inputs(policy, s, 4, &mut rng);
        for p in 1..=s {
            let got = all_reduce(policy, p, &inputs).unwrap();
            let rounds = &got.outcome.rounds;
            assert!(rounds.iter().all(|r| r.respects_single_port(p)), "S={s} P={p}");
            assert_eq!(got.reduce_rounds + got.broadcast_rounds, rounds.len());
            assert_eq!(got.broadcast_rounds, p.next_power_of_two().trailing_zeros() as usize);
            assert!(got.levels_with_messages <= got.levels);
            // Measured: with contiguous ownership no level needs a second round.
            assert_eq!(got.serialization_extra_rounds(), 0, "S={s} P={p}");
            for (i, r) in rounds.iter().enumerate() {
                let reduce = i < got.reduce_rounds;
                assert_eq!(r.kind == TransferKind::Reduce, reduce);
                let bytes = if reduce { 4 } else { 2 };
                assert!(r
                    .transfers
                    .iter()
                    .all(|t| t.bytes_per_element == bytes && t.elements == 4));
            }
        }
    }
}

#[test]
fn the_serializer_splits_conflicting_messages_into_extra_rounds() {
    // Rank 0 sends twice and rank 3 receives twice: at least two rounds.
    let messages = [(0, 1), (0, 2), (2, 3), (1, 3), (4, 5)];
    let rounds = serialize(&messages);
    assert_eq!(rounds, vec![vec![0, 2, 4], vec![1, 3]]);
    // Random message sets: every message appears once and every round is single-port.
    let mut rng = SplitMix64::new(11);
    for _ in 0..500 {
        let world = 2 + (rng.next_u64() % 9) as usize;
        let count = (rng.next_u64() % 20) as usize;
        let messages: Vec<(usize, usize)> = (0..count)
            .map(|_| {
                let src = (rng.next_u64() as usize) % world;
                let dst = (src + 1 + (rng.next_u64() as usize) % (world - 1)) % world;
                (src, dst)
            })
            .collect();
        let rounds = serialize(&messages);
        let mut seen: Vec<usize> = rounds.concat();
        seen.sort_unstable();
        assert_eq!(seen, (0..count).collect::<Vec<_>>());
        let degree = (0..world)
            .map(|r| {
                let out = messages.iter().filter(|m| m.0 == r).count();
                let inn = messages.iter().filter(|m| m.1 == r).count();
                out.max(inn)
            })
            .max()
            .unwrap_or(0);
        assert!(rounds.len() >= degree);
        for round in &rounds {
            for (i, &a) in round.iter().enumerate() {
                for &b in &round[i + 1..] {
                    assert_ne!(messages[a].0, messages[b].0);
                    assert_ne!(messages[a].1, messages[b].1);
                }
            }
        }
    }
}

#[test]
fn integer_inputs_sum_exactly() {
    for s in SHARD_COUNTS {
        for format in Format::ALL {
            let inputs: Vec<Vec<u32>> = (0..s)
                .map(|i| vec![from_f64(format, i as f64 - 2.0), from_f64(format, 1.0)])
                .collect();
            let expected = [
                from_f64(format, (0..s).map(|i| i as f64 - 2.0).sum()),
                from_f64(format, s as f64),
            ];
            for p in 1..=s {
                let got = all_reduce(WirePolicy::uniform(format), p, &inputs).unwrap();
                assert_eq!(got.outcome.results[0], expected, "{} S={s} P={p}", format.name());
            }
        }
    }
}

#[test]
fn roundings_per_element_are_s_minus_1_plus_the_output_conversion() {
    let mut rng = SplitMix64::new(2);
    for s in SHARD_COUNTS {
        for policy in policies() {
            let inputs = shard_inputs(policy, s, 3, &mut rng);
            let got = all_reduce(policy, s.min(3), &inputs).unwrap();
            let conversion = u32::from(!policy.output.contains(policy.accumulator));
            let expected = (s - 1) as u32 + conversion;
            assert_eq!(
                got.outcome.roundings_range(),
                Some((expected, expected)),
                "{}",
                policy.name()
            );
        }
    }
}

#[test]
fn lossy_wires_and_invalid_requests_are_rejected() {
    let inputs = vec![vec![0u32; 2]; 4];
    assert_eq!(
        all_reduce(WirePolicy::BF16_WIRE_FP32_ADD, 2, &inputs).unwrap_err(),
        ShardTreeError::LossyReduceWire {
            accumulator: Format::Fp32,
            reduce_wire: Format::Bf16
        }
    );
    let narrow_gather = WirePolicy {
        accumulator: Format::Fp32,
        reduce_wire: Format::Fp32,
        gather_wire: Format::Bf16,
        output: Format::Fp32,
    };
    assert_eq!(
        all_reduce(narrow_gather, 2, &inputs).unwrap_err(),
        ShardTreeError::LossyGatherWire {
            output: Format::Fp32,
            gather_wire: Format::Bf16
        }
    );
    let fp32 = WirePolicy::uniform(Format::Fp32);
    assert_eq!(all_reduce(fp32, 0, &inputs).unwrap_err(), ShardTreeError::EmptyWorld);
    assert_eq!(
        all_reduce(fp32, 5, &inputs).unwrap_err(),
        ShardTreeError::TooManyRanks { world: 5, shards: 4 }
    );
    assert_eq!(all_reduce(fp32, 1, &[]).unwrap_err(), ShardTreeError::NoShards);
    assert_eq!(
        all_reduce(fp32, 1, &[vec![0], vec![0, 0]]).unwrap_err(),
        ShardTreeError::RaggedInputs
    );
    let bf16 = WirePolicy::uniform(Format::Bf16);
    assert_eq!(
        all_reduce(bf16, 1, &[vec![0], vec![0x1_0000]]).unwrap_err(),
        ShardTreeError::InvalidRaw {
            shard: 1,
            index: 0,
            raw: 0x1_0000
        }
    );
}

#[test]
fn independent_python_tree_vectors() {
    // tests/vectors/tree-sums.txt comes from scripts/gen_tree_vectors.py, which
    // evaluates the same tree with Fraction arithmetic and shares no code.
    let text = include_str!("vectors/tree-sums.txt");
    let mut cases = 0;
    for line in text.lines() {
        let mut fields = line.split(' ');
        let (formats, leaves, expected) = (fields.next().unwrap(), fields.next().unwrap(), fields.next().unwrap());
        let (acc, out) = formats.split_once('>').unwrap();
        let (acc, out) = (Format::parse(acc).unwrap(), Format::parse(out).unwrap());
        let policy = WirePolicy {
            accumulator: acc,
            reduce_wire: acc,
            gather_wire: out,
            output: out,
        };
        let shards: Vec<Vec<u32>> = leaves
            .split(',')
            .map(|h| vec![u32::from_str_radix(h, 16).unwrap()])
            .collect();
        let expected = u32::from_str_radix(expected, 16).unwrap();
        let s = shards.len();
        let mut worlds = vec![1, 2, 3, s];
        worlds.retain(|&p| p <= s);
        worlds.dedup();
        for p in worlds {
            let got = all_reduce(policy, p, &shards).unwrap();
            for result in &got.outcome.results {
                assert_eq!(result[0], expected, "{line} (P={p})");
            }
        }
        cases += 1;
    }
    assert_eq!(cases, 6000);
}

#[test]
fn with_one_shard_per_rank_the_tree_is_the_binomial_tree() {
    // At P = S each rank owns one shard, and the shard tree pairs i with i + 2^k
    // exactly as binomial-tree reduce does, so the two independent
    // implementations must agree bit for bit.
    use fp_play_runtime::collectives::{all_reduce_with, Algorithm};
    let mut rng = SplitMix64::new(77);
    for s in [2, 3, 5, 8, 16, 48, 64] {
        for policy in policies() {
            let inputs = shard_inputs(policy, s, 23, &mut rng);
            let tree = all_reduce(policy, s, &inputs).unwrap();
            let binomial = all_reduce_with(Algorithm::BinomialTree, policy, &inputs).unwrap();
            assert_eq!(tree.outcome.results, binomial.results, "{} S=P={s}", policy.name());
            assert_eq!(tree.outcome.roundings, binomial.roundings, "{} S=P={s}", policy.name());
        }
    }
}

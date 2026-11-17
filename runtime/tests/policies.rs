//! Wire policies: hand-derived bits, identities against direct operations,
//! consistency across ranks, and per-phase byte accounting.

use fp_play_runtime::collectives::{all_reduce, all_reduce_with, Algorithm, WirePolicy};
use fp_play_runtime::cost::{estimate_recorded, LinkModel};
use fp_play_runtime::exact::ExactSum;
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::{add, convert, from_f64};
use fp_play_runtime::rng::SplitMix64;

const MIXED: [WirePolicy; 2] = [WirePolicy::FP32_RS_BF16_AG, WirePolicy::BF16_WIRE_FP32_ADD];

fn fp32_normal_inputs(world: usize, len: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = SplitMix64::new(seed);
    (0..world)
        .map(|_| (0..len).map(|_| from_f64(Format::Fp32, rng.next_normalish())).collect())
        .collect()
}

#[test]
fn three_policies_give_three_hand_derived_answers() {
    // FP32 inputs 256, 1 and 1 + 2^-8 on three ranks, one element per ring chunk.
    // BF16 has 8 significant bits: 257 is a tie between 256 and 258 (rounds to
    // even, 256), and 1 + 2^-8 is a tie between 1 and 1 + 2^-7 (rounds to 1).
    // Ring folds chunk c starting at rank c and ends at rank c - 1 (mod 3).
    let (x0, x1, x2) = (0x4380_0000, 0x3f80_0000, 0x3f80_8000);
    let fp32_inputs = vec![vec![x0; 3], vec![x1; 3], vec![x2; 3]];
    let (b256, b258) = (0x4380, 0x4381);

    // R1's BF16 wire: the partials are rounded to BF16 first (256, 1, 1).
    // chunk 0: (256 + 1) + 1 = 256; chunk 1: (1 + 1) + 256 = 258; chunk 2: (1 + 256) + 1 = 256.
    let bf16_inputs: Vec<Vec<u32>> = fp32_inputs
        .iter()
        .map(|b| b.iter().map(|&v| convert(Format::Fp32, Format::Bf16, v)).collect())
        .collect();
    let r1 = all_reduce(Algorithm::Ring, Format::Bf16, &bf16_inputs).unwrap();
    assert_eq!(r1.results[0], vec![b256, b258, b256]);

    // FP32 reduce-scatter: every order sums exactly to 258.00390625, one rounding to BF16 gives 258.
    let fp32_rs = all_reduce_with(Algorithm::Ring, WirePolicy::FP32_RS_BF16_AG, &fp32_inputs).unwrap();
    assert_eq!(fp32_rs.results[0], vec![b258, b258, b258]);

    // BF16 wire, FP32 add at each hop; a rank's own FP32 partial is never quantized:
    // chunk 0: bf16(256) + 1 = 257 -> bf16 256, + (1 + 2^-8) = 257.0039 -> 258
    // chunk 1: bf16(1) + (1 + 2^-8) = 2.0039 -> bf16 2, + 256 = 258 -> 258
    // chunk 2: bf16(1 + 2^-8) = 1, + 256 = 257 -> bf16 256, + 1 = 257 -> tie -> 256
    let per_hop = all_reduce_with(Algorithm::Ring, WirePolicy::BF16_WIRE_FP32_ADD, &fp32_inputs).unwrap();
    assert_eq!(per_hop.results[0], vec![b258, b258, b256]);

    for out in [&r1, &fp32_rs, &per_hop] {
        assert!(out.consistent());
    }
}

#[test]
fn fp32_reduce_phases_equal_the_fp32_collective_rounded_once_to_bf16() {
    // For every schedule that reduces through partial sums, FP32 reduce phases
    // perform exactly the FP32 collective's additions; the only other rounding
    // is the final conversion to BF16.
    for world in 1..=16 {
        let inputs = fp32_normal_inputs(world, 23, world as u64);
        for algorithm in Algorithm::ALL
            .into_iter()
            .filter(|a| a.supports(world) && *a != Algorithm::ExactGather)
        {
            let fp32 = all_reduce(algorithm, Format::Fp32, &inputs).unwrap();
            let mixed = all_reduce_with(algorithm, WirePolicy::FP32_RS_BF16_AG, &inputs).unwrap();
            let expected: Vec<u32> = fp32.results[0]
                .iter()
                .map(|&v| convert(Format::Fp32, Format::Bf16, v))
                .collect();
            for rank in &mixed.results {
                assert_eq!(rank, &expected, "{algorithm} P={world}");
            }
        }
    }
}

#[test]
fn exact_gather_under_fp32_reduce_phases_is_the_exact_sum_rounded_once() {
    for world in 1..=9 {
        let inputs = fp32_normal_inputs(world, 31, 100 + world as u64);
        let out = all_reduce_with(Algorithm::ExactGather, WirePolicy::FP32_RS_BF16_AG, &inputs).unwrap();
        for e in 0..31 {
            let mut sum = ExactSum::new();
            for input in &inputs {
                sum.add_raw(Format::Fp32, input[e]);
            }
            assert_eq!(out.results[0][e], sum.round(Format::Bf16), "P={world} e={e}");
        }
        assert!(out.consistent());
    }
}

#[test]
fn bf16_wire_with_fp32_add_on_two_ranks_matches_direct_operations() {
    // Ring with two ranks: chunk c is sent by rank c (rounded to BF16) and
    // added in FP32 by rank 1 - c, which rounds the FP32 sum to BF16 once.
    let inputs = fp32_normal_inputs(2, 64, 7);
    let out = all_reduce_with(Algorithm::Ring, WirePolicy::BF16_WIRE_FP32_ADD, &inputs).unwrap();
    for e in 0..64 {
        let chunk = usize::from(e >= 32);
        let (sender, owner) = (chunk, 1 - chunk);
        let sent = convert(
            Format::Bf16,
            Format::Fp32,
            convert(Format::Fp32, Format::Bf16, inputs[sender][e]),
        );
        let sum = add(Format::Fp32, inputs[owner][e], sent);
        assert_eq!(out.results[0][e], convert(Format::Fp32, Format::Bf16, sum), "e={e}");
    }
    assert!(out.consistent());
}

#[test]
fn mixed_policies_keep_ranks_consistent_and_sum_exact_data_exactly() {
    for policy in MIXED {
        for world in 1..=16 {
            // Small integers: every partial sum is exact in FP32 and BF16 (|sum| <= 80).
            let ints: Vec<Vec<u32>> = (0..world)
                .map(|r| {
                    (0..37)
                        .map(|e| from_f64(Format::Fp32, ((r * 7 + e * 3) % 11) as f64 - 5.0))
                        .collect()
                })
                .collect();
            let random = fp32_normal_inputs(world, 37, 1000 + world as u64);
            for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                let exact = all_reduce_with(algorithm, policy, &ints).unwrap();
                for e in 0..37 {
                    let want: f64 = (0..world).map(|r| ((r * 7 + e * 3) % 11) as f64 - 5.0).sum();
                    let got = fp_play_runtime::format::to_f64(Format::Bf16, exact.results[0][e]);
                    assert_eq!(got, want, "{} {algorithm} P={world} e={e}", policy.name());
                }
                let out = all_reduce_with(algorithm, policy, &random).unwrap();
                assert!(out.consistent(), "{} {algorithm} P={world}", policy.name());
                assert!(out.results.iter().flatten().all(|&v| v <= Format::Bf16.max_raw()));
            }
        }
    }
}

#[test]
fn reduce_phases_carry_fp32_bytes_only_under_fp32_rs() {
    let len = 4096;
    let link = LinkModel::default();
    for world in [2usize, 4, 8, 16] {
        let inputs = fp32_normal_inputs(world, len, 3);
        let bf16_inputs: Vec<Vec<u32>> = inputs
            .iter()
            .map(|b| b.iter().map(|&v| convert(Format::Fp32, Format::Bf16, v)).collect())
            .collect();
        // Ring and halving-doubling each send (P - 1) / P of the buffer per rank per phase.
        let share = (world - 1) * len / world;
        for algorithm in [Algorithm::Ring, Algorithm::HalvingDoubling] {
            let cases = [
                (all_reduce(algorithm, Format::Bf16, &bf16_inputs).unwrap(), 2, 2),
                (
                    all_reduce_with(algorithm, WirePolicy::FP32_RS_BF16_AG, &inputs).unwrap(),
                    4,
                    2,
                ),
                (
                    all_reduce_with(algorithm, WirePolicy::BF16_WIRE_FP32_ADD, &inputs).unwrap(),
                    2,
                    2,
                ),
            ];
            for (out, reduce_bytes, gather_bytes) in cases {
                let cost = estimate_recorded(&out.rounds, world, link);
                assert_eq!(
                    cost.max_reduce_bytes_per_rank,
                    share * reduce_bytes,
                    "{algorithm} P={world}"
                );
                assert_eq!(
                    cost.max_gather_bytes_per_rank,
                    share * gather_bytes,
                    "{algorithm} P={world}"
                );
            }
        }
    }
}

#[test]
fn policy_names_round_trip() {
    for policy in MIXED
        .into_iter()
        .chain(Format::ALL.into_iter().map(WirePolicy::uniform))
    {
        assert_eq!(WirePolicy::parse(&policy.name()), Some(policy));
    }
    assert_eq!(WirePolicy::parse("nope"), None);
}

/// Roundings per element derived by hand from each reduction tree: P - 1
/// additions; under bf16-wire/fp32-add also one BF16 rounding per send of a
/// partial (P - 1 of them); one rounding to BF16 for the mixed policies.
/// Exact-gather: one rounding of the exact sum, plus one per input converted
/// to a narrower wire. With one rank only the output conversion remains.
fn expected_roundings(policy: WirePolicy, algorithm: Algorithm, world: u32) -> u32 {
    let mixed = !policy.is_uniform();
    if world == 1 {
        return u32::from(mixed);
    }
    match (policy, algorithm) {
        (p, Algorithm::ExactGather) if p == WirePolicy::BF16_WIRE_FP32_ADD => world + 1,
        (_, Algorithm::ExactGather) => 1,
        (p, _) if p == WirePolicy::BF16_WIRE_FP32_ADD => 2 * world - 1,
        (p, _) if p == WirePolicy::FP32_RS_BF16_AG => world,
        _ => world - 1,
    }
}

#[test]
fn measured_roundings_match_the_hand_derived_counts() {
    for policy in MIXED
        .into_iter()
        .chain([WirePolicy::uniform(Format::Bf16), WirePolicy::uniform(Format::Fp32)])
    {
        for world in 1..=16usize {
            let inputs = if policy.is_uniform() {
                (0..world).map(|_| vec![0; 37]).collect()
            } else {
                fp32_normal_inputs(world, 37, world as u64)
            };
            for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                let out = all_reduce_with(algorithm, policy, &inputs).unwrap();
                let want = expected_roundings(policy, algorithm, world as u32);
                assert_eq!(
                    out.roundings_range(),
                    Some((want, want)),
                    "{} {algorithm} P={world}",
                    policy.name()
                );
                if policy.is_uniform() {
                    assert_eq!(want as usize, algorithm.roundings_per_element(world), "R1 formula");
                }
            }
        }
    }
}

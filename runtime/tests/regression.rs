//! The generalized simulator must reproduce the R1 simulator exactly under a
//! uniform wire policy. `legacy` is a frozen copy of the R1 code.

mod legacy;

use fp_play_runtime::collectives::{all_reduce, all_reduce_with, Algorithm, TransferKind, WirePolicy};
use fp_play_runtime::cost::{estimate, estimate_recorded, LinkModel};
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::from_f64;
use fp_play_runtime::rng::SplitMix64;

const LENGTHS: [usize; 7] = [0, 1, 2, 5, 9, 16, 37];

fn integer_inputs(format: Format, world: usize, len: usize) -> Vec<Vec<u32>> {
    (0..world)
        .map(|r| {
            (0..len)
                .map(|e| from_f64(format, ((r * 7 + e * 3) % 11) as f64 - 5.0))
                .collect()
        })
        .collect()
}

/// Uniformly random raw patterns over the whole encoding space: NaNs (with
/// payloads), infinities, subnormals and both zeros all occur.
fn raw_inputs(format: Format, world: usize, len: usize, rng: &mut SplitMix64) -> Vec<Vec<u32>> {
    (0..world)
        .map(|_| (0..len).map(|_| (rng.next_u64() as u32) & format.max_raw()).collect())
        .collect()
}

#[test]
fn uniform_policy_reproduces_the_r1_simulator_exactly() {
    let mut rng = SplitMix64::new(0x5eed_0001);
    let mut cases = 0;
    for format in Format::ALL {
        for world in 1..=16 {
            for len in LENGTHS {
                for inputs in [
                    integer_inputs(format, world, len),
                    raw_inputs(format, world, len, &mut rng),
                ] {
                    for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                        let new = all_reduce(algorithm, format, &inputs).unwrap();
                        let old = legacy::all_reduce(algorithm, format, &inputs);
                        let what = format!("{algorithm} {} P={world} len={len}", format.name());
                        assert_eq!(new.results, old.results, "{what}: bits");
                        assert_eq!(new.additions, old.additions, "{what}: additions");
                        assert_eq!(new.rounds.len(), old.rounds.len(), "{what}: round count");
                        for (n, o) in new.rounds.iter().zip(&old.rounds) {
                            assert_eq!(n.phase, o.phase, "{what}");
                            let transfers: Vec<_> = n.transfers.iter().map(|t| (t.src, t.dst, t.elements)).collect();
                            assert_eq!(transfers, o.transfers, "{what}: transfers in {}", n.phase);
                            assert!(n.transfers.iter().all(|t| t.bytes_per_element == format.bytes()));
                        }
                        // The policy entry point with a uniform policy is the same call.
                        let via_policy = all_reduce_with(algorithm, WirePolicy::uniform(format), &inputs).unwrap();
                        assert_eq!(via_policy.results, new.results, "{what}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 2_500, "only {cases} cases ran");
}

#[test]
fn recorded_bytes_match_the_r1_cost_interface_under_a_uniform_policy() {
    let link = LinkModel::default();
    for format in Format::ALL {
        for world in [2usize, 3, 4, 8] {
            let inputs = integer_inputs(format, world, 37);
            for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                let out = all_reduce(algorithm, format, &inputs).unwrap();
                let r1 = estimate(&out.rounds, world, format.bytes(), link);
                let recorded = estimate_recorded(&out.rounds, world, link);
                assert_eq!(r1, recorded, "{algorithm} P={world}");
                // Every byte is attributed to exactly one phase kind.
                let reduce_rounds = out.rounds.iter().filter(|r| r.kind == TransferKind::Reduce).count();
                let gather_rounds = out.rounds.len() - reduce_rounds;
                assert_eq!(
                    reduce_rounds > 0,
                    recorded.max_reduce_bytes_per_rank > 0,
                    "{algorithm} P={world}"
                );
                assert_eq!(
                    gather_rounds > 0,
                    recorded.max_gather_bytes_per_rank > 0,
                    "{algorithm} P={world}"
                );
            }
        }
    }
}

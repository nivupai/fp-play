use fp_play_runtime::collectives::{all_reduce, chunk_range, Algorithm, CollectiveError};
use fp_play_runtime::cost::{estimate, LinkModel};
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::from_f64;

fn ceil_log2(n: usize) -> usize {
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

fn integer_inputs(format: Format, world: usize, len: usize) -> Vec<Vec<u32>> {
    // Small integers: every partial and total is exact in all three formats.
    (0..world)
        .map(|r| {
            (0..len)
                .map(|e| from_f64(format, ((r * 7 + e * 3) % 11) as f64 - 5.0))
                .collect()
        })
        .collect()
}

#[test]
fn exact_data_gives_the_exact_sum_for_every_schedule() {
    for format in Format::ALL {
        for world in 1..=9 {
            for len in [0, 1, 2, 5, 9, 16, 37] {
                let inputs = integer_inputs(format, world, len);
                let expected: Vec<u32> = (0..len)
                    .map(|e| from_f64(format, (0..world).map(|r| ((r * 7 + e * 3) % 11) as f64 - 5.0).sum()))
                    .collect();
                for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
                    let out = all_reduce(algorithm, format, &inputs).unwrap();
                    assert!(out.consistent(), "{algorithm} P={world} len={len}");
                    // compare numerically: +0 and -0 may differ in sign for exact zeros
                    for (got, want) in out.results[0].iter().zip(&expected) {
                        assert_eq!(
                            fp_play_runtime::format::to_f64(format, *got),
                            fp_play_runtime::format::to_f64(format, *want),
                            "{algorithm} {} P={world} len={len}",
                            format.name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn schedules_obey_the_single_port_model_and_expected_round_counts() {
    for world in 2..=16 {
        let inputs = integer_inputs(Format::Bf16, world, 23);
        for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
            let out = all_reduce(algorithm, Format::Bf16, &inputs).unwrap();
            for round in &out.rounds {
                assert!(round.respects_single_port(world), "{algorithm} P={world} {round:?}");
            }
            let expected_rounds = match algorithm {
                Algorithm::Sequential => (world - 1) + ceil_log2(world),
                Algorithm::Ring => 2 * (world - 1),
                Algorithm::BinomialTree => 2 * ceil_log2(world),
                Algorithm::HalvingDoubling => 2 * world.trailing_zeros() as usize,
                Algorithm::ExactGather => world - 1,
            };
            assert_eq!(out.rounds.len(), expected_rounds, "{algorithm} P={world}");
            if algorithm != Algorithm::ExactGather {
                assert_eq!(out.additions, (world - 1) * 23, "{algorithm} P={world}");
            }
        }
    }
}

#[test]
fn bandwidth_optimal_schedules_send_under_twice_the_buffer() {
    let len = 1 << 12;
    for world in [2usize, 4, 8, 16] {
        let inputs = integer_inputs(Format::Fp32, world, len);
        for algorithm in [Algorithm::Ring, Algorithm::HalvingDoubling] {
            let out = all_reduce(algorithm, Format::Fp32, &inputs).unwrap();
            let cost = estimate(&out.rounds, world, 4, LinkModel::default());
            let optimal = 2 * (world - 1) * len * 4 / world;
            assert_eq!(cost.max_bytes_sent_per_rank, optimal, "{algorithm} P={world}");
        }
    }
}

#[test]
fn ring_chunks_fold_from_different_ranks() {
    // Hand-derived: FP32 values 2^24 (rank 0), 1 (rank 1), 1 (rank 2), one element per chunk.
    // (2^24 + 1) + 1 = 2^24 because each +1 is a tie that rounds to even, while
    // (1 + 1) + 2^24 = 2^24 + 2 exactly.
    let big = from_f64(Format::Fp32, 16_777_216.0);
    let one = from_f64(Format::Fp32, 1.0);
    let inputs = vec![vec![big; 3], vec![one; 3], vec![one; 3]];
    let lo = big;
    let hi = from_f64(Format::Fp32, 16_777_218.0);

    // chunk 0 folds x0,x1,x2; chunk 1 folds x1,x2,x0; chunk 2 folds x2,x0,x1.
    let ring = all_reduce(Algorithm::Ring, Format::Fp32, &inputs).unwrap();
    assert_eq!(ring.results[0], vec![lo, hi, lo]);
    let sequential = all_reduce(Algorithm::Sequential, Format::Fp32, &inputs).unwrap();
    assert_eq!(sequential.results[0], vec![lo, lo, lo]);
    let exact = all_reduce(Algorithm::ExactGather, Format::Fp32, &inputs).unwrap();
    assert_eq!(exact.results[0], vec![hi, hi, hi]);
    for out in [ring, sequential, exact] {
        assert!(out.consistent());
    }
}

#[test]
fn exact_gather_ignores_rank_order_but_ring_does_not() {
    let big = from_f64(Format::Fp32, 16_777_216.0);
    let one = from_f64(Format::Fp32, 1.0);
    let a = vec![vec![big; 3], vec![one; 3], vec![one; 3]];
    let b = vec![vec![one; 3], vec![big; 3], vec![one; 3]];
    let exact = |inputs| {
        all_reduce(Algorithm::ExactGather, Format::Fp32, inputs)
            .unwrap()
            .results[0]
            .clone()
    };
    let ring = |inputs| all_reduce(Algorithm::Ring, Format::Fp32, inputs).unwrap().results[0].clone();
    assert_eq!(exact(&a), exact(&b));
    assert_ne!(ring(&a), ring(&b));
}

#[test]
fn invalid_requests_are_rejected() {
    let f = Format::Fp16;
    assert_eq!(
        all_reduce(Algorithm::Ring, f, &[]).unwrap_err(),
        CollectiveError::EmptyWorld
    );
    assert_eq!(
        all_reduce(Algorithm::Ring, f, &[vec![0, 0], vec![0]]).unwrap_err(),
        CollectiveError::RaggedInputs
    );
    assert_eq!(
        all_reduce(Algorithm::Ring, f, &[vec![0], vec![0x1_0000]]).unwrap_err(),
        CollectiveError::InvalidRaw {
            rank: 1,
            index: 0,
            raw: 0x1_0000
        }
    );
    assert_eq!(
        all_reduce(Algorithm::HalvingDoubling, f, &[vec![0], vec![0], vec![0]]).unwrap_err(),
        CollectiveError::UnsupportedWorld {
            algorithm: Algorithm::HalvingDoubling,
            world: 3
        }
    );
}

#[test]
fn chunk_ranges_partition_the_buffer() {
    for len in 0..40 {
        for parts in 1..10 {
            let mut next = 0;
            for i in 0..parts {
                let r = chunk_range(len, parts, i);
                assert_eq!(r.start, next);
                assert!(r.len() == len / parts || r.len() == len / parts + 1);
                next = r.end;
            }
            assert_eq!(next, len);
        }
    }
}

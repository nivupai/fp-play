//! R6a: association traces from the executed collectives.

use fp_play_runtime::collectives::{schedule_rounds, trace_association, Algorithm, Assoc, WirePolicy};
use fp_play_runtime::format::Format;

fn leaf(r: usize) -> Assoc {
    Assoc::Leaf(r)
}

fn add(a: Assoc, b: Assoc) -> Assoc {
    Assoc::Add(Box::new(a), Box::new(b))
}

/// The same tree with every addition's operands put in a canonical order
/// (one addition is commutative).
fn canonical(a: &Assoc) -> String {
    match a {
        Assoc::Leaf(r) => format!("x{r}"),
        Assoc::Exact(v) => format!("exact{v:?}"),
        Assoc::Add(l, r) => {
            let (l, r) = (canonical(l), canonical(r));
            if l <= r {
                format!("({l} + {r})")
            } else {
                format!("({r} + {l})")
            }
        }
    }
}

#[test]
fn hand_derived_associations() {
    // Ring, 3 ranks, 3 elements: chunk c's fold starts at rank c.
    let t = trace_association(Algorithm::Ring, 3, 3).unwrap();
    assert_eq!(t.per_rank[0][0], add(leaf(2), add(leaf(1), leaf(0))));
    assert_eq!(t.per_rank[0][1], add(leaf(0), add(leaf(2), leaf(1))));
    assert_eq!(t.per_rank[0][2], add(leaf(1), add(leaf(0), leaf(2))));
    // Sequential: a left fold in rank order, then a broadcast.
    let t = trace_association(Algorithm::Sequential, 3, 2).unwrap();
    assert_eq!(t.per_rank[1][0], add(leaf(2), add(leaf(1), leaf(0))));
    // Binomial tree, 4 ranks.
    let t = trace_association(Algorithm::BinomialTree, 4, 1).unwrap();
    assert_eq!(t.per_rank[3][0], add(add(leaf(0), leaf(1)), add(leaf(2), leaf(3))));
    // Exact-gather.
    let t = trace_association(Algorithm::ExactGather, 4, 2).unwrap();
    assert_eq!(t.per_rank[2][1], Assoc::Exact(vec![0, 1, 2, 3]));
    // One rank: nothing is combined.
    assert_eq!(
        trace_association(Algorithm::Ring, 1, 2).unwrap().per_rank[0][1],
        leaf(0)
    );
}

/// The claim R3 relied on: only the ring's association depends on the
/// element's index (up to commutativity).
#[test]
fn only_the_ring_varies_by_element() {
    for world in [2usize, 4, 8] {
        for algorithm in Algorithm::ALL {
            let t = trace_association(algorithm, world, 16).unwrap();
            let forms: Vec<String> = t.per_rank[0].iter().map(canonical).collect();
            let uniform = forms.windows(2).all(|w| w[0] == w[1]);
            let expect_uniform = !(algorithm == Algorithm::Ring && world > 2);
            assert_eq!(uniform, expect_uniform, "{algorithm} P = {world}: {forms:?}");
        }
    }
}

#[test]
fn traces_follow_the_executed_schedule() {
    for world in 1..=8usize {
        for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
            let t = trace_association(algorithm, world, 11).unwrap();
            assert_eq!(
                t.rounds,
                schedule_rounds(algorithm, WirePolicy::uniform(Format::Fp32), world, 11).unwrap()
            );
            assert_eq!(t.ranges.len(), t.rounds.len());
            for (round, ranges) in t.rounds.iter().zip(&t.ranges) {
                assert_eq!(round.transfers.len(), ranges.len());
                for (tr, r) in round.transfers.iter().zip(ranges) {
                    assert_eq!(tr.elements, r.len());
                }
            }
            // Every rank ends with the same association, using every rank once.
            assert!(t.per_rank.windows(2).all(|w| w[0] == w[1]), "{algorithm} {world}");
            for a in &t.per_rank[0] {
                let mut leaves = Vec::new();
                fn walk(a: &Assoc, out: &mut Vec<usize>) {
                    match a {
                        Assoc::Leaf(r) => out.push(*r),
                        Assoc::Exact(v) => out.extend(v),
                        Assoc::Add(l, r) => {
                            walk(l, out);
                            walk(r, out);
                        }
                    }
                }
                walk(a, &mut leaves);
                leaves.sort_unstable();
                assert_eq!(leaves, (0..world).collect::<Vec<_>>());
            }
        }
    }
}

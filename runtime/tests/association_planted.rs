//! R6 planted bug (compiled only with `--features planted-bugs`): caught by
//! the hand-derived association check (the same expectation as
//! `tests/association.rs`).
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::collectives::{trace_association_planted, Algorithm, Assoc};

#[test]
fn swapped_operands_are_caught_by_the_hand_derived_check() {
    let leaf = Assoc::Leaf;
    let add = |a, b| Assoc::Add(Box::new(a), Box::new(b));
    let t = trace_association_planted(Algorithm::Ring, 3, 3).unwrap();
    let expected = add(leaf(2), add(leaf(1), leaf(0)));
    eprintln!("AssocSwapsOperands: element 0 traced as {:?}", t.per_rank[0][0]);
    assert_ne!(t.per_rank[0][0], expected, "not caught");
}

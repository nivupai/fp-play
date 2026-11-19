//! Planted training bugs, each caught by a named check (`planted-bugs` builds only).
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::train::checks::{finite_difference_check_planted, inference_forward_check_planted};
use fp_play_runtime::train::net::TrainMutation;

/// Caught by the finite-difference check: the forward pass is right, the
/// backward pass rotates RoPE gradients the wrong way.
#[test]
fn rope_backward_bug_is_caught_by_finite_differences() {
    let m = TrainMutation::RopeBackwardNotTransposed;
    assert!(finite_difference_check_planted(m).is_err());
    assert!(inference_forward_check_planted(m).is_ok());
}

/// Caught by the inference-forward check: forward and backward agree with
/// each other (finite differences pass), but the model is not the one the
/// inference code runs.
#[test]
fn rope_pairing_bug_is_caught_by_the_inference_forward_check() {
    let m = TrainMutation::RopeInterleavedPairs;
    assert!(finite_difference_check_planted(m).is_ok());
    assert!(inference_forward_check_planted(m).is_err());
}

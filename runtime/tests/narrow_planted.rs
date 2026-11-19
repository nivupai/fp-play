//! Planted FP8/INT8 quantizer bugs, each caught by a named check (`planted-bugs` builds only).
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::precision::narrow::{midpoint_check, quantize_planted, vector_check, Narrow, NarrowMutation};

const VECTORS: &str = include_str!("vectors/narrow.txt");

/// Ties away from zero: caught by the exhaustive midpoint check.
#[test]
fn ties_away_from_zero_is_caught_by_the_midpoint_check() {
    let q = |f: Narrow, w: f32, s: f32| quantize_planted(f, w, s, NarrowMutation::TiesAwayFromZero);
    assert!(midpoint_check(q).is_err());
}

/// An asymmetric INT8 clamp: caught by the Python vectors (quotients far
/// below −127) and by the midpoint check's saturation cases.
#[test]
fn int8_clamp_to_minus_128_is_caught_by_the_python_vectors() {
    let q = |f: Narrow, w: f32, s: f32| quantize_planted(f, w, s, NarrowMutation::Int8ClampsToMinus128);
    assert!(vector_check(VECTORS, q).is_err());
    assert!(midpoint_check(q).is_err());
}

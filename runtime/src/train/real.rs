//! The arithmetic the trainer is generic over: binary32 for training (the
//! recorded run) and binary64 for the finite-difference gradient check.
//!
//! Only IEEE 754 basic operations (`+ − × ÷`, `sqrt`) and the crate's
//! deterministic `exp` ([`crate::decoder::mathf`]) are used, so a binary32
//! training run is bit-identical on every platform: Rust never contracts
//! `a * b + c` into a fused multiply–add and never reassociates.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::decoder::mathf;

pub trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    /// Round a binary64 value to this format (round to nearest even).
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    /// e^x, deterministic on every platform (0 far below the range, ∞ far above).
    fn exp(self) -> Self;
    fn max(self, other: Self) -> Self {
        if other > self {
            other
        } else {
            self
        }
    }
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn exp(self) -> f32 {
        mathf::exp_f32(self)
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn exp(self) -> f64 {
        if self < -700.0 {
            0.0
        } else if self > 700.0 {
            f64::INFINITY
        } else {
            mathf::exp(self)
        }
    }
}

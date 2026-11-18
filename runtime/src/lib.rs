//! FP Play runtime: exact simulation of multi-device inference collectives.
//!
//! Milestone R1 models how AllReduce schedules change the bits of a
//! tensor-parallel layer. Modules:
//!
//! * [`format`] — FP16 / BF16 / FP32 encodings and exact decoding
//! * [`exact`] — exact superaccumulator with one nearest-even rounding
//! * [`ops`] — correctly rounded add, convert, fused-chain and exact dot products
//! * [`collectives`] — message-level AllReduce schedules over virtual ranks
//! * [`cost`] — α–β latency/bandwidth estimate of a schedule
//! * [`tensor_parallel`] — the row-parallel linear-layer experiment
//! * [`report`] / [`json`] — deterministic artifacts
//! * [`serving`] — R2: the deterministic serving simulator

/// Whether this build compiles in the deliberate bugs used by the
/// planted-bug tests (the `planted-bugs` Cargo feature). The `experiment`
/// and `traces` commands refuse to run when it is true.
pub const PLANTED_BUGS: bool = cfg!(feature = "planted-bugs");

pub mod collectives;
pub mod cost;
pub mod exact;
pub mod format;
pub mod json;
pub mod ops;
pub mod policies;
pub mod report;
pub mod rng;
pub mod serving;
pub mod shard_experiment;
pub mod shard_tree;
pub mod sweep;
pub mod tensor_parallel;

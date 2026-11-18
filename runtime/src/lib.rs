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

pub mod collectives;
pub mod cost;
pub mod exact;
pub mod format;
pub mod json;
pub mod ops;
pub mod policies;
pub mod report;
pub mod rng;
pub mod shard_tree;
pub mod tensor_parallel;

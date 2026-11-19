//! Milestone R4: per-layer precision policies for the R3 decoder.
//!
//! * [`mac`] — a fast fused multiply–accumulate in FP32, BF16 or FP16,
//!   bit-identical to the exact fused chain
//! * [`narrow`] — FP8 (E4M3, E5M2) and INT8 with explicit scales
//! * [`policy`] — precision levels (storage and accumulator) per site
//! * [`model`] — FP32 master weights and the decoder under a policy
//! * [`metrics`] — logit drift, KL divergence, top-1 agreement, perplexity
//! * [`experiment`] / [`report`] — the recorded experiment (results/r4-precision.*)

pub mod experiment;
pub mod mac;
pub mod metrics;
pub mod model;
pub mod narrow;
pub mod policy;
pub mod report;

//! Milestone R2: a deterministic serving simulator.
//!
//! * [`model`] — model dimensions, device constants, KV capacity
//! * [`detmath`] — a platform-independent `ln` for Poisson arrivals
//! * [`workload`] — the deterministic request generator and synthetic tokens
//! * [`kv`] — the paged KV cache: block pool, block tables, prefix cache,
//!   copy-on-write and a host swap pool, with invariant checks
//! * [`step_time`] — roofline compute plus the real AllReduce schedules
//! * [`scheduler`] — continuous batching, chunked prefill, preemption
//! * [`engine`] — the discrete-event loop and per-step invariant checks
//! * [`metrics`] — TTFT, TPOT, throughput, percentiles, counters
//!
//! Time is integer nanoseconds. Every random draw comes from
//! [`crate::rng::SplitMix64`]; every floating-point operation is a basic
//! IEEE 754 operation, so the same configuration gives byte-identical
//! results on every platform.

pub mod detmath;
pub mod engine;
pub mod kv;
pub mod metrics;
pub mod model;
pub mod reference;
pub mod scheduler;
pub mod step_time;
pub mod workload;

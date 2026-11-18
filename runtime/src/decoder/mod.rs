//! Milestone R3: a tiny Llama-style decoder whose tokens come from real
//! arithmetic, run tensor-parallel through the R1 collectives.
//!
//! * [`config`] — dimensions, seeds and the logit scale
//! * [`mathf`] — `exp`, `sin` and `cos` from basic binary64 operations
//! * [`weights`] — deterministic random BF16 weights
//! * [`forward`] — one step's forward pass in fixed-order binary32
//! * [`parallel`] — row-parallel projections and their reductions
//! * [`decode`] — greedy decoding over a static batch
//! * [`serve`] — the decoder inside the R2 engine, with a paged K/V tensor store
//! * [`variants`] — the R3 workload and its batch-composition variants
//! * [`experiment`] / [`report`] — the recorded experiment (results/r3-decode.*)

pub mod config;
pub mod decode;
pub mod experiment;
pub mod forward;
pub mod mathf;
pub mod parallel;
pub mod report;
pub mod serve;
pub mod variants;
pub mod weights;

/// Deliberate bugs for the planted-bug tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderMutation {
    /// The tensor store ignores copy-on-write block copies.
    SkipKvTensorCow,
    /// Recomputed positions are rotated as if one position later.
    RopeOffByOneOnRecompute,
    /// Shard-tree partials of odd tokens in the step buffer are summed in
    /// reverse order (a batch-dependent order in a batch-invariant path).
    BatchDependentShardOrder,
}

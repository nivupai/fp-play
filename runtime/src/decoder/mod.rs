//! Milestone R3: a tiny Llama-style decoder whose tokens come from real
//! arithmetic, run tensor-parallel through the R1 collectives.
//!
//! * [`config`] — dimensions, seeds and the logit scale
//! * [`mathf`] — `exp`, `sin` and `cos` from basic binary64 operations
//! * [`weights`] — deterministic random BF16 weights
//! * [`forward`] — one step's forward pass in fixed-order binary32
//! * [`parallel`] — row-parallel projections and their reductions
//! * [`decode`] — greedy decoding over a static batch

pub mod config;
pub mod decode;
pub mod forward;
pub mod mathf;
pub mod parallel;
pub mod weights;

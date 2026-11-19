//! Milestone R5: mapping a transformer layer onto a toy fixed-function
//! accelerator.
//!
//! * [`graph`] — the static op graph of one layer, prefill or decode
//! * [`machine`] — the illustrative machine (MAC array, vector unit, SRAM, HBM)
//! * [`sim`] — tiling, chunking and the double-buffered timing recurrence
//! * [`reference`] — a cycle-by-cycle simulation that checks the recurrence
//! * [`experiment`] / [`report`] — the recorded experiment (results/r5-dataflow.*)

pub mod experiment;
pub mod graph;
pub mod machine;
pub mod reference;
pub mod report;
pub mod sim;

/// Deliberate bugs for the planted-bug tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataflowMutation {
    /// A buffer is treated as free one chunk early (as if triple-buffered).
    BufferFreedEarly,
    /// Decode attention reads no K/V bytes from HBM.
    DecodeSkipsKvReads,
}

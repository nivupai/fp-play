//! Row-parallel projections and their reductions.
//!
//! Partial sums are binary32 (the local accumulator). They enter the
//! reduction in the wire policy's accumulator format (one correctly rounded
//! conversion if that is narrower), and the result leaves in its output
//! format, widened exactly to binary32 for the residual add. All of this
//! runs through the R1 collective simulator ([`crate::collectives`]), so
//! every rounding comes from the executed schedule.

use super::forward::{matvec, RowReduce, Site};
use super::weights::Matrix;
use crate::collectives::{all_reduce_with, transcode, Algorithm, WirePolicy};
use crate::format::Format;

/// Binary32 partials to raw patterns in `to`.
fn encode(partials: &[f32], to: Format) -> Vec<u32> {
    partials
        .iter()
        .map(|v| transcode(Format::Fp32, to, v.to_bits()))
        .collect()
}

/// Raw patterns in `from` widened exactly to binary32.
fn decode(raw: &[u32], from: Format) -> Vec<f32> {
    raw.iter()
        .map(|&r| f32::from_bits(transcode(from, Format::Fp32, r)))
        .collect()
}

/// One device (P = 1): the full sequential sum per output, converted once
/// to the policy's output format (through its accumulator format). No
/// collective runs; the simulator with one rank only converts.
#[derive(Clone, Copy, Debug)]
pub struct Unsharded {
    pub policy: WirePolicy,
}

impl RowReduce for Unsharded {
    fn project(&mut self, _site: Site, m: &Matrix, x: &[f32], tokens: usize) -> Vec<f32> {
        let mut partial = vec![0.0f32; tokens * m.rows];
        for i in 0..tokens {
            matvec(
                m,
                &x[i * m.cols..(i + 1) * m.cols],
                0..m.cols,
                &mut partial[i * m.rows..(i + 1) * m.rows],
            );
        }
        let input = encode(&partial, self.policy.accumulator);
        let outcome = all_reduce_with(Algorithm::Sequential, self.policy, &[input]).expect("one-rank reduction");
        decode(&outcome.results[0], self.policy.output)
    }
}

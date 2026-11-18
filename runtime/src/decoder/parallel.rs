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
use crate::collectives::{all_reduce_with, chunk_range, transcode, Algorithm, WirePolicy};
use crate::format::Format;
use crate::shard_tree;

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

/// How a row-parallel projection's partial sums are reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// Each rank sums its contiguous slice of the input dimension
    /// sequentially; the P partials go through `algorithm`.
    Collective(Algorithm),
    /// The input dimension is split into `shards` contiguous shards, fixed
    /// independently of P; each shard's partial is a sequential sum over its
    /// slice in increasing index order; ranks own contiguous blocks of shards
    /// and the fixed shard tree combines them ([`crate::shard_tree`]).
    ShardTree { shards: usize },
}

impl Schedule {
    pub fn name(&self) -> String {
        match self {
            Schedule::Collective(a) => a.name().to_string(),
            Schedule::ShardTree { shards } => format!("shard-tree (S = {shards})"),
        }
    }
}

/// Counters of the reductions a [`TensorParallel`] ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReduceStats {
    pub collectives: u64,
    /// Elements per rank summed over every collective.
    pub elements: u64,
    /// Correctly rounded accumulator additions across all ranks.
    pub additions: u64,
}

/// Tensor parallelism over `world` ranks. Column-parallel work (Q/K/V,
/// attention by head, gate and up, the vocabulary-parallel head) is
/// computed once: each value is a whole dot product on whichever rank owns
/// it, so it does not depend on P. Only the row-parallel projections
/// depend on P, through their partial sums and the reduction.
#[derive(Clone, Debug)]
pub struct TensorParallel {
    pub world: usize,
    pub schedule: Schedule,
    pub policy: WirePolicy,
    pub stats: ReduceStats,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<super::DecoderMutation>,
}

impl TensorParallel {
    pub fn new(world: usize, schedule: Schedule, policy: WirePolicy) -> Result<TensorParallel, String> {
        match schedule {
            Schedule::Collective(a) if !a.supports(world) => {
                return Err(format!("{a} does not support {world} ranks"));
            }
            Schedule::ShardTree { shards }
                if shards == 0 || world > shards || !policy.reduce_wire.contains(policy.accumulator) =>
            {
                return Err(format!(
                    "shard tree with S = {shards} cannot run {world} ranks under {}",
                    policy.name()
                ));
            }
            _ => {}
        }
        if world == 0 {
            return Err("world must be at least 1".into());
        }
        Ok(TensorParallel {
            world,
            schedule,
            policy,
            stats: ReduceStats::default(),
            #[cfg(feature = "planted-bugs")]
            planted: None,
        })
    }

    /// Partial sums of `parts` contiguous slices of the input dimension,
    /// each `tokens × m.rows`, in the policy's accumulator format.
    fn partials(&self, m: &Matrix, x: &[f32], tokens: usize, parts: usize) -> Vec<Vec<u32>> {
        (0..parts)
            .map(|p| {
                let ks = chunk_range(m.cols, parts, p);
                let mut partial = vec![0.0f32; tokens * m.rows];
                for i in 0..tokens {
                    let (xi, out) = (
                        &x[i * m.cols..(i + 1) * m.cols],
                        &mut partial[i * m.rows..(i + 1) * m.rows],
                    );
                    #[cfg(feature = "planted-bugs")]
                    if i % 2 == 1
                        && matches!(self.schedule, Schedule::ShardTree { .. })
                        && self.planted == Some(super::DecoderMutation::BatchDependentShardOrder)
                    {
                        out.fill(0.0);
                        for k in ks.clone().rev() {
                            for (o, acc) in out.iter_mut().enumerate() {
                                *acc += m.t[k * m.rows + o] * xi[k];
                            }
                        }
                        continue;
                    }
                    matvec(m, xi, ks.clone(), out);
                }
                encode(&partial, self.policy.accumulator)
            })
            .collect()
    }
}

impl RowReduce for TensorParallel {
    fn project(&mut self, _site: Site, m: &Matrix, x: &[f32], tokens: usize) -> Vec<f32> {
        let outcome = match self.schedule {
            Schedule::Collective(algorithm) => {
                let inputs = self.partials(m, x, tokens, self.world);
                all_reduce_with(algorithm, self.policy, &inputs).expect("validated collective")
            }
            Schedule::ShardTree { shards } => {
                let inputs = self.partials(m, x, tokens, shards);
                shard_tree::all_reduce(self.policy, self.world, &inputs)
                    .expect("validated shard tree")
                    .outcome
            }
        };
        assert!(outcome.consistent(), "ranks disagree after the reduction");
        self.stats.collectives += 1;
        self.stats.elements += (tokens * m.rows) as u64;
        self.stats.additions += outcome.additions as u64;
        decode(&outcome.results[0], self.policy.output)
    }
}

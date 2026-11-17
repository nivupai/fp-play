//! An α–β (latency–bandwidth) estimate of a simulated schedule.
//!
//! Each round costs `latency + largest_message_bytes / bandwidth`. The model
//! assumes full-duplex point-to-point links with no contention, no compute
//! cost for additions, and no overlap between rounds. Defaults are
//! illustrative, not measurements of any particular interconnect or Etched
//! hardware.

use crate::collectives::{Round, Transfer, TransferKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkModel {
    /// Per-message latency (α) in microseconds.
    pub latency_us: f64,
    /// Per-link bandwidth (β⁻¹) in gigabytes per second (10^9 bytes/s).
    pub bandwidth_gbps: f64,
}

impl Default for LinkModel {
    fn default() -> Self {
        LinkModel {
            latency_us: 2.0,
            bandwidth_gbps: 100.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostEstimate {
    pub rounds: usize,
    pub estimated_us: f64,
    /// Largest total bytes any single rank sends over the whole collective.
    pub max_bytes_sent_per_rank: usize,
    pub total_bytes: usize,
    /// Largest total bytes any single rank sends in reduce phases.
    pub max_reduce_bytes_per_rank: usize,
    /// Largest total bytes any single rank sends in gather phases.
    pub max_gather_bytes_per_rank: usize,
}

/// Estimate with every element costing `bytes_per_element` (the R1 interface;
/// correct when the whole schedule uses one wire format).
pub fn estimate(rounds: &[Round], world: usize, bytes_per_element: usize, link: LinkModel) -> CostEstimate {
    estimate_by(rounds, world, link, |t| t.elements * bytes_per_element)
}

/// Estimate using each transfer's recorded wire format, so schedules whose
/// phases use different formats are costed correctly.
pub fn estimate_recorded(rounds: &[Round], world: usize, link: LinkModel) -> CostEstimate {
    estimate_by(rounds, world, link, Transfer::bytes)
}

fn estimate_by(rounds: &[Round], world: usize, link: LinkModel, bytes_of: impl Fn(&Transfer) -> usize) -> CostEstimate {
    let bytes_per_us = link.bandwidth_gbps * 1e3;
    let mut sent = vec![0usize; world];
    let mut reduce_sent = vec![0usize; world];
    let mut gather_sent = vec![0usize; world];
    let mut total_bytes = 0;
    let mut estimated_us = 0.0;
    for round in rounds {
        let mut largest = 0;
        for t in &round.transfers {
            let bytes = bytes_of(t);
            sent[t.src] += bytes;
            match round.kind {
                TransferKind::Reduce => reduce_sent[t.src] += bytes,
                TransferKind::Gather => gather_sent[t.src] += bytes,
            }
            total_bytes += bytes;
            largest = largest.max(bytes);
        }
        estimated_us += link.latency_us + largest as f64 / bytes_per_us;
    }
    let max = |v: Vec<usize>| v.into_iter().max().unwrap_or(0);
    CostEstimate {
        rounds: rounds.len(),
        estimated_us,
        max_bytes_sent_per_rank: max(sent),
        total_bytes,
        max_reduce_bytes_per_rank: max(reduce_sent),
        max_gather_bytes_per_rank: max(gather_sent),
    }
}

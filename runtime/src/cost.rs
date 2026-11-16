//! An α–β (latency–bandwidth) estimate of a simulated schedule.
//!
//! Each round costs `latency + largest_message_bytes / bandwidth`. The model
//! assumes full-duplex point-to-point links with no contention, no compute
//! cost for additions, and no overlap between rounds. Defaults are
//! illustrative, not measurements of any particular interconnect or Etched
//! hardware.

use crate::collectives::Round;

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
}

pub fn estimate(rounds: &[Round], world: usize, bytes_per_element: usize, link: LinkModel) -> CostEstimate {
    let bytes_per_us = link.bandwidth_gbps * 1e3;
    let mut sent = vec![0usize; world];
    let mut total_bytes = 0;
    let mut estimated_us = 0.0;
    for round in rounds {
        let mut largest = 0;
        for t in &round.transfers {
            let bytes = t.elements * bytes_per_element;
            sent[t.src] += bytes;
            total_bytes += bytes;
            largest = largest.max(bytes);
        }
        estimated_us += link.latency_us + largest as f64 / bytes_per_us;
    }
    CostEstimate {
        rounds: rounds.len(),
        estimated_us,
        max_bytes_sent_per_rank: sent.into_iter().max().unwrap_or(0),
        total_bytes,
    }
}

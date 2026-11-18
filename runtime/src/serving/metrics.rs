//! Latency, throughput and counters from a finished run.
//!
//! Definitions (all times integer nanoseconds):
//!
//! * **TTFT** (per request): first committed token − arrival. All samples
//!   commit their first token in the same step.
//! * **TPOT** (per sample with at least two output tokens):
//!   (last commit − first commit) ÷ (output tokens − 1), rounded down.
//! * **End-to-end** (per request): finish − arrival.
//! * **Percentiles** by nearest rank: the p-th percentile of n sorted values
//!   is the value at 1-based rank ⌈p·n ÷ 100⌉.
//! * **Throughput**: committed output tokens (and finished requests) per
//!   second over the busy span, first arrival to last finish.
//! * **KV utilization**: device blocks in use at each step's end, weighted
//!   by step duration, over the capacity.
//! * **Compute and communication**: summed step-time parts.

use super::engine::RunResult;
use super::scheduler::Phase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Percentiles {
    pub n: usize,
    pub p50: u64,
    pub p99: u64,
    pub max: u64,
}

/// Nearest-rank percentile of sorted values (`p` in 1..=100).
pub fn nearest_rank(sorted: &[u64], p: u64) -> u64 {
    assert!(!sorted.is_empty() && (1..=100).contains(&p));
    let rank = (p * sorted.len() as u64).div_ceil(100);
    sorted[rank as usize - 1]
}

pub fn percentiles(mut values: Vec<u64>) -> Option<Percentiles> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(Percentiles {
        n: values.len(),
        p50: nearest_rank(&values, 50),
        p99: nearest_rank(&values, 99),
        max: *values.last().expect("non-empty"),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    pub requests: usize,
    pub finished: usize,
    pub rejected: usize,
    pub output_tokens: u64,
    pub ttft: Option<Percentiles>,
    pub tpot: Option<Percentiles>,
    pub e2e: Option<Percentiles>,
    pub span_ns: u64,
    pub tokens_per_s: f64,
    pub requests_per_s: f64,
    pub steps: usize,
    pub mean_step_tokens: f64,
    pub max_step_tokens: u64,
    pub compute_ns: f64,
    pub comm_ns: f64,
    pub copy_ns: f64,
    pub overhead_ns: f64,
    /// Communication ÷ total step time.
    pub comm_share: f64,
    pub preemptions_recompute: u64,
    pub preemptions_swap: u64,
    pub swap_fallbacks: u64,
    pub stalls: u64,
    pub recomputed_tokens: u64,
    pub swapped_blocks: u64,
    pub cow_copies: u64,
    pub evictions: u64,
    pub prefix_eligible_blocks: u64,
    pub prefix_hit_blocks: u64,
    pub kv_util_mean: f64,
    pub kv_peak_blocks: usize,
}

impl Metrics {
    pub fn prefix_hit_rate(&self) -> Option<f64> {
        (self.prefix_eligible_blocks > 0).then(|| self.prefix_hit_blocks as f64 / self.prefix_eligible_blocks as f64)
    }
}

pub fn metrics(run: &RunResult) -> Metrics {
    let finished: Vec<_> = run.requests.iter().filter(|r| r.phase == Phase::Finished).collect();
    let ttft = percentiles(
        finished
            .iter()
            .map(|r| r.first_token_ns.expect("finished") - r.request.arrival_ns)
            .collect(),
    );
    let e2e = percentiles(
        finished
            .iter()
            .map(|r| r.finish_ns.expect("finished") - r.request.arrival_ns)
            .collect(),
    );
    let tpot = percentiles(
        finished
            .iter()
            .filter(|r| r.request.output_len >= 2)
            .flat_map(|r| {
                let first = r.first_token_ns.expect("finished");
                r.samples
                    .iter()
                    .map(move |s| (s.last_commit_ns - first) / u64::from(r.request.output_len - 1))
            })
            .collect(),
    );
    let output_tokens: u64 = run
        .requests
        .iter()
        .flat_map(|r| r.samples.iter())
        .map(|s| u64::from(s.committed))
        .sum();
    let first_arrival = run.requests.iter().map(|r| r.request.arrival_ns).min().unwrap_or(0);
    let last_finish = finished
        .iter()
        .filter_map(|r| r.finish_ns)
        .max()
        .unwrap_or(first_arrival);
    let span_ns = last_finish - first_arrival;
    let per_s = |x: f64| if span_ns == 0 { 0.0 } else { x * 1e9 / span_ns as f64 };
    let sum = |f: fn(&super::engine::StepRecord) -> f64| run.steps.iter().map(f).sum::<f64>();
    let (compute_ns, comm_ns) = (sum(|s| s.time.compute_ns), sum(|s| s.time.comm_ns));
    let (copy_ns, overhead_ns) = (sum(|s| s.time.copy_ns), sum(|s| s.time.overhead_ns));
    let total_ns = compute_ns + comm_ns + copy_ns + overhead_ns;
    let busy_ns: u64 = run.steps.iter().map(|s| s.end_ns - s.start_ns).sum();
    let weighted: f64 = run
        .steps
        .iter()
        .map(|s| s.used_blocks as f64 * (s.end_ns - s.start_ns) as f64)
        .sum();
    let step_tokens: Vec<u64> = run.steps.iter().map(|s| s.plan.tokens()).collect();
    Metrics {
        requests: run.requests.len(),
        finished: finished.len(),
        rejected: run.requests.iter().filter(|r| r.phase == Phase::Rejected).count(),
        output_tokens,
        ttft,
        tpot,
        e2e,
        span_ns,
        tokens_per_s: per_s(output_tokens as f64),
        requests_per_s: per_s(finished.len() as f64),
        steps: run.steps.len(),
        mean_step_tokens: if step_tokens.is_empty() {
            0.0
        } else {
            step_tokens.iter().sum::<u64>() as f64 / step_tokens.len() as f64
        },
        max_step_tokens: step_tokens.iter().copied().max().unwrap_or(0),
        compute_ns,
        comm_ns,
        copy_ns,
        overhead_ns,
        comm_share: if total_ns > 0.0 { comm_ns / total_ns } else { 0.0 },
        preemptions_recompute: run.counters.preemptions_recompute,
        preemptions_swap: run.counters.preemptions_swap,
        swap_fallbacks: run.counters.swap_fallbacks,
        stalls: run.counters.stalls,
        recomputed_tokens: run.counters.recomputed_tokens,
        swapped_blocks: run.kv_stats.swapped_out_blocks + run.kv_stats.swapped_in_blocks,
        cow_copies: run.kv_stats.cow_copies,
        evictions: run.kv_stats.evictions,
        prefix_eligible_blocks: run.kv_stats.prefix_eligible_blocks,
        prefix_hit_blocks: run.kv_stats.prefix_hit_blocks,
        kv_util_mean: if busy_ns == 0 || run.capacity_blocks == 0 {
            0.0
        } else {
            weighted / (busy_ns as f64 * run.capacity_blocks as f64)
        },
        kv_peak_blocks: run.steps.iter().map(|s| s.used_blocks).max().unwrap_or(0),
    }
}

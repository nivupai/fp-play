//! The discrete-event loop: arrivals, steps, and invariant checks.
//!
//! Events are ordered by (time, sequence number): every arrival is queued
//! at the start (sequence numbers 0…n−1) and each step's end when the step
//! starts, so an arrival at the same instant as a step's end is handled
//! first. All events at one instant are handled before the next step is
//! built. Arrivals during a step wait for its end. Time is integer
//! nanoseconds.
//!
//! With checks on ([`EngineConfig::kv_check_interval`] ≥ 1), the scheduler
//! and token-stream invariants are verified after each step is built and
//! after it completes, and the KV-cache invariants after every k-th step's
//! completion and after the last step:
//! ([`KvCache::check`](super::kv::KvCache::check)) against each sequence's
//! deterministic tokens, plus
//!
//! * **scheduler:** a step's tokens ≤ the budget, running sequences ≤ the
//!   maximum, and the sequences on the device (in swap) are exactly the live
//!   sequences of the running (swapped) requests;
//! * **token stream:** each sample commits output indices 0, 1, 2, … with no
//!   gap or repeat, never more than its output length;
//! * **events:** the clock never moves backwards.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;

use super::kv::{KvCache, KvConfig, KvStats, Mutation, SeqId};
use super::scheduler::{
    Phase, RequestState, Scheduler, SchedulerConfig, SchedulerCounters, SchedulerMutation, StepOutcome, StepPlan,
};
use super::step_time::{StepTime, StepTimer};
use super::workload::Request;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    pub scheduler: SchedulerConfig,
    pub kv: KvConfig,
    /// 0: no invariant checks. k ≥ 1: the scheduler and token-stream
    /// invariants after every step, the KV-cache invariants after every k-th
    /// step and after the last one.
    pub kv_check_interval: u64,
    /// A deliberate bug for tests (never set otherwise).
    #[doc(hidden)]
    pub planted: Option<Planted>,
}

/// A deliberate bug planted by tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planted {
    Kv(Mutation),
    Scheduler(SchedulerMutation),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    pub index: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub plan: StepPlan,
    pub outcome: StepOutcome,
    pub time: StepTime,
    /// Device blocks referenced at the end of the step.
    pub used_blocks: usize,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub requests: Vec<RequestState>,
    pub steps: Vec<StepRecord>,
    pub counters: SchedulerCounters,
    pub kv_stats: KvStats,
    pub capacity_blocks: usize,
    pub end_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    Config(String),
    /// An invariant failed; `step` is the step index.
    Invariant {
        step: u64,
        what: String,
    },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Config(m) => f.write_str(m),
            EngineError::Invariant { step, what } => write!(f, "step {step}: invariant broken: {what}"),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    Arrival(usize),
    StepEnd,
}

/// Every sequence's full deterministic token stream (prompt, then that
/// sample's outputs), precomputed so the per-step content check is a lookup.
fn token_streams(requests: &[Request]) -> Vec<Vec<Vec<u32>>> {
    requests
        .iter()
        .map(|r| {
            (0..r.samples)
                .map(|s| {
                    (0..r.prompt_len + r.output_len)
                        .map(|p| r.sequence_token(s, p))
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Check the scheduler and token-stream invariants, and with `streams` the
/// KV-cache invariants too.
fn check(
    s: &Scheduler,
    plan: Option<&StepPlan>,
    next_commit: &BTreeMap<SeqId, u32>,
    streams: Option<&[Vec<Vec<u32>>]>,
    step: u64,
) -> Result<(), EngineError> {
    let fail = |what: String| Err(EngineError::Invariant { step, what });
    if let Some(plan) = plan {
        if plan.tokens() > s.config.token_budget as u64 {
            return fail(format!("scheduler: {} tokens over the budget", plan.tokens()));
        }
    }
    if s.running_seqs() > s.config.max_seqs {
        return fail(format!("scheduler: {} running sequences", s.running_seqs()));
    }
    let mut device = BTreeSet::new();
    let mut swapped = BTreeSet::new();
    for r in &s.requests {
        let live = || {
            let unfinished = (0..r.samples.len()).filter(|&i| !r.samples[i].finished);
            if r.forked {
                unfinished.collect::<Vec<_>>()
            } else {
                unfinished.take(1).collect()
            }
        };
        match r.phase {
            Phase::Running => device.extend(live().into_iter().map(|i| r.seq(i))),
            Phase::Swapped => swapped.extend(live().into_iter().map(|i| r.seq(i))),
            _ => {}
        }
        for (i, sample) in r.samples.iter().enumerate() {
            let next = next_commit.get(&r.seq(i)).copied().unwrap_or(0);
            if sample.committed != next || sample.committed > r.request.output_len {
                return fail(format!(
                    "token stream: {} committed {} tokens, the log shows {next}",
                    r.seq(i),
                    sample.committed
                ));
            }
            if sample.finished != (sample.committed == r.request.output_len) {
                return fail(format!("token stream: {} finished flag is wrong", r.seq(i)));
            }
        }
    }
    if !s.kv.sequences().eq(device.iter().copied()) {
        return fail("scheduler: device sequences differ from the running requests' live sequences".into());
    }
    if !s.kv.swapped_sequences().eq(swapped.iter().copied()) {
        return fail("scheduler: swapped sequences differ from the swapped requests' live sequences".into());
    }
    match streams {
        Some(streams) => {
            s.kv.check(&|seq, p| streams[seq.request as usize][seq.sample as usize][p])
                .or_else(|v| fail(format!("kv: {v}")))
        }
        None => Ok(()),
    }
}

/// Record commits in the log, rejecting a gap or a repeat.
fn log_commits(
    outcome: &StepOutcome,
    next_commit: &mut BTreeMap<SeqId, u32>,
    step: u64,
    checking: bool,
) -> Result<(), EngineError> {
    for &(seq, k) in &outcome.commits {
        let next = next_commit.entry(seq).or_insert(0);
        if checking && k != *next {
            return Err(EngineError::Invariant {
                step,
                what: format!("token stream: {seq} committed output {k}, expected {next}"),
            });
        }
        *next += 1;
    }
    Ok(())
}

/// Run `requests` (ids 0, 1, … in order) to completion.
pub fn run(requests: &[Request], config: &EngineConfig, timer: &dyn StepTimer) -> Result<RunResult, EngineError> {
    if requests.iter().enumerate().any(|(i, r)| r.id as usize != i) {
        return Err(EngineError::Config("request ids must be 0, 1, … in order".into()));
    }
    if config.scheduler.token_budget == 0 || config.scheduler.max_seqs == 0 {
        return Err(EngineError::Config(
            "token budget and maximum sequences must be positive".into(),
        ));
    }
    let mut kv = KvCache::new(config.kv);
    if let Some(Planted::Kv(m)) = config.planted {
        kv.plant(m);
    }
    let mut s = Scheduler::new(config.scheduler, kv, requests);
    if let Some(Planted::Scheduler(m)) = config.planted {
        s.mutation = Some(m);
    }
    let checking = config.kv_check_interval > 0;
    let interval = config.kv_check_interval.max(1);
    let streams = if checking { token_streams(requests) } else { Vec::new() };
    let mut events = BinaryHeap::new();
    let mut seqno = 0u64;
    for (i, r) in requests.iter().enumerate() {
        events.push(Reverse((r.arrival_ns, seqno, Event::Arrival(i))));
        seqno += 1;
    }
    let mut now = 0u64;
    let mut steps: Vec<StepRecord> = Vec::new();
    let mut current: Option<(StepPlan, u64, StepTime)> = None;
    let mut next_commit = BTreeMap::new();
    while let Some(Reverse((t, _, event))) = events.pop() {
        let index = steps.len() as u64;
        if t < now {
            return Err(EngineError::Invariant {
                step: index,
                what: format!("events: clock moved back from {now} to {t}"),
            });
        }
        now = t;
        let mut batch = vec![event];
        while events.peek().is_some_and(|Reverse((t2, _, _))| *t2 == now) {
            batch.push(events.pop().expect("peeked").0 .2);
        }
        for event in batch {
            match event {
                Event::Arrival(i) => s.arrive(i),
                Event::StepEnd => {
                    let (plan, start, time) = current.take().expect("a step is running");
                    let outcome = s.complete_step(&plan, now).map_err(|e| EngineError::Invariant {
                        step: index,
                        what: format!("kv: {e}"),
                    })?;
                    log_commits(&outcome, &mut next_commit, index, checking)?;
                    if checking {
                        let kv = (index % interval == 0).then_some(streams.as_slice());
                        check(&s, None, &next_commit, kv, index)?;
                    }
                    steps.push(StepRecord {
                        index,
                        start_ns: start,
                        end_ns: now,
                        plan,
                        outcome,
                        time,
                        used_blocks: s.kv.used_blocks(),
                    });
                }
            }
        }
        if current.is_none() {
            let mut plan = s.build_step();
            while plan.work.is_empty() && s.running_seqs() > 0 {
                let mut stalled = plan;
                s.break_stall(&mut stalled);
                let next = s.build_step();
                plan = merge(stalled, next);
            }
            if checking {
                // The KV cache is checked once per step, after it completes.
                check(&s, Some(&plan), &next_commit, None, steps.len() as u64)?;
            }
            if !plan.work.is_empty() {
                let time = timer.time(&s.step_work(&plan));
                if time.total_ns == 0 {
                    return Err(EngineError::Config("a step must take time".into()));
                }
                events.push(Reverse((now + time.total_ns, seqno, Event::StepEnd)));
                seqno += 1;
                current = Some((plan, now, time));
            } else if !plan.preempted.is_empty() || !plan.swapped_in.is_empty() {
                return Err(EngineError::Invariant {
                    step: steps.len() as u64,
                    what: "scheduler: a step with no work changed state".into(),
                });
            }
        }
    }
    if checking {
        check(&s, None, &next_commit, Some(&streams), steps.len() as u64)?;
    }
    if s.has_unfinished_work() {
        return Err(EngineError::Invariant {
            step: steps.len() as u64,
            what: "events ran out with requests unfinished".into(),
        });
    }
    Ok(RunResult {
        capacity_blocks: config.kv.device_blocks,
        kv_stats: s.kv.stats().clone(),
        counters: s.counters,
        requests: s.requests,
        steps,
        end_ns: now,
    })
}

/// A stall-breaking preemption followed by the rebuilt step.
fn merge(mut first: StepPlan, next: StepPlan) -> StepPlan {
    first.swapped_in.extend(next.swapped_in);
    first.admitted.extend(next.admitted);
    first.preempted.extend(next.preempted);
    first.stalls += next.stalls;
    first.work = next.work;
    first.cow_blocks += next.cow_blocks;
    first.swap_blocks += next.swap_blocks;
    first
}

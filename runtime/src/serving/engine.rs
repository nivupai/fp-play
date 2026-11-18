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
//!
//! # Failures
//!
//! A [`FaultConfig`] lists fail-stop rank failures, each at (step index,
//! layer, which of the layer's two AllReduces). That step stops at that
//! collective: it is charged its fixed overhead and copies plus the share of
//! its compute and communication up to and including that collective
//! (`(2·layer + which + 1) ÷ (2·layers)`), then the detection timeout and the
//! recovery time, rounded once to whole nanoseconds. **The step commits
//! nothing.** Losing a rank loses every device KV block (each holds a shard
//! on every rank); the host swap pool survives. Requests that were running
//! are handled by the [`FaultPolicy`]. With checks on, these guarantees are
//! verified (a violation ends the run with an error):
//!
//! * **G1, no leaked blocks:** right after the failure no device block is in
//!   use, and every KV-cache invariant (including the swap pool) holds.
//! * **G2, no duplicated tokens:** the token-stream invariant above, after
//!   every step including recovery.
//! * **G3, nothing lost under retry:** every retried sample ends with at
//!   least the tokens it had committed when the failure happened.
//! * **G4, exactly one ending:** at the end every request has finished,
//!   failed or been rejected, exactly once.
//!
//! G5, determinism, is checked by running twice (tests and the recorded
//! failure scenario compare the traces).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;

#[cfg(feature = "planted-bugs")]
use super::kv::Mutation;
use super::kv::{KvCache, KvConfig, KvStats, SeqId};
#[cfg(feature = "planted-bugs")]
use super::scheduler::SchedulerMutation;
use super::scheduler::{
    FaultOutcome, FaultPolicy, Phase, RequestState, Scheduler, SchedulerConfig, SchedulerCounters, StepOutcome,
    StepPlan,
};
use super::step_time::{StepTime, StepTimer};
use super::workload::Request;

/// A fail-stop rank failure during step `step`, in layer `layer`
/// (0-based) at its first (`allreduce = 0`, after attention) or second
/// (`1`, after the MLP) AllReduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    pub step: u64,
    pub layer: u32,
    pub allreduce: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultConfig {
    /// Failures, by step index (each step at most once).
    pub plan: Vec<Fault>,
    pub policy: FaultPolicy,
    pub detection_ns: u64,
    pub recovery_ns: u64,
    /// Retries allowed per request before it fails.
    pub retry_limit: u32,
    /// Layers in the model (to place the failing collective).
    pub layers: u32,
}

impl FaultConfig {
    pub fn none() -> FaultConfig {
        FaultConfig {
            plan: Vec::new(),
            policy: FaultPolicy::Retry,
            detection_ns: 0,
            recovery_ns: 0,
            retry_limit: 0,
            layers: 1,
        }
    }

    fn at(&self, step: u64) -> Option<Fault> {
        self.plan.iter().copied().find(|f| f.step == step)
    }

    /// Wall time of a step that fails at `fault`: see the module documentation.
    pub fn failed_step_ns(&self, time: &StepTime, fault: Fault) -> u64 {
        let done = f64::from(2 * fault.layer + u32::from(fault.allreduce) + 1) / f64::from(2 * self.layers);
        let elapsed = time.overhead_ns + time.copy_ns + done * (time.compute_ns + time.comm_ns);
        (elapsed + self.detection_ns as f64 + self.recovery_ns as f64).round() as u64
    }
}

/// What a failure did, recorded on the failed step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultRecord {
    pub layer: u32,
    pub allreduce: u8,
    pub outcome: FaultOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    pub scheduler: SchedulerConfig,
    pub kv: KvConfig,
    pub faults: FaultConfig,
    /// 0: no invariant checks. k ≥ 1: the scheduler and token-stream
    /// invariants after every step, the KV-cache invariants after every k-th
    /// step and after the last one.
    pub kv_check_interval: u64,
}

/// A deliberate bug planted by tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
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
    /// Set if a rank failed during this step (which then committed nothing).
    pub fault: Option<FaultRecord>,
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
    streams: Option<&Streams>,
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
        Some(Streams::Synthetic(streams)) => {
            s.kv.check(&|seq, p| streams[seq.request as usize][seq.sample as usize][p])
                .or_else(|v| fail(format!("kv: {v}")))
        }
        Some(Streams::Model) => {
            s.kv.check(&|seq, p| s.requests[seq.request as usize].token_at(seq.sample as usize, p as u64))
                .or_else(|v| fail(format!("kv: {v}")))
        }
        None => Ok(()),
    }
}

/// Where the KV content check gets each sequence's expected tokens.
enum Streams {
    /// The precomputed synthetic streams.
    Synthetic(Vec<Vec<Vec<u32>>>),
    /// The requests' model tokens (prompt plus committed outputs).
    Model,
}

/// Runs the model for a step (R3). With an executor, token values come from
/// the model: prompts are given up front and each committed output is the
/// executor's token.
pub trait StepExecutor {
    /// Step `index` (`plan`) has completed without a failure; its commits
    /// are not applied yet. KV appends for the plan were made when it was
    /// built, so the block tables already cover every position it computes.
    fn execute(&mut self, index: u64, plan: &StepPlan, scheduler: &mut Scheduler) -> Result<(), String>;
    /// The token `seq` commits as output `k` in the step just executed.
    fn output(&mut self, seq: SeqId, k: u32) -> Result<u32, String>;
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
    run_inner(requests, config, timer, None, |_| {})
}

/// [`run`] with token values from a model: `prompts[i]` is request i's
/// prompt, and `executor` computes every completed step and its outputs.
/// The KV cache records block copies for the executor to replay.
pub fn run_model(
    requests: &[Request],
    prompts: &[Vec<u32>],
    config: &EngineConfig,
    timer: &dyn StepTimer,
    executor: &mut dyn StepExecutor,
) -> Result<RunResult, EngineError> {
    run_inner(requests, config, timer, Some((prompts, executor)), |_| {})
}

/// [`run`] with one planted bug switched on (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
pub fn run_planted(
    requests: &[Request],
    config: &EngineConfig,
    timer: &dyn StepTimer,
    planted: Planted,
) -> Result<RunResult, EngineError> {
    run_inner(requests, config, timer, None, |s| match planted {
        Planted::Kv(m) => s.kv.plant(m),
        Planted::Scheduler(m) => s.mutation = Some(m),
    })
}

fn run_inner(
    requests: &[Request],
    config: &EngineConfig,
    timer: &dyn StepTimer,
    model: Option<(&[Vec<u32>], &mut dyn StepExecutor)>,
    plant: impl FnOnce(&mut Scheduler),
) -> Result<RunResult, EngineError> {
    if requests.iter().enumerate().any(|(i, r)| r.id as usize != i) {
        return Err(EngineError::Config("request ids must be 0, 1, … in order".into()));
    }
    if config.scheduler.token_budget == 0 || config.scheduler.max_seqs == 0 {
        return Err(EngineError::Config(
            "token budget and maximum sequences must be positive".into(),
        ));
    }
    let mut s = Scheduler::new(config.scheduler, KvCache::new(config.kv), requests);
    plant(&mut s);
    let mut executor = match model {
        Some((prompts, executor)) => {
            s.use_model_tokens(prompts).map_err(EngineError::Config)?;
            s.kv.record_copies(true);
            Some(executor)
        }
        None => None,
    };
    let faults = &config.faults;
    if faults.plan.iter().any(|f| f.layer >= faults.layers || f.allreduce > 1) {
        return Err(EngineError::Config(
            "a fault names a layer or AllReduce that does not exist".into(),
        ));
    }
    // Committed tokens of each retried sample at its failure (G3).
    let mut retried_floor: BTreeMap<SeqId, u32> = BTreeMap::new();
    let checking = config.kv_check_interval > 0;
    let interval = config.kv_check_interval.max(1);
    let streams = if executor.is_some() {
        Streams::Model
    } else if checking {
        Streams::Synthetic(token_streams(requests))
    } else {
        Streams::Synthetic(Vec::new())
    };
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
                Event::Arrival(i) => s.arrive(i, now),
                Event::StepEnd if faults.at(index).is_some() => {
                    let fault = faults.at(index).expect("checked");
                    let (plan, start, time) = current.take().expect("a step is running");
                    // Committed tokens before the failure, for G3.
                    let before: Vec<Vec<u32>> = s
                        .requests
                        .iter()
                        .map(|r| r.samples.iter().map(|x| x.committed).collect())
                        .collect();
                    let outcome = s.on_fault(faults.policy, faults.retry_limit, now);
                    for &id in &outcome.retried {
                        let r = &s.requests[id as usize];
                        for (i, &committed) in before[id as usize].iter().enumerate() {
                            retried_floor.entry(r.seq(i)).or_insert(committed);
                        }
                    }
                    if checking {
                        if s.kv.used_blocks() != 0 {
                            return Err(EngineError::Invariant {
                                step: index,
                                what: format!(
                                    "G1: {} device blocks still in use after the failure",
                                    s.kv.used_blocks()
                                ),
                            });
                        }
                        check(&s, None, &next_commit, Some(&streams), index)?;
                    }
                    steps.push(StepRecord {
                        index,
                        start_ns: start,
                        end_ns: now,
                        plan,
                        outcome: StepOutcome::default(),
                        time,
                        used_blocks: s.kv.used_blocks(),
                        fault: Some(FaultRecord {
                            layer: fault.layer,
                            allreduce: fault.allreduce,
                            outcome,
                        }),
                    });
                }
                Event::StepEnd => {
                    let (plan, start, time) = current.take().expect("a step is running");
                    let model_error = |what: String| EngineError::Invariant {
                        step: index,
                        what: format!("model: {what}"),
                    };
                    if let Some(x) = executor.as_deref_mut() {
                        x.execute(index, &plan, &mut s).map_err(model_error)?;
                    }
                    let outcome = s.complete_step(&plan, now).map_err(|e| EngineError::Invariant {
                        step: index,
                        what: format!("kv: {e}"),
                    })?;
                    if let Some(x) = executor.as_deref_mut() {
                        for &(seq, k) in &outcome.commits {
                            let token = x.output(seq, k).map_err(model_error)?;
                            s.record_output(seq, k, token).map_err(model_error)?;
                        }
                    }
                    log_commits(&outcome, &mut next_commit, index, checking)?;
                    if checking {
                        let kv = (index % interval == 0).then_some(&streams);
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
                        fault: None,
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
                let duration = match faults.at(steps.len() as u64) {
                    Some(fault) => faults.failed_step_ns(&time, fault),
                    None => time.total_ns,
                };
                if duration == 0 {
                    return Err(EngineError::Config("a step must take time".into()));
                }
                events.push(Reverse((now + duration, seqno, Event::StepEnd)));
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
    if checking {
        let fail = |what: String| {
            Err(EngineError::Invariant {
                step: steps.len() as u64,
                what,
            })
        };
        for (&seq, &floor) in &retried_floor {
            let r = &s.requests[seq.request as usize];
            if r.phase != Phase::Failed && r.samples[seq.sample as usize].committed < floor {
                return fail(format!("G3: {seq} lost committed tokens across a retry"));
            }
        }
        if let Some(r) = s.requests.iter().find(|r| !r.phase.is_terminal() || r.endings != 1) {
            return fail(format!(
                "G4: request {} ended {} times ({:?})",
                r.request.id, r.endings, r.phase
            ));
        }
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

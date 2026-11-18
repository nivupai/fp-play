//! Continuous batching: which sequences run in the next step.
//!
//! # The policy (the reference scheduler implements this same text)
//!
//! **Order.** Requests are ordered by priority (higher first), then arrival
//! time, then ID. "In order" below always means this order; the
//! **preemption victim** is the *last* running request in this order (lowest
//! priority, then latest arrival) that has no work in the step being built
//! and is not the request asking for blocks.
//!
//! **Sequences.** A request with n samples has one sequence per unfinished
//! sample. Its prompt is computed once, by the *carrier* (its lowest
//! unfinished sample); while other unfinished samples have not been forked
//! yet, the carrier's work stops at the end of the prompt. A sample's
//! *known* tokens are the prompt plus its committed outputs, and its *work*
//! is known tokens minus the tokens in its KV cache (capped at the prompt
//! for an unforked carrier). Work of 1 is a decode; more is a prefill.
//!
//! **One step** with a budget of `token_budget` tokens and at most
//! `max_seqs` running sequences:
//!
//! 1. *Swap in.* Swapped requests in order, each with all its sequences,
//!    while the sequences fit under `max_seqs` and the blocks leave at least
//!    `watermark_blocks` available. Stop at the first that does not fit.
//! 2. *Decodes.* Running requests in order: all of a request's decode
//!    sequences together, one token each. If they exceed the remaining
//!    budget, phases 2–4 stop. If their blocks (exact, copy-on-write
//!    included) are not available, preempt victims until they are; with no
//!    victim left, preempt the request itself.
//! 3. *Prefill chunks.* Running requests in order, their prefill sequences
//!    in sample order: a chunk of min(work, remaining budget) tokens. If its
//!    blocks are not available, preempt victims until they are; with no
//!    victim left, skip this sequence for this step.
//! 4. *Admission*, only if no request is still swapped and nothing was
//!    preempted in this step (so a victim is not readmitted at once, and
//!    memory pressure stops new work). Waiting requests in
//!    order: the carrier starts, adopts cached prefix blocks, and gets a
//!    first chunk of min(work, remaining budget) tokens, if its unfinished
//!    samples fit under `max_seqs` and the chunk's new blocks (plus retained
//!    blocks the adoption takes) leave at least `watermark_blocks`
//!    available. Stop at the first that does not fit.
//!
//! **Preemption.** *Recompute* frees every sequence of the victim; it
//! returns to waiting and keeps its committed tokens. *Swap* copies every
//! sequence to the host swap pool (charged to the step); if the swap pool
//! cannot hold them all, the victim is recomputed instead and the fallback
//! is counted.
//!
//! **Step end.** A sequence whose KV now holds all its known tokens samples
//! one token (it is committed). When an unforked carrier's KV reaches the end
//! of the prompt, its unfinished siblings fork from it, and siblings with no
//! committed token commit their first one from the same logits. A sample with
//! all its output tokens is finished and its sequence freed; a request is
//! finished when every sample is.
//!
//! KV appends for scheduled work are made while the step is built (their
//! tokens are known), so block availability is exact at every decision.
//!
//! **Trace order.** A step's swap-ins, admissions, preemptions and work are
//! listed in the order they happen above. Its commits are listed in work
//! order, then the first tokens of newly forked siblings; requests are
//! handled at step end in order of their first work item, and finished
//! requests are listed in that order. If a step would have no work while
//! requests are running, the last running request in order is preempted
//! (a *stall*, counted) and the step is built again; the preemptions of
//! both attempts are listed together.

use std::cmp::Reverse;

use super::kv::{KvCache, KvError, SeqId};
use super::step_time::{SeqWork, StepWork};
use super::workload::Request;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreemptionMode {
    Recompute,
    Swap,
}

impl PreemptionMode {
    pub fn name(self) -> &'static str {
        match self {
            PreemptionMode::Recompute => "recompute",
            PreemptionMode::Swap => "swap",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub token_budget: usize,
    pub max_seqs: usize,
    pub watermark_blocks: usize,
    pub preemption: PreemptionMode,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            token_budget: 512,
            max_seqs: 64,
            watermark_blocks: 0,
            preemption: PreemptionMode::Recompute,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// Not yet arrived.
    Pending,
    Waiting,
    Running,
    Swapped,
    Finished,
    /// Could never fit in the KV cache; rejected on arrival.
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleState {
    pub committed: u32,
    pub finished: bool,
    /// Commit time of the last committed token.
    pub last_commit_ns: u64,
    /// Most tokens this sample's KV has ever held (for counting recomputation).
    pub peak_kv: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestState {
    pub request: Request,
    pub phase: Phase,
    pub samples: Vec<SampleState>,
    /// Every unfinished sample has its own sequence (or there is at most one).
    pub forked: bool,
    pub first_token_ns: Option<u64>,
    pub finish_ns: Option<u64>,
    pub preemptions: u32,
}

impl RequestState {
    pub fn new(request: Request) -> RequestState {
        let samples = (0..request.samples)
            .map(|_| SampleState {
                committed: 0,
                finished: false,
                last_commit_ns: 0,
                peak_kv: 0,
            })
            .collect();
        RequestState {
            forked: request.samples <= 1,
            request,
            phase: Phase::Pending,
            samples,
            first_token_ns: None,
            finish_ns: None,
            preemptions: 0,
        }
    }

    fn key(&self) -> (Reverse<u8>, u64, u32) {
        (Reverse(self.request.priority), self.request.arrival_ns, self.request.id)
    }

    pub fn seq(&self, sample: usize) -> SeqId {
        SeqId {
            request: self.request.id,
            sample: sample as u32,
        }
    }

    fn unfinished(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.samples.len()).filter(|&s| !self.samples[s].finished)
    }

    pub fn unfinished_count(&self) -> usize {
        self.unfinished().count()
    }

    fn carrier(&self) -> Option<usize> {
        self.unfinished().next()
    }

    /// Samples that currently have a sequence (on the device or swapped).
    fn live(&self) -> Vec<usize> {
        if self.forked {
            self.unfinished().collect()
        } else {
            self.carrier().into_iter().collect()
        }
    }

    fn known(&self, sample: usize) -> u64 {
        u64::from(self.request.prompt_len + self.samples[sample].committed)
    }

    /// Where this sample's work stops this step.
    fn target(&self, sample: usize) -> u64 {
        if self.forked {
            self.known(sample)
        } else {
            u64::from(self.request.prompt_len)
        }
    }

    fn token(&self, sample: usize, position: u64) -> u32 {
        self.request.sequence_token(sample as u32, position as u32)
    }
}

/// How a preemption ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreemptKind {
    Recompute,
    Swap,
    /// Swap was chosen but the swap pool was too small.
    SwapFallback,
}

impl PreemptKind {
    pub fn name(self) -> &'static str {
        match self {
            PreemptKind::Recompute => "recompute",
            PreemptKind::Swap => "swap",
            PreemptKind::SwapFallback => "swap-full-recompute",
        }
    }
}

/// One sequence's work in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Work {
    pub seq: SeqId,
    /// KV tokens before the step.
    pub context: u64,
    pub tokens: u64,
}

/// Every decision of one step: the trace the reference scheduler must match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepPlan {
    pub swapped_in: Vec<u32>,
    pub admitted: Vec<u32>,
    pub preempted: Vec<(u32, PreemptKind)>,
    /// Requests preempted because a step would otherwise be empty.
    pub stalls: u32,
    pub work: Vec<Work>,
    pub cow_blocks: u64,
    pub swap_blocks: u64,
}

impl StepPlan {
    pub fn tokens(&self) -> u64 {
        self.work.iter().map(|w| w.tokens).sum()
    }
}

/// What a step's completion changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepOutcome {
    /// (sequence, output index) of every committed token, in commit order.
    pub commits: Vec<(SeqId, u32)>,
    pub finished: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SchedulerCounters {
    pub preemptions_recompute: u64,
    pub preemptions_swap: u64,
    pub swap_fallbacks: u64,
    pub stalls: u64,
    /// Tokens processed again because their KV had been computed before.
    pub recomputed_tokens: u64,
    pub rejected: u64,
}

/// Deliberate scheduler bugs that tests plant to show the checks catch
/// them. Never set outside tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerMutation {
    /// Preempt the *first* eligible running request instead of the last.
    VictimFirst,
    /// Give a prefill chunk one token more than the remaining budget.
    BudgetOverrun,
}

pub struct Scheduler {
    pub config: SchedulerConfig,
    pub kv: KvCache,
    pub requests: Vec<RequestState>,
    pub counters: SchedulerCounters,
    #[doc(hidden)]
    pub mutation: Option<SchedulerMutation>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, kv: KvCache, requests: &[Request]) -> Scheduler {
        Scheduler {
            config,
            kv,
            requests: requests.iter().cloned().map(RequestState::new).collect(),
            counters: SchedulerCounters::default(),
            mutation: None,
        }
    }

    /// Most device blocks a request could ever need (no sharing assumed).
    pub fn max_blocks(&self, request: &Request) -> usize {
        let b = self.kv.config().block_size as u64;
        let tokens = u64::from(request.prompt_len + request.output_len - 1);
        (tokens.div_ceil(b) * u64::from(request.samples)) as usize
    }

    /// A request arrives: it waits, or is rejected if it could never fit.
    pub fn arrive(&mut self, index: usize) {
        let fits = self.max_blocks(&self.requests[index].request) + self.config.watermark_blocks
            <= self.kv.config().device_blocks
            && self.requests[index].request.samples as usize <= self.config.max_seqs;
        let r = &mut self.requests[index];
        debug_assert_eq!(r.phase, Phase::Pending);
        if fits {
            r.phase = Phase::Waiting;
        } else {
            r.phase = Phase::Rejected;
            self.counters.rejected += 1;
        }
    }

    fn in_order(&self, phase: Phase) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.requests.len())
            .filter(|&i| self.requests[i].phase == phase)
            .collect();
        v.sort_by_key(|&i| self.requests[i].key());
        v
    }

    pub fn running_seqs(&self) -> usize {
        self.requests
            .iter()
            .filter(|r| r.phase == Phase::Running)
            .map(RequestState::unfinished_count)
            .sum()
    }

    pub fn has_unfinished_work(&self) -> bool {
        self.requests
            .iter()
            .any(|r| matches!(r.phase, Phase::Waiting | Phase::Running | Phase::Swapped))
    }

    fn kv_len(&self, r: usize, sample: usize) -> u64 {
        self.kv.table(self.requests[r].seq(sample)).map_or(0, |t| t.len as u64)
    }

    fn work_of(&self, r: usize, sample: usize) -> u64 {
        self.requests[r].target(sample) - self.kv_len(r, sample)
    }

    fn append_work(&mut self, plan: &mut StepPlan, r: usize, sample: usize, n: u64) {
        let state = &self.requests[r];
        let seq = state.seq(sample);
        let context = self.kv_len(r, sample);
        let tokens: Vec<u32> = (context..context + n).map(|p| state.token(sample, p)).collect();
        let report = self
            .kv
            .append(seq, &tokens)
            .expect("availability checked before appending");
        plan.cow_blocks += u64::from(report.cow);
        let peak = &mut self.requests[r].samples[sample].peak_kv;
        self.counters.recomputed_tokens += (*peak).min(context + n).saturating_sub(context);
        *peak = (*peak).max(context + n);
        plan.work.push(Work {
            seq,
            context,
            tokens: n,
        });
    }

    fn preempt(&mut self, plan: &mut StepPlan, r: usize) {
        let live = self.requests[r].live();
        let seqs: Vec<SeqId> = live.iter().map(|&s| self.requests[r].seq(s)).collect();
        let mut kind = match self.config.preemption {
            PreemptionMode::Recompute => PreemptKind::Recompute,
            PreemptionMode::Swap => PreemptKind::Swap,
        };
        if kind == PreemptKind::Swap {
            let need: usize = seqs
                .iter()
                .map(|&s| self.kv.table(s).map_or(0, |t| t.blocks.len()))
                .sum();
            if need > self.kv.swap_free_blocks() {
                kind = PreemptKind::SwapFallback;
            }
        }
        for &seq in &seqs {
            if kind == PreemptKind::Swap {
                plan.swap_blocks += self.kv.swap_out(seq).expect("swap space checked") as u64;
            } else {
                self.kv.free(seq).expect("live sequences are on the device");
            }
        }
        let state = &mut self.requests[r];
        state.preemptions += 1;
        if kind == PreemptKind::Swap {
            state.phase = Phase::Swapped;
            self.counters.preemptions_swap += 1;
        } else {
            state.phase = Phase::Waiting;
            state.forked = state.unfinished_count() <= 1;
            self.counters.preemptions_recompute += 1;
            if kind == PreemptKind::SwapFallback {
                self.counters.swap_fallbacks += 1;
            }
        }
        plan.preempted.push((state.request.id, kind));
    }

    /// The last running request in order with no work this step, other than `except`.
    fn victim(&self, plan: &StepPlan, except: usize) -> Option<usize> {
        let busy = |r: usize| plan.work.iter().any(|w| w.seq.request == self.requests[r].request.id);
        let mut order = self.in_order(Phase::Running);
        if self.mutation != Some(SchedulerMutation::VictimFirst) {
            order.reverse();
        }
        order.into_iter().find(|&r| r != except && !busy(r))
    }

    /// Build the next step (see the module documentation). KV appends for
    /// the scheduled work are made now.
    pub fn build_step(&mut self) -> StepPlan {
        let mut plan = StepPlan::default();
        let mut budget = self.config.token_budget as u64;
        let watermark = self.config.watermark_blocks;

        // 1. Swap in.
        for r in self.in_order(Phase::Swapped) {
            let live = self.requests[r].live();
            let seqs: Vec<SeqId> = live.iter().map(|&s| self.requests[r].seq(s)).collect();
            let need: usize = seqs
                .iter()
                .map(|&s| self.kv.swapped_table(s).expect("swapped").blocks.len())
                .sum();
            if self.running_seqs() + seqs.len() > self.config.max_seqs || self.kv.available() < need + watermark {
                break;
            }
            for seq in seqs {
                plan.swap_blocks += self.kv.swap_in(seq).expect("blocks checked") as u64;
            }
            self.requests[r].phase = Phase::Running;
            plan.swapped_in.push(self.requests[r].request.id);
        }

        // 2. Decodes.
        let mut out_of_budget = false;
        for r in self.in_order(Phase::Running) {
            if self.requests[r].phase != Phase::Running {
                continue; // preempted as a victim earlier in this step
            }
            let decodes: Vec<usize> = self.requests[r]
                .live()
                .into_iter()
                .filter(|&s| self.work_of(r, s) == 1)
                .collect();
            if decodes.is_empty() {
                continue;
            }
            if decodes.len() as u64 > budget {
                out_of_budget = true;
                break;
            }
            let batch: Vec<(SeqId, usize)> = decodes.iter().map(|&s| (self.requests[r].seq(s), 1)).collect();
            let cost = self.kv.append_cost_batch(&batch).expect("live sequences");
            while cost > self.kv.available() {
                match self.victim(&plan, r) {
                    Some(v) => self.preempt(&mut plan, v),
                    None => break,
                }
            }
            if cost > self.kv.available() {
                self.preempt(&mut plan, r);
                continue;
            }
            for s in decodes {
                self.append_work(&mut plan, r, s, 1);
            }
            budget -= batch.len() as u64;
        }

        // 3. Prefill chunks.
        if !out_of_budget {
            'requests: for r in self.in_order(Phase::Running) {
                for s in self.requests[r].live() {
                    if self.requests[r].phase != Phase::Running {
                        continue 'requests;
                    }
                    let work = self.work_of(r, s);
                    if work <= 1 {
                        continue;
                    }
                    if budget == 0 {
                        break 'requests;
                    }
                    let mut chunk = work.min(budget);
                    if self.mutation == Some(SchedulerMutation::BudgetOverrun) && work > budget {
                        chunk += 1;
                    }
                    let seq = self.requests[r].seq(s);
                    let cost = self.kv.append_cost(seq, chunk as usize).expect("live sequence");
                    while cost > self.kv.available() {
                        match self.victim(&plan, r) {
                            Some(v) => self.preempt(&mut plan, v),
                            None => break,
                        }
                    }
                    if cost > self.kv.available() {
                        continue;
                    }
                    self.append_work(&mut plan, r, s, chunk);
                    budget = budget.saturating_sub(chunk);
                }
            }
        }

        // 4. Admission.
        if !out_of_budget && plan.preempted.is_empty() && self.in_order(Phase::Swapped).is_empty() {
            for r in self.in_order(Phase::Waiting) {
                if budget == 0 {
                    break;
                }
                let state = &self.requests[r];
                if self.running_seqs() + state.unfinished_count() > self.config.max_seqs {
                    break;
                }
                let carrier = state.carrier().expect("a waiting request has an unfinished sample");
                let target = state.target(carrier);
                let tokens: Vec<u32> = (0..target).map(|p| state.token(carrier, p)).collect();
                let (adopted, retained) = self.kv.peek_prefix(&tokens);
                let chunk = (target - adopted as u64).min(budget);
                let new_blocks = (chunk as usize).div_ceil(self.kv.config().block_size);
                if self.kv.available() < retained + new_blocks + watermark {
                    break;
                }
                let seq = state.seq(carrier);
                self.kv.start(seq).expect("waiting requests hold no sequence");
                let got = self.kv.adopt_prefix(seq, &tokens).expect("fresh sequence");
                debug_assert_eq!(got, adopted);
                // Adopted tokens are not processed, so they are never counted as recomputed.
                let peak = &mut self.requests[r].samples[carrier].peak_kv;
                *peak = (*peak).max(adopted as u64);
                self.requests[r].phase = Phase::Running;
                plan.admitted.push(self.requests[r].request.id);
                self.append_work(&mut plan, r, carrier, chunk);
                budget -= chunk;
            }
        }
        plan
    }

    /// Preempt the last running request (used when a step would be empty).
    pub fn break_stall(&mut self, plan: &mut StepPlan) -> bool {
        match self.in_order(Phase::Running).last() {
            Some(&r) => {
                self.preempt(plan, r);
                plan.stalls += 1;
                self.counters.stalls += 1;
                true
            }
            None => false,
        }
    }

    /// The work of a plan, for the step-time model.
    pub fn step_work(&self, plan: &StepPlan) -> StepWork {
        StepWork {
            seqs: plan
                .work
                .iter()
                .map(|w| {
                    let r = &self.requests[w.seq.request as usize];
                    SeqWork {
                        context: w.context,
                        tokens: w.tokens,
                        samples: w.context + w.tokens == r.known(w.seq.sample as usize),
                    }
                })
                .collect(),
            cow_blocks: plan.cow_blocks,
            swap_blocks: plan.swap_blocks,
        }
    }

    fn commit(&mut self, out: &mut StepOutcome, r: usize, sample: usize, now: u64) {
        let state = &mut self.requests[r];
        let k = state.samples[sample].committed;
        out.commits.push((state.seq(sample), k));
        state.samples[sample].committed += 1;
        state.samples[sample].last_commit_ns = now;
        state.first_token_ns.get_or_insert(now);
    }

    /// Apply the end of a step at time `now`.
    pub fn complete_step(&mut self, plan: &StepPlan, now: u64) -> Result<StepOutcome, KvError> {
        let mut out = StepOutcome::default();
        for w in &plan.work {
            let (r, s) = (w.seq.request as usize, w.seq.sample as usize);
            if self.requests[r].phase != Phase::Running {
                continue;
            }
            if w.context + w.tokens == self.requests[r].known(s) {
                self.commit(&mut out, r, s, now);
            }
        }
        // Forks at the end of the prompt, then finished samples.
        // Requests with work, in order of first appearance in the plan.
        let mut touched: Vec<usize> = Vec::new();
        for w in &plan.work {
            if !touched.contains(&(w.seq.request as usize)) {
                touched.push(w.seq.request as usize);
            }
        }
        for r in touched {
            if self.requests[r].phase != Phase::Running {
                continue;
            }
            let carrier = self.requests[r].carrier();
            if let Some(c) = carrier {
                let prompt = u64::from(self.requests[r].request.prompt_len);
                if !self.requests[r].forked && self.kv_len(r, c) == prompt {
                    let siblings: Vec<usize> = self.requests[r].unfinished().filter(|&s| s != c).collect();
                    for s in siblings {
                        self.kv.fork(self.requests[r].seq(c), self.requests[r].seq(s))?;
                        let peak = &mut self.requests[r].samples[s].peak_kv;
                        *peak = (*peak).max(prompt);
                        if self.requests[r].samples[s].committed == 0 {
                            self.commit(&mut out, r, s, now);
                        }
                    }
                    self.requests[r].forked = true;
                }
            }
            for s in self.requests[r].live() {
                let state = &self.requests[r];
                if state.samples[s].committed == state.request.output_len {
                    self.kv.free(state.seq(s))?;
                    self.requests[r].samples[s].finished = true;
                }
            }
            if self.requests[r].unfinished_count() == 0 {
                self.requests[r].phase = Phase::Finished;
                self.requests[r].finish_ns = Some(now);
                out.finished.push(self.requests[r].request.id);
            }
        }
        Ok(out)
    }
}

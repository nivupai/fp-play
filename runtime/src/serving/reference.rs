//! A slow, obvious reference for the scheduler and KV cache: the test oracle.
//!
//! It implements the policy written in [`super::scheduler`] and the cache
//! semantics written in [`super::kv`] from scratch, sharing no state or logic
//! with them:
//!
//! * blocks are a `BTreeMap` from block ID to tokens; reference counts are
//!   found by scanning every block table; a block in the prefix cache is
//!   identified by the *whole token prefix* it ends, compared as tokens, not
//!   by a hash;
//! * the cost of an append is measured by performing it on a copy of the
//!   cache and counting the blocks taken;
//! * the request order is re-sorted with plain loops whenever it is needed;
//! * time advances to the earlier of the next arrival and the running
//!   step's end, by scanning the requests.
//!
//! Only the step-time model and the workload's token functions are shared.
//! `tests/serving_engine.rs` requires its step trace (every decision, time
//! and commit) to equal the engine's exactly.

use std::collections::BTreeMap;

use super::engine::{FaultConfig, FaultRecord, StepRecord};
use super::kv::{KvConfig, SeqId};
use super::scheduler::{
    FaultOutcome, FaultPolicy, PreemptKind, PreemptionMode, SchedulerConfig, StepOutcome, StepPlan, Work,
};
use super::step_time::{SeqWork, StepTimer, StepWork};
use super::workload::Request;

#[derive(Clone)]
struct Block {
    tokens: Vec<u32>,
    /// The whole token prefix this block ends, if it is in the prefix cache.
    prefix: Option<Vec<u32>>,
    released_at: u64,
}

#[derive(Clone)]
struct Kv {
    b: usize,
    capacity: usize,
    caching: bool,
    swap_capacity: usize,
    /// Blocks that hold tokens (referenced or retained).
    blocks: BTreeMap<u32, Block>,
    /// Unused blocks; the last element is handed out first.
    free: Vec<u32>,
    tables: BTreeMap<SeqId, Vec<u32>>,
    swapped: BTreeMap<SeqId, Vec<Vec<u32>>>,
    tick: u64,
    taken: usize,
    /// On a copy used to measure a cost: hand out virtual blocks when the
    /// pool runs out, so the count can exceed what is available.
    unbounded: bool,
}

impl Kv {
    fn new(c: KvConfig) -> Kv {
        Kv {
            b: c.block_size,
            capacity: c.device_blocks,
            caching: c.prefix_caching,
            swap_capacity: c.swap_blocks,
            blocks: BTreeMap::new(),
            free: (0..c.device_blocks as u32).rev().collect(),
            tables: BTreeMap::new(),
            swapped: BTreeMap::new(),
            tick: 0,
            taken: 0,
            unbounded: false,
        }
    }

    fn refcount(&self, id: u32) -> usize {
        self.tables
            .values()
            .map(|t| t.iter().filter(|&&b| b == id).count())
            .sum()
    }

    fn retained(&self) -> Vec<u32> {
        self.blocks
            .iter()
            .filter(|(id, blk)| blk.prefix.is_some() && self.refcount(**id) == 0)
            .map(|(id, _)| *id)
            .collect()
    }

    fn available(&self) -> usize {
        self.free.len() + self.retained().len()
    }

    fn swap_used(&self) -> usize {
        self.swapped.values().map(Vec::len).sum()
    }

    fn len(&self, seq: SeqId) -> usize {
        self.tables[&seq].iter().map(|b| self.blocks[b].tokens.len()).sum()
    }

    /// The sequence's tokens, block by block.
    fn tokens(&self, seq: SeqId) -> Vec<u32> {
        self.tables[&seq]
            .iter()
            .flat_map(|b| self.blocks[b].tokens.iter().copied())
            .collect()
    }

    fn take(&mut self) -> u32 {
        self.taken += 1;
        if let Some(id) = self.free.pop() {
            self.blocks.insert(
                id,
                Block {
                    tokens: Vec::new(),
                    prefix: None,
                    released_at: 0,
                },
            );
            return id;
        }
        // Evict the retained block released earliest, lower ID first.
        let mut best: Option<u32> = None;
        for id in self.retained() {
            let better = match best {
                None => true,
                Some(b) => (self.blocks[&id].released_at, id) < (self.blocks[&b].released_at, b),
            };
            if better {
                best = Some(id);
            }
        }
        let id = match best {
            Some(id) => id,
            None if self.unbounded => u32::MAX - self.taken as u32,
            None => panic!("no block available: the caller must check first"),
        };
        self.blocks.insert(
            id,
            Block {
                tokens: Vec::new(),
                prefix: None,
                released_at: 0,
            },
        );
        id
    }

    /// After a table lost its references to `ids`, free the unreferenced ones.
    fn release_all(&mut self, ids: &[u32]) {
        self.tick += 1;
        for &id in ids {
            if self.refcount(id) == 0 {
                if self.blocks[&id].prefix.is_some() {
                    self.blocks.get_mut(&id).expect("held").released_at = self.tick;
                } else {
                    self.blocks.remove(&id);
                    self.free.push(id);
                }
            }
        }
    }

    fn append(&mut self, seq: SeqId, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        let b = self.b;
        if let Some(&last) = self.tables[&seq].last() {
            if self.blocks[&last].tokens.len() < b && self.refcount(last) > 1 {
                let copy = self.take();
                let content = self.blocks[&last].tokens.clone();
                self.blocks.get_mut(&copy).expect("taken").tokens = content;
                let table = self.tables.get_mut(&seq).expect("known");
                *table.last_mut().expect("non-empty") = copy;
            }
        }
        for &t in tokens {
            let need_new = match self.tables[&seq].last() {
                None => true,
                Some(last) => self.blocks[last].tokens.len() == b,
            };
            if need_new {
                let id = self.take();
                self.tables.get_mut(&seq).expect("known").push(id);
            }
            let last = *self.tables[&seq].last().expect("ensured");
            self.blocks.get_mut(&last).expect("held").tokens.push(t);
            if self.caching && self.blocks[&last].tokens.len() == b {
                let prefix = self.tokens(seq);
                if !self.blocks.values().any(|blk| blk.prefix.as_ref() == Some(&prefix)) {
                    self.blocks.get_mut(&last).expect("held").prefix = Some(prefix);
                }
            }
        }
    }

    /// Blocks appending would take: measured on a copy.
    fn cost(&self, work: &[(SeqId, Vec<u32>)]) -> usize {
        let mut copy = self.clone();
        copy.taken = 0;
        copy.unbounded = true;
        for (seq, tokens) in work {
            copy.append(*seq, tokens);
        }
        copy.taken
    }

    /// Registered blocks matching the leading full blocks of `tokens`,
    /// leaving at least one token.
    fn prefix_match(&self, tokens: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        if !self.caching {
            return out;
        }
        let eligible = tokens.len().saturating_sub(1) / self.b;
        for i in 1..=eligible {
            let want = &tokens[..i * self.b];
            match self.blocks.iter().find(|(_, blk)| blk.prefix.as_deref() == Some(want)) {
                Some((id, _)) => out.push(*id),
                None => break,
            }
        }
        out
    }

    fn free_seq(&mut self, seq: SeqId) {
        let ids = self.tables.remove(&seq).expect("on device");
        self.release_all(&ids);
    }

    fn swap_out(&mut self, seq: SeqId) -> usize {
        let ids = self.tables.remove(&seq).expect("on device");
        let contents = ids.iter().map(|b| self.blocks[b].tokens.clone()).collect();
        self.swapped.insert(seq, contents);
        self.release_all(&ids);
        ids.len()
    }

    fn swap_in(&mut self, seq: SeqId) -> usize {
        let contents = self.swapped.remove(&seq).expect("swapped");
        let mut ids = Vec::new();
        for c in &contents {
            let id = self.take();
            self.blocks.get_mut(&id).expect("taken").tokens = c.clone();
            ids.push(id);
        }
        self.tables.insert(seq, ids);
        contents.len()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum St {
    Pending,
    Waiting,
    Running,
    Swapped,
    Finished,
    Rejected,
    Failed,
}

struct Req {
    r: Request,
    st: St,
    committed: Vec<u32>,
    done: Vec<bool>,
    forked: bool,
    admitted: bool,
    retries: u32,
}

impl Req {
    fn seq(&self, s: usize) -> SeqId {
        SeqId {
            request: self.r.id,
            sample: s as u32,
        }
    }

    fn unfinished(&self) -> Vec<usize> {
        (0..self.done.len()).filter(|&s| !self.done[s]).collect()
    }

    fn live(&self) -> Vec<usize> {
        let u = self.unfinished();
        if self.forked {
            u
        } else {
            u.into_iter().take(1).collect()
        }
    }

    fn known(&self, s: usize) -> u64 {
        u64::from(self.r.prompt_len + self.committed[s])
    }

    fn target(&self, s: usize) -> u64 {
        if self.forked {
            self.known(s)
        } else {
            u64::from(self.r.prompt_len)
        }
    }
}

struct Sim<'a> {
    cfg: SchedulerConfig,
    kv: Kv,
    reqs: Vec<Req>,
    timer: &'a dyn StepTimer,
}

impl Sim<'_> {
    /// Indices of requests in `st`, in order (priority desc, arrival, id), by
    /// selection: repeatedly take the smallest remaining.
    fn ordered(&self, st: St) -> Vec<usize> {
        let mut rest: Vec<usize> = (0..self.reqs.len()).filter(|&i| self.reqs[i].st == st).collect();
        let mut out = Vec::new();
        while !rest.is_empty() {
            let mut best = 0;
            for j in 1..rest.len() {
                let (a, b) = (&self.reqs[rest[j]].r, &self.reqs[rest[best]].r);
                let before = a.priority > b.priority
                    || (a.priority == b.priority
                        && (a.arrival_ns < b.arrival_ns || (a.arrival_ns == b.arrival_ns && a.id < b.id)));
                if before {
                    best = j;
                }
            }
            out.push(rest.remove(best));
        }
        out
    }

    fn running_seqs(&self) -> usize {
        self.reqs
            .iter()
            .filter(|r| r.st == St::Running)
            .map(|r| r.unfinished().len())
            .sum()
    }

    fn kv_len(&self, i: usize, s: usize) -> u64 {
        let seq = self.reqs[i].seq(s);
        if self.kv.tables.contains_key(&seq) {
            self.kv.len(seq) as u64
        } else {
            0
        }
    }

    fn tokens(&self, i: usize, s: usize, from: u64, n: u64) -> Vec<u32> {
        (from..from + n)
            .map(|p| self.reqs[i].r.sequence_token(s as u32, p as u32))
            .collect()
    }

    fn preempt(&mut self, plan: &mut StepPlan, i: usize) {
        let seqs: Vec<SeqId> = self.reqs[i].live().iter().map(|&s| self.reqs[i].seq(s)).collect();
        let mut kind = if self.cfg.preemption == PreemptionMode::Swap {
            PreemptKind::Swap
        } else {
            PreemptKind::Recompute
        };
        if kind == PreemptKind::Swap {
            let need: usize = seqs.iter().map(|s| self.kv.tables[s].len()).sum();
            if self.kv.swap_used() + need > self.kv.swap_capacity {
                kind = PreemptKind::SwapFallback;
            }
        }
        for seq in seqs {
            if kind == PreemptKind::Swap {
                plan.swap_blocks += self.kv.swap_out(seq) as u64;
            } else {
                self.kv.free_seq(seq);
            }
        }
        let r = &mut self.reqs[i];
        if kind == PreemptKind::Swap {
            r.st = St::Swapped;
        } else {
            r.st = St::Waiting;
            r.forked = r.unfinished().len() <= 1;
        }
        plan.preempted.push((r.r.id, kind));
    }

    fn victim(&self, plan: &StepPlan, except: usize) -> Option<usize> {
        let mut order = self.ordered(St::Running);
        order.reverse();
        order
            .into_iter()
            .find(|&i| i != except && !plan.work.iter().any(|w| w.seq.request == self.reqs[i].r.id))
    }

    fn add_work(&mut self, plan: &mut StepPlan, i: usize, s: usize, n: u64) {
        let seq = self.reqs[i].seq(s);
        let context = self.kv_len(i, s);
        let shared_partial = self.kv.tables[&seq]
            .last()
            .is_some_and(|last| self.kv.blocks[last].tokens.len() < self.kv.b && self.kv.refcount(*last) > 1);
        let toks = self.tokens(i, s, context, n);
        self.kv.append(seq, &toks);
        plan.cow_blocks += u64::from(shared_partial);
        plan.work.push(Work {
            seq,
            context,
            tokens: n,
        });
    }

    fn build(&mut self) -> StepPlan {
        let mut plan = StepPlan::default();
        let mut budget = self.cfg.token_budget as u64;
        let wm = self.cfg.watermark_blocks;
        for i in self.ordered(St::Swapped) {
            let seqs: Vec<SeqId> = self.reqs[i].live().iter().map(|&s| self.reqs[i].seq(s)).collect();
            let need: usize = seqs.iter().map(|s| self.kv.swapped[s].len()).sum();
            if self.running_seqs() + seqs.len() > self.cfg.max_seqs || self.kv.available() < need + wm {
                break;
            }
            for seq in seqs {
                plan.swap_blocks += self.kv.swap_in(seq) as u64;
            }
            self.reqs[i].st = St::Running;
            plan.swapped_in.push(self.reqs[i].r.id);
        }
        let mut stop = false;
        for i in self.ordered(St::Running) {
            if self.reqs[i].st != St::Running {
                continue;
            }
            let decodes: Vec<usize> = self.reqs[i]
                .live()
                .into_iter()
                .filter(|&s| self.reqs[i].target(s) - self.kv_len(i, s) == 1)
                .collect();
            if decodes.is_empty() {
                continue;
            }
            if decodes.len() as u64 > budget {
                stop = true;
                break;
            }
            let work: Vec<(SeqId, Vec<u32>)> = decodes
                .iter()
                .map(|&s| (self.reqs[i].seq(s), self.tokens(i, s, self.kv_len(i, s), 1)))
                .collect();
            let cost = self.kv.cost(&work);
            while cost > self.kv.available() {
                match self.victim(&plan, i) {
                    Some(v) => self.preempt(&mut plan, v),
                    None => break,
                }
            }
            if cost > self.kv.available() {
                self.preempt(&mut plan, i);
                continue;
            }
            for s in decodes {
                self.add_work(&mut plan, i, s, 1);
                budget -= 1;
            }
        }
        if !stop {
            'outer: for i in self.ordered(St::Running) {
                for s in self.reqs[i].live() {
                    if self.reqs[i].st != St::Running {
                        continue 'outer;
                    }
                    let work = self.reqs[i].target(s) - self.kv_len(i, s);
                    if work <= 1 {
                        continue;
                    }
                    if budget == 0 {
                        break 'outer;
                    }
                    let n = work.min(budget);
                    let seq = self.reqs[i].seq(s);
                    let cost = self.kv.cost(&[(seq, self.tokens(i, s, self.kv_len(i, s), n))]);
                    while cost > self.kv.available() {
                        match self.victim(&plan, i) {
                            Some(v) => self.preempt(&mut plan, v),
                            None => break,
                        }
                    }
                    if cost > self.kv.available() {
                        continue;
                    }
                    self.add_work(&mut plan, i, s, n);
                    budget -= n;
                }
            }
        }
        if !stop && plan.preempted.is_empty() && self.ordered(St::Swapped).is_empty() {
            for i in self.ordered(St::Waiting) {
                if budget == 0 || self.running_seqs() + self.reqs[i].unfinished().len() > self.cfg.max_seqs {
                    break;
                }
                let c = self.reqs[i].unfinished()[0];
                let target = self.reqs[i].target(c);
                let toks = self.tokens(i, c, 0, target);
                let hits = self.kv.prefix_match(&toks);
                let retained = hits.iter().filter(|&&b| self.kv.refcount(b) == 0).count();
                let adopted = (hits.len() * self.kv.b) as u64;
                let n = (target - adopted).min(budget);
                if self.kv.available() < retained + (n as usize).div_ceil(self.kv.b) + wm {
                    break;
                }
                let seq = self.reqs[i].seq(c);
                self.kv.tables.insert(seq, hits);
                self.reqs[i].st = St::Running;
                self.reqs[i].admitted = true;
                plan.admitted.push(self.reqs[i].r.id);
                self.add_work(&mut plan, i, c, n);
                budget -= n;
            }
        }
        plan
    }

    fn complete(&mut self, plan: &StepPlan) -> StepOutcome {
        let mut out = StepOutcome::default();
        let mut order: Vec<usize> = Vec::new();
        for w in &plan.work {
            let (i, s) = (w.seq.request as usize, w.seq.sample as usize);
            if !order.contains(&i) {
                order.push(i);
            }
            if w.context + w.tokens == self.reqs[i].known(s) {
                out.commits.push((w.seq, self.reqs[i].committed[s]));
                self.reqs[i].committed[s] += 1;
            }
        }
        for i in order {
            let c = self.reqs[i].unfinished()[0];
            if !self.reqs[i].forked && self.kv_len(i, c) == u64::from(self.reqs[i].r.prompt_len) {
                let blocks = self.kv.tables[&self.reqs[i].seq(c)].clone();
                for s in self.reqs[i].unfinished() {
                    if s == c {
                        continue;
                    }
                    self.kv.tables.insert(self.reqs[i].seq(s), blocks.clone());
                    if self.reqs[i].committed[s] == 0 {
                        out.commits.push((self.reqs[i].seq(s), 0));
                        self.reqs[i].committed[s] = 1;
                    }
                }
                self.reqs[i].forked = true;
            }
            for s in self.reqs[i].live() {
                if self.reqs[i].committed[s] == self.reqs[i].r.output_len {
                    self.kv.free_seq(self.reqs[i].seq(s));
                    self.reqs[i].done[s] = true;
                }
            }
            if self.reqs[i].unfinished().is_empty() {
                self.reqs[i].st = St::Finished;
                out.finished.push(self.reqs[i].r.id);
            }
        }
        out
    }

    /// A rank failed: forget every device block, then apply the policy.
    fn fault(&mut self, policy: FaultPolicy, retry_limit: u32, draining: &mut bool) -> FaultOutcome {
        let mut out = FaultOutcome::default();
        self.kv.tables.clear();
        self.kv.blocks.clear();
        self.kv.free = (0..self.kv.capacity as u32).rev().collect();
        for i in self.ordered(St::Running) {
            if policy == FaultPolicy::Abort || self.reqs[i].retries >= retry_limit {
                self.reqs[i].st = St::Failed;
                out.failed.push(self.reqs[i].r.id);
            } else {
                let r = &mut self.reqs[i];
                r.retries += 1;
                r.st = St::Waiting;
                r.forked = r.unfinished().len() <= 1;
                out.retried.push(r.r.id);
            }
        }
        if policy == FaultPolicy::Drain {
            *draining = true;
            for i in self.ordered(St::Waiting) {
                if !self.reqs[i].admitted {
                    self.reqs[i].st = St::Rejected;
                    out.rejected.push(self.reqs[i].r.id);
                }
            }
        }
        out
    }

    fn step_work(&self, plan: &StepPlan) -> StepWork {
        StepWork {
            seqs: plan
                .work
                .iter()
                .map(|w| {
                    let r = &self.reqs[w.seq.request as usize];
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
}

/// Run the reference over `requests` and return its step trace.
/// `used_blocks` in each record is the device blocks referenced at step end.
pub fn run(
    requests: &[Request],
    scheduler: SchedulerConfig,
    kv: KvConfig,
    faults: &FaultConfig,
    timer: &dyn StepTimer,
) -> Vec<StepRecord> {
    let b = kv.block_size as u64;
    let mut sim = Sim {
        cfg: scheduler,
        kv: Kv::new(kv),
        reqs: requests
            .iter()
            .map(|r| Req {
                r: r.clone(),
                st: St::Pending,
                committed: vec![0; r.samples as usize],
                done: vec![false; r.samples as usize],
                forked: r.samples <= 1,
                admitted: false,
                retries: 0,
            })
            .collect(),
        timer,
    };
    let mut records = Vec::new();
    let mut running: Option<(StepPlan, u64, u64, super::step_time::StepTime)> = None;
    let mut draining = false;
    let fault_at = |step: usize| faults.plan.iter().copied().find(|f| f.step == step as u64);
    loop {
        let next_arrival = sim
            .reqs
            .iter()
            .filter(|r| r.st == St::Pending)
            .map(|r| r.r.arrival_ns)
            .min();
        let now = match (next_arrival, &running) {
            (None, None) => break,
            (Some(a), None) => a,
            (None, Some((_, _, end, _))) => *end,
            (Some(a), Some((_, _, end, _))) => a.min(*end),
        };
        for i in 0..sim.reqs.len() {
            if sim.reqs[i].st == St::Pending && sim.reqs[i].r.arrival_ns == now {
                let r = &sim.reqs[i].r;
                let need = u64::from(r.prompt_len + r.output_len - 1).div_ceil(b) * u64::from(r.samples);
                let fits = need as usize + scheduler.watermark_blocks <= kv.device_blocks
                    && r.samples as usize <= scheduler.max_seqs;
                sim.reqs[i].st = if fits && !draining { St::Waiting } else { St::Rejected };
            }
        }
        if running.as_ref().is_some_and(|(_, _, end, _)| *end == now) {
            let (plan, start, end, time) = running.take().expect("checked");
            let (outcome, fault) = match fault_at(records.len()) {
                Some(f) => {
                    let o = sim.fault(faults.policy, faults.retry_limit, &mut draining);
                    (
                        StepOutcome::default(),
                        Some(FaultRecord {
                            layer: f.layer,
                            allreduce: f.allreduce,
                            outcome: o,
                        }),
                    )
                }
                None => (sim.complete(&plan), None),
            };
            let used = (0..kv.device_blocks as u32)
                .filter(|&id| sim.kv.refcount(id) > 0)
                .count();
            records.push(StepRecord {
                index: records.len() as u64,
                start_ns: start,
                end_ns: end,
                plan,
                outcome,
                time,
                used_blocks: used,
                fault,
            });
        }
        if running.is_none() {
            let mut plan = sim.build();
            while plan.work.is_empty() && sim.running_seqs() > 0 {
                let last = *sim.ordered(St::Running).last().expect("running");
                let mut stalled = plan;
                sim.preempt(&mut stalled, last);
                stalled.stalls += 1;
                let next = sim.build();
                stalled.swapped_in.extend(next.swapped_in);
                stalled.admitted.extend(next.admitted);
                stalled.preempted.extend(next.preempted);
                stalled.stalls += next.stalls;
                stalled.work = next.work;
                stalled.cow_blocks += next.cow_blocks;
                stalled.swap_blocks += next.swap_blocks;
                plan = stalled;
            }
            if !plan.work.is_empty() {
                let time = sim.timer.time(&sim.step_work(&plan));
                let duration = match fault_at(records.len()) {
                    Some(f) => faults.failed_step_ns(&time, f),
                    None => time.total_ns,
                };
                running = Some((plan, now, now + duration, time));
            }
        }
    }
    records
}

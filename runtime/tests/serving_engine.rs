//! R2: the scheduler and engine.
//!
//! * A three-request trace derived by hand from the written policy.
//! * Hundreds of small random workloads: every invariant after every step,
//!   and the step trace equal to the slow reference scheduler's, decision by
//!   decision, under both preemption modes, with and without prefix caching.

use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::engine::{run, EngineConfig, FaultConfig, StepRecord};
use fp_play_runtime::serving::kv::{KvConfig, SeqId};
use fp_play_runtime::serving::metrics::{metrics, nearest_rank};
use fp_play_runtime::serving::reference;
use fp_play_runtime::serving::scheduler::{Phase, PreemptionMode, SchedulerConfig, Work};
use fp_play_runtime::serving::step_time::LinearTimer;
use fp_play_runtime::serving::workload::{LengthDist, Request, WorkloadConfig};

fn sq(request: u32, sample: u32) -> SeqId {
    SeqId { request, sample }
}

fn work(request: u32, sample: u32, context: u64, tokens: u64) -> Work {
    Work {
        seq: sq(request, sample),
        context,
        tokens,
    }
}

fn request(id: u32, arrival_ns: u64, prompt_len: u32, output_len: u32, samples: u32) -> Request {
    Request {
        id,
        arrival_ns,
        priority: 0,
        prefix_group: None,
        prefix_len: 0,
        prompt_len,
        output_len,
        samples,
    }
}

const TIMER: LinearTimer = LinearTimer {
    base_ns: 100,
    per_token_ns: 10,
};

/// Worked by hand from the policy in `serving::scheduler` (B = 4, 8 blocks,
/// budget 6, at most 4 sequences, 100 ns + 10 ns per token):
///
/// * step 0 (0–150): r0 admitted, its whole 5-token prompt; commits r0 #0.
/// * step 1 (150–310): r0 decodes; r1 (2 samples) is admitted with its
///   3-token prompt; r2 gets the last 2 tokens of budget. r0 finishes; r1's
///   carrier commits #0, forks, and sample 1 commits #0 from the same logits.
/// * step 2 (310–470): both r1 samples decode; sample 0 copies the shared
///   partial block (one copy-on-write), sample 1 then owns it; r2 prefills 4.
/// * step 3 (470–610): r1 decodes into two new blocks; r2's last 2 prompt
///   tokens commit its only output. Both finish.
#[test]
fn hand_derived_three_request_trace() {
    let requests = vec![request(0, 0, 5, 2, 1), request(1, 50, 3, 3, 2), request(2, 60, 8, 1, 1)];
    let config = EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: 6,
            max_seqs: 4,
            watermark_blocks: 0,
            preemption: PreemptionMode::Recompute,
        },
        kv: KvConfig {
            block_size: 4,
            device_blocks: 8,
            swap_blocks: 0,
            prefix_caching: false,
        },
        faults: FaultConfig::none(),
        kv_check_interval: 1,
    };
    let result = run(&requests, &config, &TIMER).unwrap();
    let s = &result.steps;
    assert_eq!(s.len(), 4);
    let spans: Vec<(u64, u64)> = s.iter().map(|r| (r.start_ns, r.end_ns)).collect();
    assert_eq!(spans, vec![(0, 150), (150, 310), (310, 470), (470, 610)]);
    assert_eq!(s[0].plan.admitted, vec![0]);
    assert_eq!(s[0].plan.work, vec![work(0, 0, 0, 5)]);
    assert_eq!(s[0].outcome.commits, vec![(sq(0, 0), 0)]);
    assert_eq!(s[1].plan.admitted, vec![1, 2]);
    assert_eq!(
        s[1].plan.work,
        vec![work(0, 0, 5, 1), work(1, 0, 0, 3), work(2, 0, 0, 2)]
    );
    assert_eq!(s[1].outcome.commits, vec![(sq(0, 0), 1), (sq(1, 0), 0), (sq(1, 1), 0)]);
    assert_eq!(s[1].outcome.finished, vec![0]);
    assert_eq!(
        s[2].plan.work,
        vec![work(1, 0, 3, 1), work(1, 1, 3, 1), work(2, 0, 2, 4)]
    );
    assert_eq!(s[2].plan.cow_blocks, 1);
    assert_eq!(s[2].outcome.commits, vec![(sq(1, 0), 1), (sq(1, 1), 1)]);
    assert_eq!(
        s[3].plan.work,
        vec![work(1, 0, 4, 1), work(1, 1, 4, 1), work(2, 0, 6, 2)]
    );
    assert_eq!(s[3].outcome.commits, vec![(sq(1, 0), 2), (sq(1, 1), 2), (sq(2, 0), 0)]);
    assert_eq!(s[3].outcome.finished, vec![1, 2]);
    let used: Vec<usize> = s.iter().map(|r| r.used_blocks).collect();
    assert_eq!(used, vec![2, 2, 4, 0]);
    // TTFT: r0 150 − 0, r1 310 − 50, r2 610 − 60.
    let firsts: Vec<u64> = result.requests.iter().map(|r| r.first_token_ns.unwrap()).collect();
    assert_eq!(firsts, vec![150, 310, 610]);
    let m = metrics(&result);
    assert_eq!(m.output_tokens, 2 + 3 * 2 + 1);
    assert_eq!(m.ttft.unwrap().p50, 260);
    assert_eq!(m.cow_copies, 1);
    assert_eq!(
        reference::run(&requests, config.scheduler, config.kv, &config.faults, &TIMER),
        result.steps
    );
}

#[test]
fn nearest_rank_percentiles() {
    let v: Vec<u64> = (1..=10).collect();
    assert_eq!(nearest_rank(&v, 50), 5);
    assert_eq!(nearest_rank(&v, 99), 10);
    assert_eq!(nearest_rank(&[7], 99), 7);
    let w: Vec<u64> = (1..=200).collect();
    assert_eq!(nearest_rank(&w, 99), 198);
}

/// A small random configuration and workload.
fn random_case(seed: u64) -> (Vec<Request>, EngineConfig) {
    let mut rng = SplitMix64::new(seed);
    let block_size = [1, 2, 4, 8][rng.below(4) as usize];
    let device_blocks = 24 + rng.below(80) as usize;
    let workload = WorkloadConfig {
        seed: rng.next_u64(),
        requests: 6 + rng.below(30) as usize,
        rate_per_s: 2e5 + rng.below(2_000_000) as f64,
        prompt: LengthDist::Uniform {
            lo: 1,
            hi: 4 + rng.below(40) as u32,
        },
        output: LengthDist::Uniform {
            lo: 1,
            hi: 2 + rng.below(30) as u32,
        },
        prefix_groups: 2,
        prefix_len: 3 + rng.below(12) as u32,
        prefix_permille: rng.below(900) as u32,
        samples: vec![(2, 1), (1, 2), (1, 4)],
        priorities: vec![(3, 0), (1, 1)],
    };
    let config = EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: 4 + rng.below(40) as usize,
            max_seqs: 4 + rng.below(12) as usize,
            watermark_blocks: rng.below(3) as usize,
            preemption: if rng.below(2) == 0 {
                PreemptionMode::Recompute
            } else {
                PreemptionMode::Swap
            },
        },
        kv: KvConfig {
            block_size,
            device_blocks,
            swap_blocks: rng.below(60) as usize,
            prefix_caching: rng.below(3) != 0,
        },
        faults: FaultConfig::none(),
        kv_check_interval: 1,
    };
    (workload.generate().unwrap(), config)
}

fn timer_for(seed: u64) -> LinearTimer {
    LinearTimer {
        base_ns: 1_000 + seed % 7,
        per_token_ns: 10 + seed % 5,
    }
}

#[test]
fn random_workloads_keep_every_invariant_and_match_the_reference() {
    let mut totals = (0u64, 0u64, 0u64, 0u64, 0u64);
    for seed in 0..300 {
        let (requests, config) = random_case(seed);
        let timer = timer_for(seed);
        let result = run(&requests, &config, &timer).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        for r in &result.requests {
            assert!(matches!(r.phase, Phase::Finished | Phase::Rejected), "seed {seed}");
        }
        let expected = reference::run(&requests, config.scheduler, config.kv, &config.faults, &timer);
        if let Some(i) = (0..result.steps.len().max(expected.len())).find(|&i| result.steps.get(i) != expected.get(i)) {
            panic!(
                "seed {seed}: step {i} differs from the reference\nengine:    {:?}\nreference: {:?}",
                result.steps.get(i),
                expected.get(i)
            );
        }
        let m = metrics(&result);
        totals.0 += m.preemptions_recompute + m.preemptions_swap;
        totals.1 += m.cow_copies;
        totals.2 += m.prefix_hit_blocks;
        totals.3 += m.swapped_blocks;
        totals.4 += m.swap_fallbacks;
    }
    // The random cases exercise preemption, copy-on-write, prefix hits, swap and its fallback.
    assert!(
        totals.0 > 100 && totals.1 > 100 && totals.2 > 100 && totals.3 > 100 && totals.4 > 0,
        "{totals:?}"
    );
}

#[test]
fn runs_are_deterministic() {
    let (requests, config) = random_case(7);
    let a = run(&requests, &config, &TIMER).unwrap();
    let b = run(&requests, &config, &TIMER).unwrap();
    assert_eq!(a.steps, b.steps);
}

#[test]
fn step_records_are_well_formed() {
    let (requests, config) = random_case(11);
    let result = run(&requests, &config, &TIMER).unwrap();
    let mut last_end = 0;
    for (i, s) in result.steps.iter().enumerate() {
        let r: &StepRecord = s;
        assert_eq!(r.index, i as u64);
        assert!(r.start_ns >= last_end && r.end_ns > r.start_ns);
        assert!(r.plan.tokens() <= config.scheduler.token_budget as u64);
        last_end = r.end_ns;
    }
}

// ---------------------------------------------------------------------------
// Planted bugs (compiled only with `--features planted-bugs`)
// ---------------------------------------------------------------------------

#[cfg(feature = "planted-bugs")]
mod planted {
    use super::*;
    use fp_play_runtime::serving::engine::{run_planted, EngineError, Planted};
    use fp_play_runtime::serving::kv::Mutation;
    use fp_play_runtime::serving::scheduler::SchedulerMutation;

    /// Which check first caught a planted bug on the random cases.
    #[derive(Debug, PartialEq, Eq)]
    enum Caught {
        /// An engine invariant; the message starts with this prefix.
        Invariant(String),
        /// Every invariant held, but the trace differs from the reference.
        Reference,
    }

    fn first_catch(planted: Planted) -> Option<(u64, Caught)> {
        for seed in 0..300 {
            let (requests, config) = random_case(seed);
            let timer = timer_for(seed);
            match run_planted(&requests, &config, &timer, planted) {
                Err(EngineError::Invariant { what, .. }) => return Some((seed, Caught::Invariant(what))),
                Err(e) => panic!("seed {seed}: {e}"),
                Ok(result) => {
                    if result.steps != reference::run(&requests, config.scheduler, config.kv, &config.faults, &timer) {
                        return Some((seed, Caught::Reference));
                    }
                }
            }
        }
        None
    }

    #[test]
    fn every_planted_bug_is_caught_inside_the_engine() {
        let expect = |s: &str| Caught::Invariant(s.to_string());
        let cases = [
            (Planted::Kv(Mutation::LeakOnFree), expect("kv: Refcount")),
            (Planted::Kv(Mutation::SkipCow), expect("kv: Content")),
            (Planted::Kv(Mutation::SwapOutKeepsSharedRef), expect("kv: Refcount")),
            (Planted::Kv(Mutation::EarlyBlockFull), expect("kv: Shape")),
            (Planted::Kv(Mutation::StalePrefixKey), expect("kv: PrefixCache")),
            (
                Planted::Scheduler(SchedulerMutation::BudgetOverrun),
                expect("scheduler:"),
            ),
            (Planted::Scheduler(SchedulerMutation::VictimFirst), Caught::Reference),
        ];
        for (planted, want) in cases {
            let (seed, caught) = first_catch(planted).unwrap_or_else(|| panic!("{planted:?} was not caught"));
            eprintln!("{planted:?}: caught at seed {seed} by {caught:?}");
            match (&caught, &want) {
                (Caught::Invariant(got), Caught::Invariant(w)) => {
                    assert!(got.starts_with(w.as_str()), "{planted:?}: {got}");
                    if w == "scheduler:" {
                        assert!(got.contains("over the budget"), "{got}");
                    }
                }
                _ => assert_eq!(caught, want, "{planted:?}"),
            }
        }
    }
}

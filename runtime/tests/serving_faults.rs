//! R2c: rank failures, the three policies, and guarantees G1–G5.

use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::engine::{run, EngineConfig, EngineError, Fault, FaultConfig};
use fp_play_runtime::serving::kv::{KvConfig, SeqId};
use fp_play_runtime::serving::reference;
use fp_play_runtime::serving::scheduler::{FaultPolicy, Phase, PreemptionMode, SchedulerConfig};
use fp_play_runtime::serving::step_time::LinearTimer;
use fp_play_runtime::serving::traces::{three_requests_failing, to_jsonl, LINEAR};
use fp_play_runtime::serving::workload::{LengthDist, Request, WorkloadConfig};

fn sq(request: u32, sample: u32) -> SeqId {
    SeqId { request, sample }
}

/// The hand-derived three-request case with a failure in step 2, at layer 0's
/// second AllReduce of a 2-layer model, 1000 ns detection and 2000 ns recovery.
fn failing(policy: FaultPolicy) -> (Vec<Request>, EngineConfig) {
    three_requests_failing(policy)
}

/// Derived by hand from the written policy (see `tests/serving_engine.rs` for
/// steps 0 and 1). Step 2 (310 ns) runs 6 tokens: 100 ns overhead + 60 ns
/// compute; it fails after (2·0 + 1 + 1) ÷ (2·2) = half its compute, so it is
/// charged 100 + 30 + 1000 + 2000 = 3130 ns and ends at 3440 with no commits.
/// Under retry, r1 and r2 (running, in order) go back to waiting with their
/// committed tokens: r1's samples hold 1 token each, r2 none.
/// * step 3 (3440–3600): r1's carrier re-prefills the 3-token prompt and r2
///   the first 3 of its 8; r1 forks (its siblings already hold token 0, so
///   nothing is committed).
/// * step 4 (3600–3760): both r1 samples recompute token 0's KV (one
///   copy-on-write) and commit token 1, as in the run without a failure; r2
///   prefills 4.
/// * step 5 (3760–3890): r1 commits token 2 for both samples, r2 its only
///   token; both finish.
#[test]
fn hand_derived_retry() {
    let (requests, config) = failing(FaultPolicy::Retry);
    let result = run(&requests, &config, &LINEAR).unwrap();
    let s = &result.steps;
    assert_eq!(s.len(), 6);
    let fault = s[2].fault.as_ref().expect("step 2 failed");
    assert_eq!((fault.layer, fault.allreduce), (0, 1));
    assert_eq!(fault.outcome.retried, vec![1, 2]);
    assert!(fault.outcome.failed.is_empty() && fault.outcome.rejected.is_empty());
    assert_eq!((s[2].start_ns, s[2].end_ns), (310, 3440));
    assert!(s[2].outcome.commits.is_empty());
    assert_eq!(s[2].used_blocks, 0);
    let spans: Vec<(u64, u64)> = s[3..].iter().map(|r| (r.start_ns, r.end_ns)).collect();
    assert_eq!(spans, vec![(3440, 3600), (3600, 3760), (3760, 3890)]);
    assert_eq!(s[3].plan.admitted, vec![1, 2]);
    assert!(s[3].outcome.commits.is_empty());
    assert_eq!(s[4].plan.cow_blocks, 1);
    assert_eq!(s[4].outcome.commits, vec![(sq(1, 0), 1), (sq(1, 1), 1)]);
    assert_eq!(s[5].outcome.commits, vec![(sq(1, 0), 2), (sq(1, 1), 2), (sq(2, 0), 0)]);
    assert_eq!(s[5].outcome.finished, vec![1, 2]);
    assert!(result
        .requests
        .iter()
        .all(|r| r.phase == Phase::Finished && r.endings == 1));
    assert_eq!(
        reference::run(&requests, config.scheduler, config.kv, &config.faults, &LINEAR),
        result.steps
    );
}

#[test]
fn hand_derived_abort_and_drain() {
    let (requests, config) = failing(FaultPolicy::Abort);
    let result = run(&requests, &config, &LINEAR).unwrap();
    assert_eq!(result.steps.len(), 3);
    assert_eq!(result.steps[2].fault.as_ref().unwrap().outcome.failed, vec![1, 2]);
    let phases: Vec<Phase> = result.requests.iter().map(|r| r.phase).collect();
    assert_eq!(phases, vec![Phase::Finished, Phase::Failed, Phase::Failed]);
    assert_eq!(result.end_ns, 3440);
    // Drain retries the admitted requests; nothing un-admitted is waiting here.
    let (requests, config) = failing(FaultPolicy::Drain);
    let drained = run(&requests, &config, &LINEAR).unwrap();
    let retried = run(&requests, &failing(FaultPolicy::Retry).1, &LINEAR).unwrap();
    assert_eq!(to_jsonl(&drained.steps), to_jsonl(&retried.steps));
}

#[test]
fn drain_rejects_new_arrivals_and_unadmitted_requests() {
    let (mut requests, config) = failing(FaultPolicy::Drain);
    // r3 waits behind the budget at the failure (never admitted); r4 arrives later.
    requests.push(Request {
        id: 3,
        arrival_ns: 320,
        priority: 0,
        prefix_group: None,
        prefix_len: 0,
        prompt_len: 4,
        output_len: 1,
        samples: 1,
    });
    requests.push(Request {
        arrival_ns: 5000,
        id: 4,
        ..requests[3].clone()
    });
    let result = run(&requests, &config, &LINEAR).unwrap();
    let fault = result.steps[2].fault.as_ref().unwrap();
    assert_eq!(fault.outcome.retried, vec![1, 2]);
    assert_eq!(fault.outcome.rejected, vec![3]);
    let phases: Vec<Phase> = result.requests.iter().map(|r| r.phase).collect();
    assert_eq!(
        phases,
        vec![
            Phase::Finished,
            Phase::Finished,
            Phase::Finished,
            Phase::Rejected,
            Phase::Rejected
        ]
    );
    assert!(result.requests.iter().all(|r| r.endings == 1));
}

fn random_case(seed: u64) -> (Vec<Request>, EngineConfig, LinearTimer) {
    let mut rng = SplitMix64::new(seed ^ 0xfa01);
    let workload = WorkloadConfig {
        seed: rng.next_u64(),
        requests: 6 + rng.below(24) as usize,
        rate_per_s: 2e5 + rng.below(2_000_000) as f64,
        prompt: LengthDist::Uniform {
            lo: 1,
            hi: 4 + rng.below(30) as u32,
        },
        output: LengthDist::Uniform {
            lo: 1,
            hi: 2 + rng.below(20) as u32,
        },
        prefix_groups: 2,
        prefix_len: 3 + rng.below(10) as u32,
        prefix_permille: rng.below(900) as u32,
        samples: vec![(2, 1), (1, 2), (1, 4)],
        priorities: vec![(3, 0), (1, 1)],
    };
    let policy = [FaultPolicy::Abort, FaultPolicy::Retry, FaultPolicy::Drain][rng.below(3) as usize];
    let faults = (0..1 + rng.below(3))
        .map(|_| Fault {
            step: rng.below(40),
            layer: rng.below(4) as u32,
            allreduce: rng.below(2) as u8,
        })
        .collect::<Vec<_>>();
    let mut plan: Vec<Fault> = Vec::new();
    for f in faults {
        if !plan.iter().any(|g| g.step == f.step) {
            plan.push(f);
        }
    }
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
            block_size: [1, 2, 4, 8][rng.below(4) as usize],
            device_blocks: 24 + rng.below(80) as usize,
            swap_blocks: rng.below(60) as usize,
            prefix_caching: rng.below(3) != 0,
        },
        faults: FaultConfig {
            plan,
            policy,
            detection_ns: rng.below(5000),
            recovery_ns: 1 + rng.below(20_000),
            retry_limit: rng.below(3) as u32,
            layers: 4,
        },
        kv_check_interval: 1,
    };
    let timer = LinearTimer {
        base_ns: 1_000 + seed % 7,
        per_token_ns: 10 + seed % 5,
    };
    (workload.generate().unwrap(), config, timer)
}

/// G1–G4 are checked inside every run (an error fails the test); the trace
/// must equal the reference's; G5: a second run is byte-identical.
#[test]
fn random_failures_keep_g1_to_g5_and_match_the_reference() {
    let mut totals = [0usize; 4];
    for seed in 0..300 {
        let (requests, config, timer) = random_case(seed);
        let result = run(&requests, &config, &timer).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let expected = reference::run(&requests, config.scheduler, config.kv, &config.faults, &timer);
        if let Some(i) = (0..result.steps.len().max(expected.len())).find(|&i| result.steps.get(i) != expected.get(i)) {
            panic!(
                "seed {seed}: step {i} differs from the reference\nengine:    {:?}\nreference: {:?}",
                result.steps.get(i),
                expected.get(i)
            );
        }
        let again = run(&requests, &config, &timer).unwrap();
        assert_eq!(to_jsonl(&again.steps), to_jsonl(&result.steps), "seed {seed}: G5");
        for r in &result.requests {
            assert!(r.phase.is_terminal() && r.endings == 1, "seed {seed}");
        }
        let faults: Vec<_> = result.steps.iter().filter_map(|s| s.fault.as_ref()).collect();
        totals[0] += faults.len();
        totals[1] += faults.iter().map(|f| f.outcome.failed.len()).sum::<usize>();
        totals[2] += faults.iter().map(|f| f.outcome.retried.len()).sum::<usize>();
        totals[3] += faults.iter().map(|f| f.outcome.rejected.len()).sum::<usize>();
    }
    // Failures fire, and every policy outcome occurs.
    assert!(totals.iter().all(|&t| t > 50), "{totals:?}");
}

/// Compiled only with `--features planted-bugs`.
#[cfg(feature = "planted-bugs")]
#[test]
fn double_commit_on_retry_is_caught() {
    use fp_play_runtime::serving::engine::{run_planted, Planted};
    use fp_play_runtime::serving::scheduler::SchedulerMutation;
    for seed in 0..300 {
        let (requests, config, timer) = random_case(seed);
        let planted = Planted::Scheduler(SchedulerMutation::DoubleCommitOnRetry);
        if let Err(EngineError::Invariant { what, step }) = run_planted(&requests, &config, &timer, planted) {
            eprintln!("DoubleCommitOnRetry: caught at seed {seed}, step {step}: {what}");
            assert!(what.starts_with("token stream:"), "{what}");
            return;
        }
    }
    panic!("DoubleCommitOnRetry was not caught");
}

#[test]
fn faults_must_name_a_real_collective() {
    let (requests, mut config) = failing(FaultPolicy::Retry);
    config.faults.plan[0].layer = 2;
    assert!(matches!(run(&requests, &config, &LINEAR), Err(EngineError::Config(_))));
}

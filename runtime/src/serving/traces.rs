//! Step traces as JSON lines, and the golden traces pinned in `tests/traces/`.
//!
//! One line per step: its times, every scheduling decision, the work of each
//! sequence, copies, swaps, commits, finished requests and blocks in use.
//! The golden traces are small fixed scenarios; `tests/serving_traces.rs`
//! compares them byte for byte, and `fp-play-runtime traces --out
//! tests/traces` regenerates them. One of them uses the roofline step-time
//! model, so its times also pin the floating-point path across platforms.

use crate::collectives::{Algorithm, WirePolicy};
use crate::cost::LinkModel;
use crate::format::Format;
use crate::json::Json;

use super::engine::{run, EngineConfig, Fault, FaultConfig, StepRecord};
use super::kv::{KvConfig, SeqId};
use super::model::{kv_capacity_blocks, DeviceConfig, ModelConfig};
use super::scheduler::{FaultPolicy, PreemptionMode, SchedulerConfig};
use super::step_time::{Collective, LinearTimer, StepModel, StepTimer};
use super::workload::{LengthDist, Request, WorkloadConfig};

fn seq(s: SeqId) -> Vec<Json> {
    vec![Json::Int(i64::from(s.request)), Json::Int(i64::from(s.sample))]
}

fn ids(v: &[u32]) -> Json {
    Json::Arr(v.iter().map(|&i| Json::Int(i64::from(i))).collect())
}

/// One JSON object per step, one per line.
pub fn to_jsonl(steps: &[StepRecord]) -> String {
    let mut out = String::new();
    for s in steps {
        let p = &s.plan;
        let line = Json::obj(vec![
            ("step", Json::Int(s.index as i64)),
            ("start_ns", Json::Int(s.start_ns as i64)),
            ("end_ns", Json::Int(s.end_ns as i64)),
            ("swapped_in", ids(&p.swapped_in)),
            ("admitted", ids(&p.admitted)),
            (
                "preempted",
                Json::Arr(
                    p.preempted
                        .iter()
                        .map(|(r, k)| Json::Arr(vec![Json::Int(i64::from(*r)), Json::str(k.name())]))
                        .collect(),
                ),
            ),
            ("stalls", Json::Int(i64::from(p.stalls))),
            (
                "work",
                Json::Arr(
                    p.work
                        .iter()
                        .map(|w| {
                            let mut v = seq(w.seq);
                            v.extend([Json::Int(w.context as i64), Json::Int(w.tokens as i64)]);
                            Json::Arr(v)
                        })
                        .collect(),
                ),
            ),
            ("cow_blocks", Json::Int(p.cow_blocks as i64)),
            ("swap_blocks", Json::Int(p.swap_blocks as i64)),
            (
                "commits",
                Json::Arr(
                    s.outcome
                        .commits
                        .iter()
                        .map(|(q, k)| {
                            let mut v = seq(*q);
                            v.push(Json::Int(i64::from(*k)));
                            Json::Arr(v)
                        })
                        .collect(),
                ),
            ),
            ("finished", ids(&s.outcome.finished)),
            ("used_blocks", Json::uint(s.used_blocks)),
            (
                "fault",
                match &s.fault {
                    None => Json::Null,
                    Some(f) => Json::obj(vec![
                        ("layer", Json::Int(i64::from(f.layer))),
                        ("allreduce", Json::Int(i64::from(f.allreduce))),
                        ("failed", ids(&f.outcome.failed)),
                        ("retried", ids(&f.outcome.retried)),
                        ("rejected", ids(&f.outcome.rejected)),
                    ]),
                },
            ),
        ]);
        out.push_str(&line.to_compact());
        out.push('\n');
    }
    out
}

/// The hand-derived three-request case of `tests/serving_engine.rs`.
pub fn three_requests() -> (Vec<Request>, EngineConfig) {
    let req = |id, arrival_ns, prompt_len, output_len, samples| Request {
        id,
        arrival_ns,
        priority: 0,
        prefix_group: None,
        prefix_len: 0,
        prompt_len,
        output_len,
        samples,
    };
    (
        vec![req(0, 0, 5, 2, 1), req(1, 50, 3, 3, 2), req(2, 60, 8, 1, 1)],
        EngineConfig {
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
        },
    )
}

/// The linear timer used by the hand-derived and small golden traces.
pub const LINEAR: LinearTimer = LinearTimer {
    base_ns: 100,
    per_token_ns: 10,
};

fn pressure(preemption: PreemptionMode) -> (Vec<Request>, EngineConfig) {
    let workload = WorkloadConfig {
        seed: 0x7ace_0002,
        requests: 20,
        rate_per_s: 1e8,
        prompt: LengthDist::Uniform { lo: 3, hi: 12 },
        output: LengthDist::Uniform { lo: 2, hi: 10 },
        prefix_groups: 2,
        prefix_len: 8,
        prefix_permille: 600,
        samples: vec![(2, 1), (1, 2)],
        priorities: vec![(3, 0), (1, 1)],
    };
    let config = EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: 24,
            max_seqs: 10,
            watermark_blocks: 1,
            preemption,
        },
        kv: KvConfig {
            block_size: 4,
            device_blocks: 18,
            swap_blocks: 12,
            prefix_caching: true,
        },
        faults: FaultConfig::none(),
        kv_check_interval: 1,
    };
    (workload.generate().expect("valid workload"), config)
}

/// A few requests on the roofline model at P = 2 with ring AllReduces.
fn roofline() -> (Vec<Request>, EngineConfig, StepModel) {
    let model = ModelConfig::EIGHT_B_CLASS;
    let device = DeviceConfig::default();
    let workload = WorkloadConfig {
        seed: 0x7ace_0003,
        requests: 6,
        rate_per_s: 50.0,
        prompt: LengthDist::Uniform { lo: 20, hi: 120 },
        output: LengthDist::Uniform { lo: 2, hi: 6 },
        samples: vec![(1, 1), (1, 2)],
        ..WorkloadConfig::default()
    };
    let capacity = kv_capacity_blocks(&model, &device, 2, 16).expect("default device fits");
    let config = EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: 64,
            ..SchedulerConfig::default()
        },
        kv: KvConfig {
            block_size: 16,
            device_blocks: capacity as usize,
            swap_blocks: 0,
            prefix_caching: false,
        },
        faults: FaultConfig::none(),
        kv_check_interval: 1,
    };
    let timer = StepModel::new(
        model,
        device,
        2,
        16,
        Collective::Schedule(Algorithm::Ring),
        WirePolicy::uniform(Format::Bf16),
        LinkModel::default(),
        20_000.0,
    )
    .expect("valid step model");
    (workload.generate().expect("valid workload"), config, timer)
}

/// The three-request case with a rank failure in step 2 (layer 0's second
/// AllReduce of a 2-layer model; 1000 ns detection, 2000 ns recovery).
pub fn three_requests_failing(policy: FaultPolicy) -> (Vec<Request>, EngineConfig) {
    let (requests, mut config) = three_requests();
    config.faults = FaultConfig {
        plan: vec![Fault {
            step: 2,
            layer: 0,
            allreduce: 1,
        }],
        policy,
        detection_ns: 1000,
        recovery_ns: 2000,
        retry_limit: 1,
        layers: 2,
    };
    (requests, config)
}

/// Every golden trace: (file name, contents).
pub fn golden() -> Result<Vec<(&'static str, String)>, String> {
    let trace = |requests: &[Request], config: &EngineConfig, timer: &dyn StepTimer| {
        run(requests, config, timer)
            .map(|r| to_jsonl(&r.steps))
            .map_err(|e| e.to_string())
    };
    let (a, ac) = three_requests();
    let (b, bc) = pressure(PreemptionMode::Recompute);
    let (c, cc) = pressure(PreemptionMode::Swap);
    let (d, dc, dt) = roofline();
    let (e, ec) = three_requests_failing(FaultPolicy::Retry);
    let (f, mut fc) = pressure(PreemptionMode::Swap);
    fc.faults = FaultConfig {
        plan: vec![
            Fault {
                step: 6,
                layer: 3,
                allreduce: 0,
            },
            Fault {
                step: 15,
                layer: 1,
                allreduce: 1,
            },
        ],
        policy: FaultPolicy::Drain,
        detection_ns: 500,
        recovery_ns: 1500,
        retry_limit: 1,
        layers: 4,
    };
    Ok(vec![
        ("three-requests.jsonl", trace(&a, &ac, &LINEAR)?),
        ("pressure-recompute.jsonl", trace(&b, &bc, &LINEAR)?),
        ("pressure-swap.jsonl", trace(&c, &cc, &LINEAR)?),
        ("roofline-p2-ring.jsonl", trace(&d, &dc, &dt)?),
        ("three-requests-fault-retry.jsonl", trace(&e, &ec, &LINEAR)?),
        ("pressure-swap-faults-drain.jsonl", trace(&f, &fc, &LINEAR)?),
    ])
}

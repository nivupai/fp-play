//! The recorded serving scenarios (results/r2-serving.*).
//!
//! 1. **Load sweep**: arrival rate × tensor-parallel size P.
//! 2. **Preemption under memory pressure**: recompute versus swap, with a
//!    device memory small enough that preemption is frequent, plus swap at
//!    faster host links and both modes with one sample per request.
//! 3. **Prefix sharing**: shared system prompts with the prefix cache on and off.
//! 4. **Collective choice**: ring, halving-doubling and the shard tree, under
//!    a BF16 wire and FP32 reduce-scatter / BF16 all-gather.
//! 5. **Failures**: one rank failure mid-run under abort, retry and drain,
//!    against the same run without a failure. Each failure run is run twice
//!    and its traces compared (G5).
//!
//! Every scenario uses the 8B-class model, illustrative device constants,
//! 16-token blocks, a 512-token step budget and at most 64 sequences.
//! Device memory is a per-scenario setting. Every invariant is checked during
//! every run (see [`ScenarioRun::kv_check_interval`] for the KV-cache check's
//! cadence).

use crate::collectives::{Algorithm, WirePolicy};
use crate::cost::LinkModel;
use crate::format::Format;
use crate::shard_tree::DEFAULT_SHARDS;

use super::engine::{run, EngineConfig, Fault, FaultConfig, RunResult};
use super::kv::KvConfig;
use super::metrics::{metrics, Metrics};
use super::model::{kv_capacity_blocks, DeviceConfig, ModelConfig};
use super::scheduler::{FaultPolicy, PreemptionMode, SchedulerConfig};
use super::step_time::{Collective, StepModel};
use super::traces::to_jsonl;
use super::workload::{LengthDist, WorkloadConfig};

pub const BLOCK_SIZE: u64 = 16;
/// Per-step overhead, nanoseconds (illustrative).
pub const STEP_OVERHEAD_NS: f64 = 20_000.0;

#[derive(Clone, Debug)]
pub struct ServingConfig {
    pub model: ModelConfig,
    pub device: DeviceConfig,
    pub link: LinkModel,
    pub scheduler: SchedulerConfig,
    /// Workload shared by every scenario (each overrides some fields).
    pub workload: WorkloadConfig,
    pub load_rates: Vec<f64>,
    pub load_worlds: Vec<u64>,
    pub pressure_memory_bytes: u64,
    pub pressure_swap_blocks: usize,
    /// `(weight, class)` priorities in the pressure scenario.
    pub pressure_priorities: Vec<(u32, u8)>,
    /// Further host-link bandwidths (bytes/s) for swap in the pressure scenario.
    pub swap_bandwidths: Vec<f64>,
    pub prefix_memory_bytes: u64,
    pub prefix_groups: u32,
    pub prefix_len: u32,
    pub prefix_permille: u32,
    pub collective_worlds: Vec<u64>,
    pub collective_rate: f64,
    /// KV-cache check cadence for scenarios with a large, nearly empty pool.
    pub sparse_check_interval: u64,
    pub failure_world: u64,
    pub failure_rate: f64,
    pub failure: Fault,
    pub failure_detection_ns: u64,
    pub failure_recovery_ns: u64,
    pub failure_retry_limit: u32,
}

impl Default for ServingConfig {
    fn default() -> Self {
        ServingConfig {
            model: ModelConfig::EIGHT_B_CLASS,
            device: DeviceConfig::default(),
            link: LinkModel::default(),
            scheduler: SchedulerConfig::default(),
            workload: WorkloadConfig {
                seed: 0x5e7e_0001,
                requests: 200,
                rate_per_s: 10.0,
                prompt: LengthDist::Uniform { lo: 64, hi: 1024 },
                output: LengthDist::Uniform { lo: 16, hi: 256 },
                prefix_groups: 0,
                prefix_len: 0,
                prefix_permille: 0,
                samples: vec![(2, 1), (1, 2), (1, 4)],
                priorities: vec![(1, 0)],
            },
            load_rates: vec![5.0, 10.0, 20.0],
            load_worlds: vec![1, 2, 4, 8],
            pressure_memory_bytes: 21_000_000_000,
            pressure_swap_blocks: 1024,
            pressure_priorities: vec![(4, 0), (1, 1)],
            swap_bandwidths: vec![100e9, 400e9, 1.6e12],
            prefix_memory_bytes: 30_000_000_000,
            prefix_groups: 4,
            prefix_len: 512,
            prefix_permille: 800,
            collective_worlds: vec![4, 8],
            collective_rate: 5.0,
            sparse_check_interval: 256,
            failure_world: 4,
            failure_rate: 20.0,
            failure: Fault {
                step: 1500,
                layer: 16,
                allreduce: 1,
            },
            failure_detection_ns: 500_000_000,
            failure_recovery_ns: 5_000_000_000,
            failure_retry_limit: 1,
        }
    }
}

/// The collectives scenario 4 compares.
pub const COLLECTIVES: [Collective; 3] = [
    Collective::Schedule(Algorithm::Ring),
    Collective::Schedule(Algorithm::HalvingDoubling),
    Collective::ShardTree(DEFAULT_SHARDS),
];

/// The wire policies scenario 4 compares.
pub const POLICIES: [WirePolicy; 2] = [WirePolicy::uniform(Format::Bf16), WirePolicy::FP32_RS_BF16_AG];

#[derive(Clone, Debug)]
pub struct ScenarioRun {
    pub label: String,
    pub world: u64,
    pub rate_per_s: f64,
    pub memory_bytes: u64,
    pub capacity_blocks: u64,
    pub swap_blocks: usize,
    pub preemption: PreemptionMode,
    pub prefix_caching: bool,
    pub collective: Collective,
    pub policy: WirePolicy,
    /// The KV-cache invariants were checked after every k-th step and the
    /// last; the scheduler and token-stream invariants after every step.
    pub kv_check_interval: u64,
    /// Host-link bandwidth used for swapping, bytes/s.
    pub host_bytes_per_s: f64,
    /// `(weight, n)` samples per request.
    pub samples: Vec<(u32, u32)>,
    /// The failure policy, if this run has a failure plan.
    pub fault_policy: Option<FaultPolicy>,
    /// (failed, retried, rejected) request counts at each failure that fired.
    pub fault_outcomes: Vec<(usize, usize, usize)>,
    /// For runs with failures: a second run gave a byte-identical trace (G5).
    pub rerun_identical: Option<bool>,
    pub metrics: Metrics,
}

#[derive(Clone, Debug)]
pub struct ServingResult {
    pub config: ServingConfig,
    pub load: Vec<ScenarioRun>,
    pub pressure: Vec<ScenarioRun>,
    /// Swap at faster host links, then recompute and swap with one sample per request.
    pub swap_sensitivity: Vec<ScenarioRun>,
    pub prefix: Vec<ScenarioRun>,
    pub collectives: Vec<ScenarioRun>,
    pub failures: Vec<ScenarioRun>,
}

/// One run's settings.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub label: String,
    pub world: u64,
    pub memory_bytes: u64,
    pub workload: WorkloadConfig,
    pub preemption: PreemptionMode,
    pub swap_blocks: usize,
    pub prefix_caching: bool,
    pub collective: Collective,
    pub policy: WirePolicy,
    pub kv_check_interval: u64,
    pub faults: FaultConfig,
    /// Host-link bandwidth for swapping, bytes/s (default: the device's).
    pub host_bytes_per_s: Option<f64>,
}

/// Run one scenario point; also returns the full run for tests.
pub fn run_spec(config: &ServingConfig, spec: &RunSpec) -> Result<(ScenarioRun, RunResult), String> {
    let device = DeviceConfig {
        memory_bytes: spec.memory_bytes,
        host_bytes_per_s: spec.host_bytes_per_s.unwrap_or(config.device.host_bytes_per_s),
        ..config.device
    };
    let capacity = kv_capacity_blocks(&config.model, &device, spec.world, BLOCK_SIZE).map_err(|e| e.to_string())?;
    let timer = StepModel::new(
        config.model,
        device,
        spec.world,
        BLOCK_SIZE,
        spec.collective,
        spec.policy,
        config.link,
        STEP_OVERHEAD_NS,
    )
    .map_err(|e| e.to_string())?;
    let engine = EngineConfig {
        scheduler: SchedulerConfig {
            preemption: spec.preemption,
            watermark_blocks: (capacity / 100) as usize,
            ..config.scheduler
        },
        kv: KvConfig {
            block_size: BLOCK_SIZE as usize,
            device_blocks: capacity as usize,
            swap_blocks: spec.swap_blocks,
            prefix_caching: spec.prefix_caching,
        },
        faults: spec.faults.clone(),
        kv_check_interval: spec.kv_check_interval,
    };
    let requests = spec.workload.generate().map_err(|e| e.to_string())?;
    let result = run(&requests, &engine, &timer).map_err(|e| format!("{}: {e}", spec.label))?;
    let has_faults = !spec.faults.plan.is_empty();
    let rerun_identical = if has_faults {
        let again = run(&requests, &engine, &timer).map_err(|e| format!("{}: {e}", spec.label))?;
        Some(to_jsonl(&again.steps) == to_jsonl(&result.steps))
    } else {
        None
    };
    let fault_outcomes = result
        .steps
        .iter()
        .filter_map(|s| s.fault.as_ref())
        .map(|f| {
            (
                f.outcome.failed.len(),
                f.outcome.retried.len(),
                f.outcome.rejected.len(),
            )
        })
        .collect();
    Ok((
        ScenarioRun {
            label: spec.label.clone(),
            world: spec.world,
            rate_per_s: spec.workload.rate_per_s,
            memory_bytes: spec.memory_bytes,
            capacity_blocks: capacity,
            swap_blocks: spec.swap_blocks,
            preemption: spec.preemption,
            prefix_caching: spec.prefix_caching,
            collective: spec.collective,
            policy: spec.policy,
            kv_check_interval: spec.kv_check_interval,
            host_bytes_per_s: device.host_bytes_per_s,
            samples: spec.workload.samples.clone(),
            fault_policy: has_faults.then_some(spec.faults.policy),
            fault_outcomes,
            rerun_identical,
            metrics: metrics(&result),
        },
        result,
    ))
}

impl ServingConfig {
    fn base(&self, label: String) -> RunSpec {
        RunSpec {
            label,
            world: 1,
            memory_bytes: self.device.memory_bytes,
            workload: self.workload.clone(),
            preemption: PreemptionMode::Recompute,
            swap_blocks: 0,
            prefix_caching: false,
            collective: Collective::Schedule(Algorithm::Ring),
            policy: WirePolicy::uniform(Format::Bf16),
            kv_check_interval: 1,
            faults: FaultConfig::none(),
            host_bytes_per_s: None,
        }
    }

    /// Scenario 5: no failure, then one failure under each policy.
    pub fn failure_specs(&self) -> Vec<RunSpec> {
        let policies = [
            None,
            Some(FaultPolicy::Abort),
            Some(FaultPolicy::Retry),
            Some(FaultPolicy::Drain),
        ];
        policies
            .into_iter()
            .map(|policy| {
                let name = policy.map_or("no failure", |p| p.name());
                let mut s = self.base(format!("failure {name}"));
                s.world = self.failure_world;
                s.workload.rate_per_s = self.failure_rate;
                s.kv_check_interval = self.sparse_check_interval;
                if let Some(policy) = policy {
                    s.faults = FaultConfig {
                        plan: vec![self.failure],
                        policy,
                        detection_ns: self.failure_detection_ns,
                        recovery_ns: self.failure_recovery_ns,
                        retry_limit: self.failure_retry_limit,
                        layers: self.model.layers as u32,
                    };
                }
                s
            })
            .collect()
    }

    pub fn load_specs(&self) -> Vec<RunSpec> {
        let mut out = Vec::new();
        for &world in &self.load_worlds {
            for &rate in &self.load_rates {
                let mut s = self.base(format!("load P={world} rate={rate}"));
                s.world = world;
                s.workload.rate_per_s = rate;
                s.kv_check_interval = self.sparse_check_interval;
                out.push(s);
            }
        }
        out
    }

    pub fn pressure_specs(&self) -> Vec<RunSpec> {
        [PreemptionMode::Recompute, PreemptionMode::Swap]
            .into_iter()
            .map(|mode| {
                let mut s = self.base(format!("pressure {}", mode.name()));
                s.memory_bytes = self.pressure_memory_bytes;
                s.preemption = mode;
                s.swap_blocks = self.pressure_swap_blocks;
                s.workload.priorities = self.pressure_priorities.clone();
                s
            })
            .collect()
    }

    pub fn swap_sensitivity_specs(&self) -> Vec<RunSpec> {
        let base = self.pressure_specs();
        let mut out = Vec::new();
        for &bw in &self.swap_bandwidths {
            let mut s = base[1].clone();
            s.label = format!("pressure swap host {} GB/s", bw / 1e9);
            s.host_bytes_per_s = Some(bw);
            out.push(s);
        }
        for mut s in base {
            s.label = format!("{} one sample", s.label);
            s.workload.samples = vec![(1, 1)];
            out.push(s);
        }
        out
    }

    pub fn prefix_specs(&self) -> Vec<RunSpec> {
        [false, true]
            .into_iter()
            .map(|on| {
                let mut s = self.base(format!("prefix caching {}", if on { "on" } else { "off" }));
                s.memory_bytes = self.prefix_memory_bytes;
                s.prefix_caching = on;
                s.workload.prefix_groups = self.prefix_groups;
                s.workload.prefix_len = self.prefix_len;
                s.workload.prefix_permille = self.prefix_permille;
                s
            })
            .collect()
    }

    pub fn collective_specs(&self) -> Vec<RunSpec> {
        let mut out = Vec::new();
        for &world in &self.collective_worlds {
            for policy in POLICIES {
                for collective in COLLECTIVES {
                    let mut s = self.base(format!("P={world} {} {}", collective.name(), policy.name()));
                    s.world = world;
                    s.collective = collective;
                    s.policy = policy;
                    s.workload.rate_per_s = self.collective_rate;
                    s.kv_check_interval = self.sparse_check_interval;
                    out.push(s);
                }
            }
        }
        out
    }
}

/// Run every spec, in parallel on the available cores. Runs are independent
/// and results are returned in spec order, so the output does not depend on
/// the number of threads.
pub fn run_specs(config: &ServingConfig, specs: &[RunSpec]) -> Result<Vec<ScenarioRun>, String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Result<ScenarioRun, String>>>> = Mutex::new(specs.iter().map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(specs.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= specs.len() {
                    break;
                }
                let r = run_spec(config, &specs[i]).map(|(r, _)| r);
                slots.lock().expect("no worker panicked")[i] = Some(r);
            });
        }
    });
    slots
        .into_inner()
        .expect("no worker panicked")
        .into_iter()
        .map(|r| r.expect("every spec ran"))
        .collect()
}

pub fn run_all(config: &ServingConfig) -> Result<ServingResult, String> {
    let groups = [
        config.load_specs(),
        config.pressure_specs(),
        config.swap_sensitivity_specs(),
        config.prefix_specs(),
        config.collective_specs(),
        config.failure_specs(),
    ];
    let lens: Vec<usize> = groups.iter().map(Vec::len).collect();
    let all: Vec<RunSpec> = groups.into_iter().flatten().collect();
    let mut runs = run_specs(config, &all)?.into_iter();
    let mut take = |n: usize| runs.by_ref().take(n).collect::<Vec<_>>();
    Ok(ServingResult {
        config: config.clone(),
        load: take(lens[0]),
        pressure: take(lens[1]),
        swap_sensitivity: take(lens[2]),
        prefix: take(lens[3]),
        collectives: take(lens[4]),
        failures: take(lens[5]),
    })
}

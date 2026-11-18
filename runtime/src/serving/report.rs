//! results/r2-serving.{json,md}: the recorded serving scenarios.

use crate::json::Json;

use super::experiment::{ScenarioRun, ServingResult, BLOCK_SIZE, STEP_OVERHEAD_NS};
use super::metrics::{Metrics, Percentiles};
use super::workload::LengthDist;

pub const SCHEMA: &str = "fp-play-runtime/r2-serving/v1";

fn num(x: f64) -> Json {
    Json::Num(x)
}

fn int(x: u64) -> Json {
    Json::Int(x as i64)
}

fn pct_json(p: Option<Percentiles>) -> Json {
    match p {
        None => Json::Null,
        Some(p) => Json::obj(vec![
            ("n", Json::uint(p.n)),
            ("p50", int(p.p50)),
            ("p99", int(p.p99)),
            ("max", int(p.max)),
        ]),
    }
}

fn dist_json(d: &LengthDist) -> Json {
    match d {
        LengthDist::Fixed(n) => Json::obj(vec![("fixed", int(u64::from(*n)))]),
        LengthDist::Uniform { lo, hi } => Json::obj(vec![
            ("uniform_lo", int(u64::from(*lo))),
            ("uniform_hi", int(u64::from(*hi))),
        ]),
        LengthDist::Table(t) => Json::obj(vec![(
            "table_weight_length",
            Json::Arr(
                t.iter()
                    .map(|(w, n)| Json::Arr(vec![int(u64::from(*w)), int(u64::from(*n))]))
                    .collect(),
            ),
        )]),
    }
}

/// Swap traffic summed over every device: blocks × per-device block bytes × P.
fn swap_bytes(result: &ServingResult, run: &ScenarioRun) -> u64 {
    let m = &result.config.model;
    run.metrics.swapped_blocks * BLOCK_SIZE * m.kv_bytes_per_token_per_device(run.world) * run.world
}

fn metrics_json(result: &ServingResult, run: &ScenarioRun) -> Json {
    let m: &Metrics = &run.metrics;
    Json::obj(vec![
        ("requests", Json::uint(m.requests)),
        ("finished", Json::uint(m.finished)),
        ("rejected", Json::uint(m.rejected)),
        ("failed", Json::uint(m.failed)),
        ("fault_steps", Json::uint(m.fault_steps)),
        ("fault_ns", int(m.fault_ns)),
        ("output_tokens", int(m.output_tokens)),
        ("ttft_ns", pct_json(m.ttft)),
        ("tpot_ns", pct_json(m.tpot)),
        ("e2e_ns", pct_json(m.e2e)),
        ("span_ns", int(m.span_ns)),
        ("output_tokens_per_s", num(m.tokens_per_s)),
        ("requests_per_s", num(m.requests_per_s)),
        ("steps", Json::uint(m.steps)),
        ("mean_step_tokens", num(m.mean_step_tokens)),
        ("max_step_tokens", int(m.max_step_tokens)),
        ("compute_ns", num(m.compute_ns)),
        ("comm_ns", num(m.comm_ns)),
        ("copy_ns", num(m.copy_ns)),
        ("overhead_ns", num(m.overhead_ns)),
        ("comm_share", num(m.comm_share)),
        ("preemptions_recompute", int(m.preemptions_recompute)),
        ("preemptions_swap", int(m.preemptions_swap)),
        ("swap_full_fallbacks", int(m.swap_fallbacks)),
        ("stalls", int(m.stalls)),
        ("recomputed_tokens", int(m.recomputed_tokens)),
        ("swapped_blocks", int(m.swapped_blocks)),
        ("swap_bytes_all_devices", int(swap_bytes(result, run))),
        ("cow_copies", int(m.cow_copies)),
        ("prefix_evictions", int(m.evictions)),
        ("prefix_eligible_blocks", int(m.prefix_eligible_blocks)),
        ("prefix_hit_blocks", int(m.prefix_hit_blocks)),
        ("prefix_hit_rate", m.prefix_hit_rate().map_or(Json::Null, num)),
        ("kv_utilization_mean", num(m.kv_util_mean)),
        ("kv_peak_blocks", Json::uint(m.kv_peak_blocks)),
    ])
}

fn run_json(result: &ServingResult, run: &ScenarioRun) -> Json {
    Json::obj(vec![
        ("label", Json::str(&run.label)),
        ("world", int(run.world)),
        ("rate_per_s", num(run.rate_per_s)),
        ("device_memory_bytes", int(run.memory_bytes)),
        ("kv_capacity_blocks", int(run.capacity_blocks)),
        ("kv_capacity_tokens", int(run.capacity_blocks * BLOCK_SIZE)),
        ("swap_blocks", Json::uint(run.swap_blocks)),
        ("preemption", Json::str(run.preemption.name())),
        ("prefix_caching", Json::Bool(run.prefix_caching)),
        ("collective", Json::str(run.collective.name())),
        ("wire_policy", Json::str(run.policy.name())),
        ("kv_check_interval_steps", int(run.kv_check_interval)),
        (
            "fault_policy",
            run.fault_policy.map_or(Json::Null, |p| Json::str(p.name())),
        ),
        (
            "faults_fired",
            Json::Arr(
                run.fault_outcomes
                    .iter()
                    .map(|&(f, r, j)| {
                        Json::obj(vec![
                            ("failed", Json::uint(f)),
                            ("retried", Json::uint(r)),
                            ("rejected", Json::uint(j)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("rerun_identical", run.rerun_identical.map_or(Json::Null, Json::Bool)),
        ("metrics", metrics_json(result, run)),
    ])
}

pub fn to_json(result: &ServingResult, command: &str) -> Json {
    let c = &result.config;
    let m = &c.model;
    let d = &c.device;
    let w = &c.workload;
    let runs = |v: &[ScenarioRun]| Json::Arr(v.iter().map(|r| run_json(result, r)).collect());
    Json::obj(vec![
        ("schema", Json::str(SCHEMA)),
        ("tool", Json::str(format!("fp-play-runtime {}", env!("CARGO_PKG_VERSION")))),
        ("command", Json::str(command)),
        (
            "config",
            Json::obj(vec![
                (
                    "model",
                    Json::obj(vec![
                        ("description", Json::str("8B-class public transformer dimensions; not a specific released model")),
                        ("hidden", int(m.hidden)),
                        ("layers", int(m.layers)),
                        ("q_heads", int(m.q_heads)),
                        ("kv_heads", int(m.kv_heads)),
                        ("head_dim", int(m.head_dim)),
                        ("ffn", int(m.ffn)),
                        ("vocab", int(m.vocab)),
                        ("parameters", int(m.params())),
                        ("weight_bytes_per_element", int(m.weight_bytes)),
                        ("kv_bytes_per_element", int(m.kv_bytes)),
                        ("kv_bytes_per_token_at_p1", int(m.kv_bytes_per_token_per_device(1))),
                    ]),
                ),
                (
                    "device",
                    Json::obj(vec![
                        ("description", Json::str("illustrative constants, not a measured or named device")),
                        ("default_memory_bytes", int(d.memory_bytes)),
                        ("reserve_bytes", int(d.reserve_bytes)),
                        ("peak_flops", num(d.peak_flops)),
                        ("memory_bytes_per_s", num(d.hbm_bytes_per_s)),
                        ("host_link_bytes_per_s", num(d.host_bytes_per_s)),
                    ]),
                ),
                (
                    "link",
                    Json::obj(vec![
                        ("latency_us", num(c.link.latency_us)),
                        ("bandwidth_gbps", num(c.link.bandwidth_gbps)),
                    ]),
                ),
                ("block_size_tokens", int(BLOCK_SIZE)),
                ("step_overhead_ns", num(STEP_OVERHEAD_NS)),
                (
                    "scheduler",
                    Json::obj(vec![
                        ("token_budget", Json::uint(c.scheduler.token_budget)),
                        ("max_seqs", Json::uint(c.scheduler.max_seqs)),
                        ("watermark", Json::str("floor(1% of the device blocks)")),
                    ]),
                ),
                (
                    "workload",
                    Json::obj(vec![
                        ("seed", Json::Int(w.seed as i64)),
                        ("requests", Json::uint(w.requests)),
                        ("arrivals", Json::str("Poisson")),
                        ("own_prompt_tokens", dist_json(&w.prompt)),
                        ("output_tokens_per_sample", dist_json(&w.output)),
                        (
                            "samples_weight_n",
                            Json::Arr(
                                w.samples
                                    .iter()
                                    .map(|(a, b)| Json::Arr(vec![int(u64::from(*a)), int(u64::from(*b))]))
                                    .collect(),
                            ),
                        ),
                    ]),
                ),
                (
                    "definitions",
                    Json::obj(vec![
                        ("ttft", Json::str("first committed token − arrival, per request")),
                        ("tpot", Json::str("(last commit − first commit) ÷ (output tokens − 1), per sample with ≥ 2 output tokens, rounded down")),
                        ("e2e", Json::str("finish − arrival, per request")),
                        ("percentiles", Json::str("nearest rank on integer nanoseconds")),
                        ("throughput", Json::str("committed output tokens (finished requests) per second, first arrival to last finish")),
                        ("comm_share", Json::str("AllReduce time ÷ total step time")),
                        ("kv_utilization_mean", Json::str("device blocks in use at each step's end, weighted by step duration, ÷ capacity")),
                        ("prefix_hit_rate", Json::str("prefix-cache blocks adopted ÷ full prompt blocks eligible, at admission")),
                        ("swap_bytes_all_devices", Json::str("swapped-out plus swapped-in blocks × block bytes per device × P")),
                    ]),
                ),
                (
                    "invariants",
                    Json::str("every run checks the scheduler and token-stream invariants after every step, and the KV-cache invariants after every kv_check_interval_steps-th step, after every failure and after the last step"),
                ),
                (
                    "failures",
                    Json::obj(vec![
                        ("world", int(c.failure_world)),
                        ("rate_per_s", num(c.failure_rate)),
                        ("step", int(c.failure.step)),
                        ("layer", int(u64::from(c.failure.layer))),
                        ("allreduce", int(u64::from(c.failure.allreduce))),
                        ("detection_ns", int(c.failure_detection_ns)),
                        ("recovery_ns", int(c.failure_recovery_ns)),
                        ("retry_limit", int(u64::from(c.failure_retry_limit))),
                        ("guarantees", Json::str("G1 no leaked blocks, G2 no duplicated tokens, G3 nothing lost under retry and G4 exactly one ending are checked inside every run (a violation stops the experiment with an error); G5 determinism: each failure run is run twice and its traces compared (rerun_identical)")),
                    ]),
                ),
            ]),
        ),
        (
            "scenarios",
            Json::obj(vec![
                ("load", runs(&result.load)),
                ("preemption_under_memory_pressure", runs(&result.pressure)),
                ("prefix_sharing", runs(&result.prefix)),
                ("collective_choice", runs(&result.collectives)),
                ("failures", runs(&result.failures)),
            ]),
        ),
    ])
}

/// "a, b or c (weights x : y : z)" from `(weight, value)` pairs.
fn weighted<T: std::fmt::Display>(table: &[(u32, T)]) -> String {
    let values: Vec<String> = table.iter().map(|(_, v)| v.to_string()).collect();
    let weights: Vec<String> = table.iter().map(|(w, _)| w.to_string()).collect();
    let list = match values.len() {
        0 => String::new(),
        1 => values[0].clone(),
        n => format!("{} or {}", values[..n - 1].join(", "), values[n - 1]),
    };
    format!("{list} (weights {})", weights.join(" : "))
}

fn dist_text(d: &LengthDist) -> String {
    match d {
        LengthDist::Fixed(n) => n.to_string(),
        LengthDist::Uniform { lo, hi } => format!("{lo}–{hi}"),
        LengthDist::Table(t) => weighted(&t.iter().map(|&(w, n)| (w, n)).collect::<Vec<_>>()),
    }
}

fn ms(ns: u64) -> String {
    format!("{:.2}", ns as f64 / 1e6)
}

fn p50_99(p: Option<Percentiles>) -> String {
    p.map_or("–".into(), |p| format!("{} / {}", ms(p.p50), ms(p.p99)))
}

fn gb(bytes: u64) -> String {
    format!("{:.2}", bytes as f64 / 1e9)
}

pub fn to_markdown(result: &ServingResult, command: &str) -> String {
    let c = &result.config;
    let m = &c.model;
    let mut md = String::new();
    md.push_str("# R2: a deterministic serving simulator\n\n");
    md.push_str(&format!(
        "Generated by `{command}`. Every number below comes from that run.\n\n"
    ));
    md.push_str("## Setup\n\n");
    md.push_str(&format!(
        "- Model: 8B-class dimensions (not a specific released model): hidden {}, {} layers, {} query and {} KV heads of dimension {}, FFN {}, vocabulary {}, {} parameters, BF16 weights and KV.\n",
        m.hidden, m.layers, m.q_heads, m.kv_heads, m.head_dim, m.ffn, m.vocab, m.params()
    ));
    md.push_str(&format!(
        "- Device (illustrative, not a measured or named device): {} GB memory by default (set per scenario), {} GB reserve, {:.0} TFLOP/s, {:.0} GB/s memory, {:.0} GB/s host link. Link: {} µs + bytes ÷ {} GB/s per round.\n",
        gb(c.device.memory_bytes), gb(c.device.reserve_bytes), c.device.peak_flops / 1e12,
        c.device.hbm_bytes_per_s / 1e9, c.device.host_bytes_per_s / 1e9, c.link.latency_us, c.link.bandwidth_gbps
    ));
    md.push_str(&format!(
        "- Step time: a roofline (max of FLOPs ÷ peak and bytes ÷ memory bandwidth per device) plus two AllReduces of tokens × hidden BF16 activations per layer, each costed from the schedule the collective records, plus copies, swaps and a {} µs per-step overhead. No overlap.\n",
        STEP_OVERHEAD_NS / 1e3
    ));
    md.push_str(&format!(
        "- Scheduler: {}-token budget per step, at most {} sequences, {}-token blocks, watermark 1% of the blocks. Workload: {} requests, Poisson arrivals, own prompt {} tokens, {} output tokens per sample, {} samples per request.\n",
        c.scheduler.token_budget,
        c.scheduler.max_seqs,
        BLOCK_SIZE,
        c.workload.requests,
        dist_text(&c.workload.prompt),
        dist_text(&c.workload.output),
        weighted(&c.workload.samples),
    ));
    md.push_str("- Every run checks the scheduler and token-stream invariants after every step and the KV-cache invariants after every k-th step and the last (k is listed per scenario). Latencies are in milliseconds, shown as p50 / p99 (nearest rank).\n\n");

    md.push_str("## 1. Load sweep\n\n");
    if let Some(r) = result.load.first() {
        md.push_str(&format!(
            "Device memory {} GB; ring AllReduce with a BF16 wire; KV check every {} steps.\n\n",
            gb(r.memory_bytes),
            r.kv_check_interval
        ));
    }
    md.push_str("| P | Rate (req/s) | KV blocks | TTFT p50 / p99 | TPOT p50 / p99 | Output tok/s | Comm share | Preemptions |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &result.load {
        let x = &r.metrics;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.0} | {:.1}% | {} |\n",
            r.world,
            r.rate_per_s,
            r.capacity_blocks,
            p50_99(x.ttft),
            p50_99(x.tpot),
            x.tokens_per_s,
            100.0 * x.comm_share,
            x.preemptions_recompute + x.preemptions_swap
        ));
    }

    md.push_str("\n## 2. Preemption under memory pressure\n\n");
    if let Some(r) = result.pressure.first() {
        md.push_str(&format!(
            "P = {}, {} req/s, device memory {} GB → {} KV blocks ({} tokens); {} host swap blocks; priority classes {}; KV check every {} step(s).\n\n",
            r.world,
            r.rate_per_s,
            gb(r.memory_bytes),
            r.capacity_blocks,
            r.capacity_blocks * BLOCK_SIZE,
            r.swap_blocks,
            weighted(&c.pressure_priorities),
            r.kv_check_interval
        ));
    }
    md.push_str("| Mode | Preemptions | Swap-full fallbacks | Recomputed tokens | Swap traffic (GB, all devices) | TTFT p50 / p99 | TPOT p50 / p99 | E2E p50 / p99 | Output tok/s |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &result.pressure {
        let x = &r.metrics;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.0} |\n",
            r.preemption.name(),
            x.preemptions_recompute + x.preemptions_swap,
            x.swap_fallbacks,
            x.recomputed_tokens,
            gb(swap_bytes(result, r)),
            p50_99(x.ttft),
            p50_99(x.tpot),
            p50_99(x.e2e),
            x.tokens_per_s
        ));
    }

    md.push_str("\n## 3. Prefix sharing\n\n");
    if let Some(r) = result.prefix.first() {
        md.push_str(&format!(
            "P = {}, {} req/s, device memory {} GB → {} KV blocks; {} shared system prompts of {} tokens, used by {:.1}% of requests; KV check every {} step(s).\n\n",
            r.world,
            r.rate_per_s,
            gb(r.memory_bytes),
            r.capacity_blocks,
            c.prefix_groups,
            c.prefix_len,
            f64::from(c.prefix_permille) / 10.0,
            r.kv_check_interval
        ));
    }
    md.push_str("| Prefix cache | Hit rate | Mean KV use | Peak blocks | Evictions | Preemptions | TTFT p50 / p99 | TPOT p50 / p99 | Output tok/s |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &result.prefix {
        let x = &r.metrics;
        md.push_str(&format!(
            "| {} | {} | {:.1}% | {} | {} | {} | {} | {} | {:.0} |\n",
            if r.prefix_caching { "on" } else { "off" },
            x.prefix_hit_rate().map_or("–".into(), |h| format!("{:.1}%", 100.0 * h)),
            100.0 * x.kv_util_mean,
            x.kv_peak_blocks,
            x.evictions,
            x.preemptions_recompute + x.preemptions_swap,
            p50_99(x.ttft),
            p50_99(x.tpot),
            x.tokens_per_s
        ));
    }

    md.push_str("\n## 4. Collective choice\n\n");
    if let Some(r) = result.collectives.first() {
        md.push_str(&format!(
            "{} req/s, device memory {} GB; KV check every {} steps. The TPOT change is the p50 against ring with the same P and wire policy.\n\n",
            r.rate_per_s,
            gb(r.memory_bytes),
            r.kv_check_interval
        ));
    }
    md.push_str("| P | Wire policy | Collective | Comm share | TPOT p50 / p99 | TPOT p50 vs ring | TTFT p50 / p99 |\n");
    md.push_str("|---:|---|---|---:|---:|---:|---:|\n");
    for r in &result.collectives {
        let x = &r.metrics;
        let ring = result
            .collectives
            .iter()
            .find(|o| o.world == r.world && o.policy == r.policy && o.collective.name() == "ring");
        let change = match (ring.and_then(|o| o.metrics.tpot), x.tpot) {
            (Some(a), Some(b)) => format!("{:+.1}%", 100.0 * (b.p50 as f64 - a.p50 as f64) / a.p50 as f64),
            _ => "–".into(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {:.1}% | {} | {} | {} |\n",
            r.world,
            r.policy.name(),
            r.collective.name(),
            100.0 * x.comm_share,
            p50_99(x.tpot),
            change,
            p50_99(x.ttft)
        ));
    }

    md.push_str("\n## 5. Failures\n\n");
    md.push_str(&format!(
        "P = {}, {} req/s, device memory {} GB; a fail-stop rank failure in step {} at layer {}'s {} AllReduce, {} s detection and {} s recovery (illustrative), retry limit {}. Every device KV block is lost; the failed step commits nothing. G1–G4 are checked inside each run (a violation stops the experiment); G5 by running each failure run twice. Latencies are over finished requests only.\n\n",
        c.failure_world,
        c.failure_rate,
        gb(c.device.memory_bytes),
        c.failure.step,
        c.failure.layer,
        if c.failure.allreduce == 0 { "first (after attention)" } else { "second (after the MLP)" },
        c.failure_detection_ns as f64 / 1e9,
        c.failure_recovery_ns as f64 / 1e9,
        c.failure_retry_limit
    ));
    md.push_str("| Policy | Finished | Failed | Rejected | Retried at the failure | Recomputed tokens | Time in the failed step (s) | TTFT p50 / p99 | E2E p50 / p99 | Rerun identical (G5) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for r in &result.failures {
        let x = &r.metrics;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {} | {} | {} |\n",
            r.fault_policy.map_or("no failure", |p| p.name()),
            x.finished,
            x.failed,
            x.rejected,
            r.fault_outcomes.iter().map(|o| o.1).sum::<usize>(),
            x.recomputed_tokens,
            x.fault_ns as f64 / 1e9,
            p50_99(x.ttft),
            p50_99(x.e2e),
            r.rerun_identical.map_or("–", |b| if b { "yes" } else { "no" })
        ));
    }

    md.push_str("\n## What this does and does not show\n\n");
    md.push_str("- It shows how a paged KV cache, continuous batching, preemption and the AllReduce schedule interact under one explicit, deterministic model, with every step checked against the invariants and, in the tests, against an independent reference scheduler.\n");
    md.push_str("- The compute model is a roofline with illustrative constants, not a measured device. Compute and communication do not overlap. The links have no contention.\n");
    md.push_str("- Tokens are synthetic; no model numerics run (that is R3). There is one tensor-parallel group, with no replicas or routing.\n");
    md.push_str("- P must divide the 8 KV heads; KV-head replication for larger P is not modeled.\n");
    md.push_str("- Failures are fail-stop with constant detection and recovery times; after recovery the group continues at the same P. Continuing at P − 1 is future work.\n");
    md.push_str("- No speculative decoding, LoRA or quantized KV. No hardware is measured.\n");
    md
}

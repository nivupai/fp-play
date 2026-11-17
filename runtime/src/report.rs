//! Deterministic JSON and Markdown renderings of an experiment.

use crate::collectives::{Algorithm, WirePolicy};
use crate::cost::CostEstimate;
use crate::json::Json;
use crate::policies::{self, PoliciesResult};
use crate::tensor_parallel::{Comparison, ExperimentResult, INPUT_FORMAT, LOCAL_ACCUMULATOR};

pub const ARTIFACT_SCHEMA: &str = "fp-play-runtime/tp-allreduce/v1";

fn comparison(c: &Comparison, elements: usize) -> Json {
    Json::obj(vec![
        ("mismatched", Json::uint(c.mismatched)),
        ("mismatched_fraction", Json::Num(c.mismatched as f64 / elements as f64)),
        ("max_ulp", Json::Int(c.max_ulp as i64)),
        ("mean_ulp", Json::Num(c.mean_ulp)),
        ("relative_rms_error", Json::Num(c.relative_rms_error)),
    ])
}

/// The full artifact. Contains no timestamps or host details so reruns are byte-identical.
pub fn to_json(result: &ExperimentResult, command: &str) -> Json {
    let c = &result.config;
    let n = result.elements;
    Json::obj(vec![
        ("schema", Json::str(ARTIFACT_SCHEMA)),
        (
            "tool",
            Json::str(format!("fp-play-runtime {}", env!("CARGO_PKG_VERSION"))),
        ),
        ("command", Json::str(command)),
        (
            "config",
            Json::obj(vec![
                ("tokens", Json::uint(c.tokens)),
                ("hidden", Json::uint(c.hidden)),
                ("outputs", Json::uint(c.outputs)),
                (
                    "world_sizes",
                    Json::Arr(c.world_sizes.iter().map(|&w| Json::uint(w)).collect()),
                ),
                (
                    "wire_formats",
                    Json::Arr(c.wire_formats.iter().map(|f| Json::str(f.name())).collect()),
                ),
                ("seed", Json::Int(c.seed as i64)),
                ("input_format", Json::str(INPUT_FORMAT.name())),
                ("local_accumulator", Json::str(LOCAL_ACCUMULATOR.name())),
                (
                    "local_kernel",
                    Json::str("sequential fused chain: exact products, one rounding per term, start +0"),
                ),
                ("rounding", Json::str("round-to-nearest, ties-to-even")),
                (
                    "link_model",
                    Json::obj(vec![
                        ("latency_us", Json::Num(c.link.latency_us)),
                        ("bandwidth_gbps", Json::Num(c.link.bandwidth_gbps)),
                        ("note", Json::str("illustrative alpha-beta model; not a measurement")),
                    ]),
                ),
            ]),
        ),
        ("elements_per_run", Json::uint(n)),
        (
            "baselines",
            Json::Arr(
                result
                    .baselines
                    .iter()
                    .map(|b| {
                        Json::obj(vec![
                            ("wire", Json::str(b.wire.name())),
                            ("single_device_vs_exact", comparison(&b.single_device_vs_exact, n)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "runs",
            Json::Arr(
                result
                    .worlds
                    .iter()
                    .map(|w| {
                        Json::obj(vec![
                            ("world", Json::uint(w.world)),
                            ("wire", Json::str(w.wire.name())),
                            (
                                "algorithms",
                                Json::Arr(
                                    w.algorithms
                                        .iter()
                                        .map(|a| {
                                            Json::obj(vec![
                                                ("algorithm", Json::str(a.algorithm.name())),
                                                ("consistent_across_ranks", Json::Bool(a.consistent_across_ranks)),
                                                ("roundings_per_element", Json::uint(a.roundings_per_element)),
                                                ("vs_exact", comparison(&a.vs_exact, n)),
                                                ("vs_single_device", comparison(&a.vs_single_device, n)),
                                                (
                                                    "argmax_changes_vs_single_device",
                                                    Json::uint(a.argmax_changes_vs_single_device),
                                                ),
                                                (
                                                    "cost",
                                                    Json::obj(vec![
                                                        ("rounds", Json::uint(a.cost.rounds)),
                                                        ("estimated_us", Json::Num(a.cost.estimated_us)),
                                                        (
                                                            "max_bytes_sent_per_rank",
                                                            Json::uint(a.cost.max_bytes_sent_per_rank),
                                                        ),
                                                        ("total_bytes", Json::uint(a.cost.total_bytes)),
                                                    ]),
                                                ),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "pairwise_mismatches",
                                Json::Arr(
                                    w.pairwise_mismatches
                                        .iter()
                                        .map(|(a, b, d)| {
                                            Json::obj(vec![
                                                ("a", Json::str(a.name())),
                                                ("b", Json::str(b.name())),
                                                ("mismatched", Json::uint(*d)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

fn pct(count: usize, total: usize) -> String {
    let p = 100.0 * count as f64 / total as f64;
    if count > 0 && p < 1.0 {
        format!("{p:.2}%")
    } else {
        format!("{p:.1}%")
    }
}

pub fn to_markdown(result: &ExperimentResult, command: &str) -> String {
    let c = &result.config;
    let n = result.elements;
    let mut md = String::new();
    md.push_str("# Tensor-parallel AllReduce: does the topology change the bits?\n\n");
    md.push_str(&format!(
        "Generated by `{command}` (fp-play-runtime {}). Re-running the command reproduces this file byte for byte.\n\n",
        env!("CARGO_PKG_VERSION")
    ));
    md.push_str(&format!(
        "**Setup.** `Y = X·W`, X {}×{} and W {}×{} in BF16 (seed `{:#x}`); the hidden dimension is split across P devices. \
Each device accumulates its slice in FP32 (exact products, one rounding per term), converts the partial to the wire format, \
and the partials are AllReduced. {} output elements per run. All arithmetic is exact with one round-to-nearest-even per modeled operation.\n\n",
        c.tokens, c.hidden, c.hidden, c.outputs, c.seed, n
    ));
    md.push_str("Columns: **roundings/elem** — roundings each output element undergoes inside the collective \
(not counting the conversion of the partials to the wire format before it; see the last section); \
**≠ exact** — elements differing from the exact dot product rounded once to the wire format; \
**≠ 1-device** — elements differing from the same FP32 kernel run unsharded on one device; \
**rel. RMS err** — ‖Y − Y_exact‖₂ / ‖Y_exact‖₂ computed in binary64 (a display estimate; per-element ulp statistics are in the JSON); **top-1 Δ** — tokens whose argmax output changes vs 1-device; **est. µs** — α–β estimate (");
    md.push_str(&format!(
        "α = {} µs, {} GB/s per link; illustrative, not a measurement).\n\n",
        c.link.latency_us, c.link.bandwidth_gbps
    ));

    for baseline in &result.baselines {
        let wire = baseline.wire;
        md.push_str(&format!("## Wire format: {}\n\n", wire.name().to_uppercase()));
        md.push_str(&format!(
            "Single-device baseline vs exact: {} of elements differ; relative RMS error {:.2e}.\n\n",
            pct(baseline.single_device_vs_exact.mismatched, n),
            baseline.single_device_vs_exact.relative_rms_error
        ));
        md.push_str("| P | algorithm | ranks agree | roundings/elem | ≠ exact | rel. RMS err vs exact | ≠ 1-device | top-1 Δ | rounds | est. µs | max bytes sent/rank |\n");
        md.push_str("|--:|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|\n");
        for w in result.worlds.iter().filter(|w| w.wire == wire) {
            for a in &w.algorithms {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.2e} | {} | {} | {} | {:.2} | {} |\n",
                    w.world,
                    a.algorithm.name(),
                    if a.consistent_across_ranks { "yes" } else { "**no**" },
                    a.roundings_per_element,
                    pct(a.vs_exact.mismatched, n),
                    a.vs_exact.relative_rms_error,
                    pct(a.vs_single_device.mismatched, n),
                    a.argmax_changes_vs_single_device,
                    a.cost.rounds,
                    a.cost.estimated_us,
                    a.cost.max_bytes_sent_per_rank,
                ));
            }
        }
        md.push('\n');
        md.push_str(
            "Pairwise disagreement between the four reduction schedules at the same P (elements whose bits differ; \
pairs that include exact-gather are in the JSON):\n\n",
        );
        for w in result.worlds.iter().filter(|w| w.wire == wire) {
            let parts: Vec<String> = w
                .pairwise_mismatches
                .iter()
                .filter(|(a, b, _)| *a != Algorithm::ExactGather && *b != Algorithm::ExactGather)
                .map(|(a, b, d)| format!("{} vs {}: {}", a.name(), b.name(), pct(*d, n)))
                .collect();
            md.push_str(&format!("- P = {}: {}\n", w.world, parts.join("; ")));
        }
        md.push('\n');
    }

    md.push_str("## What this does and does not show\n\n");
    md.push_str("- Inside the collective, every reduction schedule performs the same number of wire-format roundings per element \
(P − 1 additions) and exact-gather performs one; the differences between schedules come only from association order, \
which each schedule fixes differently. Ring folds every chunk starting at a different rank.\n");
    md.push_str("- Before the collective, each device's FP32 partial is converted to the wire format: exact for the FP32 wire, \
but one rounding of each of the P partials for the BF16 wire. That conversion, not the collective, is why BF16 exact-gather \
stays far from the single-device baseline even though it rounds only once.\n");
    md.push_str("- exact-gather is topology-independent: for fixed partials its result does not depend on the schedule, at the cost of all-gathering every input.\n");
    md.push_str("- Changing P changes the local partial sums themselves, so no AllReduce can make a sharded run match the unsharded kernel bit for bit.\n");
    md.push_str("- Inputs are synthetic (Irwin–Hall normal, weights scaled by 1/√hidden); the local kernel is one sequential order, not a blocked GEMM; \
the cost model ignores contention, compute and overlap. No hardware, interconnect or Etched-system measurement is claimed.\n");
    md
}

// ---------------------------------------------------------------------------
// R1.1: wire policies and reduce-scatter on its own (results/r1.1-policies.*)
// ---------------------------------------------------------------------------

pub const POLICIES_SCHEMA: &str = "fp-play-runtime/r1.1-policies/v1";

fn cost_json(c: &CostEstimate) -> Json {
    Json::obj(vec![
        ("rounds", Json::uint(c.rounds)),
        ("estimated_us", Json::Num(c.estimated_us)),
        ("max_bytes_sent_per_rank", Json::uint(c.max_bytes_sent_per_rank)),
        ("max_reduce_bytes_per_rank", Json::uint(c.max_reduce_bytes_per_rank)),
        ("max_gather_bytes_per_rank", Json::uint(c.max_gather_bytes_per_rank)),
        ("total_bytes", Json::uint(c.total_bytes)),
    ])
}

fn policy_json(p: &WirePolicy) -> Json {
    Json::obj(vec![
        ("name", Json::str(p.name())),
        ("accumulator", Json::str(p.accumulator.name())),
        ("reduce_wire", Json::str(p.reduce_wire.name())),
        ("gather_wire", Json::str(p.gather_wire.name())),
        ("output", Json::str(p.output.name())),
    ])
}

fn optional(x: Option<f64>) -> Json {
    x.map_or(Json::Null, Json::Num)
}

/// The R1.1 policies artifact. No timestamps or host details, so reruns are byte-identical.
pub fn policies_to_json(result: &PoliciesResult, command: &str) -> Json {
    let c = &result.config;
    let n = result.elements;
    Json::obj(vec![
        ("schema", Json::str(POLICIES_SCHEMA)),
        (
            "tool",
            Json::str(format!("fp-play-runtime {}", env!("CARGO_PKG_VERSION"))),
        ),
        ("command", Json::str(command)),
        (
            "config",
            Json::obj(vec![
                ("tokens", Json::uint(c.tokens)),
                ("hidden", Json::uint(c.hidden)),
                ("outputs", Json::uint(c.outputs)),
                (
                    "world_sizes",
                    Json::Arr(c.world_sizes.iter().map(|&w| Json::uint(w)).collect()),
                ),
                ("seed", Json::Int(c.seed as i64)),
                ("input_format", Json::str(INPUT_FORMAT.name())),
                ("local_accumulator", Json::str(LOCAL_ACCUMULATOR.name())),
                (
                    "local_kernel",
                    Json::str("sequential fused chain: exact products, one rounding per term, start +0"),
                ),
                ("output_format", Json::str(policies::OUTPUT.name())),
                (
                    "policies",
                    Json::Arr(policies::POLICIES.iter().map(policy_json).collect()),
                ),
                (
                    "ideal",
                    Json::str("exact sum of the P FP32 partials rounded once to the output format"),
                ),
                (
                    "recovered_fraction",
                    Json::str("(err_bf16 - err_policy) / (err_bf16 - err_ideal), errors against exact; null when err_bf16 <= err_ideal"),
                ),
                (
                    "vs_bf16_exact_gather_note",
                    Json::str("same ratio with bf16 exact-gather as the target; it can exceed 1 because bf16 exact-gather rounds each FP32 partial to BF16 before its one rounding"),
                ),
                (
                    "phases",
                    Json::Arr(
                        Algorithm::ALL
                            .iter()
                            .map(|a| {
                                let (reduce, gather) = a.phases();
                                Json::obj(vec![
                                    ("algorithm", Json::str(a.name())),
                                    ("reduce_phase", Json::str(reduce)),
                                    ("gather_phase", Json::str(gather)),
                                    ("reduce_scatter_only", Json::Bool(policies::SPLIT_ALGORITHMS.contains(a))),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("rounding", Json::str("round-to-nearest, ties-to-even")),
                (
                    "link_model",
                    Json::obj(vec![
                        ("latency_us", Json::Num(c.link.latency_us)),
                        ("bandwidth_gbps", Json::Num(c.link.bandwidth_gbps)),
                        ("note", Json::str("illustrative alpha-beta model; not a measurement")),
                    ]),
                ),
            ]),
        ),
        ("elements_per_run", Json::uint(n)),
        ("single_device_vs_exact", comparison(&result.single_device_vs_exact, n)),
        (
            "runs",
            Json::Arr(
                result
                    .worlds
                    .iter()
                    .map(|w| {
                        Json::obj(vec![
                            ("world", Json::uint(w.world)),
                            ("ideal_vs_exact", comparison(&w.ideal_vs_exact, n)),
                            (
                                "algorithms",
                                Json::Arr(
                                    w.runs
                                        .iter()
                                        .map(|r| {
                                            Json::obj(vec![
                                                ("policy", Json::str(r.policy.name())),
                                                ("algorithm", Json::str(r.algorithm.name())),
                                                ("consistent_across_ranks", Json::Bool(r.consistent_across_ranks)),
                                                (
                                                    "roundings_per_element",
                                                    Json::obj(vec![
                                                        ("before_collective", Json::uint(r.input_roundings as usize)),
                                                        (
                                                            "collective_min",
                                                            Json::uint(r.collective_roundings.0 as usize),
                                                        ),
                                                        (
                                                            "collective_max",
                                                            Json::uint(r.collective_roundings.1 as usize),
                                                        ),
                                                    ]),
                                                ),
                                                ("vs_exact", comparison(&r.vs_exact, n)),
                                                ("vs_ideal", comparison(&r.vs_ideal, n)),
                                                ("vs_single_device", comparison(&r.vs_single_device, n)),
                                                (
                                                    "argmax_changes_vs_single_device",
                                                    Json::uint(r.argmax_changes_vs_single_device),
                                                ),
                                                ("cost", cost_json(&r.cost)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "recovery",
                                Json::Arr(
                                    w.recovery
                                        .iter()
                                        .map(|r| {
                                            Json::obj(vec![
                                                ("algorithm", Json::str(r.algorithm.name())),
                                                ("policy", Json::str(r.policy.name())),
                                                ("bf16_vs_exact", comparison(&r.baseline, n)),
                                                ("policy_vs_exact", comparison(&r.policy_result, n)),
                                                ("ideal_vs_exact", comparison(&r.ideal, n)),
                                                ("recovered_fraction_mismatched", optional(r.mismatch_fraction)),
                                                ("recovered_fraction_rms", optional(r.rms_fraction)),
                                                (
                                                    "vs_bf16_exact_gather",
                                                    Json::obj(vec![
                                                        ("bf16_exact_gather_vs_exact", comparison(&r.bf16_exact_gather, n)),
                                                        (
                                                            "recovered_fraction_mismatched",
                                                            optional(r.mismatch_fraction_vs_bf16_exact_gather),
                                                        ),
                                                        (
                                                            "recovered_fraction_rms",
                                                            optional(r.rms_fraction_vs_bf16_exact_gather),
                                                        ),
                                                    ]),
                                                ),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            (
                                "reduce_scatter_only",
                                Json::Arr(
                                    w.split
                                        .iter()
                                        .map(|s| {
                                            Json::obj(vec![
                                                ("algorithm", Json::str(s.algorithm.name())),
                                                ("policy", Json::str(s.policy.name())),
                                                ("bits_match_all_reduce", Json::Bool(s.bits_match_all_reduce)),
                                                ("reduce_scatter", cost_json(&s.reduce_scatter)),
                                                ("all_reduce", cost_json(&s.all_reduce)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

/// `lo–hi` over the given values, or one value when they agree.
fn pct_range(values: impl Iterator<Item = usize>, total: usize) -> String {
    let values: Vec<usize> = values.collect();
    let (lo, hi) = (values.iter().min(), values.iter().max());
    match (lo, hi) {
        (Some(lo), Some(hi)) if lo == hi => pct(*lo, total),
        (Some(lo), Some(hi)) => format!("{}–{}", pct(*lo, total), pct(*hi, total)),
        _ => "—".to_string(),
    }
}

/// Absolute error and bytes, per P, over the four reduction schedules.
fn headline(result: &PoliciesResult) -> String {
    let n = result.elements;
    let bf16 = policies::POLICIES[0];
    let mut md = String::from(
        "Elements of Y that differ from the exact result, over the reduction schedules (sequential, ring, binomial-tree and, \
for powers of two, halving-doubling), and the most bytes any rank sends in a ring AllReduce relative to `bf16`. Policies \
are defined below.\n\n",
    );
    md.push_str("| P | `bf16` | `fp32-rs/bf16-ag` | `bf16-wire/fp32-add` | ideal | `bf16` exact-gather | ring bytes/rank: `fp32-rs/bf16-ag` | ring bytes/rank: `bf16-wire/fp32-add` |\n");
    md.push_str("|--:|--:|--:|--:|--:|--:|--:|--:|\n");
    for w in &result.worlds {
        let wrong = |p: WirePolicy| {
            pct_range(
                w.runs
                    .iter()
                    .filter(|r| r.policy == p && r.algorithm != Algorithm::ExactGather)
                    .map(|r| r.vs_exact.mismatched),
                n,
            )
        };
        let ring_bytes = |p: WirePolicy| w.run(p, Algorithm::Ring).map(|r| r.cost.max_bytes_sent_per_rank as f64);
        let ratio = |p: WirePolicy| match (ring_bytes(p), ring_bytes(bf16)) {
            (Some(a), Some(b)) if b > 0.0 => format!("{:.2}×", a / b),
            _ => "—".to_string(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            w.world,
            wrong(bf16),
            wrong(WirePolicy::FP32_RS_BF16_AG),
            wrong(WirePolicy::BF16_WIRE_FP32_ADD),
            pct(w.ideal_vs_exact.mismatched, n),
            w.run(bf16, Algorithm::ExactGather)
                .map_or("—".to_string(), |r| pct(r.vs_exact.mismatched, n)),
            ratio(WirePolicy::FP32_RS_BF16_AG),
            ratio(WirePolicy::BF16_WIRE_FP32_ADD),
        ));
    }
    md.push('\n');
    if let Some(w) = result.worlds.last() {
        let range = |p: WirePolicy| {
            pct_range(
                w.runs
                    .iter()
                    .filter(|r| r.policy == p && r.algorithm != Algorithm::ExactGather)
                    .map(|r| r.vs_exact.mismatched),
                n,
            )
        };
        let bytes = |p: WirePolicy| w.run(p, Algorithm::Ring).map_or(0, |r| r.cost.max_bytes_sent_per_rank);
        md.push_str(&format!(
            "At P = {}: {} of elements wrong under `bf16` become {} under `fp32-rs/bf16-ag`, for {:.2}× the bytes per rank \
in a ring AllReduce; `bf16-wire/fp32-add` sends the same bytes as `bf16` and gets {} wrong.\n\n",
            w.world,
            range(bf16),
            range(WirePolicy::FP32_RS_BF16_AG),
            bytes(WirePolicy::FP32_RS_BF16_AG) as f64 / bytes(bf16).max(1) as f64,
            range(WirePolicy::BF16_WIRE_FP32_ADD),
        ));
    }
    md
}

fn percent(fraction: Option<f64>) -> String {
    fraction.map_or("—".to_string(), |f| format!("{:.1}%", 100.0 * f))
}

pub fn policies_to_markdown(result: &PoliciesResult, command: &str) -> String {
    let c = &result.config;
    let n = result.elements;
    let mut md = String::new();
    md.push_str("# R1.1: BF16 output with FP32 where it can help\n\n");
    md.push_str(&format!(
        "Generated by `{command}` (fp-play-runtime {}). Re-running the command reproduces this file byte for byte.\n\n",
        env!("CARGO_PKG_VERSION")
    ));
    md.push_str(&format!(
        "**Setup.** The problem of [tp-allreduce.md](tp-allreduce.md): `Y = X·W`, X {}×{} and W {}×{} in BF16 (seed `{:#x}`), \
the hidden dimension split across P devices, each computing an FP32 partial over its slice (exact products, one FP32 rounding \
per term). The output is BF16. {} output elements per run. All arithmetic is exact with one round-to-nearest-even per \
modeled operation.\n\n",
        c.tokens, c.hidden, c.hidden, c.outputs, c.seed, n
    ));
    md.push_str("## Headline\n\n");
    md.push_str(&headline(result));
    md.push_str("**Policies.** These are modeled policies, not descriptions of any particular library.\n\n");
    md.push_str("- `bf16` — R1's BF16 wire: every FP32 partial is rounded to BF16 before the collective, and every combine is a BF16 \
addition. 2 bytes per element in every phase.\n");
    md.push_str("- `fp32-rs/bf16-ag` — FP32 partials and FP32 additions. Reduce phases (reduce-scatter, chain or tree reduce, \
exact-gather's input gather) send FP32, 4 bytes per element; a finished value is rounded once to BF16, and gather phases \
(all-gather, broadcast) send BF16.\n");
    md.push_str("- `bf16-wire/fp32-add` — FP32 partials and FP32 additions, but every message is BF16: each hop rounds the \
sender's FP32 partial to BF16 and the receiver adds it to its own FP32 partial. 2 bytes per element in every phase.\n\n");
    md.push_str("**Where each policy applies.** Every policy applies to all five schedules: each round is a reduce phase \
(its payload is combined or summed; `reduce_wire`) or a gather phase (its payload is a finished value that is only copied; \
`gather_wire`). Only ring and halving-doubling have a reduce-scatter that can be used on its own.\n\n");
    md.push_str("| schedule | reduce phase | gather phase | reduce-scatter only |\n|---|---|---|:-:|\n");
    for a in Algorithm::ALL {
        let (reduce, gather) = a.phases();
        md.push_str(&format!(
            "| {} | {reduce} | {gather} | {} |\n",
            a.name(),
            if policies::SPLIT_ALGORITHMS.contains(&a) {
                "yes"
            } else {
                "no"
            }
        ));
    }
    md.push('\n');
    md.push_str(&format!(
        "**References.** **≠ exact** — elements differing from the exact dot product rounded once to BF16. \
**≠ ideal** — elements differing from the exact sum of the P FP32 partials rounded once to BF16: the best any collective \
can do once the local kernels have run. The unsharded FP32 kernel rounded to BF16 differs from exact on {} of elements.\n\n",
        pct(result.single_device_vs_exact.mismatched, n)
    ));
    md.push_str("Columns: **roundings/elem** — operations that can round in each output element's reduction tree, \
before the collective (the conversion of the P partials) + inside it (measured from the schedule; additions, roundings of \
an exact sum, and narrowing conversions); **rel. RMS err** — ‖Y − Y_exact‖₂ / ‖Y_exact‖₂ in binary64 (a display estimate); \
**top-1 Δ** — tokens whose argmax output changes vs the unsharded kernel; **reduce / gather B/rank** — the most bytes any \
rank sends in reduce phases and in gather phases; **est. µs** — α–β estimate (");
    md.push_str(&format!(
        "α = {} µs, {} GB/s per link; illustrative, not a measurement).\n\n",
        c.link.latency_us, c.link.bandwidth_gbps
    ));

    md.push_str("## Results\n\n");
    for w in &result.worlds {
        md.push_str(&format!(
            "- P = {}: the ideal for these partials differs from exact on {} of elements.\n",
            w.world,
            pct(w.ideal_vs_exact.mismatched, n)
        ));
    }
    md.push('\n');
    md.push_str("| P | policy | algorithm | ranks agree | roundings/elem | ≠ exact | ≠ ideal | rel. RMS err | top-1 Δ | rounds | reduce B/rank | gather B/rank | est. µs |\n");
    md.push_str("|--:|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n");
    for w in &result.worlds {
        for r in &w.runs {
            let (lo, hi) = r.collective_roundings;
            let collective = if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}–{hi}")
            };
            md.push_str(&format!(
                "| {} | `{}` | {} | {} | {} + {} | {} | {} | {:.2e} | {} | {} | {} | {} | {:.2} |\n",
                w.world,
                r.policy.name(),
                r.algorithm.name(),
                if r.consistent_across_ranks { "yes" } else { "**no**" },
                r.input_roundings,
                collective,
                pct(r.vs_exact.mismatched, n),
                pct(r.vs_ideal.mismatched, n),
                r.vs_exact.relative_rms_error,
                r.argmax_changes_vs_single_device,
                r.cost.rounds,
                r.cost.max_reduce_bytes_per_rank,
                r.cost.max_gather_bytes_per_rank,
                r.cost.estimated_us,
            ));
        }
    }
    md.push('\n');

    md.push_str("## How much of the gap to the ideal each mixed policy closes\n\n");
    md.push_str("For each reduction schedule: **closed** = (err under `bf16` − err under the policy) ÷ (err under `bf16` \
− err of the ideal), errors measured against exact, on mismatch count and (in parentheses) on relative RMS error. 100% means \
the policy is as close to exact as the ideal for these partials; above 100% would mean it is closer by chance. The same ratio \
with BF16 exact-gather as the target is in the JSON only (`vs_bf16_exact_gather`); it exceeds 100% because BF16 \
exact-gather rounds each FP32 partial to BF16 before its one rounding.\n\n");
    md.push_str("| P | algorithm | ≠ exact: `bf16` | ≠ exact: ideal | ≠ exact: `fp32-rs/bf16-ag` | closed | ≠ exact: `bf16-wire/fp32-add` | closed |\n");
    md.push_str("|--:|---|--:|--:|--:|--:|--:|--:|\n");
    for w in &result.worlds {
        for a in Algorithm::ALL {
            let find = |p: WirePolicy| w.recovery.iter().find(|r| r.algorithm == a && r.policy == p);
            let (Some(rs), Some(hop)) = (find(WirePolicy::FP32_RS_BF16_AG), find(WirePolicy::BF16_WIRE_FP32_ADD))
            else {
                continue;
            };
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} ({}) | {} | {} ({}) |\n",
                w.world,
                a.name(),
                pct(rs.baseline.mismatched, n),
                pct(rs.ideal.mismatched, n),
                pct(rs.policy_result.mismatched, n),
                percent(rs.mismatch_fraction),
                percent(rs.rms_fraction),
                pct(hop.policy_result.mismatched, n),
                percent(hop.mismatch_fraction),
                percent(hop.rms_fraction),
            ));
        }
    }
    md.push('\n');

    md.push_str("## Reduce-scatter only\n\n");
    md.push_str(
        "Sequence-parallel output keeps each device's owned slice instead of all-gathering it. **bits = slice** — \
the owned slices, placed together, equal the AllReduce output bit for bit (the all-gather only copies).\n\n",
    );
    md.push_str("| P | algorithm | policy | bits = slice | RS rounds | RS est. µs | RS max bytes/rank | AllReduce rounds | AllReduce est. µs | AllReduce max bytes/rank |\n");
    md.push_str("|--:|---|---|:-:|--:|--:|--:|--:|--:|--:|\n");
    for w in &result.worlds {
        for s in &w.split {
            md.push_str(&format!(
                "| {} | {} | `{}` | {} | {} | {:.2} | {} | {} | {:.2} | {} |\n",
                w.world,
                s.algorithm.name(),
                s.policy.name(),
                if s.bits_match_all_reduce { "yes" } else { "**no**" },
                s.reduce_scatter.rounds,
                s.reduce_scatter.estimated_us,
                s.reduce_scatter.max_bytes_sent_per_rank,
                s.all_reduce.rounds,
                s.all_reduce.estimated_us,
                s.all_reduce.max_bytes_sent_per_rank,
            ));
        }
    }
    md.push('\n');

    md.push_str("## What this does and does not show\n\n");
    md.push_str(
        "- The two mixed policies are modeled from their definitions above; neither is attributed to any library or \
hardware.\n",
    );
    md.push_str("- **≠ ideal** isolates the collective: the ideal already includes every rounding of the local FP32 kernels, so \
the gap from ideal to exact is the kernels', not the collective's. Exact-gather under `fp32-rs/bf16-ag` sums the FP32 \
partials exactly and rounds once, so it is the ideal by construction.\n");
    md.push_str("- `bf16-wire/fp32-add` performs as many roundings per element as `bf16` (one per partial plus the additions); \
the difference is where they happen. In the reduction schedules additions are FP32 and the receiving device's own partial is never \
quantized. Exact-gather uses every input in its wire form, the device's own included, so under this policy it gives exactly \
the `bf16` exact-gather output.\n");
    md.push_str(
        "- Reduce-scatter chunks are ranges of the flattened, token-major output. They line up with token boundaries \
only when P divides the token count, so this is not exactly a sequence-parallel layout for every P.\n",
    );
    md.push_str("- Inputs are synthetic (Irwin–Hall normal, weights scaled by 1/√hidden); the local kernel is one sequential \
order, not a blocked GEMM; the cost model ignores contention, compute and overlap. No hardware or interconnect measurement \
is claimed.\n");
    md
}

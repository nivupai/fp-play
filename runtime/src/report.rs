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
                                                ("bf16_mismatched", Json::uint(r.baseline_mismatched)),
                                                ("policy_mismatched", Json::uint(r.policy_mismatched)),
                                                ("bf16_exact_gather_mismatched", Json::uint(r.exact_gather_mismatched)),
                                                ("recovered_fraction_mismatched", optional(r.mismatch_fraction)),
                                                ("recovered_fraction_rms", optional(r.rms_fraction)),
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

fn percent(fraction: Option<f64>) -> String {
    fraction.map_or("—".to_string(), |f| format!("{:.0}%", 100.0 * f))
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
    md.push_str("**Policies.** These are modeled policies, not descriptions of any particular library.\n\n");
    md.push_str("- `bf16` — R1's BF16 wire: every FP32 partial is rounded to BF16 before the collective, and every combine is a BF16 \
addition. 2 bytes per element in every phase.\n");
    md.push_str("- `fp32-rs/bf16-ag` — FP32 partials and FP32 additions. Reduce phases (reduce-scatter, chain or tree reduce, \
exact-gather's input gather) send FP32, 4 bytes per element; a finished value is rounded once to BF16, and gather phases \
(all-gather, broadcast) send BF16.\n");
    md.push_str("- `bf16-wire/fp32-add` — FP32 partials and FP32 additions, but every message is BF16: each hop rounds the \
sender's FP32 partial to BF16 and the receiver adds it to its own FP32 partial. 2 bytes per element in every phase.\n\n");
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

    md.push_str("## How much of exact-gather's accuracy the mixed policies recover\n\n");
    md.push_str("For each reduction schedule: **recovered** = (≠ exact under `bf16` − ≠ exact under the policy) ÷ \
(≠ exact under `bf16` − ≠ exact for `bf16` exact-gather), the share of the gap between R1's BF16 wire and BF16 exact-gather \
that the policy closes; the same ratio on relative RMS error is in parentheses. Above 100% means the policy is closer to \
exact than BF16 exact-gather. That is possible because BF16 exact-gather still rounds each FP32 partial to BF16 before its \
single rounding. — means there is no gap: with two devices every `bf16` schedule computes the same sum as `bf16` exact-gather.\n\n");
    md.push_str("| P | algorithm | ≠ exact: `bf16` | ≠ exact: `bf16` exact-gather | ≠ exact: `fp32-rs/bf16-ag` | recovered | ≠ exact: `bf16-wire/fp32-add` | recovered |\n");
    md.push_str("|--:|---|--:|--:|--:|--:|--:|--:|\n");
    for w in &result.worlds {
        let algorithms: Vec<Algorithm> = Algorithm::ALL
            .into_iter()
            .filter(|a| w.recovery.iter().any(|r| r.algorithm == *a))
            .collect();
        for a in algorithms {
            let find = |p: WirePolicy| w.recovery.iter().find(|r| r.algorithm == a && r.policy == p);
            let (Some(rs), Some(hop)) = (find(WirePolicy::FP32_RS_BF16_AG), find(WirePolicy::BF16_WIRE_FP32_ADD))
            else {
                continue;
            };
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} ({}) | {} | {} ({}) |\n",
                w.world,
                a.name(),
                pct(rs.baseline_mismatched, n),
                pct(rs.exact_gather_mismatched, n),
                pct(rs.policy_mismatched, n),
                percent(rs.mismatch_fraction),
                percent(rs.rms_fraction),
                pct(hop.policy_mismatched, n),
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

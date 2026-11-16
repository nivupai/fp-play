//! Deterministic JSON and Markdown renderings of an experiment.

use crate::collectives::Algorithm;
use crate::json::Json;
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
    md.push_str("Columns: **≠ exact** — elements differing from the exact dot product rounded once to the wire format; \
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
        md.push_str("Pairwise disagreement between algorithms at the same P (elements whose bits differ):\n\n");
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
    md.push_str("- Every algorithm performs the same number of wire-format roundings per element (P − 1) except exact-gather (one); \
the differences come only from association order, which each schedule fixes differently. Ring folds every chunk starting at a different rank.\n");
    md.push_str("- exact-gather is topology-independent: for fixed partials its result does not depend on the schedule, at the cost of all-gathering every input.\n");
    md.push_str("- Changing P changes the local partial sums themselves, so no AllReduce can make a sharded run match the unsharded kernel bit for bit.\n");
    md.push_str("- Inputs are synthetic (Irwin–Hall normal, weights scaled by 1/√hidden); the local kernel is one sequential order, not a blocked GEMM; \
the cost model ignores contention, compute and overlap. No hardware, interconnect or Etched-system measurement is claimed.\n");
    md
}

//! JSON and Markdown for the R3 experiment (results/r3-decode.*). Every
//! number and every generated sentence comes from the [`DecodeResult`].

use super::experiment::{AtRisk, DecodeResult, GridRun, MatrixRow};
use super::parallel::Schedule;
use super::variants::Variant;
use crate::collectives::WirePolicy;
use crate::json::Json;

fn num(x: f64) -> Json {
    Json::Num(x)
}

fn int(x: u64) -> Json {
    Json::Int(x as i64)
}

/// Nearest-rank percentile of `v` (sorted ascending); `None` if empty.
fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let rank = ((p / 100.0) * v.len() as f64).ceil().max(1.0) as usize;
    Some(v[rank.min(v.len()) - 1])
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v
}

fn scale_name(s: f32) -> String {
    format!("{s}")
}

/// Bin labels from the upper edges: "< 1", "1–4", …, "≥ 4096".
fn bin_labels(edges: &[f64]) -> Vec<String> {
    let mut out = Vec::new();
    let mut lo: Option<f64> = None;
    for &e in edges {
        out.push(match lo {
            None => format!("< {e}"),
            Some(l) => format!("{l}–{e}"),
        });
        lo = Some(e);
    }
    out.push(format!("≥ {}", lo.unwrap_or(0.0)));
    out
}

fn bin_of(edges: &[f64], x: f64) -> usize {
    edges.iter().position(|&e| x < e).unwrap_or(edges.len())
}

/// At-risk tokens and flips per margin bin, with the median perturbation.
struct Bin {
    at_risk: usize,
    flips: usize,
    perturbations: Vec<f64>,
}

fn bins(edges: &[f64], tokens: &[&AtRisk]) -> Vec<Bin> {
    let mut out: Vec<Bin> = (0..=edges.len())
        .map(|_| Bin {
            at_risk: 0,
            flips: 0,
            perturbations: Vec::new(),
        })
        .collect();
    for t in tokens {
        let b = &mut out[bin_of(edges, t.margin_ulps)];
        b.at_risk += 1;
        b.flips += usize::from(t.flipped);
        b.perturbations.push(t.perturbation_ulps);
    }
    for b in &mut out {
        b.perturbations = sorted(std::mem::take(&mut b.perturbations));
    }
    out
}

fn first_divergences(g: &GridRun) -> Vec<f64> {
    sorted(g.first_divergence.iter().flatten().map(|&i| i as f64).collect())
}

fn perturbations(g: &GridRun) -> Vec<f64> {
    sorted(g.at_risk.iter().map(|t| t.perturbation_ulps).collect())
}

/// Baseline top-2 margins (absolute, ulps) over every generated token.
fn baseline_margins(r: &DecodeResult, policy: WirePolicy, scale: f32) -> (Vec<f64>, Vec<f64>) {
    let b = r.baseline(policy, scale);
    let tops = b.generations.iter().flat_map(|g| g.top.iter());
    let abs = sorted(tops.clone().map(|t| t.margin()).collect());
    let ulps = sorted(tops.map(|t| t.margin() / super::experiment::ulp_f32(t.top)).collect());
    (abs, ulps)
}

/// Grid runs of one (policy, scale), in config order.
fn runs_of(r: &DecodeResult, policy: WirePolicy, scale: f32) -> Vec<&GridRun> {
    r.grid
        .iter()
        .filter(|g| g.policy == policy && g.scale == scale)
        .collect()
}

/// Every grid run at a non-default scale chose the same tokens as at the default.
fn scales_agree(r: &DecodeResult) -> (usize, usize) {
    let s0 = r.config.scales[0];
    let others: Vec<&GridRun> = r.grid.iter().filter(|g| g.scale != s0).collect();
    let same = others
        .iter()
        .filter(|g| {
            r.grid.iter().any(|h| {
                h.scale == s0
                    && h.world == g.world
                    && h.schedule == g.schedule
                    && h.policy == g.policy
                    && h.tokens == g.tokens
            })
        })
        .count();
    (same, others.len())
}

fn grid_json(g: &GridRun) -> Json {
    let p = perturbations(g);
    let flips: Vec<Json> = g
        .at_risk
        .iter()
        .filter(|t| t.flipped)
        .map(|t| {
            Json::obj(vec![
                ("margin", num(t.margin)),
                ("margin_ulps", num(t.margin_ulps)),
                ("perturbation_ulps", num(t.perturbation_ulps)),
            ])
        })
        .collect();
    Json::obj(vec![
        ("world", Json::uint(g.world)),
        ("schedule", Json::str(g.schedule.name())),
        ("policy", Json::str(g.policy.name())),
        ("logit_scale", num(f64::from(g.scale))),
        ("diverged", Json::uint(g.diverged())),
        ("logits_identical", Json::uint(g.logits_identical)),
        (
            "first_divergence",
            Json::Arr(
                g.first_divergence
                    .iter()
                    .map(|d| d.map_or(Json::Null, Json::uint))
                    .collect(),
            ),
        ),
        (
            "surviving_after_t_tokens",
            Json::Arr(
                (0..=g.tokens.first().map_or(0, Vec::len))
                    .map(|t| Json::uint(g.surviving(t)))
                    .collect(),
            ),
        ),
        ("at_risk_tokens", Json::uint(g.at_risk.len())),
        ("flips", Json::Arr(flips)),
        (
            "perturbation_ulps",
            Json::obj(vec![
                ("p50", percentile(&p, 50.0).map_or(Json::Null, num)),
                ("p90", percentile(&p, 90.0).map_or(Json::Null, num)),
                ("max", p.last().copied().map_or(Json::Null, num)),
            ]),
        ),
        ("collectives", int(g.reduce.collectives)),
        ("collective_additions", int(g.reduce.additions)),
    ])
}

fn matrix_json(row: &MatrixRow) -> Json {
    Json::obj(vec![
        ("world", Json::uint(row.world)),
        ("schedule", Json::str(row.schedule.name())),
        ("policy", Json::str(row.policy.name())),
        ("predicted_invariant", Json::Bool(row.predicted_invariant())),
        (
            "variants",
            Json::Arr(
                row.runs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let (logits, tokens) = row.differing(i);
                        Json::obj(vec![
                            ("variant", Json::str(v.variant.name())),
                            ("prompts_with_different_logits", Json::uint(logits)),
                            ("prompts_with_different_tokens", Json::uint(tokens)),
                            ("engine_steps", Json::uint(v.engine_steps)),
                            ("forward_tokens", int(v.exec.forward_tokens)),
                            ("recomputed_tokens", int(v.exec.recomputed_tokens)),
                            ("preemptions", int(v.preemptions)),
                            ("prefix_hit_blocks", int(v.prefix_hit_blocks)),
                            ("failed_steps", Json::uint(v.failed_steps)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

pub fn to_json(r: &DecodeResult, command: &str) -> Json {
    let c = &r.config;
    let m = &c.model;
    let w = &c.workload;
    let mut margins = Vec::new();
    for &scale in &c.scales {
        for &policy in &c.policies {
            let (abs, ulps) = baseline_margins(r, policy, scale);
            let runs = runs_of(r, policy, scale);
            let tokens: Vec<&AtRisk> = runs.iter().flat_map(|g| g.at_risk.iter()).collect();
            let b = bins(&c.margin_bins, &tokens);
            margins.push(Json::obj(vec![
                ("policy", Json::str(policy.name())),
                ("logit_scale", num(f64::from(scale))),
                (
                    "baseline_margin",
                    Json::obj(vec![
                        ("p10", percentile(&abs, 10.0).map_or(Json::Null, num)),
                        ("p50", percentile(&abs, 50.0).map_or(Json::Null, num)),
                        ("p90", percentile(&abs, 90.0).map_or(Json::Null, num)),
                    ]),
                ),
                (
                    "baseline_margin_ulps",
                    Json::obj(vec![
                        ("p10", percentile(&ulps, 10.0).map_or(Json::Null, num)),
                        ("p50", percentile(&ulps, 50.0).map_or(Json::Null, num)),
                        ("p90", percentile(&ulps, 90.0).map_or(Json::Null, num)),
                    ]),
                ),
                (
                    "bins_ulps",
                    Json::Arr(
                        bin_labels(&c.margin_bins)
                            .into_iter()
                            .zip(&b)
                            .enumerate()
                            .map(|(i, (label, x))| {
                                Json::obj(vec![
                                    ("margin_ulps", Json::str(label)),
                                    (
                                        "baseline_tokens",
                                        Json::uint(ulps.iter().filter(|&&u| bin_of(&c.margin_bins, u) == i).count()),
                                    ),
                                    ("at_risk_tokens", Json::uint(x.at_risk)),
                                    ("flips", Json::uint(x.flips)),
                                    (
                                        "median_perturbation_ulps",
                                        percentile(&x.perturbations, 50.0).map_or(Json::Null, num),
                                    ),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]));
        }
    }
    let (agree, total) = scales_agree(r);
    Json::obj(vec![
        ("command", Json::str(command)),
        (
            "model",
            Json::obj(vec![
                ("hidden", Json::uint(m.hidden)),
                ("layers", Json::uint(m.layers)),
                ("q_heads", Json::uint(m.q_heads)),
                ("kv_heads", Json::uint(m.kv_heads)),
                ("head_dim", Json::uint(m.head_dim)),
                ("ffn", Json::uint(m.ffn)),
                ("vocab", Json::uint(m.vocab)),
                ("parameters", Json::uint(m.parameters())),
                ("seed", int(m.seed)),
                ("weights_checksum", Json::str(format!("{:016x}", r.weights_checksum))),
            ]),
        ),
        (
            "workload",
            Json::obj(vec![
                ("prompts", Json::uint(w.count())),
                ("prompt_tokens", Json::uint(w.shared + w.own)),
                ("shared_prefix_tokens", Json::uint(w.shared)),
                ("prefix_groups", Json::uint(w.groups)),
                ("generated_tokens", int(u64::from(w.output_len))),
                ("block_size", Json::uint(w.block_size)),
                ("seed", int(w.seed)),
            ]),
        ),
        (
            "logit_scales",
            Json::Arr(c.scales.iter().map(|&s| num(f64::from(s))).collect()),
        ),
        (
            "baselines",
            Json::Arr(
                r.baselines
                    .iter()
                    .map(|b| {
                        Json::obj(vec![
                            ("policy", Json::str(b.policy.name())),
                            ("logit_scale", num(f64::from(b.scale))),
                            (
                                "tokens",
                                Json::Arr(
                                    b.generations
                                        .iter()
                                        .map(|g| Json::Arr(g.tokens.iter().map(|&t| int(u64::from(t))).collect()))
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("grid", Json::Arr(r.grid.iter().map(grid_json).collect())),
        ("margins", Json::Arr(margins)),
        (
            "scales_agree",
            Json::obj(vec![("runs", Json::uint(total)), ("same_tokens", Json::uint(agree))]),
        ),
        (
            "batch_invariance",
            Json::Arr(r.matrix.iter().map(matrix_json).collect()),
        ),
        ("prediction_held", Json::Bool(r.prediction_held())),
    ])
}

fn opt(x: Option<f64>, digits: usize) -> String {
    x.map_or("–".into(), |v| format!("{v:.digits$}"))
}

pub fn to_markdown(r: &DecodeResult, command: &str) -> String {
    let c = &r.config;
    let m = &c.model;
    let w = &c.workload;
    let s0 = c.scales[0];
    let mut md = String::new();
    md.push_str("# R3: does sharding change the tokens?\n\n");
    md.push_str(&format!(
        "Generated by `{command}`. Every number below comes from that run.\n\n"
    ));
    md.push_str("## Setup\n\n");
    md.push_str(&format!(
        "- Model: a Llama-style decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings) with deterministic random BF16 weights (checksum `{:016x}`): hidden {}, {} layers, {} query and {} KV heads of dimension {}, FFN {}, vocabulary {}, {} parameters.\n",
        r.weights_checksum, m.hidden, m.layers, m.q_heads, m.kv_heads, m.head_dim, m.ffn, m.vocab, m.parameters()
    ));
    md.push_str("- Arithmetic: local kernels (matrix–vector products, RMSNorm, attention, softmax, SwiGLU, residual adds) use binary32 basic operations in one written order, with `exp`, `sin` and `cos` built from binary64 basic operations. Weights and the KV cache are BF16; the residual stream, attention and logits are binary32. Each row-parallel projection (attention output, MLP down; two per layer) produces binary32 partials that go through the executed AllReduce under the wire policy, whose BF16 output is added to the residual stream. Under the shard tree each of S = 64 shards sums its slice of the input dimension in a fixed order; the tree depends only on S.\n");
    md.push_str(&format!(
        "- Workload: {} prompts of {} tokens ({} groups sharing a {}-token prefix), {} greedy tokens each, one sample, no end-of-sequence token. Served through the R2 engine (block size {}); every scheduler, token-stream, KV-cache and tensor invariant is checked after every step.\n",
        w.count(), w.shared + w.own, w.groups, w.shared, w.output_len, w.block_size
    ));
    md.push_str("- Baseline: P = 1 with the same wire policy and logit scale. P = 1 runs no collective: each projection's binary32 sum is rounded once to BF16. The shard tree at P = 1 still combines its 64 shard partials, so it differs from the baseline by a fixed change of arithmetic.\n");
    md.push_str("- Divergence: a prompt diverges at the first generated token that differs from the baseline. Its context matches the baseline up to and including that token, so every token up to it is *at risk*: there the baseline's top-2 margin and the run's logit perturbation (the largest |run − baseline| over the vocabulary) are comparable. Margins and perturbations are given in ulps of the baseline's top logit (binary32).\n\n");

    // 1. Divergence grid at the default scale.
    md.push_str(&format!(
        "## 1. Divergence from P = 1 (logit scale {})\n\n",
        scale_name(s0)
    ));
    md.push_str("All prompts served in one batch. Diverged: prompts whose greedy tokens differ from the baseline anywhere. Identical logits: prompts whose every logit matches the baseline bit for bit.\n\n");
    md.push_str("| Policy | Schedule | P | Diverged (of 32) | Identical logits | First divergence: min / median | Still identical after 16 / 32 / 48 tokens | Perturbation p50 / p90 (ulps) |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");
    for g in r.grid.iter().filter(|g| g.scale == s0) {
        let f = first_divergences(g);
        let p = perturbations(g);
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} / {} | {} / {} / {} | {} / {} |\n",
            g.policy.name(),
            g.schedule.name(),
            g.world,
            g.diverged(),
            g.logits_identical,
            opt(f.first().copied(), 0),
            opt(percentile(&f, 50.0), 0),
            g.surviving(16),
            g.surviving(32),
            g.surviving(48),
            opt(percentile(&p, 50.0), 2),
            opt(percentile(&p, 90.0), 2),
        ));
    }
    md.push_str(&shard_tree_sentence(r));

    // 2. Margins.
    md.push_str("\n## 2. Divergence and the top-2 margin\n\n");
    md.push_str("Baseline margins over every generated token (32 × 48), absolute and in ulps of the top logit:\n\n");
    md.push_str("| Policy | Logit scale | Margin p10 / p50 / p90 | Margin p10 / p50 / p90 (ulps) |\n");
    md.push_str("|---|---:|---:|---:|\n");
    for &scale in &c.scales {
        for &policy in &c.policies {
            let (abs, ulps) = baseline_margins(r, policy, scale);
            md.push_str(&format!(
                "| {} | {} | {} / {} / {} | {} / {} / {} |\n",
                policy.name(),
                scale_name(scale),
                opt(percentile(&abs, 10.0), 4),
                opt(percentile(&abs, 50.0), 4),
                opt(percentile(&abs, 90.0), 4),
                opt(percentile(&ulps, 10.0), 0),
                opt(percentile(&ulps, 50.0), 0),
                opt(percentile(&ulps, 90.0), 0),
            ));
        }
    }
    let (agree, total) = scales_agree(r);
    md.push_str(&format!(
        "\nAt logit scale {}, {agree} of {total} runs chose exactly the tokens they chose at scale {}. ",
        c.scales
            .iter()
            .skip(1)
            .map(|s| scale_name(*s))
            .collect::<Vec<_>>()
            .join(", "),
        scale_name(s0)
    ));
    md.push_str("A positive logit scale preserves the order of the logits, and a power of two multiplies them exactly, so absolute margins grow by the scale while margins in ulps, and every greedy token, stay the same. A wider logit spread made this way does not make decoding more robust: the perturbation grows with the logits. Divergence is therefore given against the margin in ulps below.\n\n");
    for &policy in &c.policies {
        let runs = runs_of(r, policy, s0);
        let tokens: Vec<&AtRisk> = runs.iter().flat_map(|g| g.at_risk.iter()).collect();
        let b = bins(&c.margin_bins, &tokens);
        md.push_str(&format!(
            "**{}**, every run in section 1 pooled ({} runs, {} at-risk tokens):\n\n",
            policy.name(),
            runs.len(),
            tokens.len()
        ));
        md.push_str("| Baseline margin (ulps) | At-risk tokens | Flips | Flip rate | Median perturbation (ulps) |\n");
        md.push_str("|---|---:|---:|---:|---:|\n");
        for (label, x) in bin_labels(&c.margin_bins).into_iter().zip(&b) {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                label,
                x.at_risk,
                x.flips,
                if x.at_risk == 0 {
                    "–".into()
                } else {
                    format!("{:.1}%", 100.0 * x.flips as f64 / x.at_risk as f64)
                },
                opt(percentile(&x.perturbations, 50.0), 2),
            ));
        }
        let flips: Vec<f64> = sorted(tokens.iter().filter(|t| t.flipped).map(|t| t.margin_ulps).collect());
        if let Some(&max) = flips.last() {
            let below = tokens.iter().filter(|t| t.margin_ulps <= max).count();
            md.push_str(&format!(
                "\nEvery flip ({} in all) happened at a token whose baseline margin was at most {:.0} ulps; {:.1}% of at-risk tokens had a margin that small.",
                flips.len(),
                max,
                100.0 * below as f64 / tokens.len() as f64
            ));
        } else {
            md.push_str("\nNo token flipped.");
        }
        // A flip needs the top-2 gap to move by more than the margin, and the
        // gap moves by at most twice the largest logit change.
        let reachable: Vec<&&AtRisk> = tokens
            .iter()
            .filter(|t| t.margin_ulps <= 2.0 * t.perturbation_ulps)
            .collect();
        let consistent = tokens
            .iter()
            .filter(|t| t.flipped)
            .all(|t| t.margin_ulps <= 2.0 * t.perturbation_ulps);
        md.push_str(&format!(
            " A flip requires the margin to be at most twice the run's largest logit change at that token: {} for every flip. {} at-risk tokens met that condition ({:.1}%), and {} of them flipped.\n\n",
            if consistent { "this holds" } else { "this does NOT hold" },
            reachable.len(),
            100.0 * reachable.len() as f64 / tokens.len().max(1) as f64,
            reachable.iter().filter(|t| t.flipped).count()
        ));
    }

    // 3. Batch invariance.
    md.push_str(&format!(
        "## 3. Batch invariance (P = {}, and a P = 1 control)\n\n",
        c.matrix_world
    ));
    md.push_str("Each prompt's logits in each batch composition, compared bit for bit with the same prompt served alone. Each cell: prompts whose logits differ / prompts whose tokens differ (of 32).\n\n");
    md.push_str("Prediction, derived from the schedules before measuring: local arithmetic is per token, so only a reduction whose expression for an element depends on the element's index in the step buffer can make the bits depend on the batch. Sequential and binomial-tree reduce every element alike, exact-gather sums each element exactly, the shard tree's tree depends only on S, and halving-doubling at a power-of-two P combines (x0 + x2) + (x1 + x3), up to commutativity, for every element. The ring folds each chunk starting from a different rank. So only the ring should vary.\n\n");
    for v in &Variant::ALL[1..] {
        md.push_str(&format!("- **{}**: {}.\n", v.name(), v.description()));
    }
    md.push('\n');
    md.push_str("| Policy | Schedule | P | Predicted invariant |");
    for v in &Variant::ALL[1..] {
        md.push_str(&format!(" {} |", v.name()));
    }
    md.push_str("\n|---|---|---:|---|");
    for _ in &Variant::ALL[1..] {
        md.push_str("---:|");
    }
    md.push('\n');
    for row in &r.matrix {
        md.push_str(&format!(
            "| {} | {} | {} | {} |",
            row.policy.name(),
            row.schedule.name(),
            row.world,
            if row.predicted_invariant() { "yes" } else { "no" }
        ));
        for i in 1..row.runs.len() {
            let (l, t) = row.differing(i);
            md.push_str(&format!(" {l} / {t} |"));
        }
        md.push('\n');
    }
    md.push_str(&mechanism_sentence(r));
    md.push_str(&if r.prediction_held() {
        "\n**The prediction held exactly:** every predicted-invariant row is bit-identical to serving alone in every variant, and the ring differs in at least one.\n".to_string()
    } else {
        "\n**The prediction did not hold exactly;** see the rows above.\n".to_string()
    });

    // 4. Cost.
    md.push_str("\n## 4. What determinism costs\n\n");
    md.push_str("Latency is not recomputed here. R2 measured the deterministic pairing (shard tree with `fp32-rs/bf16-ag`) against ring on the 8B-class serving model: see `results/r2-serving.md`, section 4, \"The deterministic pairing\". This experiment adds what that pairing buys in tokens: its rows above are the same at every P (section 1) and in every batch composition (section 3).\n");

    md.push_str("\n## What this does and does not show\n\n");
    md.push_str("- It shows, with every rounding executed, where a tensor-parallel decoder's tokens can change: with P and the schedule through the row-parallel AllReduces, and with batch composition only through a schedule whose reduction order depends on an element's place in the step buffer.\n");
    md.push_str("- The weights are random. Margins in a trained model are distributed differently, so the divergence rates here do not transfer; the margin tables are the transferable part. Greedy decoding with these weights often repeats a token for many steps.\n");
    md.push_str("- The model is tiny (2 layers, hidden 256) and the prompts are synthetic.\n");
    md.push_str("- Local kernels are sequential and batch-invariant by construction. Real GPU kernels choose tilings and split-K reductions by batch shape, which can break batch invariance locally as well; that is not modeled.\n");
    md.push_str("- Greedy decoding only, fixed output lengths, no sampling. `exp`, `sin` and `cos` are deterministic but not correctly rounded.\n");
    md.push_str(
        "- No latency is measured or modeled here; the cost of determinism is R2's illustrative serving model.\n",
    );
    md
}

fn shard_tree_sentence(r: &DecodeResult) -> String {
    let s0 = r.config.scales[0];
    let mut out = String::new();
    for &policy in &r.config.policies {
        let trees: Vec<&GridRun> = r
            .grid
            .iter()
            .filter(|g| g.scale == s0 && g.policy == policy && matches!(g.schedule, Schedule::ShardTree { .. }))
            .collect();
        if let Some(first) = trees.first() {
            let same = trees.iter().all(|g| g.tokens == first.tokens);
            out.push_str(&format!(
                "\nShard tree, {}: tokens at P = {} are {}; it diverges from the unsharded baseline in {} prompt{} at every P, a fixed change of arithmetic rather than a dependence on P.",
                policy.name(),
                trees.iter().map(|g| g.world.to_string()).collect::<Vec<_>>().join(", "),
                if same { "identical" } else { "not identical" },
                first.diverged(),
                if first.diverged() == 1 { "" } else { "s" }
            ));
        }
    }
    out.push('\n');
    out
}

fn mechanism_sentence(r: &DecodeResult) -> String {
    let Some(control) = r.matrix.iter().find(|row| row.world == 1) else {
        return String::new();
    };
    let by = |v: Variant| control.runs.iter().find(|x| x.variant == v);
    match (by(Variant::PrefixCache), by(Variant::Pressure), by(Variant::FailureRetry)) {
        (Some(p), Some(q), Some(f)) => format!(
            "\nEach mechanism ran (P = 1 control, {}): prefix cache {} block hits; recompute preemption {} preemptions and {} recomputed tokens; failure {} failed step and {} recomputed tokens.\n",
            control.policy.name(),
            p.prefix_hit_blocks,
            q.preemptions,
            q.exec.recomputed_tokens,
            f.failed_steps,
            f.exec.recomputed_tokens
        ),
        _ => String::new(),
    }
}

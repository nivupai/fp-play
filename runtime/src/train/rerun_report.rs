//! JSON and Markdown for the reruns on the trained model
//! (results/r3-trained.*, results/r4-trained.*). Every number and every
//! generated sentence comes from the [`R3Rerun`] / [`R4Rerun`].

use super::rerun::{R3Rerun, R4Rerun, COPY_WORKLOAD, R4_PROMPTS, TASK};
use crate::collectives::WirePolicy;
use crate::decoder::experiment::{DecodeResult, GridRun};
use crate::decoder::variants::Variant;
use crate::json::Json;
use crate::precision::experiment::{Kind, PolicyRun, PrecisionResult};
use crate::precision::policy::{Level, Site};

fn num(x: f64) -> Json {
    Json::Num(x)
}

/// Nearest-rank quantile of a sorted list: element round((n − 1) · p).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty());
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v
}

const MODELS: [&str; 2] = ["random", "trained"];

fn r3_models(r: &R3Rerun) -> [(&'static str, &DecodeResult); 2] {
    [(MODELS[0], &r.random), (MODELS[1], &r.trained)]
}

/// Margins at the P = 1 baseline of one policy: absolute (logits) and in ulps of the top logit.
pub struct MarginStats {
    pub tokens: usize,
    pub abs: Vec<f64>,
    pub ulps: Vec<f64>,
    pub copy_correct: usize,
}

pub fn margins(d: &DecodeResult, policy: WirePolicy, symbols: &[Vec<u32>]) -> MarginStats {
    let b = d.baseline(policy, d.config.scales[0]);
    let abs: Vec<f64> = b
        .generations
        .iter()
        .flat_map(|g| g.top.iter().map(|t| t.margin()))
        .collect();
    let ulps: Vec<f64> = b
        .generations
        .iter()
        .flat_map(|g| {
            g.top
                .iter()
                .map(|t| t.margin() / crate::decoder::experiment::ulp_f32(t.top))
        })
        .collect();
    let copy_correct = b
        .generations
        .iter()
        .zip(symbols)
        .map(|(g, s)| g.tokens.iter().zip(s).filter(|(a, b)| a == b).count())
        .sum();
    MarginStats {
        tokens: abs.len(),
        abs: sorted(abs),
        ulps: sorted(ulps),
        copy_correct,
    }
}

/// Divergence pooled over every grid run of one policy.
pub struct Pooled {
    pub runs: usize,
    pub runs_diverged: usize,
    pub prompts: usize,
    pub prompts_diverged: usize,
    pub logits_identical: usize,
    pub at_risk: usize,
    pub flips: usize,
    /// At-risk tokens whose margin is at most twice the run's perturbation.
    pub within_reach: usize,
    pub perturbation_ulps: Vec<f64>,
}

impl Pooled {
    pub fn flip_rate(&self) -> f64 {
        self.flips as f64 / self.at_risk.max(1) as f64
    }
}

fn runs_of(d: &DecodeResult, policy: WirePolicy) -> Vec<&GridRun> {
    d.grid.iter().filter(|g| g.policy == policy).collect()
}

pub fn pooled(d: &DecodeResult, policy: WirePolicy) -> Pooled {
    let runs = runs_of(d, policy);
    let mut p = Pooled {
        runs: runs.len(),
        runs_diverged: runs.iter().filter(|g| g.diverged() > 0).count(),
        prompts: runs.iter().map(|g| g.first_divergence.len()).sum(),
        prompts_diverged: runs.iter().map(|g| g.diverged()).sum(),
        logits_identical: runs.iter().map(|g| g.logits_identical).sum(),
        at_risk: 0,
        flips: 0,
        within_reach: 0,
        perturbation_ulps: Vec::new(),
    };
    for g in &runs {
        for a in &g.at_risk {
            p.at_risk += 1;
            p.flips += usize::from(a.flipped);
            p.within_reach += usize::from(a.margin_ulps <= 2.0 * a.perturbation_ulps);
            p.perturbation_ulps.push(a.perturbation_ulps);
        }
    }
    p.perturbation_ulps = sorted(p.perturbation_ulps);
    p
}

/// Batch-invariance summary of one model.
pub struct Invariance {
    pub rows: usize,
    pub predicted_rows: usize,
    /// Predicted-invariant rows with any variant whose logits differ from `alone`.
    pub predicted_rows_differing: usize,
    pub ring_rows: usize,
    /// Ring rows (P > 1) with a variant whose logits differ.
    pub ring_rows_differing: usize,
    /// (variant, prompt) cells whose tokens differ from `alone`, over every row.
    pub token_cells_differing: usize,
    pub held: bool,
}

pub fn invariance(d: &DecodeResult) -> Invariance {
    let mut v = Invariance {
        rows: d.matrix.len(),
        predicted_rows: 0,
        predicted_rows_differing: 0,
        ring_rows: 0,
        ring_rows_differing: 0,
        token_cells_differing: 0,
        held: d.prediction_held(),
    };
    for row in &d.matrix {
        let differs = (1..row.runs.len()).any(|i| row.differing(i).0 > 0);
        v.token_cells_differing += (1..row.runs.len()).map(|i| row.differing(i).1).sum::<usize>();
        if row.predicted_invariant() {
            v.predicted_rows += 1;
            v.predicted_rows_differing += usize::from(differs);
        } else {
            v.ring_rows += 1;
            v.ring_rows_differing += usize::from(differs);
        }
    }
    v
}

fn margin_json(m: &MarginStats) -> Json {
    Json::obj(vec![
        ("tokens", Json::uint(m.tokens)),
        ("copy_tokens_correct", Json::uint(m.copy_correct)),
        ("margin_min", num(m.abs[0])),
        ("margin_p10", num(quantile(&m.abs, 0.1))),
        ("margin_p50", num(quantile(&m.abs, 0.5))),
        ("margin_p90", num(quantile(&m.abs, 0.9))),
        ("margin_ulps_p10", num(quantile(&m.ulps, 0.1))),
        ("margin_ulps_p50", num(quantile(&m.ulps, 0.5))),
    ])
}

fn pooled_json(p: &Pooled) -> Json {
    Json::obj(vec![
        ("runs", Json::uint(p.runs)),
        ("runs_with_divergence", Json::uint(p.runs_diverged)),
        ("prompts", Json::uint(p.prompts)),
        ("prompts_diverged", Json::uint(p.prompts_diverged)),
        ("prompts_logits_identical", Json::uint(p.logits_identical)),
        ("at_risk_tokens", Json::uint(p.at_risk)),
        ("flips", Json::uint(p.flips)),
        ("flip_rate", num(p.flip_rate())),
        ("within_reach", Json::uint(p.within_reach)),
        ("perturbation_ulps_p50", num(quantile(&p.perturbation_ulps, 0.5))),
        ("perturbation_ulps_p90", num(quantile(&p.perturbation_ulps, 0.9))),
    ])
}

pub fn r3_to_json(r: &R3Rerun, command: &str) -> Json {
    let policies = &r.random.config.policies;
    let models = r3_models(r)
        .into_iter()
        .map(|(name, d)| {
            let inv = invariance(d);
            Json::obj(vec![
                ("model", Json::str(name)),
                ("weights_checksum", Json::str(format!("{:016x}", d.weights_checksum))),
                (
                    "policies",
                    Json::Arr(
                        policies
                            .iter()
                            .map(|&p| {
                                Json::obj(vec![
                                    ("policy", Json::str(p.name())),
                                    ("baseline", margin_json(&margins(d, p, &r.symbols))),
                                    ("divergence", pooled_json(&pooled(d, p))),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "grid",
                    Json::Arr(
                        d.grid
                            .iter()
                            .map(|g| {
                                Json::obj(vec![
                                    ("policy", Json::str(g.policy.name())),
                                    ("schedule", Json::str(g.schedule.name())),
                                    ("world", Json::uint(g.world)),
                                    ("prompts_diverged", Json::uint(g.diverged())),
                                    ("prompts_logits_identical", Json::uint(g.logits_identical)),
                                    ("at_risk_tokens", Json::uint(g.at_risk.len())),
                                    ("flips", Json::uint(g.at_risk.iter().filter(|a| a.flipped).count())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "batch_invariance",
                    Json::obj(vec![
                        ("rows", Json::uint(inv.rows)),
                        ("prediction_held", Json::Bool(inv.held)),
                        ("predicted_invariant_rows", Json::uint(inv.predicted_rows)),
                        (
                            "predicted_invariant_rows_differing",
                            Json::uint(inv.predicted_rows_differing),
                        ),
                        ("ring_rows", Json::uint(inv.ring_rows)),
                        ("ring_rows_differing", Json::uint(inv.ring_rows_differing)),
                        ("token_cells_differing", Json::uint(inv.token_cells_differing)),
                        (
                            "rows_detail",
                            Json::Arr(
                                d.matrix
                                    .iter()
                                    .map(|row| {
                                        Json::obj(vec![
                                            ("world", Json::uint(row.world)),
                                            ("schedule", Json::str(row.schedule.name())),
                                            ("policy", Json::str(row.policy.name())),
                                            (
                                                "variants",
                                                Json::Arr(
                                                    (1..row.runs.len())
                                                        .map(|i| {
                                                            let (l, t) = row.differing(i);
                                                            Json::obj(vec![
                                                                ("variant", Json::str(row.runs[i].variant.name())),
                                                                ("prompts_logits_differ", Json::uint(l)),
                                                                ("prompts_tokens_differ", Json::uint(t)),
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
                    ]),
                ),
            ])
        })
        .collect();
    let w = &COPY_WORKLOAD;
    Json::obj(vec![
        ("command", Json::str(command)),
        (
            "workload",
            Json::obj(vec![
                ("task", Json::str("copy: BOS s1 … s24 SEP, 24 greedy tokens")),
                ("prompts", Json::uint(w.count())),
                ("prompt_tokens", Json::uint(w.shared + w.own)),
                ("groups", Json::uint(w.groups)),
                ("shared_prefix_tokens", Json::uint(w.shared)),
                ("generated_tokens", Json::uint(w.output_len as usize)),
                ("block_size", Json::uint(w.block_size)),
            ]),
        ),
        (
            "worlds",
            Json::Arr(r.random.config.worlds.iter().map(|&x| Json::uint(x)).collect()),
        ),
        ("matrix_world", Json::uint(r.random.config.matrix_world)),
        ("models", Json::Arr(models)),
    ])
}

fn sci(x: f64) -> String {
    if x == 0.0 {
        "0".into()
    } else {
        format!("{x:.2e}")
    }
}

fn pct(n: usize, d: usize) -> String {
    format!("{:.2}%", 100.0 * n as f64 / d.max(1) as f64)
}

pub fn r3_to_markdown(r: &R3Rerun, command: &str) -> String {
    let w = &COPY_WORKLOAD;
    let c = &r.random.config;
    let mut md = String::new();
    md.push_str("# R3 rerun on the trained model: margins, flips and batch invariance\n\n");
    md.push_str(&format!(
        "Generated by `{command}`. Every number below comes from that run.\n\n"
    ));
    md.push_str("## Setup\n\n");
    md.push_str(&format!(
        "- Models: R3's architecture with **random** BF16 weights (the `r3-decode` model, checksum `{:016x}`) and with the **trained** weights of `r4-training` rounded once to BF16 (checksum `{:016x}`). Everything else is R3's: binary32 local kernels, BF16 KV cache, the executed AllReduce under each wire policy, the R2 engine with every invariant checked after every step.\n",
        r.random.weights_checksum, r.trained.weights_checksum
    ));
    md.push_str(&format!(
        "- Workload (both models): {} copy prompts `BOS s1 … s{L} SEP` of {} tokens in {} groups sharing their first {} tokens, {} greedy tokens each (the copy), block size {}.\n",
        w.count(),
        w.shared + w.own,
        w.groups,
        w.shared,
        w.output_len,
        w.block_size,
        L = TASK.len
    ));
    md.push_str(&format!(
        "- Grid: P ∈ {{{}}} for every AllReduce schedule, plus the shard tree at P = 1, under both wire policies, against the P = 1 baseline of the same policy (logit scale 1 only: a positive scale cannot change greedy tokens). Batch invariance at P = {} and the P = 1 control, as in R3.\n",
        c.worlds.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
        c.matrix_world
    ));
    md.push_str("- At-risk tokens, flips and perturbations are defined as in `r3-decode`: every generated token up to and including a prompt's first divergence; a flip is the token where it diverges; the perturbation is the largest |run − baseline| logit there, in ulps of the baseline's top logit. *Within reach*: margin ≤ 2 × perturbation (a flip implies it, by construction).\n\n");

    md.push_str("## 1. Top-2 margins at P = 1\n\n| Model | Policy | Tokens | Copy tokens correct | Margin min / p10 / p50 / p90 (logits) | Margin p10 / p50 (ulps of top logit) |\n|---|---|---:|---:|---:|---:|\n");
    for (name, d) in r3_models(r) {
        for &p in &c.policies {
            let m = margins(d, p, &r.symbols);
            md.push_str(&format!(
                "| {name} | {} | {} | {} | {:.4} / {:.4} / {:.4} / {:.4} | {:.0} / {:.0} |\n",
                p.name(),
                m.tokens,
                m.copy_correct,
                m.abs[0],
                quantile(&m.abs, 0.1),
                quantile(&m.abs, 0.5),
                quantile(&m.abs, 0.9),
                quantile(&m.ulps, 0.1),
                quantile(&m.ulps, 0.5)
            ));
        }
    }
    md.push_str(&margin_sentence(r));

    md.push_str("\n## 2. Divergence from P = 1, pooled over the grid\n\n| Model | Policy | Runs | Runs with a divergence | Prompts diverged | At-risk tokens | Flips | Flip rate | Within reach | Perturbation p50 / p90 (ulps) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (name, d) in r3_models(r) {
        for &p in &c.policies {
            let x = pooled(d, p);
            md.push_str(&format!(
                "| {name} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {:.0} / {:.0} |\n",
                p.name(),
                x.runs,
                x.runs_diverged,
                x.prompts_diverged,
                x.prompts,
                x.at_risk,
                x.flips,
                pct(x.flips, x.at_risk),
                pct(x.within_reach, x.at_risk),
                quantile(&x.perturbation_ulps, 0.5),
                quantile(&x.perturbation_ulps, 0.9)
            ));
        }
    }
    md.push_str(&flip_sentence(r));
    md.push_str("\nPer run (prompts diverged of 32, random / trained):\n\n| Policy | Schedule | P | Random | Trained |\n|---|---|---:|---:|---:|\n");
    for (g, t) in r.random.grid.iter().zip(&r.trained.grid) {
        assert_eq!((g.policy, g.schedule, g.world), (t.policy, t.schedule, t.world));
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            g.policy.name(),
            g.schedule.name(),
            g.world,
            g.diverged(),
            t.diverged()
        ));
    }

    md.push_str(&format!(
        "\n## 3. Batch invariance (P = {} and the P = 1 control)\n\nEach prompt's logits in each of the {} batch compositions are compared bit for bit with the prompt served alone.\n\n| Model | Rows | Prediction held | Predicted-invariant rows with differing logits | Ring rows (P = {}) with differing logits | (variant, prompt) cells with differing tokens |\n|---|---:|---|---:|---:|---:|\n",
        c.matrix_world,
        Variant::ALL.len() - 1,
        c.matrix_world
    ));
    for (name, d) in r3_models(r) {
        let v = invariance(d);
        md.push_str(&format!(
            "| {name} | {} | {} | {} of {} | {} of {} | {} |\n",
            v.rows,
            if v.held { "yes" } else { "no" },
            v.predicted_rows_differing,
            v.predicted_rows,
            v.ring_rows_differing,
            v.ring_rows,
            v.token_cells_differing
        ));
    }
    md.push_str("\nThe prediction (only the ring's bits may depend on batch composition) is about bits, so it does not depend on the weights; the trained model tests it again on other values.\n");

    let identical: usize = r3_models(r)
        .iter()
        .flat_map(|(_, d)| d.grid.iter().map(|g| g.logits_identical))
        .sum();
    md.push_str("\n## What this does and does not show\n\n");
    let reach = |d: &DecodeResult| {
        d.config.policies.iter().fold((0, 0), |(w, a), &p| {
            let x = pooled(d, p);
            (w + x.within_reach, a + x.at_risk)
        })
    };
    let (rw, ra) = reach(&r.random);
    let (tw, ta) = reach(&r.trained);
    md.push_str(&format!(
        "- The same arithmetic, schedules and batch compositions run on both models. {} Within reach (margin ≤ 2 × perturbation, which every flip requires): {rw} of {ra} at-risk tokens with random weights, {tw} of {ta} with trained weights.\n",
        if identical == 0 {
            "No prompt's logits match its baseline bit for bit in any grid run, for either model, so sharding still changes the trained model's logits."
        } else {
            "Some prompts' logits match their baseline bit for bit (see the JSON)."
        }
    ));
    md.push_str("- The trained model solves one synthetic copy task with confident predictions. This does not say how often a language model's tokens have small margins, so it is not a prediction for one.\n");
    let (vr, vt) = (invariance(&r.random), invariance(&r.trained));
    md.push_str(&format!(
        "- Batch composition changed the ring's logits in {} of {} ring rows with random weights and {} of {} with trained weights; it changed tokens in {} and {} (variant, prompt) cells.\n",
        vr.ring_rows_differing, vr.ring_rows, vt.ring_rows_differing, vt.ring_rows, vr.token_cells_differing, vt.token_cells_differing
    ));
    md
}

fn margin_sentence(r: &R3Rerun) -> String {
    let p = r.random.config.policies[0];
    let a = margins(&r.random, p, &r.symbols);
    let b = margins(&r.trained, p, &r.symbols);
    format!(
        "\nMedian top-2 margin under `{}`: {:.4} logits with random weights, {:.4} with trained weights ({:.0}×); the trained model's smallest margin, {:.4}, is {:.0}× the random model's median.\n",
        p.name(),
        quantile(&a.abs, 0.5),
        quantile(&b.abs, 0.5),
        quantile(&b.abs, 0.5) / quantile(&a.abs, 0.5),
        b.abs[0],
        b.abs[0] / quantile(&a.abs, 0.5)
    )
}

fn flip_sentence(r: &R3Rerun) -> String {
    let mut out = String::new();
    for &p in &r.random.config.policies {
        let (a, b) = (pooled(&r.random, p), pooled(&r.trained, p));
        out.push_str(&format!(
            "\nUnder `{}`: random weights flip {} of {} at-risk tokens ({}), trained weights {} of {} ({}); the perturbation medians are {:.0} and {:.0} ulps of the top logit.",
            p.name(),
            a.flips,
            a.at_risk,
            pct(a.flips, a.at_risk),
            b.flips,
            b.at_risk,
            pct(b.flips, b.at_risk),
            quantile(&a.perturbation_ulps, 0.5),
            quantile(&b.perturbation_ulps, 0.5)
        ));
    }
    out.push('\n');
    out
}

// ---------------------------------------------------------------- R4 rerun

fn r4_models(r: &R4Rerun) -> [(&'static str, &PrecisionResult); 2] {
    [(MODELS[0], &r.random), (MODELS[1], &r.trained)]
}

fn uniform(p: &PrecisionResult, level: Level) -> &PolicyRun {
    p.runs
        .iter()
        .find(|x| x.kind == Kind::Uniform(level))
        .expect("every level has a uniform run")
}

fn one_site(p: &PrecisionResult, site: Site, level: Level) -> Option<&PolicyRun> {
    p.runs.iter().find(|x| x.kind == Kind::OneSite(site, level))
}

pub fn r4_to_json(r: &R4Rerun, command: &str) -> Json {
    let c = &r.trained.config;
    Json::obj(vec![
        ("command", Json::str(command)),
        (
            "prompts",
            Json::obj(vec![
                ("task", Json::str("copy: BOS s1 … s24 SEP")),
                ("count", Json::uint(R4_PROMPTS)),
                ("prompt_tokens", Json::uint(c.prompt_len)),
                ("continuation", Json::uint(c.continuation)),
            ]),
        ),
        (
            "levels",
            Json::Arr(Level::EVERY.iter().map(|l| Json::str(l.name())).collect()),
        ),
        (
            "models",
            Json::Arr(
                r4_models(r)
                    .into_iter()
                    .zip([r.reference_copy_correct.0, r.reference_copy_correct.1])
                    .map(|((name, p), correct)| {
                        Json::obj(vec![
                            ("model", Json::str(name)),
                            ("reference_copy_tokens_correct", Json::uint(correct)),
                            ("reference_perplexity", num(p.reference_perplexity)),
                            (
                                "runs",
                                Json::Arr(p.runs.iter().map(crate::precision::report::run_json).collect()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

pub fn r4_to_markdown(r: &R4Rerun, command: &str) -> String {
    let c = &r.trained.config;
    let scored = uniform(&r.trained, Level::Fp32).drift.scored;
    let mut md = String::new();
    md.push_str("# R4 rerun on the trained model: precision levels, FP8 and INT8 included\n\n");
    md.push_str(&format!(
        "Generated by `{command}`. Every number below comes from that run.\n\n"
    ));
    md.push_str("## Setup\n\n");
    md.push_str("- Models: R4's random FP32 master weights (the `r4-precision` model) and the trained weights of `r4-training`, same architecture.\n");
    md.push_str(&format!(
        "- Prompts (both models): {} copy prompts `BOS s1 … s{L} SEP` ({} tokens). The all-FP32 reference continues each greedily for {} tokens; it copies {} of {} tokens correctly with random weights and {} of {} with trained weights. Every policy is compared with the reference teacher-forced on that continuation ({} scored tokens) and also decodes greedily on its own.\n",
        R4_PROMPTS,
        c.prompt_len,
        c.continuation,
        r.reference_copy_correct.0,
        R4_PROMPTS * c.continuation,
        r.reference_copy_correct.1,
        R4_PROMPTS * c.continuation,
        scored,
        L = TASK.len
    ));
    md.push_str("- Levels: the five R4 float levels and three 8-bit levels with explicit binary32 scales (`precision/narrow.rs`): `fp8-e4m3`, `fp8-e5m2` (FP32 accumulation) and `int8` (exact integer accumulation), with one scale per weight row, per activation vector and per (token, KV head) slice of K and V.\n\n");

    md.push_str(&format!(
        "## 1. Every site at one level\n\n| Level | Random: mean KL | Random: top-1 (of {scored}) | Random: greedy diverged (of {R4_PROMPTS}) | Trained: mean KL | Trained: top-1 | Trained: greedy diverged |\n|---|---:|---:|---:|---:|---:|---:|\n"
    ));
    for level in Level::EVERY {
        let (a, b) = (uniform(&r.random, level), uniform(&r.trained, level));
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            level.name(),
            sci(a.drift.mean_kl),
            a.drift.top1_agree,
            a.diverged(),
            sci(b.drift.mean_kl),
            b.drift.top1_agree,
            b.diverged()
        ));
    }
    md.push_str(&level_sentences(r));

    md.push_str("\n## 2. One site at a time (mean KL; every other site FP32)\n\nThe KV cache only stores values, so its `-acc` levels are omitted (–).\n");
    let layers = c.model.layers;
    for (name, p) in r4_models(r) {
        md.push_str(&format!("\n**{name} weights**\n\n| Site |"));
        for level in &Level::EVERY[1..] {
            md.push_str(&format!(" {} |", level.name()));
        }
        md.push_str("\n|---|");
        md.push_str(&"---:|".repeat(Level::EVERY.len() - 1));
        md.push('\n');
        for site in Site::all(layers) {
            md.push_str(&format!("| {} |", site.name()));
            for &level in &Level::EVERY[1..] {
                md.push_str(&match one_site(p, site, level) {
                    Some(x) => format!(" {} |", sci(x.drift.mean_kl)),
                    None => " – |".into(),
                });
            }
            md.push('\n');
        }
    }
    md.push_str(&site_sentences(r));

    md.push_str("\n## 3. Mixed policies\n\n| Policy | Random: mean KL | Random: top-1 | Trained: mean KL | Trained: top-1 |\n|---|---:|---:|---:|---:|\n");
    let mixed = |p: &PrecisionResult| -> Vec<(String, f64, usize)> {
        p.runs
            .iter()
            .filter(|x| x.kind == Kind::Mixed)
            .map(|x| (x.policy.name.clone(), x.drift.mean_kl, x.drift.top1_agree))
            .collect()
    };
    for (a, b) in mixed(&r.random).into_iter().zip(mixed(&r.trained)) {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            a.0,
            sci(a.1),
            a.2,
            sci(b.1),
            b.2
        ));
    }
    let nonfinite: usize = r4_models(r)
        .iter()
        .flat_map(|(_, p)| p.runs.iter().map(|x| x.drift.nonfinite))
        .sum();
    md.push_str(&format!(
        "\nNon-finite logits over every run of both models: {nonfinite}.\n"
    ));
    md.push_str("\n## What this does and does not show\n\n");
    md.push_str("- It compares how far each precision level moves the logits (KL against FP32) and whether it changes greedy tokens, for one architecture with random and with trained weights.\n");
    md.push_str("- The scaling scheme is one simple choice (per-row and per-vector absmax scales in binary32); block scales, per-channel activation scales, calibration and outlier handling are not modeled.\n");
    md.push_str("- The trained model's task is synthetic and its predictions are confident; these numbers do not measure the FP8/INT8 sensitivity of a language model.\n");
    md
}

fn level_sentences(r: &R4Rerun) -> String {
    let mut out = String::from("\n");
    for level in Level::SCALED {
        let (a, b) = (uniform(&r.random, level), uniform(&r.trained, level));
        out.push_str(&format!(
            "- `{}` everywhere: mean KL {} with random weights and {} with trained weights; top-1 agreement {} and {}; greedy continuations changed {} and {} of {}.\n",
            level.name(),
            sci(a.drift.mean_kl),
            sci(b.drift.mean_kl),
            a.drift.top1_agree,
            b.drift.top1_agree,
            a.diverged(),
            b.diverged(),
            R4_PROMPTS
        ));
    }
    let worst = |p: &PrecisionResult| {
        Level::EVERY[1..]
            .iter()
            .copied()
            .max_by(|&x, &y| {
                uniform(p, x)
                    .drift
                    .mean_kl
                    .partial_cmp(&uniform(p, y).drift.mean_kl)
                    .expect("finite")
            })
            .expect("levels")
    };
    out.push_str(&format!(
        "- The level with the largest mean KL is `{}` for random weights and `{}` for trained weights.\n",
        worst(&r.random).name(),
        worst(&r.trained).name()
    ));
    out
}

fn site_sentences(r: &R4Rerun) -> String {
    let layers = r.trained.config.model.layers;
    let mut out = String::from("\nMost sensitive site when lowered alone:\n\n");
    for level in Level::SCALED {
        let pick = |p: &PrecisionResult| {
            Site::all(layers)
                .into_iter()
                .filter_map(|s| one_site(p, s, level).map(|x| (s, x.drift.mean_kl)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"))
                .expect("sites")
        };
        let (a, b) = (pick(&r.random), pick(&r.trained));
        out.push_str(&format!(
            "- `{}`: {} for random weights (mean KL {}), {} for trained weights ({}).\n",
            level.name(),
            a.0.name(),
            sci(a.1),
            b.0.name(),
            sci(b.1)
        ));
    }
    out
}

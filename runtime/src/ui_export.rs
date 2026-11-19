//! R6: the data the FP Play `#collectives` workspace reads
//! (results/r6-collectives.json), written by the experiment command.
//!
//! * every schedule's executed rounds with each transfer's element range,
//!   and every element's association, from
//!   [`crate::collectives::trace_association`] (the real simulator, tracing
//!   beside the values);
//! * the shard tree's rounds (its dry run) and its fixed tree
//!   ([`crate::shard_tree::tree`], the list the executor iterates);
//! * the error of every schedule against the exact sum at every P, copied
//!   from the tp-allreduce result computed in the same run.

use crate::collectives::{trace_association, Algorithm, Assoc, Round, TransferKind, WirePolicy};
use crate::json::Json;
use crate::shard_tree;
use crate::tensor_parallel::ExperimentResult;

/// Ranks shown in the schedule view.
pub const WORLDS: [usize; 4] = [2, 3, 4, 8];
/// Elements per buffer in the schedule view.
pub const ELEMENTS: usize = 8;
/// Shards of the small shard tree shown.
pub const SHARDS: usize = 8;

fn assoc_json(a: &Assoc) -> Json {
    match a {
        Assoc::Leaf(r) => Json::uint(*r),
        Assoc::Add(l, r) => Json::Arr(vec![assoc_json(l), assoc_json(r)]),
        Assoc::Exact(v) => Json::obj(vec![("exact", Json::Arr(v.iter().map(|&r| Json::uint(r)).collect()))]),
    }
}

fn rounds_json(rounds: &[Round], ranges: Option<&[Vec<std::ops::Range<usize>>]>) -> Json {
    Json::Arr(
        rounds
            .iter()
            .enumerate()
            .map(|(i, r)| {
                Json::obj(vec![
                    ("phase", Json::str(r.phase)),
                    (
                        "kind",
                        Json::str(match r.kind {
                            TransferKind::Reduce => "reduce",
                            TransferKind::Gather => "gather",
                        }),
                    ),
                    (
                        "transfers",
                        Json::Arr(
                            r.transfers
                                .iter()
                                .enumerate()
                                .map(|(j, t)| {
                                    let first = ranges.map_or(0, |rs| rs[i][j].start);
                                    Json::obj(vec![
                                        ("src", Json::uint(t.src)),
                                        ("dst", Json::uint(t.dst)),
                                        ("first", Json::uint(first)),
                                        ("elements", Json::uint(t.elements)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect(),
    )
}

pub fn to_json(tp: &ExperimentResult, command: &str) -> Result<Json, String> {
    let mut schedules = Vec::new();
    for &world in &WORLDS {
        for algorithm in Algorithm::ALL.into_iter().filter(|a| a.supports(world)) {
            let t = trace_association(algorithm, world, ELEMENTS).map_err(|e| e.to_string())?;
            let agree = t.per_rank.windows(2).all(|w| w[0] == w[1]);
            schedules.push(Json::obj(vec![
                ("algorithm", Json::str(algorithm.name())),
                ("description", Json::str(algorithm.description())),
                ("world", Json::uint(world)),
                ("elements", Json::uint(ELEMENTS)),
                ("rounds", rounds_json(&t.rounds, Some(&t.ranges))),
                ("ranks_agree", Json::Bool(agree)),
                ("association", Json::Arr(t.per_rank[0].iter().map(assoc_json).collect())),
            ]));
        }
    }
    let mut trees = Vec::new();
    for world in [1usize, 2, 4, 8] {
        let sched = shard_tree::schedule_rounds(WirePolicy::FP32_RS_BF16_AG, world, SHARDS, ELEMENTS)
            .map_err(|e| e.to_string())?;
        let mut nodes: Vec<Assoc> = (0..SHARDS).map(Assoc::Leaf).collect();
        let combines = shard_tree::tree(SHARDS);
        for c in &combines {
            let right = nodes[c.right].clone();
            let left = std::mem::replace(&mut nodes[c.left], Assoc::Leaf(0));
            nodes[c.left] = Assoc::Add(Box::new(left), Box::new(right));
        }
        trees.push(Json::obj(vec![
            ("world", Json::uint(world)),
            ("shards", Json::uint(SHARDS)),
            (
                "owners",
                Json::Arr(
                    (0..SHARDS)
                        .map(|s| Json::uint(shard_tree::shard_owner(SHARDS, world, s)))
                        .collect(),
                ),
            ),
            (
                "combines",
                Json::Arr(
                    combines
                        .iter()
                        .map(|c| {
                            Json::obj(vec![
                                ("level", Json::uint(c.level as usize)),
                                ("left", Json::uint(c.left)),
                                ("right", Json::uint(c.right)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("rounds", rounds_json(&sched.rounds, None)),
            ("association", assoc_json(&nodes[0])),
        ]));
    }
    let errors: Vec<Json> = tp
        .worlds
        .iter()
        .flat_map(|w| {
            w.algorithms.iter().map(move |a| {
                Json::obj(vec![
                    ("wire", Json::str(w.wire.name())),
                    ("world", Json::uint(w.world)),
                    ("algorithm", Json::str(a.algorithm.name())),
                    ("relative_rms_error", Json::Num(a.vs_exact.relative_rms_error)),
                    ("mismatched", Json::uint(a.vs_exact.mismatched)),
                    ("max_ulp", Json::Int(a.vs_exact.max_ulp as i64)),
                    ("mean_ulp", Json::Num(a.vs_exact.mean_ulp)),
                ])
            })
        })
        .collect();
    let baselines: Vec<Json> = tp
        .baselines
        .iter()
        .map(|b| {
            Json::obj(vec![
                ("wire", Json::str(b.wire.name())),
                (
                    "relative_rms_error",
                    Json::Num(b.single_device_vs_exact.relative_rms_error),
                ),
                ("mismatched", Json::uint(b.single_device_vs_exact.mismatched)),
            ])
        })
        .collect();
    let c = &tp.config;
    Ok(Json::obj(vec![
        ("schema", Json::str("fp-play-runtime/r6-collectives/v1")),
        ("command", Json::str(command)),
        (
            "note",
            Json::str("Schedules and associations come from the executed R1 simulator; errors from the R1 tensor-parallel experiment in the same run. Illustrative, not a measurement of any system."),
        ),
        ("schedules", Json::Arr(schedules)),
        ("shard_trees", Json::Arr(trees)),
        (
            "errors",
            Json::obj(vec![
                (
                    "problem",
                    Json::obj(vec![
                        ("tokens", Json::uint(c.tokens)),
                        ("hidden", Json::uint(c.hidden)),
                        ("outputs", Json::uint(c.outputs)),
                        ("elements", Json::uint(tp.elements)),
                    ]),
                ),
                ("single_device", Json::Arr(baselines)),
                ("runs", Json::Arr(errors)),
            ]),
        ),
    ]))
}

pub fn to_markdown(command: &str) -> String {
    format!(
        "# R6: data for the `#collectives` workspace\n\nGenerated by `{command}`. `r6-collectives.json` is read by the FP Play app's `#collectives` workspace (`src/lib/collectives/`).\n\n- **Schedules:** for P = {} and every schedule that supports P, the executed rounds of an AllReduce over {} elements, each transfer's element range, and each element's association (which ranks' partial sums are added, in which order), traced beside the real simulation.\n- **Shard tree:** S = {} shards at P = 1, 2, 4 and 8: the fixed tree, shard owners and scheduled rounds.\n- **Errors:** each schedule's error against the exact sum at every P, copied from `tp-allreduce` in the same run.\n\nNothing here is a measurement of any system.\n",
        WORLDS.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(", "),
        ELEMENTS,
        SHARDS
    )
}

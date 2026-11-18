//! A deterministic AllReduce over a fixed logical tree of S shards.
//!
//! The reduction is split into **S logical shards** (default
//! [`DEFAULT_SHARDS`] = 64), fixed independently of the number of devices.
//! Each shard contributes one buffer (its partial sum). The shards are
//! combined by one **fixed binary tree** that depends only on S. At level
//! `k` (`k = 0, 1, …`) node `i` absorbs node `i + 2^k` for every `i` that is
//! a multiple of `2^(k+1)` with `i + 2^k < S`, so shard 0's node ends as the
//! root. P devices own contiguous blocks of shards (rank `r` owns
//! [`owned_shards`]`(S, P, r)`), so any `1 ≤ P ≤ S` works. The whole buffer
//! moves at every tree node; a chunked (pipelined) version is not modeled.
//!
//! **Invariant: every node of the tree is computed the same way whether its
//! two children are on one rank or on different ranks.** A node is always one
//! correctly rounded accumulator-format addition `left + right`. When the
//! right child lives on another rank it arrives in a message in the policy's
//! reduce-wire format, so [`all_reduce`] accepts only policies whose reduce
//! wire represents every accumulator value exactly
//! ([`Format::contains`]); a lossy wire (such as `bf16-wire/fp32-add`) would
//! round a child only because it crossed ranks, and is rejected with
//! [`ShardTreeError::LossyReduceWire`]. With that invariant, the output bits
//! depend on S and the shard inputs only, never on P. `tests/shard_tree.rs`
//! checks this for every P from 1 to S.
//!
//! **Rounds.** A tree level whose combines need messages becomes one or more
//! synchronous rounds. The messages of a level are packed into rounds first
//! fit, so that each rank sends at most one and receives at most one message
//! per round (the same single-port model as [`crate::collectives`]). Extra
//! rounds are added only when a level's messages would otherwise conflict.
//! The reported round count is the one actually scheduled. With contiguous
//! ownership, each rank sends and receives at most one message per level, so
//! no level needs more than one round. `tests/shard_tree.rs` measures this
//! for every P ≤ S. The root (rank 0) then converts its value to the output
//! format once, and a binomial broadcast copies it to every rank in the
//! gather-wire format, which must hold every output value exactly.

use std::fmt;
use std::ops::Range;

use crate::collectives::{chunk_range, transcode, AllReduceOutcome, Round, Transfer, TransferKind, WirePolicy};
use crate::format::Format;
use crate::ops;

/// The default number of logical shards.
pub const DEFAULT_SHARDS: usize = 64;

/// Shards owned by `rank` when `shards` shards are split over `world` ranks.
pub fn owned_shards(shards: usize, world: usize, rank: usize) -> Range<usize> {
    chunk_range(shards, world, rank)
}

/// The rank that owns `shard`.
pub fn shard_owner(shards: usize, world: usize, shard: usize) -> usize {
    (0..world)
        .find(|&r| owned_shards(shards, world, r).contains(&shard))
        .expect("every shard has an owner")
}

/// One node of the fixed tree: at `level`, node `left` absorbs node `right`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Combine {
    pub level: u32,
    pub left: usize,
    pub right: usize,
}

/// The fixed tree over `shards` leaves, level by level. It depends only on `shards`.
pub fn tree(shards: usize) -> Vec<Combine> {
    let mut combines = Vec::new();
    let mut level = 0;
    while (1usize << level) < shards {
        let half = 1usize << level;
        for left in (0..shards).step_by(2 * half) {
            if left + half < shards {
                combines.push(Combine {
                    level,
                    left,
                    right: left + half,
                });
            }
        }
        level += 1;
    }
    combines
}

/// Pack messages `(src, dst)` into rounds, first fit, so that every rank
/// sends at most one and receives at most one message per round. Returns the
/// message indices of each round, in order. Every message lands in exactly
/// one round. First fit is not guaranteed to use the fewest rounds for an
/// arbitrary message set.
pub fn serialize(messages: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut rounds: Vec<Vec<usize>> = Vec::new();
    for (index, &(src, dst)) in messages.iter().enumerate() {
        let fits = |round: &Vec<usize>| {
            round
                .iter()
                .all(|&other| messages[other].0 != src && messages[other].1 != dst)
        };
        match rounds.iter_mut().find(|r| fits(r)) {
            Some(round) => round.push(index),
            None => rounds.push(vec![index]),
        }
    }
    rounds
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardTreeError {
    NoShards,
    EmptyWorld,
    /// Every rank must own at least one shard.
    TooManyRanks {
        world: usize,
        shards: usize,
    },
    RaggedInputs,
    InvalidRaw {
        shard: usize,
        index: usize,
        raw: u32,
    },
    /// The reduce wire cannot carry every accumulator value exactly, so a
    /// node would round differently when its child arrives in a message.
    LossyReduceWire {
        accumulator: Format,
        reduce_wire: Format,
    },
    /// The gather wire cannot carry every output value exactly.
    LossyGatherWire {
        output: Format,
        gather_wire: Format,
    },
}

impl fmt::Display for ShardTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardTreeError::NoShards => write!(f, "the shard tree needs at least one shard"),
            ShardTreeError::EmptyWorld => write!(f, "world must contain at least one rank"),
            ShardTreeError::TooManyRanks { world, shards } => {
                write!(f, "{world} ranks cannot each own one of {shards} shards")
            }
            ShardTreeError::RaggedInputs => write!(f, "every shard must contribute a buffer of the same length"),
            ShardTreeError::InvalidRaw { shard, index, raw } => {
                write!(
                    f,
                    "shard {shard} element {index}: raw pattern {raw:#x} does not fit the accumulator format"
                )
            }
            ShardTreeError::LossyReduceWire {
                accumulator,
                reduce_wire,
            } => write!(
                f,
                "the shard tree needs a reduce wire that holds every {} value exactly, not {}",
                accumulator.name(),
                reduce_wire.name()
            ),
            ShardTreeError::LossyGatherWire { output, gather_wire } => write!(
                f,
                "the shard tree needs a gather wire that holds every {} value exactly, not {}",
                output.name(),
                gather_wire.name()
            ),
        }
    }
}

impl std::error::Error for ShardTreeError {}

#[derive(Clone, Debug)]
pub struct ShardTreeOutcome {
    /// Results, roundings, rounds and additions, as for the other collectives.
    pub outcome: AllReduceOutcome,
    pub shards: usize,
    /// Tree levels (`ceil(log2 S)`).
    pub levels: usize,
    /// Levels whose combines needed at least one message.
    pub levels_with_messages: usize,
    /// Rounds of the tree reduce after serialization.
    pub reduce_rounds: usize,
    /// Rounds of the binomial broadcast of the root.
    pub broadcast_rounds: usize,
}

impl ShardTreeOutcome {
    /// Rounds added by serialization beyond one per level with messages.
    pub fn serialization_extra_rounds(&self) -> usize {
        self.reduce_rounds - self.levels_with_messages
    }
}

/// One rounding is counted for a conversion to a format that does not contain the value's format.
fn conversion_roundings(from: Format, to: Format) -> u32 {
    u32::from(!to.contains(from))
}

/// AllReduce `shards` (one buffer per shard, in the policy's accumulator
/// format) over `world` ranks with the fixed shard tree.
pub fn all_reduce(policy: WirePolicy, world: usize, shards: &[Vec<u32>]) -> Result<ShardTreeOutcome, ShardTreeError> {
    let s = shards.len();
    if s == 0 {
        return Err(ShardTreeError::NoShards);
    }
    if world == 0 {
        return Err(ShardTreeError::EmptyWorld);
    }
    if world > s {
        return Err(ShardTreeError::TooManyRanks { world, shards: s });
    }
    let len = shards[0].len();
    if shards.iter().any(|b| b.len() != len) {
        return Err(ShardTreeError::RaggedInputs);
    }
    let acc = policy.accumulator;
    for (shard, buffer) in shards.iter().enumerate() {
        if let Some((index, &raw)) = buffer.iter().enumerate().find(|(_, &v)| v > acc.max_raw()) {
            return Err(ShardTreeError::InvalidRaw { shard, index, raw });
        }
    }
    if !policy.reduce_wire.contains(acc) {
        return Err(ShardTreeError::LossyReduceWire {
            accumulator: acc,
            reduce_wire: policy.reduce_wire,
        });
    }
    if !policy.gather_wire.contains(policy.output) {
        return Err(ShardTreeError::LossyGatherWire {
            output: policy.output,
            gather_wire: policy.gather_wire,
        });
    }

    let wire = policy.reduce_wire;
    let mut nodes: Vec<Vec<u32>> = shards.to_vec();
    // Roundings in each node's tree. Every element of a node has the same count.
    let mut node_roundings = vec![0u32; s];
    let mut rounds = Vec::new();
    let mut additions = 0;
    let combines = tree(s);
    let levels = combines.last().map_or(0, |c| c.level as usize + 1);
    let mut levels_with_messages = 0;
    for level in 0..levels as u32 {
        let at_level: Vec<Combine> = combines.iter().copied().filter(|c| c.level == level).collect();
        // Every combine of this level reads only nodes that no other combine of
        // this level writes, so their order within the level does not matter.
        let mut messages = Vec::new();
        for c in &at_level {
            let (src, dst) = (shard_owner(s, world, c.right), shard_owner(s, world, c.left));
            let right: Vec<u32> = if src == dst {
                nodes[c.right].clone()
            } else {
                messages.push((src, dst));
                // Sent in the reduce wire and widened back: exact by the check above.
                nodes[c.right]
                    .iter()
                    .map(|&v| transcode(wire, acc, transcode(acc, wire, v)))
                    .collect()
            };
            // The node computation itself never depends on where `right` came from.
            for (l, r) in nodes[c.left].iter_mut().zip(&right) {
                *l = ops::add(acc, *l, *r);
            }
            // One addition; the message round trip is exact (the value is in
            // `acc`, which `wire` contains), so it adds no rounding.
            node_roundings[c.left] += node_roundings[c.right] + 1;
            additions += len;
        }
        if !messages.is_empty() {
            levels_with_messages += 1;
        }
        for round in serialize(&messages) {
            rounds.push(Round {
                phase: "tree-reduce",
                kind: TransferKind::Reduce,
                transfers: round
                    .iter()
                    .map(|&m| Transfer {
                        src: messages[m].0,
                        dst: messages[m].1,
                        elements: len,
                        bytes_per_element: wire.bytes(),
                    })
                    .collect(),
            });
        }
    }
    let reduce_rounds = rounds.len();

    // The root (rank 0, which owns shard 0) converts once to the output format.
    let output = policy.output;
    let root: Vec<u32> = nodes[0].iter().map(|&v| transcode(acc, output, v)).collect();
    let root_roundings = node_roundings[0] + conversion_roundings(acc, output);
    let gather = policy.gather_wire;
    let mut results: Vec<Option<Vec<u32>>> = vec![None; world];
    results[0] = Some(root);
    let mut distance = 1;
    while distance < world {
        let plan: Vec<(usize, usize)> = (0..distance)
            .filter(|r| r + distance < world)
            .map(|r| (r, r + distance))
            .collect();
        for &(src, dst) in &plan {
            let payload = results[src].as_ref().expect("sender holds the root value");
            let copied = payload
                .iter()
                .map(|&v| transcode(gather, output, transcode(output, gather, v)))
                .collect();
            results[dst] = Some(copied);
        }
        rounds.push(Round {
            phase: "broadcast",
            kind: TransferKind::Gather,
            transfers: plan
                .iter()
                .map(|&(src, dst)| Transfer {
                    src,
                    dst,
                    elements: len,
                    bytes_per_element: gather.bytes(),
                })
                .collect(),
        });
        distance *= 2;
    }
    let broadcast_rounds = rounds.len() - reduce_rounds;
    // Broadcast copies are exact (the value is in `output`, which `gather` contains).
    let roundings = vec![vec![root_roundings; len]; world];
    Ok(ShardTreeOutcome {
        outcome: AllReduceOutcome {
            results: results
                .into_iter()
                .map(|r| r.expect("broadcast reaches every rank"))
                .collect(),
            roundings,
            rounds,
            additions,
        },
        shards: s,
        levels,
        levels_with_messages,
        reduce_rounds,
        broadcast_rounds,
    })
}

//! A message-level AllReduce simulator.
//!
//! Each algorithm is executed as synchronous rounds of point-to-point
//! transfers between virtual ranks. Every addition a rank performs is a real
//! correctly rounded addition, so the association order each algorithm
//! implies — and therefore the output bits — come from the simulated schedule
//! itself, never from a separate formula.
//!
//! Round semantics: all transfers in a round read the senders' buffers as they
//! were at the start of the round, then every receiver applies what it got.
//! Every schedule here obeys a single-port model (each rank sends at most one
//! message and receives at most one message per round); tests enforce this.
//!
//! # Wire policies
//!
//! A [`WirePolicy`] separates where numbers live from what crosses the wire:
//!
//! * every rank's buffer, and every combine, is in the **accumulator** format
//!   (one correctly rounded addition per combine);
//! * a message in a reduce phase (its payload will be combined or summed) is
//!   converted to the **reduce wire** format when sent, and back to the
//!   accumulator format when received;
//! * a message in a gather phase (its payload is a finished value that is only
//!   copied) uses the **gather wire** format the same way;
//! * at the end every rank converts its buffer to the **output** format.
//!
//! A conversion between two different formats is one correctly rounded
//! conversion ([`ops::convert`]); between equal formats it copies the raw bits,
//! so NaN payloads survive copies exactly as in R1. [`all_reduce`] uses a
//! uniform policy (all four formats equal), which reproduces the R1 simulator
//! bit for bit; `tests/regression.rs` checks this against a frozen copy.

use std::fmt;
use std::ops::Range;

use crate::exact::ExactSum;
use crate::format::Format;
use crate::ops;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Algorithm {
    /// Chain reduce 0→1→…→N−1 (a left fold), then binomial broadcast.
    Sequential,
    /// Ring reduce-scatter then ring all-gather over N chunks.
    Ring,
    /// Binomial-tree reduce to rank 0, then binomial broadcast.
    BinomialTree,
    /// Rabenseifner: recursive-halving reduce-scatter, recursive-doubling all-gather.
    HalvingDoubling,
    /// Ring all-gather of every input, then each rank sums exactly and rounds once.
    ExactGather,
}

impl Algorithm {
    pub const ALL: [Algorithm; 5] = [
        Algorithm::Sequential,
        Algorithm::Ring,
        Algorithm::BinomialTree,
        Algorithm::HalvingDoubling,
        Algorithm::ExactGather,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Algorithm::Sequential => "sequential",
            Algorithm::Ring => "ring",
            Algorithm::BinomialTree => "binomial-tree",
            Algorithm::HalvingDoubling => "halving-doubling",
            Algorithm::ExactGather => "exact-gather",
        }
    }

    pub fn parse(text: &str) -> Option<Algorithm> {
        Algorithm::ALL.into_iter().find(|a| a.name() == text.trim())
    }

    pub const fn description(self) -> &'static str {
        match self {
            Algorithm::Sequential => "chain reduce (left fold in rank order) + binomial broadcast",
            Algorithm::Ring => "ring reduce-scatter + ring all-gather; each chunk's fold starts at a different rank",
            Algorithm::BinomialTree => "binomial-tree reduce to rank 0 + binomial broadcast",
            Algorithm::HalvingDoubling => {
                "recursive-halving reduce-scatter + recursive-doubling all-gather (power-of-two worlds)"
            }
            Algorithm::ExactGather => "all-gather every input, exact sum, one rounding; topology-independent",
        }
    }

    pub fn supports(self, world: usize) -> bool {
        match self {
            Algorithm::HalvingDoubling => world.is_power_of_two(),
            _ => world >= 1,
        }
    }

    /// Roundings each output element undergoes inside the collective under a
    /// uniform wire policy (additions only; any conversion of the inputs to
    /// the wire format before the collective is not counted here).
    pub fn roundings_per_element(self, world: usize) -> usize {
        match self {
            Algorithm::ExactGather => usize::from(world > 1),
            _ => world.saturating_sub(1),
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Formats used by a collective; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WirePolicy {
    /// Format of every rank's buffer and of every combine.
    pub accumulator: Format,
    /// Message format in reduce phases (payload is combined or summed).
    pub reduce_wire: Format,
    /// Message format in gather phases (payload is a finished value, copied).
    pub gather_wire: Format,
    /// Format of the final buffers.
    pub output: Format,
}

impl WirePolicy {
    /// Everything in one format: the R1 behaviour.
    pub const fn uniform(format: Format) -> WirePolicy {
        WirePolicy {
            accumulator: format,
            reduce_wire: format,
            gather_wire: format,
            output: format,
        }
    }

    pub fn is_uniform(&self) -> bool {
        *self == WirePolicy::uniform(self.accumulator)
    }
}

/// Convert a raw pattern between formats: a bit copy when the formats are
/// equal, otherwise one correctly rounded conversion.
pub fn transcode(from: Format, to: Format, raw: u32) -> u32 {
    if from == to {
        raw
    } else {
        ops::convert(from, to, raw)
    }
}

/// Whether a phase's messages are reduced at the receiver or only copied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferKind {
    /// The payload is an input to a reduction (sent in the reduce wire format).
    Reduce,
    /// The payload is a finished value that is copied (sent in the gather wire format).
    Gather,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transfer {
    pub src: usize,
    pub dst: usize,
    pub elements: usize,
    /// Bytes per element of the message's wire format.
    pub bytes_per_element: usize,
}

impl Transfer {
    pub fn bytes(&self) -> usize {
        self.elements * self.bytes_per_element
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub phase: &'static str,
    pub kind: TransferKind,
    pub transfers: Vec<Transfer>,
}

impl Round {
    /// Each rank sends at most one message and receives at most one.
    pub fn respects_single_port(&self, world: usize) -> bool {
        let mut sends = vec![0u32; world];
        let mut receives = vec![0u32; world];
        for t in &self.transfers {
            if t.src >= world || t.dst >= world || t.src == t.dst {
                return false;
            }
            sends[t.src] += 1;
            receives[t.dst] += 1;
        }
        sends.iter().chain(&receives).all(|&n| n <= 1)
    }
}

#[derive(Clone, Debug)]
pub struct AllReduceOutcome {
    /// Final buffer on every rank, in the policy's output format.
    pub results: Vec<Vec<u32>>,
    pub rounds: Vec<Round>,
    /// Correctly rounded accumulator-format additions performed across all ranks.
    pub additions: usize,
}

impl AllReduceOutcome {
    /// True if every rank ended with bit-identical buffers.
    pub fn consistent(&self) -> bool {
        self.results.windows(2).all(|w| w[0] == w[1])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollectiveError {
    EmptyWorld,
    RaggedInputs,
    InvalidRaw { rank: usize, index: usize, raw: u32 },
    UnsupportedWorld { algorithm: Algorithm, world: usize },
}

impl fmt::Display for CollectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CollectiveError::EmptyWorld => write!(f, "world must contain at least one rank"),
            CollectiveError::RaggedInputs => write!(f, "every rank must contribute a buffer of the same length"),
            CollectiveError::InvalidRaw { rank, index, raw } => {
                write!(
                    f,
                    "rank {rank} element {index}: raw pattern {raw:#x} does not fit the input format"
                )
            }
            CollectiveError::UnsupportedWorld { algorithm, world } => {
                write!(f, "{algorithm} does not support a world of {world} ranks")
            }
        }
    }
}

impl std::error::Error for CollectiveError {}

/// Element range of chunk `i` when `len` elements are split into `parts` near-equal chunks.
pub fn chunk_range(len: usize, parts: usize, i: usize) -> Range<usize> {
    let base = len / parts;
    let extra = len % parts;
    let start = i * base + i.min(extra);
    start..start + base + usize::from(i < extra)
}

struct Message {
    src: usize,
    dst: usize,
    range: Range<usize>,
    data: Vec<u32>,
}

struct Simulator {
    policy: WirePolicy,
    /// Every rank's buffer, in the accumulator format (or, once `final_output`
    /// is set, already in the output format).
    buffers: Vec<Vec<u32>>,
    rounds: Vec<Round>,
    additions: usize,
    /// Set by exact-gather, which rounds straight to the output format.
    final_output: bool,
}

impl Simulator {
    fn new(policy: WirePolicy, inputs: &[Vec<u32>]) -> Simulator {
        Simulator {
            policy,
            buffers: inputs.to_vec(),
            rounds: Vec::new(),
            additions: 0,
            final_output: false,
        }
    }

    fn world(&self) -> usize {
        self.buffers.len()
    }

    fn len(&self) -> usize {
        self.buffers[0].len()
    }

    fn wire(&self, kind: TransferKind) -> Format {
        match kind {
            TransferKind::Reduce => self.policy.reduce_wire,
            TransferKind::Gather => self.policy.gather_wire,
        }
    }

    /// Execute one round. `plan` lists (src, dst, element range).
    fn round(&mut self, phase: &'static str, kind: TransferKind, plan: &[(usize, usize, Range<usize>)]) {
        let acc = self.policy.accumulator;
        let wire = self.wire(kind);
        let messages: Vec<Message> = plan
            .iter()
            .map(|(src, dst, range)| Message {
                src: *src,
                dst: *dst,
                range: range.clone(),
                data: self.buffers[*src][range.clone()]
                    .iter()
                    .map(|&v| transcode(acc, wire, v))
                    .collect(),
            })
            .collect();
        for m in &messages {
            let target = &mut self.buffers[m.dst][m.range.clone()];
            match kind {
                TransferKind::Reduce => {
                    for (local, &incoming) in target.iter_mut().zip(&m.data) {
                        *local = ops::add(acc, *local, transcode(wire, acc, incoming));
                    }
                    self.additions += m.data.len();
                }
                TransferKind::Gather => {
                    for (local, &incoming) in target.iter_mut().zip(&m.data) {
                        *local = transcode(wire, acc, incoming);
                    }
                }
            }
        }
        self.rounds.push(Round {
            phase,
            kind,
            transfers: messages
                .iter()
                .map(|m| Transfer {
                    src: m.src,
                    dst: m.dst,
                    elements: m.data.len(),
                    bytes_per_element: wire.bytes(),
                })
                .collect(),
        });
    }

    fn binomial_broadcast(&mut self, root: usize) {
        let n = self.world();
        let all = 0..self.len();
        let mut distance = 1;
        while distance < n {
            let plan: Vec<_> = (0..distance)
                .filter(|rel| rel + distance < n)
                .map(|rel| ((rel + root) % n, (rel + distance + root) % n, all.clone()))
                .collect();
            self.round("broadcast", TransferKind::Gather, &plan);
            distance *= 2;
        }
    }

    fn sequential(&mut self) {
        let n = self.world();
        let all = 0..self.len();
        for r in 1..n {
            self.round("chain-reduce", TransferKind::Reduce, &[(r - 1, r, all.clone())]);
        }
        self.binomial_broadcast(n - 1);
    }

    fn binomial_tree(&mut self) {
        let n = self.world();
        let all = 0..self.len();
        let mut distance = 1;
        while distance < n {
            let plan: Vec<_> = (0..n)
                .step_by(2 * distance)
                .filter(|r| r + distance < n)
                .map(|r| (r + distance, r, all.clone()))
                .collect();
            self.round("tree-reduce", TransferKind::Reduce, &plan);
            distance *= 2;
        }
        self.binomial_broadcast(0);
    }

    fn ring_reduce_scatter(&mut self) {
        let n = self.world();
        let len = self.len();
        for step in 0..n - 1 {
            let plan: Vec<_> = (0..n)
                .map(|r| (r, (r + 1) % n, chunk_range(len, n, (r + n - step) % n)))
                .collect();
            self.round("reduce-scatter", TransferKind::Reduce, &plan);
        }
    }

    fn ring_all_gather(&mut self) {
        let n = self.world();
        let len = self.len();
        for step in 0..n - 1 {
            let plan: Vec<_> = (0..n)
                .map(|r| (r, (r + 1) % n, chunk_range(len, n, (r + 1 + n - step) % n)))
                .collect();
            self.round("all-gather", TransferKind::Gather, &plan);
        }
    }

    fn ring(&mut self) {
        self.ring_reduce_scatter();
        self.ring_all_gather();
    }

    /// Element range covered by a contiguous range of chunk indices.
    fn chunk_elements(len: usize, n: usize, chunks: &Range<usize>) -> Range<usize> {
        if chunks.is_empty() {
            0..0
        } else {
            chunk_range(len, n, chunks.start).start..chunk_range(len, n, chunks.end - 1).end
        }
    }

    /// Recursive halving; returns the chunk-index range each rank owns afterwards.
    fn halving_reduce_scatter(&mut self) -> Vec<Range<usize>> {
        let n = self.world();
        let len = self.len();
        let mut owned: Vec<Range<usize>> = vec![0..n; n];
        let mut distance = n / 2;
        while distance >= 1 {
            let mut plan = Vec::with_capacity(n);
            let mut next = owned.clone();
            for r in 0..n {
                let partner = r ^ distance;
                let current = owned[r].clone();
                let mid = (current.start + current.end) / 2;
                let (keep, give) = if r & distance == 0 {
                    (current.start..mid, mid..current.end)
                } else {
                    (mid..current.end, current.start..mid)
                };
                plan.push((r, partner, Simulator::chunk_elements(len, n, &give)));
                next[r] = keep;
            }
            self.round("reduce-scatter", TransferKind::Reduce, &plan);
            owned = next;
            distance /= 2;
        }
        owned
    }

    /// Recursive doubling from the ownership left by [`Self::halving_reduce_scatter`].
    fn doubling_all_gather(&mut self, mut owned: Vec<Range<usize>>) {
        let n = self.world();
        let len = self.len();
        let mut distance = 1;
        while distance < n {
            let plan: Vec<_> = (0..n)
                .map(|r| (r, r ^ distance, Simulator::chunk_elements(len, n, &owned[r])))
                .collect();
            self.round("all-gather", TransferKind::Gather, &plan);
            owned = (0..n)
                .map(|r| {
                    let (a, b) = (&owned[r], &owned[r ^ distance]);
                    a.start.min(b.start)..a.end.max(b.end)
                })
                .collect();
            distance *= 2;
        }
        debug_assert!(owned.iter().all(|r| *r == (0..n)));
    }

    fn halving_doubling(&mut self) {
        let owned = self.halving_reduce_scatter();
        self.doubling_all_gather(owned);
    }

    /// Every rank receives every input in the reduce wire format, then sums
    /// all of them exactly and rounds once to the output format. A rank's own
    /// input is also taken in its wire form, so every rank sums the same
    /// gathered tensor and all ranks agree.
    fn exact_gather(&mut self) {
        let n = self.world();
        let len = self.len();
        let acc = self.policy.accumulator;
        let wire = self.policy.reduce_wire;
        let gathered: Vec<Vec<u32>> = self
            .buffers
            .iter()
            .map(|b| b.iter().map(|&v| transcode(acc, wire, v)).collect())
            .collect();
        // have[r][s]: rank r holds rank s's input. Ring all-gather of whole buffers.
        let mut have = vec![vec![false; n]; n];
        for (r, row) in have.iter_mut().enumerate() {
            row[r] = true;
        }
        for step in 0..n - 1 {
            let mut transfers = Vec::with_capacity(n);
            for r in 0..n {
                let block = (r + n - step) % n;
                let dst = (r + 1) % n;
                debug_assert!(have[r][block]);
                have[dst][block] = true;
                transfers.push(Transfer {
                    src: r,
                    dst,
                    elements: len,
                    bytes_per_element: wire.bytes(),
                });
            }
            self.rounds.push(Round {
                phase: "all-gather-inputs",
                kind: TransferKind::Reduce,
                transfers,
            });
        }
        debug_assert!(have.iter().all(|row| row.iter().all(|&h| h)));
        for r in 0..n {
            for e in 0..len {
                let mut sum = ExactSum::new();
                for input in &gathered {
                    sum.add_raw(wire, input[e]);
                }
                self.buffers[r][e] = sum.round(self.policy.output);
            }
        }
        self.final_output = true;
    }

    fn into_outcome(self) -> AllReduceOutcome {
        let (acc, output) = (self.policy.accumulator, self.policy.output);
        let results = if self.final_output {
            self.buffers
        } else {
            self.buffers
                .into_iter()
                .map(|b| b.into_iter().map(|v| transcode(acc, output, v)).collect())
                .collect()
        };
        AllReduceOutcome {
            results,
            rounds: self.rounds,
            additions: self.additions,
        }
    }
}

fn validate(algorithm: Algorithm, input_format: Format, inputs: &[Vec<u32>]) -> Result<(), CollectiveError> {
    let world = inputs.len();
    if world == 0 {
        return Err(CollectiveError::EmptyWorld);
    }
    if inputs.iter().any(|b| b.len() != inputs[0].len()) {
        return Err(CollectiveError::RaggedInputs);
    }
    for (rank, buffer) in inputs.iter().enumerate() {
        if let Some((index, &raw)) = buffer.iter().enumerate().find(|(_, &raw)| raw > input_format.max_raw()) {
            return Err(CollectiveError::InvalidRaw { rank, index, raw });
        }
    }
    if !algorithm.supports(world) {
        return Err(CollectiveError::UnsupportedWorld { algorithm, world });
    }
    Ok(())
}

/// Run `algorithm` over `inputs` (one buffer per rank, raw patterns in `format`)
/// with everything in that one format.
pub fn all_reduce(
    algorithm: Algorithm,
    format: Format,
    inputs: &[Vec<u32>],
) -> Result<AllReduceOutcome, CollectiveError> {
    all_reduce_with(algorithm, WirePolicy::uniform(format), inputs)
}

/// Run `algorithm` under `policy`. `inputs` are raw patterns in the policy's
/// accumulator format; results are in its output format.
pub fn all_reduce_with(
    algorithm: Algorithm,
    policy: WirePolicy,
    inputs: &[Vec<u32>],
) -> Result<AllReduceOutcome, CollectiveError> {
    validate(algorithm, policy.accumulator, inputs)?;
    let mut sim = Simulator::new(policy, inputs);
    if inputs.len() > 1 {
        match algorithm {
            Algorithm::Sequential => sim.sequential(),
            Algorithm::Ring => sim.ring(),
            Algorithm::BinomialTree => sim.binomial_tree(),
            Algorithm::HalvingDoubling => sim.halving_doubling(),
            Algorithm::ExactGather => sim.exact_gather(),
        }
    }
    Ok(sim.into_outcome())
}

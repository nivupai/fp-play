//! A frozen copy of the R1 AllReduce simulator (runtime/src/collectives.rs at
//! commit 627f532), kept only as a regression reference.
//!
//! Do not edit the simulation logic below. `tests/regression.rs` checks that
//! the generalized simulator in `src/collectives.rs`, run with a uniform wire
//! policy, reproduces this code's output bits, round structure and addition
//! counts exactly. The only changes from the original are this header, the
//! imports, and the `LegacyRound` / `LegacyOutcome` result types (the crate's
//! `Round` and `Transfer` gained fields after R1); a transfer is recorded as
//! `(src, dst, elements)`.

#![allow(dead_code)]

use std::ops::Range;

use fp_play_runtime::collectives::Algorithm;
use fp_play_runtime::exact::ExactSum;
use fp_play_runtime::format::Format;
use fp_play_runtime::ops;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyRound {
    pub phase: &'static str,
    /// (src, dst, elements)
    pub transfers: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct LegacyOutcome {
    pub results: Vec<Vec<u32>>,
    pub rounds: Vec<LegacyRound>,
    pub additions: usize,
}

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
    format: Format,
    buffers: Vec<Vec<u32>>,
    rounds: Vec<LegacyRound>,
    additions: usize,
}

impl Simulator {
    fn world(&self) -> usize {
        self.buffers.len()
    }

    fn len(&self) -> usize {
        self.buffers[0].len()
    }

    /// Execute one round. `plan` lists (src, dst, element range).
    fn round(&mut self, phase: &'static str, plan: &[(usize, usize, Range<usize>)], combine: bool) {
        let messages: Vec<Message> = plan
            .iter()
            .map(|(src, dst, range)| Message {
                src: *src,
                dst: *dst,
                range: range.clone(),
                data: self.buffers[*src][range.clone()].to_vec(),
            })
            .collect();
        for m in &messages {
            let target = &mut self.buffers[m.dst][m.range.clone()];
            if combine {
                for (local, &incoming) in target.iter_mut().zip(&m.data) {
                    *local = ops::add(self.format, *local, incoming);
                }
                self.additions += m.data.len();
            } else {
                target.copy_from_slice(&m.data);
            }
        }
        self.rounds.push(LegacyRound {
            phase,
            transfers: messages.iter().map(|m| (m.src, m.dst, m.data.len())).collect(),
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
            self.round("broadcast", &plan, false);
            distance *= 2;
        }
    }

    fn sequential(&mut self) {
        let n = self.world();
        let all = 0..self.len();
        for r in 1..n {
            self.round("chain-reduce", &[(r - 1, r, all.clone())], true);
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
            self.round("tree-reduce", &plan, true);
            distance *= 2;
        }
        self.binomial_broadcast(0);
    }

    fn ring(&mut self) {
        let n = self.world();
        let len = self.len();
        for step in 0..n - 1 {
            let plan: Vec<_> = (0..n)
                .map(|r| (r, (r + 1) % n, chunk_range(len, n, (r + n - step) % n)))
                .collect();
            self.round("reduce-scatter", &plan, true);
        }
        for step in 0..n - 1 {
            let plan: Vec<_> = (0..n)
                .map(|r| (r, (r + 1) % n, chunk_range(len, n, (r + 1 + n - step) % n)))
                .collect();
            self.round("all-gather", &plan, false);
        }
    }

    fn halving_doubling(&mut self) {
        let n = self.world();
        let len = self.len();
        let elements = |chunks: &Range<usize>| -> Range<usize> {
            if chunks.is_empty() {
                0..0
            } else {
                chunk_range(len, n, chunks.start).start..chunk_range(len, n, chunks.end - 1).end
            }
        };
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
                plan.push((r, partner, elements(&give)));
                next[r] = keep;
            }
            self.round("reduce-scatter", &plan, true);
            owned = next;
            distance /= 2;
        }
        let mut distance = 1;
        while distance < n {
            let plan: Vec<_> = (0..n).map(|r| (r, r ^ distance, elements(&owned[r]))).collect();
            self.round("all-gather", &plan, false);
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

    fn exact_gather(&mut self) {
        let n = self.world();
        let len = self.len();
        let inputs = self.buffers.clone();
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
                transfers.push((r, dst, len));
            }
            self.rounds.push(LegacyRound {
                phase: "all-gather-inputs",
                transfers,
            });
        }
        debug_assert!(have.iter().all(|row| row.iter().all(|&h| h)));
        for r in 0..n {
            for e in 0..len {
                let mut sum = ExactSum::new();
                for input in &inputs {
                    sum.add_raw(self.format, input[e]);
                }
                self.buffers[r][e] = sum.round(self.format);
            }
        }
    }
}

/// The R1 entry point (input validation is left to the caller in this copy).
pub fn all_reduce(algorithm: Algorithm, format: Format, inputs: &[Vec<u32>]) -> LegacyOutcome {
    let world = inputs.len();
    let mut sim = Simulator {
        format,
        buffers: inputs.to_vec(),
        rounds: Vec::new(),
        additions: 0,
    };
    if world > 1 {
        match algorithm {
            Algorithm::Sequential => sim.sequential(),
            Algorithm::Ring => sim.ring(),
            Algorithm::BinomialTree => sim.binomial_tree(),
            Algorithm::HalvingDoubling => sim.halving_doubling(),
            Algorithm::ExactGather => sim.exact_gather(),
        }
    }
    LegacyOutcome {
        results: sim.buffers,
        rounds: sim.rounds,
        additions: sim.additions,
    }
}

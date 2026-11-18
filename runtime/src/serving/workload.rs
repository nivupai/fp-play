//! The deterministic request generator and synthetic tokens.
//!
//! Requests arrive as a Poisson process (exponential inter-arrival times
//! from [`super::detmath::ln`], rounded once to whole nanoseconds and summed
//! as integers). Lengths, priorities, shared system prompts and the number of
//! parallel samples are integer draws from [`SplitMix64`]. The same
//! configuration gives the same requests on every platform; a pinned hash in
//! `tests/serving_workload.rs` checks this on every CI run.
//!
//! Tokens are synthetic but deterministic: a prompt token is a hash of the
//! request (or of its shared prefix group) and the position, and sample `s`'s
//! `k`-th output token is a hash of (request, s, k). The KV cache stores these
//! tokens, so content checks can detect sharing and copy-on-write mistakes.

use crate::rng::SplitMix64;

use super::detmath;
use super::model::ServingError;

/// How a length is drawn. Every variant draws a whole number of tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LengthDist {
    Fixed(u32),
    /// Uniform over `lo..=hi`.
    Uniform {
        lo: u32,
        hi: u32,
    },
    /// `(weight, length)` pairs; a length is chosen with probability weight ÷ total.
    Table(Vec<(u32, u32)>),
}

impl LengthDist {
    fn draw(&self, rng: &mut SplitMix64) -> u32 {
        match self {
            LengthDist::Fixed(n) => *n,
            LengthDist::Uniform { lo, hi } => lo + rng.below(u64::from(hi - lo) + 1) as u32,
            LengthDist::Table(table) => pick(table, rng),
        }
    }

    fn check(&self, name: &str, min: u32) -> Result<(), ServingError> {
        let bad = |m: &str| Err(ServingError::InvalidConfig(format!("{name}: {m}")));
        match self {
            LengthDist::Fixed(n) if *n < min => bad("length below the minimum"),
            LengthDist::Uniform { lo, hi } if lo > hi => bad("lo > hi"),
            LengthDist::Uniform { lo, .. } if *lo < min => bad("length below the minimum"),
            LengthDist::Table(t) if t.is_empty() || t.iter().all(|(w, _)| *w == 0) => bad("empty table"),
            LengthDist::Table(t) if t.iter().any(|(_, n)| *n < min) => bad("length below the minimum"),
            _ => Ok(()),
        }
    }
}

/// Weighted choice from `(weight, value)` pairs.
fn pick<T: Copy>(table: &[(u32, T)], rng: &mut SplitMix64) -> T {
    let total: u64 = table.iter().map(|(w, _)| u64::from(*w)).sum();
    let mut x = rng.below(total);
    for (w, v) in table {
        if x < u64::from(*w) {
            return *v;
        }
        x -= u64::from(*w);
    }
    unreachable!("x < total")
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkloadConfig {
    pub seed: u64,
    pub requests: usize,
    /// Mean arrival rate, requests per second.
    pub rate_per_s: f64,
    /// Length of each request's own prompt tokens (after any shared prefix).
    pub prompt: LengthDist,
    /// Output tokens per sample.
    pub output: LengthDist,
    /// Number of shared system prompts (0 = none).
    pub prefix_groups: u32,
    /// Tokens in each shared system prompt.
    pub prefix_len: u32,
    /// Requests (per mille) that start with a shared system prompt.
    pub prefix_permille: u32,
    /// `(weight, n)`: parallel samples per request.
    pub samples: Vec<(u32, u32)>,
    /// `(weight, class)`: priority class, higher is more important.
    pub priorities: Vec<(u32, u8)>,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        WorkloadConfig {
            seed: 0x5e7e_0001,
            requests: 200,
            rate_per_s: 10.0,
            prompt: LengthDist::Uniform { lo: 64, hi: 1024 },
            output: LengthDist::Uniform { lo: 16, hi: 256 },
            prefix_groups: 0,
            prefix_len: 0,
            prefix_permille: 0,
            samples: vec![(1, 1)],
            priorities: vec![(1, 0)],
        }
    }
}

/// Allowed parallel-sample counts.
pub const SAMPLE_COUNTS: [u32; 3] = [1, 2, 4];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub id: u32,
    pub arrival_ns: u64,
    pub priority: u8,
    /// Shared system prompt this request starts with, if any.
    pub prefix_group: Option<u32>,
    /// Tokens of that shared prefix (0 without one).
    pub prefix_len: u32,
    /// Whole prompt: shared prefix plus the request's own tokens.
    pub prompt_len: u32,
    /// Output tokens per sample.
    pub output_len: u32,
    /// Parallel samples, each with its own output stream.
    pub samples: u32,
}

impl Request {
    /// Prompt token at `position` (< `prompt_len`).
    pub fn prompt_token(&self, position: u32) -> u32 {
        match self.prefix_group {
            Some(group) if position < self.prefix_len => token(&[1, u64::from(group), u64::from(position)]),
            _ => token(&[2, u64::from(self.id), u64::from(position)]),
        }
    }

    /// Sample `sample`'s output token `k`.
    pub fn output_token(&self, sample: u32, k: u32) -> u32 {
        token(&[3, u64::from(self.id), u64::from(sample), u64::from(k)])
    }

    /// Token at sequence `position` of `sample`: the prompt, then that sample's outputs.
    pub fn sequence_token(&self, sample: u32, position: u32) -> u32 {
        if position < self.prompt_len {
            self.prompt_token(position)
        } else {
            self.output_token(sample, position - self.prompt_len)
        }
    }
}

/// A synthetic token: a SplitMix64 hash of the words, truncated to 32 bits.
fn token(words: &[u64]) -> u32 {
    let mut h = 0x70c3_7e5d_u64;
    for &w in words {
        h = SplitMix64::new(h ^ w).next_u64();
    }
    (h >> 32) as u32
}

impl WorkloadConfig {
    pub fn check(&self) -> Result<(), ServingError> {
        let bad = |m: &str| Err(ServingError::InvalidConfig(m.to_string()));
        if !(self.rate_per_s.is_finite() && self.rate_per_s > 0.0) {
            return bad("arrival rate must be finite and positive");
        }
        if self.requests > u32::MAX as usize {
            return bad("too many requests");
        }
        self.prompt.check("prompt", 1)?;
        self.output.check("output", 1)?;
        if self.prefix_permille > 1000 {
            return bad("prefix_permille must be at most 1000");
        }
        if self.prefix_permille > 0 && (self.prefix_groups == 0 || self.prefix_len == 0) {
            return bad("a shared prefix needs at least one group and a positive length");
        }
        if self.samples.is_empty() || self.samples.iter().all(|(w, _)| *w == 0) {
            return bad("samples table is empty");
        }
        if self.samples.iter().any(|(_, n)| !SAMPLE_COUNTS.contains(n)) {
            return bad("samples per request must be 1, 2 or 4");
        }
        if self.priorities.is_empty() || self.priorities.iter().all(|(w, _)| *w == 0) {
            return bad("priority table is empty");
        }
        Ok(())
    }

    /// Generate the requests in arrival order (ids 0, 1, …).
    pub fn generate(&self) -> Result<Vec<Request>, ServingError> {
        self.check()?;
        let mut rng = SplitMix64::new(self.seed);
        let mut now_ns: u64 = 0;
        let mut out = Vec::with_capacity(self.requests);
        for id in 0..self.requests as u32 {
            // u in (0, 1]; inter-arrival = −ln(u)/rate seconds, rounded once to ns.
            let u = 1.0 - rng.next_unit();
            let gap_ns = (-detmath::ln(u) / self.rate_per_s * 1e9).round() as u64;
            now_ns += gap_ns;
            let own = self.prompt.draw(&mut rng);
            let output_len = self.output.draw(&mut rng);
            let prefix_group = if self.prefix_permille > 0 && rng.below(1000) < u64::from(self.prefix_permille) {
                Some(rng.below(u64::from(self.prefix_groups)) as u32)
            } else {
                None
            };
            let prefix_len = if prefix_group.is_some() { self.prefix_len } else { 0 };
            let samples = pick(&self.samples, &mut rng);
            let priority = pick(&self.priorities, &mut rng);
            out.push(Request {
                id,
                arrival_ns: now_ns,
                priority,
                prefix_group,
                prefix_len,
                prompt_len: prefix_len + own,
                output_len,
                samples,
            });
        }
        Ok(out)
    }
}

/// FNV-1a over every field of every request, for pinning a workload.
pub fn fingerprint(requests: &[Request]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for r in requests {
        eat(u64::from(r.id));
        eat(r.arrival_ns);
        eat(u64::from(r.priority));
        eat(r.prefix_group.map_or(u64::MAX, u64::from));
        eat(u64::from(r.prefix_len));
        eat(u64::from(r.prompt_len));
        eat(u64::from(r.output_len));
        eat(u64::from(r.samples));
    }
    h
}

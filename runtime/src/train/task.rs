//! The synthetic copy task.
//!
//! A sequence is `BOS s₁ … s_L SEP s₁ … s_L`: `L` symbols drawn uniformly
//! and independently from the vocabulary's symbol range, then the same
//! symbols again. The model reads every token but the last and is scored on
//! predicting the copy: the L targets after `SEP`. Copying the token L + 1
//! positions back is a fixed relative offset, which RoPE attention can
//! represent, so a 2-layer model learns it to near-perfect accuracy and with
//! large top-2 margins, unlike random weights.

use crate::rng::SplitMix64;

pub const BOS: u32 = 0;
pub const SEP: u32 = 1;
/// The first symbol ID (0 and 1 are BOS and SEP).
pub const FIRST_SYMBOL: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyTask {
    /// Symbols per sequence (L).
    pub len: usize,
    pub vocab: usize,
}

impl CopyTask {
    /// The prompt (`BOS s₁ … s_L SEP`) for given symbols.
    pub fn prompt(&self, symbols: &[u32]) -> Vec<u32> {
        assert_eq!(symbols.len(), self.len);
        let mut p = Vec::with_capacity(self.len + 2);
        p.push(BOS);
        p.extend_from_slice(symbols);
        p.push(SEP);
        p
    }

    /// `len` symbols from `rng`.
    pub fn symbols(&self, rng: &mut SplitMix64) -> Vec<u32> {
        let range = (self.vocab as u32 - FIRST_SYMBOL) as u64;
        (0..self.len).map(|_| FIRST_SYMBOL + rng.below(range) as u32).collect()
    }

    /// Model input (every token but the last) and the scored targets
    /// (position, expected next token) for given symbols.
    pub fn example(&self, symbols: &[u32]) -> (Vec<u32>, Vec<(usize, u32)>) {
        let mut input = self.prompt(symbols);
        input.extend_from_slice(&symbols[..self.len - 1]);
        let targets = (0..self.len).map(|i| (self.len + 1 + i, symbols[i])).collect();
        (input, targets)
    }

    /// The `count` symbol strings of batch `index` drawn from `seed`.
    pub fn batch(&self, seed: u64, index: u64, count: usize) -> Vec<Vec<u32>> {
        let mut rng = SplitMix64::new(seed ^ index.wrapping_mul(0xd1b5_4a32_d192_ed03));
        (0..count).map(|_| self.symbols(&mut rng)).collect()
    }
}

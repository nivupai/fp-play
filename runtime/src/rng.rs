//! Deterministic input generation that is bit-identical on every platform.
//!
//! Only IEEE-specified binary64 operations (+, ×, /, sqrt) are used — no libm
//! transcendental — so the same seed yields the same inputs on macOS and Linux.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard normal (Irwin–Hall: sum of 12 uniforms − 6).
    /// Bounded to [-6, 6]; adequate for synthetic activations and weights.
    pub fn next_normalish(&mut self) -> f64 {
        (0..12).map(|_| self.next_unit()).sum::<f64>() - 6.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        self.next_u64() % n
    }
}

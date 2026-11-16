//! Exact accumulation and a single round-to-nearest-even step.
//!
//! [`ExactSum`] is a fixed-point two's-complement integer whose least
//! significant bit weighs `2^LSB_EXPONENT`. Every finite FP16/BF16/FP32 value,
//! and every exact product of two such values, lands on this grid without
//! loss, so any number of additions is exact until [`ExactSum::round`]
//! performs the only rounding. This mirrors FP Play's TypeScript engine, which
//! uses exact BigInt dyadics and one destination rounding.
//!
//! Special values: NaN, or `+∞` together with `-∞`, give the format's
//! canonical quiet NaN (payloads are not propagated — a documented difference
//! from FP Play's FP16 NaN policy). An exactly zero finite sum is `-0` only if
//! every addend was `-0`; otherwise `+0`, as round-to-nearest-even requires.

use crate::format::{decode, Dyadic, Format, Value};

const LIMBS: usize = 12;
/// Weight of bit 0. Products of two FP32 subnormals reach `2^-298`.
pub const LSB_EXPONENT: i32 = -384;

#[derive(Clone, Debug)]
pub struct ExactSum {
    limbs: [u64; LIMBS],
    nan: bool,
    positive_infinity: bool,
    negative_infinity: bool,
    addends: usize,
    all_negative_zero: bool,
}

impl Default for ExactSum {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactSum {
    pub fn new() -> Self {
        ExactSum {
            limbs: [0; LIMBS],
            nan: false,
            positive_infinity: false,
            negative_infinity: false,
            addends: 0,
            all_negative_zero: true,
        }
    }

    pub fn add_raw(&mut self, format: Format, raw: u32) {
        self.add_value(decode(format, raw));
    }

    pub fn add_value(&mut self, value: Value) {
        self.addends += 1;
        match value {
            Value::Zero { negative } => {
                if !negative {
                    self.all_negative_zero = false;
                }
            }
            Value::Finite(d) => {
                self.all_negative_zero = false;
                self.add_dyadic(d);
            }
            Value::Infinity { negative } => {
                self.all_negative_zero = false;
                if negative {
                    self.negative_infinity = true;
                } else {
                    self.positive_infinity = true;
                }
            }
            Value::NaN => {
                self.all_negative_zero = false;
                self.nan = true;
            }
        }
    }

    fn add_dyadic(&mut self, d: Dyadic) {
        if d.significand == 0 {
            return;
        }
        let offset = d.exponent - LSB_EXPONENT;
        assert!(offset >= 0, "exponent {} below exact grid", d.exponent);
        let offset = offset as usize;
        let index = offset / 64;
        assert!(index + 2 < LIMBS, "exponent {} above exact grid", d.exponent);
        let wide = (d.significand as u128) << (offset % 64);
        let low = wide as u64;
        let high = (wide >> 64) as u64;
        if d.negative {
            self.subtract_at(index, low, high);
        } else {
            self.add_at(index, low, high);
        }
    }

    fn add_at(&mut self, index: usize, low: u64, high: u64) {
        let (sum, c0) = self.limbs[index].overflowing_add(low);
        self.limbs[index] = sum;
        let (sum, c1) = self.limbs[index + 1].overflowing_add(high);
        let (sum, c2) = sum.overflowing_add(c0 as u64);
        self.limbs[index + 1] = sum;
        let mut carry = c1 || c2;
        let mut i = index + 2;
        while carry && i < LIMBS {
            let (sum, c) = self.limbs[i].overflowing_add(1);
            self.limbs[i] = sum;
            carry = c;
            i += 1;
        }
    }

    fn subtract_at(&mut self, index: usize, low: u64, high: u64) {
        let (diff, b0) = self.limbs[index].overflowing_sub(low);
        self.limbs[index] = diff;
        let (diff, b1) = self.limbs[index + 1].overflowing_sub(high);
        let (diff, b2) = diff.overflowing_sub(b0 as u64);
        self.limbs[index + 1] = diff;
        let mut borrow = b1 || b2;
        let mut i = index + 2;
        while borrow && i < LIMBS {
            let (diff, b) = self.limbs[i].overflowing_sub(1);
            self.limbs[i] = diff;
            borrow = b;
            i += 1;
        }
    }

    /// True if the finite part is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// Round the exact sum once to `format` with round-to-nearest, ties-to-even.
    pub fn round(&self, format: Format) -> u32 {
        if self.nan || (self.positive_infinity && self.negative_infinity) {
            return format.canonical_nan();
        }
        if self.positive_infinity {
            return format.infinity(false);
        }
        if self.negative_infinity {
            return format.infinity(true);
        }
        let negative = self.limbs[LIMBS - 1] >> 63 == 1;
        let mut magnitude = self.limbs;
        if negative {
            // two's-complement negation
            let mut carry = true;
            for limb in magnitude.iter_mut() {
                let (v, c) = (!*limb).overflowing_add(carry as u64);
                *limb = v;
                carry = c;
            }
        }
        if magnitude.iter().all(|&l| l == 0) {
            return format.zero(self.addends > 0 && self.all_negative_zero);
        }
        round_magnitude(format, negative, &magnitude, LSB_EXPONENT)
    }
}

fn bit(limbs: &[u64], position: usize) -> bool {
    (limbs[position / 64] >> (position % 64)) & 1 == 1
}

fn any_below(limbs: &[u64], position: usize) -> bool {
    let index = position / 64;
    if limbs[..index].iter().any(|&l| l != 0) {
        return true;
    }
    let within = position % 64;
    within > 0 && limbs[index] & ((1u64 << within) - 1) != 0
}

fn bits(limbs: &[u64], start: usize, count: usize) -> u64 {
    debug_assert!(count <= 64);
    (0..count).fold(0u64, |acc, i| acc | ((bit(limbs, start + i) as u64) << i))
}

fn highest_bit(limbs: &[u64]) -> Option<usize> {
    limbs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &l)| l != 0)
        .map(|(i, &l)| i * 64 + 63 - l.leading_zeros() as usize)
}

/// Round `(-1)^negative × magnitude × 2^lsb_exponent` (magnitude nonzero) to
/// `format` with round-to-nearest, ties-to-even, including gradual underflow
/// and overflow to infinity.
pub fn round_magnitude(format: Format, negative: bool, magnitude: &[u64], lsb_exponent: i32) -> u32 {
    let msb = highest_bit(magnitude).expect("round_magnitude requires a nonzero magnitude");
    let precision = format.precision() as i32;
    let lead = msb as i32 + lsb_exponent;
    let mut quantum = (lead - (precision - 1)).max(format.min_quantum_exponent());
    let shift = quantum - lsb_exponent;
    let mut significand = if shift <= 0 {
        bits(magnitude, 0, msb + 1) << (-shift)
    } else {
        let s = shift as usize;
        let kept = if msb + 1 > s {
            bits(magnitude, s, msb + 1 - s)
        } else {
            0
        };
        let half = bit(magnitude, s - 1);
        let sticky = any_below(magnitude, s - 1);
        if half && (sticky || kept & 1 == 1) {
            kept + 1
        } else {
            kept
        }
    };
    if significand == 1u64 << precision {
        significand >>= 1;
        quantum += 1;
    }
    encode(format, negative, significand, quantum)
}

/// Encode `(-1)^negative × significand × 2^quantum` where the significand has
/// already been rounded to the format's precision at that quantum.
fn encode(format: Format, negative: bool, significand: u64, quantum: i32) -> u32 {
    let sign = format.zero(negative);
    if significand == 0 {
        return sign;
    }
    let fraction_bits = format.fraction_bits();
    if significand >= 1u64 << fraction_bits {
        let biased = quantum + fraction_bits as i32 + format.bias();
        if biased >= format.exponent_field_max() as i32 {
            return format.infinity(negative);
        }
        debug_assert!(biased >= 1);
        sign | ((biased as u32) << fraction_bits) | (significand as u32 & format.fraction_mask())
    } else {
        debug_assert_eq!(quantum, format.min_quantum_exponent());
        sign | significand as u32
    }
}

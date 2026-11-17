//! Fixed binary floating-point formats and exact decoding.
//!
//! Raw patterns are carried as `u32` regardless of width. Every API validates
//! that a raw pattern fits its format instead of silently masking it.

/// The three storage formats the runtime models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Format {
    /// IEEE 754 binary16.
    Fp16,
    /// bfloat16: binary32's exponent range with an 8-bit significand.
    Bf16,
    /// IEEE 754 binary32.
    Fp32,
}

impl Format {
    pub const ALL: [Format; 3] = [Format::Fp16, Format::Bf16, Format::Fp32];

    pub const fn name(self) -> &'static str {
        match self {
            Format::Fp16 => "fp16",
            Format::Bf16 => "bf16",
            Format::Fp32 => "fp32",
        }
    }

    pub fn parse(text: &str) -> Option<Format> {
        Format::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(text.trim()))
    }

    pub const fn width(self) -> u32 {
        match self {
            Format::Fp16 | Format::Bf16 => 16,
            Format::Fp32 => 32,
        }
    }

    pub const fn bytes(self) -> usize {
        (self.width() / 8) as usize
    }

    pub const fn exponent_bits(self) -> u32 {
        match self {
            Format::Fp16 => 5,
            Format::Bf16 | Format::Fp32 => 8,
        }
    }

    pub const fn fraction_bits(self) -> u32 {
        match self {
            Format::Fp16 => 10,
            Format::Bf16 => 7,
            Format::Fp32 => 23,
        }
    }

    /// Significand precision including the hidden bit.
    pub const fn precision(self) -> u32 {
        self.fraction_bits() + 1
    }

    pub const fn bias(self) -> i32 {
        (1 << (self.exponent_bits() - 1)) - 1
    }

    /// Exponent of the smallest normal number.
    pub const fn emin(self) -> i32 {
        1 - self.bias()
    }

    /// Exponent of the least significant bit of a subnormal (the finest quantum).
    pub const fn min_quantum_exponent(self) -> i32 {
        self.emin() - self.fraction_bits() as i32
    }

    pub const fn max_raw(self) -> u32 {
        if self.width() == 32 {
            u32::MAX
        } else {
            (1 << self.width()) - 1
        }
    }

    pub const fn sign_mask(self) -> u32 {
        1 << (self.width() - 1)
    }

    pub const fn exponent_field_max(self) -> u32 {
        (1 << self.exponent_bits()) - 1
    }

    pub const fn fraction_mask(self) -> u32 {
        (1 << self.fraction_bits()) - 1
    }

    /// Canonical quiet NaN produced by every runtime operation that yields NaN.
    pub const fn canonical_nan(self) -> u32 {
        match self {
            Format::Fp16 => 0x7e00,
            Format::Bf16 => 0x7fc0,
            Format::Fp32 => 0x7fc0_0000,
        }
    }

    pub const fn infinity(self, negative: bool) -> u32 {
        let magnitude = self.exponent_field_max() << self.fraction_bits();
        if negative {
            magnitude | self.sign_mask()
        } else {
            magnitude
        }
    }

    pub const fn zero(self, negative: bool) -> u32 {
        if negative {
            self.sign_mask()
        } else {
            0
        }
    }

    pub fn assert_raw(self, raw: u32) {
        assert!(
            raw <= self.max_raw(),
            "raw pattern {raw:#x} does not fit {} ({} bits)",
            self.name(),
            self.width()
        );
    }

    /// True if every value of `other` is exactly representable in `self`
    /// (so converting from `other` to `self` never rounds).
    pub const fn contains(self, other: Format) -> bool {
        matches!(
            (self, other),
            (Format::Fp32, _) | (Format::Bf16, Format::Bf16) | (Format::Fp16, Format::Fp16)
        )
    }
}

/// A nonzero finite value `(-1)^negative × significand × 2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyadic {
    pub negative: bool,
    pub significand: u64,
    pub exponent: i32,
}

/// Exact classification of an encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Zero { negative: bool },
    Finite(Dyadic),
    Infinity { negative: bool },
    NaN,
}

/// Decode a raw pattern exactly. Panics if `raw` does not fit `format`.
pub fn decode(format: Format, raw: u32) -> Value {
    format.assert_raw(raw);
    let negative = raw & format.sign_mask() != 0;
    let field = (raw >> format.fraction_bits()) & format.exponent_field_max();
    let fraction = raw & format.fraction_mask();
    if field == 0 {
        if fraction == 0 {
            Value::Zero { negative }
        } else {
            Value::Finite(Dyadic {
                negative,
                significand: fraction as u64,
                exponent: format.min_quantum_exponent(),
            })
        }
    } else if field == format.exponent_field_max() {
        if fraction == 0 {
            Value::Infinity { negative }
        } else {
            Value::NaN
        }
    } else {
        Value::Finite(Dyadic {
            negative,
            significand: (fraction | (1 << format.fraction_bits())) as u64,
            exponent: field as i32 - format.bias() - format.fraction_bits() as i32,
        })
    }
}

/// Decode an `f64` exactly (used only to quantize generated inputs).
pub fn decode_f64(x: f64) -> Value {
    let bits = x.to_bits();
    let negative = bits >> 63 == 1;
    let field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if field == 0 {
        if fraction == 0 {
            Value::Zero { negative }
        } else {
            Value::Finite(Dyadic {
                negative,
                significand: fraction,
                exponent: -1074,
            })
        }
    } else if field == 0x7ff {
        if fraction == 0 {
            Value::Infinity { negative }
        } else {
            Value::NaN
        }
    } else {
        Value::Finite(Dyadic {
            negative,
            significand: fraction | (1u64 << 52),
            exponent: field - 1075,
        })
    }
}

pub fn is_nan(format: Format, raw: u32) -> bool {
    matches!(decode(format, raw), Value::NaN)
}

/// Exact binary64 value of an encoding (every FP16/BF16/FP32 value embeds in binary64).
pub fn to_f64(format: Format, raw: u32) -> f64 {
    match decode(format, raw) {
        Value::Zero { negative } => {
            if negative {
                -0.0
            } else {
                0.0
            }
        }
        Value::Infinity { negative } => {
            if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }
        Value::NaN => f64::NAN,
        Value::Finite(d) => {
            // significand < 2^24 is exact; the power of two is built from bits so
            // no libm call is involved. Exponents here are within [-149, 104].
            let scale = f64::from_bits(((d.exponent + 1023) as u64) << 52);
            let magnitude = d.significand as f64 * scale;
            if d.negative {
                -magnitude
            } else {
                magnitude
            }
        }
    }
}

/// Distance in units of representable values between two non-NaN encodings of
/// the same format (`+0` and `-0` are distance 0 apart). `None` if either is NaN.
pub fn ulp_distance(format: Format, a: u32, b: u32) -> Option<u64> {
    if is_nan(format, a) || is_nan(format, b) {
        return None;
    }
    let key = |raw: u32| -> i64 {
        let magnitude = (raw & !format.sign_mask()) as i64;
        if raw & format.sign_mask() != 0 {
            -magnitude
        } else {
            magnitude
        }
    };
    Some(key(a).abs_diff(key(b)))
}

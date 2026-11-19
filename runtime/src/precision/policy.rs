//! Per-site precision levels and policies.

use super::narrow::Narrow;
use crate::format::Format;

/// A site's storage format (weights and the activations entering its
/// matrix products) and accumulator format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    /// FP32 storage, FP32 accumulation.
    Fp32,
    /// BF16 storage, FP32 accumulation.
    Bf16,
    /// BF16 storage, BF16 accumulation.
    Bf16Acc,
    /// FP16 storage, FP32 accumulation.
    Fp16,
    /// FP16 storage, FP16 accumulation.
    Fp16Acc,
    /// FP8 E4M3 storage with a binary32 scale per weight row and per
    /// activation vector, FP32 accumulation ([`super::narrow`]).
    Fp8E4M3,
    /// FP8 E5M2 storage with scales, FP32 accumulation.
    Fp8E5M2,
    /// INT8 storage with scales, exact integer accumulation.
    Int8,
}

impl Level {
    pub const ALL: [Level; 5] = [Level::Fp32, Level::Bf16, Level::Bf16Acc, Level::Fp16, Level::Fp16Acc];
    pub const LOW: [Level; 4] = [Level::Bf16, Level::Bf16Acc, Level::Fp16, Level::Fp16Acc];
    /// The 8-bit levels with explicit scales.
    pub const SCALED: [Level; 3] = [Level::Fp8E4M3, Level::Fp8E5M2, Level::Int8];
    /// Every level, FP32 first.
    pub const EVERY: [Level; 8] = [
        Level::Fp32,
        Level::Bf16,
        Level::Bf16Acc,
        Level::Fp16,
        Level::Fp16Acc,
        Level::Fp8E4M3,
        Level::Fp8E5M2,
        Level::Int8,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Level::Fp32 => "fp32",
            Level::Bf16 => "bf16",
            Level::Bf16Acc => "bf16-acc",
            Level::Fp16 => "fp16",
            Level::Fp16Acc => "fp16-acc",
            Level::Fp8E4M3 => "fp8-e4m3",
            Level::Fp8E5M2 => "fp8-e5m2",
            Level::Int8 => "int8",
        }
    }

    /// The 8-bit format of a scaled level.
    pub const fn narrow(self) -> Option<Narrow> {
        match self {
            Level::Fp8E4M3 => Some(Narrow::E4M3),
            Level::Fp8E5M2 => Some(Narrow::E5M2),
            Level::Int8 => Some(Narrow::Int8),
            _ => None,
        }
    }

    /// The storage format of a floating-point level. Panics for the scaled
    /// levels, which store [`Self::narrow`] values with scales.
    pub const fn storage(self) -> Format {
        match self {
            Level::Fp32 => Format::Fp32,
            Level::Bf16 | Level::Bf16Acc => Format::Bf16,
            Level::Fp16 | Level::Fp16Acc => Format::Fp16,
            Level::Fp8E4M3 | Level::Fp8E5M2 | Level::Int8 => panic!("scaled levels have no float storage format"),
        }
    }

    /// The accumulator format (FP32 for the scaled FP8 levels; INT8
    /// accumulates exactly in integers and is converted once to FP32).
    pub const fn accumulator(self) -> Format {
        match self {
            Level::Fp32 | Level::Bf16 | Level::Fp16 | Level::Fp8E4M3 | Level::Fp8E5M2 | Level::Int8 => Format::Fp32,
            Level::Bf16Acc => Format::Bf16,
            Level::Fp16Acc => Format::Fp16,
        }
    }
}

/// Where a precision level applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Site {
    /// Layer l's Q, K, V and output projections.
    Attention(usize),
    /// Layer l's gate, up and down projections.
    Mlp(usize),
    /// The KV cache (storage only; the attention core accumulates in FP32).
    KvCache,
    /// The tied embedding and output head.
    Head,
}

impl Site {
    pub fn all(layers: usize) -> Vec<Site> {
        let mut v = Vec::new();
        for l in 0..layers {
            v.push(Site::Attention(l));
            v.push(Site::Mlp(l));
        }
        v.push(Site::KvCache);
        v.push(Site::Head);
        v
    }

    pub fn name(&self) -> String {
        match self {
            Site::Attention(l) => format!("layer {l} attention"),
            Site::Mlp(l) => format!("layer {l} MLP"),
            Site::KvCache => "KV cache".into(),
            Site::Head => "embedding and head".into(),
        }
    }
}

/// A level for every site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Policy {
    pub name: String,
    pub sites: Vec<(Site, Level)>,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<PrecisionMutation>,
}

/// Deliberate bugs for the planted-bug tests (`planted-bugs` builds only).
#[cfg(feature = "planted-bugs")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMutation {
    /// BF16-accumulating sites accumulate in FP32.
    Bf16AccumulatesInFp32,
    /// The KV cache keeps binary32 values whatever its level.
    SkipKvRounding,
}

impl Policy {
    pub fn uniform(layers: usize, level: Level) -> Policy {
        Policy {
            name: format!("all {}", level.name()),
            sites: Site::all(layers).into_iter().map(|s| (s, level)).collect(),
            #[cfg(feature = "planted-bugs")]
            planted: None,
        }
    }

    /// FP32 everywhere except `site` at `level`.
    pub fn only(layers: usize, site: Site, level: Level) -> Policy {
        let mut p = Policy::uniform(layers, Level::Fp32);
        p.set(site, level);
        p.name = format!("{} at {}", site.name(), level.name());
        p
    }

    pub fn set(&mut self, site: Site, level: Level) {
        for (s, l) in &mut self.sites {
            if *s == site {
                *l = level;
            }
        }
    }

    pub fn level(&self, site: Site) -> Level {
        self.sites
            .iter()
            .find(|(s, _)| *s == site)
            .map_or(Level::Fp32, |&(_, l)| l)
    }
}

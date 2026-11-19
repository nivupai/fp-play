//! A static graph of one transformer layer (prefill or decode).
//!
//! The layer is the R2 8B-class layer: RMSNorm, the fused Q/K/V projection,
//! RoPE, attention scores, softmax, the weighted sum of V, the output
//! projection, a residual add, RMSNorm, the fused gate/up projection,
//! SiLU × up, the down projection and a residual add, plus the K/V write to
//! the cache. Elements are 2 bytes (BF16) throughout.

use crate::serving::model::ModelConfig;

/// Bytes per element of every tensor.
pub const ELEMENT_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// `tokens` prompt tokens of one sequence, attending causally to each other.
    Prefill { tokens: u64 },
    /// One new token for each of `batch` sequences, each with `context`
    /// tokens already in the KV cache.
    Decode { batch: u64, context: u64 },
}

impl Phase {
    pub fn name(&self) -> String {
        match self {
            Phase::Prefill { tokens } => format!("prefill {tokens} tokens"),
            Phase::Decode { batch, context } => format!("decode batch {batch}, context {context}"),
        }
    }

    /// Tokens computed in this phase.
    pub fn tokens(&self) -> u64 {
        match *self {
            Phase::Prefill { tokens } => tokens,
            Phase::Decode { batch, .. } => batch,
        }
    }
}

/// Where an operand lives before the op runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Model weights: always in HBM.
    Weights,
    /// An activation produced by an earlier op: in SRAM if it fits (see
    /// [`crate::dataflow::machine::Machine::resident`]), otherwise in HBM.
    Activation,
    /// The KV cache: always in HBM.
    KvCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    None,
    /// Output element (i, j) is needed only for j ≤ i (attention scores).
    CausalOutput,
    /// Reduction index k contributes to row i only for k ≤ i (weighted sum of V).
    CausalReduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// `count` independent products of an M×K operand A by a K×N operand B.
    MatMul {
        count: u64,
        m: u64,
        k: u64,
        n: u64,
        a: Source,
        b: Source,
        mask: Mask,
    },
    /// An element-wise pass over `elements` elements costing `ops` vector
    /// operations each, reading `inputs` and writing one tensor of that size.
    Vector { elements: u64, ops: u64, inputs: u64 },
    /// Write `elements` new K and V elements to the KV cache in HBM.
    KvWrite { elements: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub name: &'static str,
    pub kind: OpKind,
}

impl Op {
    /// Useful multiply–accumulates (masked-out pairs excluded).
    pub fn useful_macs(&self) -> u64 {
        match self.kind {
            OpKind::MatMul {
                count, m, k, n, mask, ..
            } => {
                count
                    * match mask {
                        Mask::None => m * k * n,
                        // Σ_i (i + 1) over rows i < m, times the other dimension.
                        Mask::CausalOutput => k * (m * (m + 1) / 2),
                        Mask::CausalReduction => n * (m * (m + 1) / 2),
                    }
            }
            _ => 0,
        }
    }
}

/// The ops of one layer in execution order.
pub fn layer(model: &ModelConfig, phase: Phase) -> Vec<Op> {
    let h = model.hidden;
    let d = model.head_dim;
    let kvw = model.kv_heads * d;
    let group = model.q_heads / model.kv_heads;
    let t = phase.tokens();
    let mm = |name, count, m, k, n, a, b, mask| Op {
        name,
        kind: OpKind::MatMul {
            count,
            m,
            k,
            n,
            a,
            b,
            mask,
        },
    };
    let vector = |name, elements, ops, inputs| Op {
        name,
        kind: OpKind::Vector { elements, ops, inputs },
    };
    let mut ops = vec![
        vector("attention RMSNorm", t * h, 4, 1),
        mm(
            "Q/K/V projection",
            1,
            t,
            h,
            h + 2 * kvw,
            Source::Activation,
            Source::Weights,
            Mask::None,
        ),
        vector("RoPE", t * (h + kvw), 6, 1),
        Op {
            name: "K/V cache write",
            kind: OpKind::KvWrite { elements: 2 * t * kvw },
        },
    ];
    match phase {
        Phase::Prefill { tokens } => {
            // Per query head: scores = Q·Kᵀ (causal), probabilities · V (causal).
            // K and V of this prompt were just computed: activations.
            ops.push(mm(
                "attention scores",
                model.q_heads,
                tokens,
                d,
                tokens,
                Source::Activation,
                Source::Activation,
                Mask::CausalOutput,
            ));
            ops.push(vector("softmax", model.q_heads * tokens * (tokens + 1) / 2, 5, 1));
            ops.push(mm(
                "attention × V",
                model.q_heads,
                tokens,
                tokens,
                d,
                Source::Activation,
                Source::Activation,
                Mask::CausalReduction,
            ));
        }
        Phase::Decode { batch, context } => {
            // Per sequence and KV head: the group's query heads share K and V,
            // read from the cache (the context plus the new token).
            let l = context + 1;
            ops.push(mm(
                "attention scores",
                batch * model.kv_heads,
                group,
                d,
                l,
                Source::Activation,
                Source::KvCache,
                Mask::None,
            ));
            ops.push(vector("softmax", batch * model.q_heads * l, 5, 1));
            ops.push(mm(
                "attention × V",
                batch * model.kv_heads,
                group,
                l,
                d,
                Source::Activation,
                Source::KvCache,
                Mask::None,
            ));
        }
    }
    ops.extend([
        mm(
            "output projection",
            1,
            t,
            h,
            h,
            Source::Activation,
            Source::Weights,
            Mask::None,
        ),
        vector("residual add", t * h, 1, 2),
        vector("MLP RMSNorm", t * h, 4, 1),
        mm(
            "gate/up projection",
            1,
            t,
            h,
            2 * model.ffn,
            Source::Activation,
            Source::Weights,
            Mask::None,
        ),
        vector("SiLU × up", t * model.ffn, 4, 2),
        mm(
            "down projection",
            1,
            t,
            model.ffn,
            h,
            Source::Activation,
            Source::Weights,
            Mask::None,
        ),
        vector("residual add", t * h, 1, 2),
    ]);
    ops
}

/// Weight bytes of one layer (the matrix products' B operands that are weights).
pub fn weight_bytes(model: &ModelConfig) -> u64 {
    let kvw = model.kv_heads * model.head_dim;
    let h = model.hidden;
    ELEMENT_BYTES * (h * (h + 2 * kvw) + h * h + h * 2 * model.ffn + model.ffn * h)
}

/// KV-cache bytes one decode layer must read: K and V of every context
/// position (plus the new token) for every sequence.
pub fn decode_kv_read_bytes(model: &ModelConfig, batch: u64, context: u64) -> u64 {
    ELEMENT_BYTES * 2 * batch * (context + 1) * model.kv_heads * model.head_dim
}

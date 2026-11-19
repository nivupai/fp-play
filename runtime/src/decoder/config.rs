//! Decoder dimensions and the seed for its deterministic random weights.

/// A Llama-style decoder: RMSNorm, rotary position embeddings (RoPE),
/// grouped-query attention and a SwiGLU MLP, with tied input and output
/// embeddings. Weights are BF16; see [`super::weights`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderConfig {
    pub hidden: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// SwiGLU intermediate width.
    pub ffn: usize,
    pub vocab: usize,
    /// RoPE base (θ).
    pub rope_base: f64,
    /// RMSNorm ε, added to the mean square in binary32.
    pub rms_eps: f32,
    /// Seed of the weights.
    pub seed: u64,
    /// Multiplies every logit (binary32) after the output projection. For a
    /// power of two the product is exact, so the order of the logits, and
    /// therefore greedy decoding, cannot change; only absolute margins scale.
    pub logit_scale: f32,
}

impl DecoderConfig {
    /// The R3 model: 2 layers, hidden 256, 16 query and 8 KV heads of
    /// dimension 16, FFN 704, vocabulary 2048. Every P in {1, 2, 4, 8}
    /// divides the heads, the FFN width and the vocabulary.
    pub const TINY: DecoderConfig = DecoderConfig {
        hidden: 256,
        layers: 2,
        q_heads: 16,
        kv_heads: 8,
        head_dim: 16,
        ffn: 704,
        vocab: 2048,
        rope_base: 10_000.0,
        rms_eps: 1e-5,
        seed: 0x00de_c0de_0003,
        logit_scale: 1.0,
    };

    /// The trained model (`crate::train`): the R3 dimensions ([`Self::TINY`])
    /// with its own seed, which sets the training's initialization, so every
    /// R3 and R4 comparison between random and trained weights uses the same
    /// architecture.
    pub const TRAINED: DecoderConfig = DecoderConfig {
        seed: 0x00de_c0de_0074,
        ..DecoderConfig::TINY
    };

    /// A reduced model with the same structure, small enough for the
    /// stdlib-only Python reference (`scripts/gen_decoder_vectors.py`).
    /// Four KV heads, so the reference can run P = 4.
    pub const REFERENCE: DecoderConfig = DecoderConfig {
        hidden: 32,
        layers: 2,
        q_heads: 8,
        kv_heads: 4,
        head_dim: 4,
        ffn: 48,
        vocab: 64,
        rope_base: 10_000.0,
        rms_eps: 1e-5,
        seed: 0x00de_c0de_00f1,
        logit_scale: 1.0,
    };

    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            self.hidden,
            self.layers,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.ffn,
            self.vocab,
        ];
        if positive.contains(&0) {
            return Err("every dimension must be positive".into());
        }
        if self.q_heads * self.head_dim != self.hidden {
            return Err("q_heads × head_dim must equal hidden".into());
        }
        if self.q_heads % self.kv_heads != 0 {
            return Err("kv_heads must divide q_heads".into());
        }
        if self.head_dim % 2 != 0 {
            return Err("RoPE needs an even head_dim".into());
        }
        if !(self.logit_scale.is_finite() && self.logit_scale > 0.0) {
            return Err("logit_scale must be finite and positive".into());
        }
        if self.vocab > u32::MAX as usize {
            return Err("vocabulary too large".into());
        }
        Ok(())
    }

    /// Width of one token's K (or V) across all KV heads.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Query heads per KV head.
    pub fn group(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Weight count (tied embeddings counted once).
    pub fn parameters(&self) -> usize {
        let per_layer = 2 * self.hidden
            + self.hidden * (self.hidden + 2 * self.kv_width())
            + self.hidden * self.hidden
            + 3 * self.hidden * self.ffn;
        self.vocab * self.hidden + self.layers * per_layer + self.hidden
    }

    /// Multiply–accumulates of the matrix products for one token (attention
    /// scores and the weighted sum of V excluded; they grow with context).
    pub fn macs_per_token(&self) -> usize {
        let per_layer =
            self.hidden * (self.hidden + 2 * self.kv_width()) + self.hidden * self.hidden + 3 * self.hidden * self.ffn;
        self.layers * per_layer + self.vocab * self.hidden
    }

    /// Whether `world` ranks can split this model: P must divide the KV
    /// heads (so every rank has whole GQA groups), the FFN width and the vocabulary.
    pub fn supports_world(&self, world: usize) -> bool {
        world >= 1 && self.kv_heads % world == 0 && self.ffn % world == 0 && self.vocab % world == 0
    }
}

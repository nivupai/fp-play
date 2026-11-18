//! Model dimensions, device constants and KV-cache capacity.
//!
//! The default model has 8B-class public transformer dimensions (hidden
//! 4096, 32 layers, 32 query heads and 8 KV heads of dimension 128, FFN
//! 14336, an illustrative 128,000-token vocabulary), BF16 weights and KV.
//! It is not a specific released model. The device constants are
//! illustrative, not a measurement or a datasheet of any product.
//!
//! Tensor parallelism over P devices splits the heads and the FFN, so each
//! device stores 1/P of the weights and of every token's KV. P must divide
//! the KV head count (8 by default): with more devices than KV heads, real
//! systems replicate KV heads, which this model does not do.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub hidden: u64,
    pub layers: u64,
    pub q_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    pub vocab: u64,
    /// Bytes per weight element.
    pub weight_bytes: u64,
    /// Bytes per K or V element.
    pub kv_bytes: u64,
}

impl ModelConfig {
    /// 8B-class dimensions; see the module documentation.
    pub const EIGHT_B_CLASS: ModelConfig = ModelConfig {
        hidden: 4096,
        layers: 32,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        ffn: 14336,
        vocab: 128_000,
        weight_bytes: 2,
        kv_bytes: 2,
    };

    /// Parameters of one transformer layer: Q and O projections, K and V
    /// projections, the gated MLP (three matrices) and two norm vectors.
    pub fn layer_params(&self) -> u64 {
        let kv_width = self.kv_heads * self.head_dim;
        let attention = 2 * self.hidden * self.q_heads * self.head_dim + 2 * self.hidden * kv_width;
        let mlp = 3 * self.hidden * self.ffn;
        attention + mlp + 2 * self.hidden
    }

    /// All parameters: layers, input embedding, output head, final norm.
    pub fn params(&self) -> u64 {
        self.layers * self.layer_params() + 2 * self.vocab * self.hidden + self.hidden
    }

    /// Weight bytes stored on each of `world` devices (rounded up).
    pub fn weight_bytes_per_device(&self, world: u64) -> u64 {
        (self.params() * self.weight_bytes).div_ceil(world)
    }

    /// KV bytes one token occupies on each of `world` devices (K and V, every layer).
    pub fn kv_bytes_per_token_per_device(&self, world: u64) -> u64 {
        2 * self.layers * (self.kv_heads / world) * self.head_dim * self.kv_bytes
    }

    /// Accept `world` only if it splits the KV heads (and so the query heads) evenly.
    pub fn check_world(&self, world: u64) -> Result<(), ServingError> {
        if world == 0
            || self.kv_heads % world != 0
            || self.q_heads % world != 0
            || self.ffn % world != 0
            || self.hidden % world != 0
        {
            return Err(ServingError::WorldDoesNotDivideHeads {
                world,
                kv_heads: self.kv_heads,
            });
        }
        Ok(())
    }
}

/// Illustrative device constants (not a named product).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceConfig {
    /// Device memory in bytes. A per-scenario setting.
    pub memory_bytes: u64,
    /// Memory held back for activations and workspace, in bytes.
    pub reserve_bytes: u64,
    /// Peak dense BF16 throughput, FLOP/s.
    pub peak_flops: f64,
    /// Device memory bandwidth, bytes/s.
    pub hbm_bytes_per_s: f64,
    /// Host link bandwidth used for swapping, bytes/s.
    pub host_bytes_per_s: f64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig {
            memory_bytes: 80_000_000_000,
            reserve_bytes: 4_000_000_000,
            peak_flops: 400e12,
            hbm_bytes_per_s: 3e12,
            host_bytes_per_s: 25e9,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServingError {
    WorldDoesNotDivideHeads {
        world: u64,
        kv_heads: u64,
    },
    /// Weights and the reserve do not fit, or no KV block does.
    NoKvMemory {
        memory_bytes: u64,
        needed_bytes: u64,
    },
    InvalidConfig(String),
}

impl fmt::Display for ServingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServingError::WorldDoesNotDivideHeads { world, kv_heads } => write!(
                f,
                "tensor-parallel size {world} must divide the {kv_heads} KV heads (and the query heads, hidden and FFN \
                 widths); KV-head replication for larger sizes is not modeled"
            ),
            ServingError::NoKvMemory {
                memory_bytes,
                needed_bytes,
            } => write!(
                f,
                "device memory {memory_bytes} B leaves no room for one KV block (weights, reserve and one block need \
                 {needed_bytes} B)"
            ),
            ServingError::InvalidConfig(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ServingError {}

/// Device KV blocks of `block_size` tokens available on each device at
/// tensor-parallel size `world`: (memory − weight shard − reserve) ÷ block bytes.
/// Every rank holds the same blocks (1/P of the heads of the same tokens), so
/// this is also the number of logical blocks.
pub fn kv_capacity_blocks(
    model: &ModelConfig,
    device: &DeviceConfig,
    world: u64,
    block_size: u64,
) -> Result<u64, ServingError> {
    model.check_world(world)?;
    if block_size == 0 {
        return Err(ServingError::InvalidConfig("block size must be at least 1".into()));
    }
    let block_bytes = block_size * model.kv_bytes_per_token_per_device(world);
    let fixed = model.weight_bytes_per_device(world) + device.reserve_bytes;
    let blocks = device.memory_bytes.saturating_sub(fixed) / block_bytes;
    if blocks == 0 {
        return Err(ServingError::NoKvMemory {
            memory_bytes: device.memory_bytes,
            needed_bytes: fixed + block_bytes,
        });
    }
    Ok(blocks)
}

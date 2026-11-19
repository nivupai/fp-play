//! Mapping ops onto the machine and timing them.
//!
//! # Mapping
//!
//! A matrix product is computed in output passes of `tm` rows (a multiple of
//! the array's rows, as many as fit a quarter of SRAM, or all of A if A is
//! resident), and within a pass in column blocks of the array's width and
//! reduction chunks of `tk` (two B chunks fit a quarter of SRAM: double
//! buffering). Each (pass, column block, reduction chunk) is one **chunk**:
//! the DMA engine loads its B part (unless B is resident), its A part on the
//! first column block of a pass (unless A is resident) and stores its
//! output on the last reduction chunk (unless the output is resident); the
//! array then spends `ceil(rows / R) × kk` cycles on it (R rows × C columns
//! of multiply–accumulates per cycle). Masked chunks entirely above the
//! causal diagonal are skipped; partly masked ones are computed in full
//! (issued) while only unmasked pairs count as useful. A vector op is split
//! into chunks of `lanes × 4096` elements costing `ceil(e / lanes) × ops`
//! vector cycles, with its inputs and output moved by DMA unless resident.
//!
//! # Timing
//!
//! One DMA engine and one compute engine (the array, or the vector unit),
//! two chunk buffers. Chunk i's load starts when the DMA is free and its
//! buffer is free (chunk i − 2's compute has ended); its compute starts when
//! its load has ended and the engine is free:
//!
//! ```text
//! L_start(i) = max(L_end(i−1), C_end(i−2));  L_end(i) = L_start(i) + l_i
//! C_start(i) = max(L_end(i), C_end(i−1));    C_end(i) = C_start(i) + c_i
//! ```
//!
//! Ops run one after another (each waits for the previous one: a data
//! dependency), so every op's first load is exposed. `reference.rs`
//! re-derives every timeline with a cycle-by-cycle simulation.

use super::graph::{Mask, Op, OpKind, Source, ELEMENT_BYTES};
use super::machine::Machine;

/// One unit of work: DMA cycles, then compute cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chunk {
    pub load: u64,
    pub compute: u64,
    pub hbm_bytes: u64,
    /// Multiply–accumulates issued on the array (including padding and masked pairs).
    pub issued_macs: u64,
    pub useful_macs: u64,
}

/// Pairs (i, j) with i in [i0, i1), j in [j0, j1) and j ≤ i.
fn lower_pairs(i0: u64, i1: u64, j0: u64, j1: u64) -> u64 {
    (i0..i1).map(|i| (i + 1).min(j1).saturating_sub(j0)).sum()
}

fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

/// The chunks of `op` on `machine`.
pub fn chunks(op: &Op, machine: &Machine) -> Vec<Chunk> {
    let bpc = machine.hbm_bytes_per_cycle;
    let cycles = |bytes: u64| ceil_div(bytes, bpc);
    let mut out = Vec::new();
    match op.kind {
        OpKind::MatMul {
            count,
            m,
            k,
            n,
            a,
            b,
            mask,
        } => {
            let (r, c) = (machine.rows, machine.cols);
            let a_resident = a == Source::Activation && machine.resident(m * k * ELEMENT_BYTES);
            let b_resident = b == Source::Activation && machine.resident(k * n * ELEMENT_BYTES);
            let out_resident = machine.resident(m * n * ELEMENT_BYTES);
            let quarter = machine.sram_bytes / 4;
            let tm = if a_resident {
                ceil_div(m, r) * r
            } else {
                (r * (quarter / (r * k * ELEMENT_BYTES)).max(1)).min(ceil_div(m, r) * r)
            };
            let tk = (quarter / (2 * c * ELEMENT_BYTES)).clamp(1, k);
            let mut instance = Vec::new();
            for p in 0..ceil_div(m, tm) {
                let (i0, i1) = (p * tm, (p * tm + tm).min(m));
                let rows = i1 - i0;
                for nb in 0..ceil_div(n, c) {
                    let (j0, j1) = (nb * c, (nb * c + c).min(n));
                    if mask == Mask::CausalOutput && j0 >= i1 {
                        continue;
                    }
                    let kcs: Vec<u64> = (0..ceil_div(k, tk))
                        .filter(|&kc| mask != Mask::CausalReduction || kc * tk < i1)
                        .collect();
                    for (idx, &kc) in kcs.iter().enumerate() {
                        let (k0, k1) = (kc * tk, (kc * tk + tk).min(k));
                        let kk = k1 - k0;
                        let mut bytes = 0;
                        if !b_resident {
                            bytes += kk * (j1 - j0) * ELEMENT_BYTES;
                        }
                        if !a_resident && nb == 0 {
                            bytes += rows * kk * ELEMENT_BYTES;
                        }
                        if !out_resident && idx + 1 == kcs.len() {
                            bytes += rows * (j1 - j0) * ELEMENT_BYTES;
                        }
                        let useful = match mask {
                            Mask::None => rows * (j1 - j0) * kk,
                            Mask::CausalOutput => lower_pairs(i0, i1, j0, j1) * kk,
                            Mask::CausalReduction => lower_pairs(i0, i1, k0, k1) * (j1 - j0),
                        };
                        instance.push(Chunk {
                            load: cycles(bytes),
                            compute: ceil_div(rows, r) * kk,
                            hbm_bytes: bytes,
                            issued_macs: ceil_div(rows, r) * r * c * kk,
                            useful_macs: useful,
                        });
                    }
                }
            }
            for _ in 0..count {
                out.extend_from_slice(&instance);
            }
        }
        OpKind::Vector { elements, ops, inputs } => {
            let resident = machine.resident(elements * ELEMENT_BYTES);
            let size = machine.lanes * 4096;
            let mut done = 0;
            while done < elements {
                let e = size.min(elements - done);
                let bytes = if resident { 0 } else { (inputs + 1) * e * ELEMENT_BYTES };
                out.push(Chunk {
                    load: cycles(bytes),
                    compute: ceil_div(e, machine.lanes) * ops,
                    hbm_bytes: bytes,
                    ..Chunk::default()
                });
                done += e;
            }
        }
        OpKind::KvWrite { elements } => {
            let size = machine.lanes * 4096;
            let mut done = 0;
            while done < elements {
                let e = size.min(elements - done);
                let bytes = e * ELEMENT_BYTES;
                out.push(Chunk {
                    load: cycles(bytes),
                    hbm_bytes: bytes,
                    ..Chunk::default()
                });
                done += e;
            }
        }
    }
    #[cfg(feature = "planted-bugs")]
    if machine.planted == Some(super::DataflowMutation::DecodeSkipsKvReads) {
        if let OpKind::MatMul { b: Source::KvCache, .. } = op.kind {
            for ch in &mut out {
                ch.hbm_bytes = 0;
                ch.load = 0;
            }
        }
    }
    out
}

/// The timeline of one op's chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timeline {
    pub cycles: u64,
    /// Compute-engine busy cycles.
    pub busy: u64,
    /// Idle cycles before the first compute (the exposed first load).
    pub fill: u64,
    /// Other idle cycles: waiting for the DMA.
    pub memory_stall: u64,
    /// DMA busy cycles.
    pub dma: u64,
}

/// Apply the double-buffering recurrence (see the module documentation).
pub fn timeline(chunks: &[Chunk], machine: &Machine) -> Timeline {
    let _ = machine;
    let (mut l_end, mut c_end) = (0u64, 0u64);
    let mut c_ends: Vec<u64> = Vec::with_capacity(chunks.len());
    let mut first_start = None;
    for (i, ch) in chunks.iter().enumerate() {
        #[allow(unused_mut)]
        let mut back = 2;
        #[cfg(feature = "planted-bugs")]
        if machine.planted == Some(super::DataflowMutation::BufferFreedEarly) {
            back = 3;
        }
        let free = if i >= back { c_ends[i - back] } else { 0 };
        let l_start = l_end.max(free);
        l_end = l_start + ch.load;
        let c_start = l_end.max(c_end);
        first_start.get_or_insert(c_start);
        c_end = c_start + ch.compute;
        c_ends.push(c_end);
    }
    let cycles = c_end.max(l_end);
    let busy: u64 = chunks.iter().map(|c| c.compute).sum();
    let fill = first_start.unwrap_or(0).min(cycles);
    Timeline {
        cycles,
        busy,
        fill,
        memory_stall: cycles - busy - fill,
        dma: chunks.iter().map(|c| c.load).sum(),
    }
}

/// One op's mapping and timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpTiming {
    pub op: Op,
    pub timeline: Timeline,
    pub chunks: usize,
    pub hbm_bytes: u64,
    pub issued_macs: u64,
    pub useful_macs: u64,
    /// Array (matrix products) or vector unit.
    pub on_array: bool,
}

pub fn time_op(op: &Op, machine: &Machine) -> OpTiming {
    let ch = chunks(op, machine);
    OpTiming {
        op: *op,
        timeline: timeline(&ch, machine),
        chunks: ch.len(),
        hbm_bytes: ch.iter().map(|c| c.hbm_bytes).sum(),
        issued_macs: ch.iter().map(|c| c.issued_macs).sum(),
        useful_macs: ch.iter().map(|c| c.useful_macs).sum(),
        on_array: matches!(op.kind, OpKind::MatMul { .. }),
    }
}

/// A whole layer, op by op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerTiming {
    pub ops: Vec<OpTiming>,
}

impl LayerTiming {
    pub fn cycles(&self) -> u64 {
        self.ops.iter().map(|o| o.timeline.cycles).sum()
    }
    pub fn hbm_bytes(&self) -> u64 {
        self.ops.iter().map(|o| o.hbm_bytes).sum()
    }
    pub fn useful_macs(&self) -> u64 {
        self.ops.iter().map(|o| o.useful_macs).sum()
    }
    pub fn issued_macs(&self) -> u64 {
        self.ops.iter().map(|o| o.issued_macs).sum()
    }
}

pub fn time_layer(ops: &[Op], machine: &Machine) -> LayerTiming {
    LayerTiming {
        ops: ops.iter().map(|o| time_op(o, machine)).collect(),
    }
}

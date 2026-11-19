//! An independent cycle-by-cycle simulation of the DMA engine, the two chunk
//! buffers and the compute engine, to check [`super::sim::timeline`].
//!
//! At every cycle boundary it first retires whatever finishes then (a load
//! fills its buffer; a compute frees its buffer and the engine), then starts
//! whatever can start (the next load, if the DMA is idle and a buffer is
//! free; the next compute, if its load is done and the engine is idle),
//! repeating until nothing changes (zero-length work finishes at once),
//! then advances one cycle.

use super::sim::Chunk;

pub fn cycles(chunks: &[Chunk]) -> u64 {
    let n = chunks.len();
    let mut t = 0u64;
    let mut next_load = 0;
    let mut next_compute = 0;
    let mut loading: Option<(usize, u64)> = None; // (chunk, end time)
    let mut computing: Option<(usize, u64)> = None;
    let mut loaded = vec![false; n];
    let mut buffers_in_use = 0;
    let mut last = 0;
    loop {
        let mut changed = true;
        while changed {
            changed = false;
            if let Some((i, end)) = loading {
                if end <= t {
                    loaded[i] = true;
                    loading = None;
                    last = last.max(end);
                    changed = true;
                }
            }
            if let Some((_, end)) = computing {
                if end <= t {
                    computing = None;
                    buffers_in_use -= 1;
                    last = last.max(end);
                    changed = true;
                }
            }
            if loading.is_none() && next_load < n && buffers_in_use < 2 {
                buffers_in_use += 1;
                loading = Some((next_load, t + chunks[next_load].load));
                next_load += 1;
                changed = true;
            }
            if computing.is_none() && next_compute < n && loaded[next_compute] {
                computing = Some((next_compute, t + chunks[next_compute].compute));
                next_compute += 1;
                changed = true;
            }
        }
        if next_compute == n && computing.is_none() && loading.is_none() {
            return last;
        }
        t += 1;
    }
}

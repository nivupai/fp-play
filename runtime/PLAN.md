# FP Play runtime — plan

Goal: a small, verifiable inference systems project in the same spirit as
FP Play. Every claim is backed by an independent check or recorded data, and
a written boundary says what is not claimed. Milestones are ordered so each
one ships something usable on its own.

## R1 — Collectives numerics ✅

Exact superaccumulator numerics; five AllReduce schedules as message-level
simulations; α–β cost model; row-parallel linear experiment; recorded
artifact with stale detection. See [README.md](README.md).

## R1.1 — Policies people actually choose (small, next)

- **Wire ≠ accumulator**: BF16 on the wire with FP32 accumulation inside
  reduce-scatter (2× bytes in that phase). Measure how much of exact-gather's
  accuracy it recovers, and at what cost.
- **Reduce-scatter only** (sequence-parallel output) and all-gather as
  separate collectives.
- **A cheaper deterministic AllReduce**: a fixed logical reduction tree over
  shards, independent of the physical topology, compared with exact-gather.
- **Sensitivity**: sweep seeds, hidden size and P up to 64; report spread,
  not one run.

## R2 — Serving runtime (idea #3)

A deterministic discrete-event simulator in Rust:

- Request lifecycle: queued → prefill → decode → finished / preempted / failed.
- **Paged KV cache**: fixed-size blocks, a free-list allocator, per-sequence
  block tables, reference-counted prefix sharing, copy-on-write.
- **Continuous batching scheduler**: a per-step token budget mixing prefill
  chunks and decode tokens; FCFS with a configurable priority policy.
- **Preemption**: recompute versus swap, with block accounting checked after
  every step.
- **Failure handling**: inject a rank failure mid-collective; abort, retry or
  drain, with invariants (no leaked blocks, no duplicated tokens).
- Metrics: time to first token, time per output token, throughput,
  p50/p99 under Poisson arrivals with configurable prompt/output lengths.
- Step time comes from the R1 cost model plus a simple compute model, so
  tensor-parallel communication shows up in latency.

Verification: property tests on allocator invariants, a reference scheduler
written as a slow obvious loop, and recorded traces.

## R3 — End to end: does sharding change the tokens?

A tiny Llama-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention)
run tensor-parallel across virtual devices inside the R2 runtime, with the
exact numerics from R1.

Headline experiments:

- Does greedy decoding diverge across P, AllReduce schedule, or **batch
  composition** (batch invariance)? After how many tokens?
- What does making it deterministic cost in latency?

Uses random weights by default; a small real checkpoint is optional (it needs
network access to download).

## R4 — Per-layer precision policies (idea #5)

Choose a storage format and accumulation width per layer (FP32 / BF16 /
FP16, later FP8 / INT8 with scales). Measure logit drift and perplexity
against FP32, reusing FP Play's mixed-precision semantics as the reference.
Report which layers are sensitive.

## R5 — Dataflow mapping simulator (idea #2)

Describe a transformer block as a static graph and map it onto a toy
fixed-function pipeline (compute units, SRAM, HBM bandwidth). Report
utilization, stall reasons and a roofline for prefill versus decode. FP Play's
Nangate45 delays can feed one unit model, clearly labeled as an assumption.

## R6 — FP Play UI integration

- A `#collectives` workspace that reads the recorded JSON: an
  animated-on-request schedule view (rank × round grid, text alternative),
  per-element association trees, and the error-versus-P chart.
- Landing and overall UI refresh (deferred by request).
- Optional: compile the crate to WebAssembly so the browser runs the real
  simulator rather than a JavaScript copy (once a wasm toolchain is
  available).

## Housekeeping

- CI job running `cargo fmt --check`, `cargo clippy -D warnings` and
  `cargo test --release`, with actions pinned to commit IDs like the existing
  workflows.
- Python bindings (PyO3) once crates can be fetched; the crate has no
  dependencies until then.

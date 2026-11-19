//! R5a: the dataflow graph, mapping and timing against independent checks.

use fp_play_runtime::dataflow::graph::{decode_kv_read_bytes, layer, weight_bytes, OpKind, Phase, Source};
use fp_play_runtime::dataflow::machine::{Machine, RECORDED_FP16_FMA_DELAY_NS};
use fp_play_runtime::dataflow::reference;
use fp_play_runtime::dataflow::sim::{chunks, time_layer, time_op, timeline, Chunk};
use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::model::ModelConfig;

/// A small layer and machine, so the cycle-by-cycle reference stays quick.
const SMALL: ModelConfig = ModelConfig {
    hidden: 256,
    layers: 1,
    q_heads: 4,
    kv_heads: 2,
    head_dim: 64,
    ffn: 512,
    vocab: 1000,
    weight_bytes: 2,
    kv_bytes: 2,
};

fn small_machine() -> Machine {
    let mut m = Machine::illustrative("small", 1.0);
    m.rows = 16;
    m.cols = 16;
    m.lanes = 32;
    m.sram_bytes = 256 << 10;
    m.hbm_bytes_per_cycle = 64;
    m
}

fn phases() -> Vec<Phase> {
    vec![
        Phase::Prefill { tokens: 1 },
        Phase::Prefill { tokens: 37 },
        Phase::Prefill { tokens: 128 },
        Phase::Decode { batch: 1, context: 63 },
        Phase::Decode { batch: 5, context: 200 },
    ]
}

#[test]
fn recurrence_equals_the_cycle_by_cycle_reference_on_random_chunks() {
    let m = small_machine();
    let mut rng = SplitMix64::new(0xdf10);
    for _ in 0..3_000 {
        let n = rng.below(40) as usize;
        let ch: Vec<Chunk> = (0..n)
            .map(|_| Chunk {
                load: if rng.below(5) == 0 { 0 } else { rng.below(50) },
                compute: if rng.below(5) == 0 { 0 } else { rng.below(50) },
                ..Chunk::default()
            })
            .collect();
        assert_eq!(timeline(&ch, &m).cycles, reference::cycles(&ch), "{ch:?}");
    }
}

#[test]
fn recurrence_equals_the_reference_on_every_op_of_small_layers() {
    let m = small_machine();
    for phase in phases() {
        for op in layer(&SMALL, phase) {
            let ch = chunks(&op, &m);
            assert_eq!(
                timeline(&ch, &m).cycles,
                reference::cycles(&ch),
                "{} {}",
                phase.name(),
                op.name
            );
        }
    }
}

#[test]
fn mapping_conserves_useful_work_and_bounds_hold() {
    for (model, machine) in [
        (SMALL, small_machine()),
        (ModelConfig::EIGHT_B_CLASS, Machine::illustrative("big", 1.0)),
    ] {
        for phase in phases().into_iter().chain([
            Phase::Prefill { tokens: 2048 },
            Phase::Decode {
                batch: 64,
                context: 1024,
            },
        ]) {
            for op in layer(&model, phase) {
                let t = time_op(&op, &machine);
                assert_eq!(t.useful_macs, op.useful_macs(), "{} {}", phase.name(), op.name);
                assert!(t.issued_macs >= t.useful_macs);
                let tl = t.timeline;
                assert_eq!(tl.busy + tl.fill + tl.memory_stall, tl.cycles);
                assert!(tl.cycles >= tl.busy.max(tl.dma), "{}", op.name);
                assert!(tl.cycles <= tl.busy + tl.dma, "{}", op.name);
            }
        }
    }
}

#[test]
fn weights_and_kv_cache_are_read_at_least_once() {
    let model = ModelConfig::EIGHT_B_CLASS;
    let machine = Machine::illustrative("big", 1.0);
    for phase in [
        Phase::Prefill { tokens: 512 },
        Phase::Decode {
            batch: 8,
            context: 1024,
        },
    ] {
        let ops = layer(&model, phase);
        let timed = time_layer(&ops, &machine);
        let weight_ops: u64 = timed
            .ops
            .iter()
            .filter(|o| matches!(o.op.kind, OpKind::MatMul { b: Source::Weights, .. }))
            .map(|o| o.hbm_bytes)
            .sum();
        assert!(weight_ops >= weight_bytes(&model), "{}", phase.name());
        if let Phase::Decode { batch, context } = phase {
            let kv: u64 = timed
                .ops
                .iter()
                .filter(|o| matches!(o.op.kind, OpKind::MatMul { b: Source::KvCache, .. }))
                .map(|o| o.hbm_bytes)
                .sum();
            assert!(kv >= decode_kv_read_bytes(&model, batch, context), "KV bytes {kv}");
        }
    }
}

/// Hand-derived: a 4×8 by 8×4 product on a 2×2 array with 1 byte/cycle.
/// A is resident (one pass of 4 rows); two column blocks of 2, one
/// reduction chunk each: B part 8·2·2 = 32 bytes → 32 load cycles;
/// compute ceil(4/2)·8 = 16 cycles. Loads at 0–32 and 32–64, computes at
/// 32–48 and 64–80: 80 cycles, fill 32, busy 32, stall 16.
#[test]
fn hand_derived_product() {
    let mut m = Machine::illustrative("tiny", 1.0);
    m.rows = 2;
    m.cols = 2;
    m.sram_bytes = 1 << 20;
    m.hbm_bytes_per_cycle = 1;
    let op = fp_play_runtime::dataflow::graph::Op {
        name: "hand",
        kind: OpKind::MatMul {
            count: 1,
            m: 4,
            k: 8,
            n: 4,
            a: Source::Activation,
            b: Source::Weights,
            mask: fp_play_runtime::dataflow::graph::Mask::None,
        },
    };
    let ch = chunks(&op, &m);
    assert_eq!(
        ch.iter().map(|c| (c.load, c.compute)).collect::<Vec<_>>(),
        vec![(32, 16), (32, 16)]
    );
    let t = timeline(&ch, &m);
    assert_eq!((t.cycles, t.fill, t.busy, t.memory_stall), (80, 32, 32, 16));
}

#[test]
fn the_recorded_fma_delay_matches_the_hardware_artifact() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/lib/showcase/hardware-summary.json"
    ))
    .expect("hardware summary");
    let at = text.find("\"id\": \"baseline-fma\"").expect("baseline-fma record");
    let rest = &text[at..];
    let d = rest.find("\"delay\": ").expect("delay field") + "\"delay\": ".len();
    let value: f64 = rest[d..].split([',', '\n']).next().unwrap().trim().parse().unwrap();
    assert_eq!(value, RECORDED_FP16_FMA_DELAY_NS);
}

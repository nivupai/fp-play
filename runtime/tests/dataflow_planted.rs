//! R5 planted bugs (compiled only with `--features planted-bugs`). Each must
//! be caught by the check named in its test.
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::dataflow::graph::{decode_kv_read_bytes, layer, OpKind, Phase, Source};
use fp_play_runtime::dataflow::machine::Machine;
use fp_play_runtime::dataflow::reference;
use fp_play_runtime::dataflow::sim::{time_layer, timeline, Chunk};
use fp_play_runtime::dataflow::DataflowMutation;
use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::serving::model::ModelConfig;

/// Caught by the recurrence-versus-reference check on random chunks (the
/// same check as `tests/dataflow.rs`): freeing a buffer one chunk early
/// lets loads run ahead of what two buffers allow.
#[test]
fn buffer_freed_early_is_caught_by_the_cycle_reference() {
    let mut m = Machine::illustrative("small", 1.0);
    m.planted = Some(DataflowMutation::BufferFreedEarly);
    let mut rng = SplitMix64::new(0xdf10);
    let mut caught = 0;
    for _ in 0..3_000 {
        let n = rng.below(40) as usize;
        let ch: Vec<Chunk> = (0..n)
            .map(|_| Chunk {
                load: if rng.below(5) == 0 { 0 } else { rng.below(50) },
                compute: if rng.below(5) == 0 { 0 } else { rng.below(50) },
                ..Chunk::default()
            })
            .collect();
        caught += usize::from(timeline(&ch, &m).cycles != reference::cycles(&ch));
    }
    eprintln!("BufferFreedEarly: {caught} of 3000 random chunk lists disagree with the cycle-by-cycle reference");
    assert!(caught > 0, "not caught");
}

/// Caught by the KV-traffic check: decode attention must read every
/// cached K and V at least once.
#[test]
fn decode_without_kv_reads_is_caught_by_the_traffic_check() {
    let mut m = Machine::illustrative("big", 1.0);
    m.planted = Some(DataflowMutation::DecodeSkipsKvReads);
    let model = ModelConfig::EIGHT_B_CLASS;
    let timed = time_layer(
        &layer(
            &model,
            Phase::Decode {
                batch: 8,
                context: 1024,
            },
        ),
        &m,
    );
    let kv: u64 = timed
        .ops
        .iter()
        .filter(|o| matches!(o.op.kind, OpKind::MatMul { b: Source::KvCache, .. }))
        .map(|o| o.hbm_bytes)
        .sum();
    let want = decode_kv_read_bytes(&model, 8, 1024);
    eprintln!("DecodeSkipsKvReads: {kv} KV bytes read, at least {want} required");
    assert!(kv < want, "not caught");
}

//! R2: the step-time model, against hand-computed values.

use fp_play_runtime::collectives::{Algorithm, WirePolicy};
use fp_play_runtime::cost::LinkModel;
use fp_play_runtime::format::Format;
use fp_play_runtime::serving::model::{DeviceConfig, ModelConfig, ServingError};
use fp_play_runtime::serving::step_time::{Collective, SeqWork, StepModel, StepTimer, StepWork};

fn model(world: u64, collective: Collective, policy: WirePolicy) -> Result<StepModel, ServingError> {
    StepModel::new(
        ModelConfig::EIGHT_B_CLASS,
        DeviceConfig::default(),
        world,
        16,
        collective,
        policy,
        LinkModel::default(),
        20_000.0,
    )
}

const BF16: WirePolicy = WirePolicy::uniform(Format::Bf16);

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

#[test]
fn one_decode_at_p1_is_memory_bound() {
    let m = model(1, Collective::Schedule(Algorithm::Ring), BF16).unwrap();
    let work = StepWork {
        seqs: vec![SeqWork {
            context: 0,
            tokens: 1,
            samples: true,
        }],
        cow_blocks: 0,
        swap_blocks: 0,
    };
    let t = m.time(&work);
    // Bytes: (8,028,164,096 − 128,000·4096 embedding) × 2 + 2 tokens × 131,072
    // = 15,008,014,336; ÷ 3e12 B/s = 5,002,671.445… ns. FLOPs (1.5e10) ÷ 4e14
    // take only 37.5 µs, so the step is memory-bound.
    assert!(close(t.compute_ns, 15_008_014_336.0 / 3e12 * 1e9), "{}", t.compute_ns);
    assert_eq!(t.comm_ns, 0.0);
    assert_eq!(t.total_ns, 5_022_671);
}

#[test]
fn ring_allreduce_cost_from_its_schedule() {
    let m = model(2, Collective::Schedule(Algorithm::Ring), BF16).unwrap();
    // P = 2 ring on 4096 BF16 elements: one reduce-scatter and one all-gather
    // round, each 2048 elements × 2 bytes = 4096 B: 2 × (2 µs + 4096 B ÷ 100 GB/s).
    assert!(close(m.allreduce_ns(4096), 2.0 * (2000.0 + 4096.0 / 100.0)));
    let one = StepWork {
        seqs: vec![SeqWork {
            context: 10,
            tokens: 1,
            samples: true,
        }],
        ..StepWork::default()
    };
    // 32 layers × 2 AllReduces per step.
    assert!(close(m.time(&one).comm_ns, 64.0 * m.allreduce_ns(4096)));
}

#[test]
fn collectives_and_policies_change_communication() {
    let tokens = 64 * 4096;
    let ring = model(8, Collective::Schedule(Algorithm::Ring), BF16).unwrap();
    let hd = model(8, Collective::Schedule(Algorithm::HalvingDoubling), BF16).unwrap();
    let tree = model(8, Collective::ShardTree(64), BF16).unwrap();
    let ring32 = model(8, Collective::Schedule(Algorithm::Ring), WirePolicy::FP32_RS_BF16_AG).unwrap();
    // Halving-doubling: 6 rounds instead of ring's 14, the same bytes.
    assert!(hd.allreduce_ns(tokens) < ring.allreduce_ns(tokens));
    // The shard tree moves whole buffers at every level.
    assert!(tree.allreduce_ns(tokens) > ring.allreduce_ns(tokens));
    // FP32 reduce-scatter messages are twice as large.
    assert!(ring32.allreduce_ns(tokens) > ring.allreduce_ns(tokens));
    // Lossy wire on the shard tree, and P not dividing the KV heads, are rejected.
    assert!(model(8, Collective::ShardTree(64), WirePolicy::BF16_WIRE_FP32_ADD).is_err());
    assert!(matches!(
        model(3, Collective::Schedule(Algorithm::Ring), BF16),
        Err(ServingError::WorldDoesNotDivideHeads { world: 3, .. })
    ));
}

#[test]
fn copies_and_swaps_are_charged() {
    let m = model(1, Collective::Schedule(Algorithm::Ring), BF16).unwrap();
    let base = m.time(&StepWork::default());
    let with = m.time(&StepWork {
        cow_blocks: 3,
        swap_blocks: 5,
        ..StepWork::default()
    });
    // A block is 16 × 131,072 = 2,097,152 bytes per device.
    let expected = 6.0 * 2_097_152.0 / 3e12 * 1e9 + 5.0 * 2_097_152.0 / 25e9 * 1e9;
    assert!(close(with.copy_ns, expected));
    assert!(with.total_ns > base.total_ns);
}

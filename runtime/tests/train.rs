//! The trainer: backward pass against finite differences, forward pass
//! against the inference model, determinism, and the committed weights.

use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::precision::model::MasterWeights;
use fp_play_runtime::train::checks::{finite_difference_check, inference_forward_check};
use fp_play_runtime::train::net::{Layout, Net, RopeTable};
use fp_play_runtime::train::task::{CopyTask, BOS, SEP};
use fp_play_runtime::train::{batch_gradient, learning_rate, load_trained, TrainConfig, TRAINED_WEIGHTS_SHA256};

#[test]
fn backward_matches_central_finite_differences_for_every_parameter() {
    let worst = finite_difference_check().unwrap();
    eprintln!("worst relative gradient error {worst:e}");
}

#[test]
fn training_forward_matches_the_inference_model() {
    let worst = inference_forward_check().unwrap();
    // Rounding only: binary32 chains with separate products vs fused chains.
    assert!(worst > 0.0 && worst < 1e-5, "{worst:e}");
}

#[test]
fn copy_examples_score_the_tokens_after_sep() {
    let task = CopyTask { len: 4, vocab: 64 };
    let (input, targets) = task.example(&[5, 9, 2, 63]);
    assert_eq!(input, vec![BOS, 5, 9, 2, 63, SEP, 5, 9, 2]);
    assert_eq!(targets, vec![(5, 5), (6, 9), (7, 2), (8, 63)]);
    let b = task.batch(7, 3, 100);
    assert!(b.iter().flatten().all(|&s| (2..64).contains(&s)));
    assert_eq!(b, task.batch(7, 3, 100));
    assert_ne!(b, task.batch(7, 4, 100));
}

#[test]
fn learning_rate_warms_up_then_decays_linearly() {
    let c = TrainConfig {
        steps: 150,
        warmup: 50,
        peak_lr: 2e-3,
        final_lr_fraction: 0.1,
        ..TrainConfig::default()
    };
    assert_eq!(learning_rate(&c, 0), 2e-3 * (1.0 / 50.0));
    assert_eq!(learning_rate(&c, 49), 2e-3);
    assert_eq!(learning_rate(&c, 50), 2e-3);
    assert!((learning_rate(&c, 100) - 1.1e-3).abs() < 1e-9);
}

/// The batch gradient (computed on many threads) is bit for bit the sum of
/// the sequences' gradients in sequence order, computed here on one thread.
#[test]
fn batch_gradient_is_the_in_order_sum_of_sequence_gradients() {
    let c = fp_play_runtime::train::checks::SMALL;
    let c = DecoderConfig { vocab: 16, ..c };
    let layout = Layout::new(&c);
    let task = CopyTask { len: 5, vocab: 16 };
    let rope = RopeTable::<f32>::new(&c, 16);
    let p = layout.flatten(&MasterWeights::generate(&c));
    let batch = task.batch(11, 0, 24);
    let (grad, loss, correct) = batch_gradient(&layout, &p, &rope, &task, &batch);
    let pt = layout.transpose(&p);
    let net = Net::new(&layout, &p, &pt, &rope);
    let weight = 1.0f32 / (batch.len() * task.len) as f32;
    let mut sum = vec![0.0f32; layout.total];
    let (mut l, mut k) = (0.0, 0);
    for s in &batch {
        let (input, targets) = task.example(s);
        let mut g = vec![0.0f32; layout.total];
        let (a, b) = net.loss_and_grad(&input, &targets, weight, &mut g);
        for (x, y) in sum.iter_mut().zip(&g) {
            *x += *y;
        }
        l += a;
        k += b;
    }
    assert!(grad.iter().zip(&sum).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!((loss, correct), (l, k));
}

#[test]
fn committed_weights_match_the_pinned_sha256() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/results/r4-trained-weights.bin");
    let bytes = std::fs::read(path).expect("results/r4-trained-weights.bin missing: run the experiment command");
    assert_eq!(fp_play_runtime::sha256::hex(&bytes), TRAINED_WEIGHTS_SHA256);
    let w = load_trained(&bytes).unwrap();
    assert_eq!(w.embed.rows, DecoderConfig::TRAINED.vocab);
    let mut corrupt = bytes.clone();
    corrupt[17] ^= 1;
    assert!(load_trained(&corrupt).is_err());
}

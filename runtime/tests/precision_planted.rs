//! R4 planted bugs (compiled only with `--features planted-bugs`). Each must
//! be caught by the check named in its test (the same check as in
//! `tests/precision.rs`, run on the planted policy).
#![cfg(feature = "planted-bugs")]

use fp_play_runtime::decoder::config::DecoderConfig;
use fp_play_runtime::decoder::decode::prompts;
use fp_play_runtime::format::Format;
use fp_play_runtime::ops::{from_f64, fused_chain_dot};
use fp_play_runtime::precision::mac::encode;
use fp_play_runtime::precision::model::{site_product, MasterWeights, PolicyModel};
use fp_play_runtime::precision::policy::{Level, Policy, PrecisionMutation, Site};
use fp_play_runtime::rng::SplitMix64;

/// Caught by the per-level exact-chain check: the BF16-accumulating site
/// no longer equals the exact BF16 fused chain.
#[test]
fn bf16_accumulating_in_fp32_is_caught_by_the_exact_chain_check() {
    let c = DecoderConfig::REFERENCE;
    let w = MasterWeights::generate(&c);
    let m = &w.layers[0].w_up;
    let mut rng = SplitMix64::new(0x51e);
    let x: Vec<f32> = (0..m.cols).map(|_| rng.next_normalish() as f32).collect();
    let level = Level::Bf16Acc;
    let mut policy = Policy::uniform(c.layers, level);
    policy.planted = Some(PrecisionMutation::Bf16AccumulatesInFp32);
    let got = site_product(&policy, m, level, &x);
    let xs: Vec<u32> = x.iter().map(|&v| from_f64(Format::Bf16, f64::from(v))).collect();
    let mismatches = got
        .iter()
        .enumerate()
        .filter(|&(o, &g)| {
            let row: Vec<u32> = m.w[o * m.cols..(o + 1) * m.cols]
                .iter()
                .map(|&v| from_f64(Format::Bf16, f64::from(v)))
                .collect();
            encode(Format::Bf16, f64::from(g)) != fused_chain_dot(Format::Bf16, Format::Bf16, &row, &xs)
        })
        .count();
    eprintln!(
        "Bf16AccumulatesInFp32: {mismatches} of {} outputs differ from the exact BF16 chain",
        got.len()
    );
    assert!(mismatches > 0, "not caught");
}

/// Caught by the every-site-affects-the-logits check: lowering the KV cache
/// no longer changes anything.
#[test]
fn skipped_kv_rounding_is_caught_by_the_site_check() {
    let c = DecoderConfig::TINY;
    let w = MasterWeights::generate(&c);
    let p = &prompts(0x51735, 1, 12, c.vocab)[0];
    let base = PolicyModel::new(c, &w, Policy::uniform(c.layers, Level::Fp32)).teacher_forced(p);
    let caught = Level::LOW.iter().all(|&level| {
        let mut policy = Policy::only(c.layers, Site::KvCache, level);
        policy.planted = Some(PrecisionMutation::SkipKvRounding);
        PolicyModel::new(c, &w, policy).teacher_forced(p) == base
    });
    eprintln!("SkipKvRounding: KV-cache levels leave the logits unchanged = {caught}");
    assert!(caught, "not caught");
}

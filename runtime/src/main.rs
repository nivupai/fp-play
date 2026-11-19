use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use fp_play_runtime::collectives::{all_reduce_with, Algorithm, WirePolicy};
use fp_play_runtime::cost::{estimate_recorded, LinkModel};
use fp_play_runtime::format::{to_f64, Format};
use fp_play_runtime::ops::from_f64;
use fp_play_runtime::policies;
use fp_play_runtime::report;
use fp_play_runtime::rng::SplitMix64;
use fp_play_runtime::shard_experiment;
use fp_play_runtime::shard_tree::DEFAULT_SHARDS;
use fp_play_runtime::sweep::{self, SweepConfig};
use fp_play_runtime::tensor_parallel::{prepare, run_prepared, ExperimentConfig};

const USAGE: &str = "\
fp-play-runtime — exact simulation of AllReduce numerics

USAGE:
  fp-play-runtime experiment [--tokens N] [--hidden N] [--outputs N]
                             [--world 2,3,4,8] [--wire fp32,bf16] [--seed N]
                             [--latency-us X] [--bandwidth-gbps X] [--shards S]
                             [--out DIR]
      Run the tensor-parallel experiments; write tp-allreduce.{json,md}
      (R1), and r1.1-policies, r1.1-shard-tree (S logical shards, default
      64) and r1.1-sweep (.json and .md each, R1.1) to DIR (default: results).
      The sweep uses its own fixed grid of seeds, sizes and world sizes.
      Also writes r2-serving (R2: the serving scenarios), r3-decode (R3:
      the tensor-parallel decoder), r4-precision (R4: per-layer precision
      policies), r5-dataflow (R5: a layer on a toy accelerator) and
      r6-collectives (R6: data for the app's #collectives workspace) and
      r4-training (the copy-task model's training run, plus its weights
      r4-trained-weights.bin), each with its own fixed configuration; the
      flags above do not apply to them.

  fp-play-runtime schedule --algorithm NAME --world N [--format bf16 | --policy NAME]
                           [--len N] [--seed N]
      Print one AllReduce schedule round by round with its result and cost.
      Algorithms: sequential, ring, binomial-tree, halving-doubling, exact-gather.
      Policies: fp16, bf16, fp32 (uniform), fp32-rs/bf16-ag, bf16-wire/fp32-add.

  fp-play-runtime train [--steps N] [--batch N] [--lr X]
      Train the copy-task model (R4 completion) and print its loss curve,
      held-out accuracy and the weights' SHA-256. Writes nothing; the
      recorded run (default settings) is part of `experiment`.

  fp-play-runtime traces [--out DIR]
      Write the golden serving step traces (JSON lines) that
      tests/serving_traces.rs pins, to DIR (default: tests/traces).
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let records = matches!(args.first().map(String::as_str), Some("experiment" | "traces"));
    if records && fp_play_runtime::PLANTED_BUGS {
        eprintln!(
            "error: this binary was built with the `planted-bugs` feature; it will not write recorded results or traces"
        );
        return ExitCode::FAILURE;
    }
    let result = match args.first().map(String::as_str) {
        Some("experiment") => experiment(&args[1..]),
        Some("schedule") => schedule(&args[1..]),
        Some("traces") => traces(&args[1..]),
        Some("train") => train(&args[1..]),
        Some("-h" | "--help") | None => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn train(args: &[String]) -> Result<(), String> {
    let flags = Flags(args);
    flags.check_known(&["--steps", "--batch", "--lr"])?;
    let d = fp_play_runtime::train::TrainConfig::default();

    let c = fp_play_runtime::train::TrainConfig {
        steps: flags.parse("--steps", d.steps)?,
        batch: flags.parse("--batch", d.batch)?,
        peak_lr: flags.parse("--lr", d.peak_lr)?,
        ..d
    };
    let start = std::time::Instant::now();
    let r = fp_play_runtime::train::train(&c);
    for s in r.steps.iter().filter(|s| s.step % 25 == 0 || s.step + 1 == c.steps) {
        println!(
            "step {:4}  lr {:.2e}  loss {:.4}  accuracy {:.3}  grad norm {:.3}",
            s.step, s.lr, s.loss, s.accuracy, s.grad_norm
        );
    }
    println!(
        "held-out loss {:.5}  accuracy {:.5}  parameters {}  sha256 {}  ({:.1} s)",
        r.eval_loss,
        r.eval_accuracy,
        r.parameters,
        r.sha256,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

struct Flags<'a>(&'a [String]);

impl Flags<'_> {
    fn get(&self, name: &str) -> Result<Option<&str>, String> {
        let mut i = 0;
        while i < self.0.len() {
            if self.0[i] == name {
                return self
                    .0
                    .get(i + 1)
                    .map(|v| Some(v.as_str()))
                    .ok_or(format!("{name} needs a value"));
            }
            i += 1;
        }
        Ok(None)
    }

    fn check_known(&self, known: &[&str]) -> Result<(), String> {
        for (i, arg) in self.0.iter().enumerate() {
            if arg.starts_with("--") && !known.contains(&arg.as_str()) {
                return Err(format!("unknown flag `{arg}`"));
            }
            if !arg.starts_with("--") && (i == 0 || !self.0[i - 1].starts_with("--")) {
                return Err(format!("unexpected argument `{arg}`"));
            }
        }
        Ok(())
    }

    fn parse<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name)? {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("invalid value `{v}` for {name}")),
        }
    }

    fn seed(&self, default: u64) -> Result<u64, String> {
        match self.get("--seed")? {
            None => Ok(default),
            Some(v) => {
                let parsed = match v.strip_prefix("0x") {
                    Some(hex) => u64::from_str_radix(hex, 16),
                    None => v.parse(),
                };
                parsed.map_err(|_| format!("invalid seed `{v}`"))
            }
        }
    }
}

fn experiment(args: &[String]) -> Result<(), String> {
    let flags = Flags(args);
    flags.check_known(&[
        "--tokens",
        "--hidden",
        "--outputs",
        "--world",
        "--wire",
        "--seed",
        "--latency-us",
        "--bandwidth-gbps",
        "--shards",
        "--out",
    ])?;
    let defaults = ExperimentConfig::default();
    let world_sizes = match flags.get("--world")? {
        None => defaults.world_sizes.clone(),
        Some(v) => v
            .split(',')
            .map(|s| s.trim().parse().map_err(|_| format!("invalid world size `{s}`")))
            .collect::<Result<_, _>>()?,
    };
    let wire_formats = match flags.get("--wire")? {
        None => defaults.wire_formats.clone(),
        Some(v) => v
            .split(',')
            .map(|s| Format::parse(s).ok_or(format!("unknown format `{s}`")))
            .collect::<Result<_, _>>()?,
    };
    let config = ExperimentConfig {
        tokens: flags.parse("--tokens", defaults.tokens)?,
        hidden: flags.parse("--hidden", defaults.hidden)?,
        outputs: flags.parse("--outputs", defaults.outputs)?,
        world_sizes,
        wire_formats,
        seed: flags.seed(defaults.seed)?,
        link: LinkModel {
            latency_us: flags.parse("--latency-us", defaults.link.latency_us)?,
            bandwidth_gbps: flags.parse("--bandwidth-gbps", defaults.link.bandwidth_gbps)?,
        },
    };
    let out = PathBuf::from(flags.get("--out")?.unwrap_or("results"));
    let command = std::iter::once("fp-play-runtime experiment".to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ");
    let prepared = prepare(&config)?;
    let result = run_prepared(&config, &prepared)?;
    let policies = policies::run_prepared(&config, &prepared)?;
    let shards = shard_experiment::run_prepared(&config, &prepared, flags.parse("--shards", DEFAULT_SHARDS)?)?;
    let sweep = sweep::run(&SweepConfig {
        shards: flags.parse("--shards", DEFAULT_SHARDS)?,
        ..SweepConfig::default()
    })?;
    let serving = fp_play_runtime::serving::experiment::run_all(&Default::default())?;
    let decode = fp_play_runtime::decoder::experiment::run(&Default::default())?;
    let precision = fp_play_runtime::precision::experiment::run(&Default::default())?;
    let dataflow = fp_play_runtime::dataflow::experiment::run(&Default::default());
    let training = fp_play_runtime::train::report::run(&Default::default());
    fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let artifacts = [
        (
            "tp-allreduce",
            report::to_json(&result, &command).to_pretty(),
            report::to_markdown(&result, &command),
        ),
        (
            "r1.1-policies",
            report::policies_to_json(&policies, &command).to_pretty(),
            report::policies_to_markdown(&policies, &command),
        ),
        (
            "r1.1-shard-tree",
            report::shard_tree_to_json(&shards, &command).to_pretty(),
            report::shard_tree_to_markdown(&shards, &command),
        ),
        (
            "r1.1-sweep",
            report::sweep_to_json(&sweep, &command).to_pretty(),
            report::sweep_to_markdown(&sweep, &command),
        ),
        (
            "r2-serving",
            fp_play_runtime::serving::report::to_json(&serving, &command).to_pretty(),
            fp_play_runtime::serving::report::to_markdown(&serving, &command),
        ),
        (
            "r3-decode",
            fp_play_runtime::decoder::report::to_json(&decode, &command).to_pretty(),
            fp_play_runtime::decoder::report::to_markdown(&decode, &command),
        ),
        (
            "r4-precision",
            fp_play_runtime::precision::report::to_json(&precision, &command).to_pretty(),
            fp_play_runtime::precision::report::to_markdown(&precision, &command),
        ),
        (
            "r6-collectives",
            fp_play_runtime::ui_export::to_json(&result, &command)?.to_pretty(),
            fp_play_runtime::ui_export::to_markdown(&command),
        ),
        (
            "r5-dataflow",
            fp_play_runtime::dataflow::report::to_json(&dataflow, &command).to_pretty(),
            fp_play_runtime::dataflow::report::to_markdown(&dataflow, &command),
        ),
        (
            "r4-training",
            fp_play_runtime::train::report::to_json(&training, &command).to_pretty(),
            fp_play_runtime::train::report::to_markdown(&training, &command),
        ),
    ];
    for (name, json, markdown) in artifacts {
        let json_path = out.join(format!("{name}.json"));
        let md_path = out.join(format!("{name}.md"));
        fs::write(&json_path, json).map_err(|e| e.to_string())?;
        fs::write(&md_path, &markdown).map_err(|e| e.to_string())?;
        print!("{markdown}");
        eprintln!("wrote {} and {}", json_path.display(), md_path.display());
    }
    let weights_path = out.join("r4-trained-weights.bin");
    fs::write(
        &weights_path,
        fp_play_runtime::train::to_bytes(&training.result.weights),
    )
    .map_err(|e| e.to_string())?;
    eprintln!("wrote {} (SHA-256 {})", weights_path.display(), training.result.sha256);
    if training.result.sha256 != fp_play_runtime::train::TRAINED_WEIGHTS_SHA256 {
        eprintln!("note: the trained weights differ from the pinned TRAINED_WEIGHTS_SHA256; update it if the change is intended");
    }
    Ok(())
}

fn traces(args: &[String]) -> Result<(), String> {
    let flags = Flags(args);
    flags.check_known(&["--out"])?;
    let out = PathBuf::from(flags.get("--out")?.unwrap_or("tests/traces"));
    fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    for (name, contents) in fp_play_runtime::serving::traces::golden()? {
        let path = out.join(name);
        fs::write(&path, contents).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn schedule(args: &[String]) -> Result<(), String> {
    let flags = Flags(args);
    flags.check_known(&["--algorithm", "--world", "--format", "--policy", "--len", "--seed"])?;
    let name = flags.get("--algorithm")?.ok_or("--algorithm is required")?;
    let algorithm = Algorithm::parse(name).ok_or(format!("unknown algorithm `{name}`"))?;
    let world: usize = flags.parse("--world", 4)?;
    let policy = match (flags.get("--format")?, flags.get("--policy")?) {
        (Some(_), Some(_)) => return Err("give --format or --policy, not both".into()),
        (None, None) => WirePolicy::uniform(Format::Bf16),
        (Some(v), None) => WirePolicy::uniform(Format::parse(v).ok_or(format!("unknown format `{v}`"))?),
        (None, Some(v)) => WirePolicy::parse(v).ok_or(format!("unknown wire policy `{v}`"))?,
    };
    let (input, output) = (policy.accumulator, policy.output);
    let len: usize = flags.parse("--len", 8)?;
    let mut rng = SplitMix64::new(flags.seed(1)?);
    let inputs: Vec<Vec<u32>> = (0..world)
        .map(|_| (0..len).map(|_| from_f64(input, rng.next_normalish())).collect())
        .collect();
    let outcome = all_reduce_with(algorithm, policy, &inputs).map_err(|e| e.to_string())?;
    println!("{algorithm}: {}", algorithm.description());
    if policy.is_uniform() {
        println!("world {world}, {len} {} elements per rank\n", input.name());
    } else {
        println!(
            "world {world}, {len} {} elements per rank, wire policy {} (reduce phases {}, gather phases {}, output {})\n",
            input.name(),
            policy.name(),
            policy.reduce_wire.name(),
            policy.gather_wire.name(),
            output.name()
        );
    }
    for (i, round) in outcome.rounds.iter().enumerate() {
        let transfers: Vec<String> = round
            .transfers
            .iter()
            .map(|t| format!("{}→{} ({})", t.src, t.dst, t.elements))
            .collect();
        println!("round {:>2} [{}] {}", i + 1, round.phase, transfers.join("  "));
    }
    let shown = |raw: u32| format!("{} ({raw:#06x})", to_f64(output, raw));
    println!(
        "\nrank 0 result: {}",
        outcome.results[0]
            .iter()
            .map(|&r| shown(r))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("ranks agree: {}", outcome.consistent());
    let cost = estimate_recorded(&outcome.rounds, world, LinkModel::default());
    println!(
        "{} rounds, est. {:.3} µs under the default link model",
        cost.rounds, cost.estimated_us
    );
    Ok(())
}

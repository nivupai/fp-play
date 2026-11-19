//! The recorded R5 experiment (results/r5-dataflow.*): one 8B-class layer,
//! prefill and decode shapes, on two variants of the illustrative machine.

use super::graph::{layer, Phase};
use super::machine::{Machine, RECORDED_FP16_FMA_DELAY_NS};
use super::sim::{time_layer, LayerTiming};
use crate::serving::model::ModelConfig;

#[derive(Clone, Debug)]
pub struct DataflowConfig {
    pub model: ModelConfig,
    pub machines: Vec<Machine>,
    pub phases: Vec<Phase>,
    /// Phases broken down op by op in the report.
    pub detail: Vec<Phase>,
}

impl Default for DataflowConfig {
    fn default() -> Self {
        let mut phases: Vec<Phase> = [128, 512, 2048].iter().map(|&t| Phase::Prefill { tokens: t }).collect();
        phases.extend([1, 8, 32, 128].iter().map(|&b| Phase::Decode {
            batch: b,
            context: 1024,
        }));
        DataflowConfig {
            model: ModelConfig::EIGHT_B_CLASS,
            machines: vec![
                Machine::illustrative("1 ns cycle (illustrative)", 1.0),
                Machine::illustrative(
                    "4.179512 ns cycle (assumed: recorded FP16 FMA path delay)",
                    RECORDED_FP16_FMA_DELAY_NS,
                ),
            ],
            phases,
            detail: vec![
                Phase::Prefill { tokens: 512 },
                Phase::Decode {
                    batch: 8,
                    context: 1024,
                },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Run {
    pub machine: usize,
    pub phase: Phase,
    pub timing: LayerTiming,
}

/// Where a layer's cycles went (they sum to the layer's cycles).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Breakdown {
    /// Array cycles doing useful multiply–accumulates.
    pub useful: f64,
    /// Array cycles on padding rows or masked pairs.
    pub waste: f64,
    pub vector: f64,
    /// Idle compute waiting for the DMA, after an op's first chunk.
    pub memory_stall: f64,
    /// Each op's exposed first load (ops run in sequence).
    pub fill: f64,
}

impl Run {
    pub fn breakdown(&self, m: &Machine) -> Breakdown {
        let peak = m.peak_macs() as f64;
        let mut b = Breakdown {
            useful: 0.0,
            waste: 0.0,
            vector: 0.0,
            memory_stall: 0.0,
            fill: 0.0,
        };
        for o in &self.timing.ops {
            if o.on_array {
                b.useful += o.useful_macs as f64 / peak;
                b.waste += (o.issued_macs - o.useful_macs) as f64 / peak;
            } else {
                b.vector += o.timeline.busy as f64;
            }
            b.memory_stall += o.timeline.memory_stall as f64;
            b.fill += o.timeline.fill as f64;
        }
        b
    }

    /// Useful multiply–accumulates per cycle over peak.
    pub fn utilization(&self, m: &Machine) -> f64 {
        self.timing.useful_macs() as f64 / (m.peak_macs() as f64 * self.timing.cycles() as f64)
    }

    /// The roofline's lower bound in cycles: max(useful MACs ÷ peak, HBM bytes ÷ bytes per cycle).
    pub fn roofline_cycles(&self, m: &Machine) -> u64 {
        self.timing
            .useful_macs()
            .div_ceil(m.peak_macs())
            .max(self.timing.hbm_bytes().div_ceil(m.hbm_bytes_per_cycle))
    }

    /// FLOPs (2 per useful multiply–accumulate) per HBM byte.
    pub fn intensity(&self) -> f64 {
        2.0 * self.timing.useful_macs() as f64 / self.timing.hbm_bytes() as f64
    }
}

#[derive(Clone, Debug)]
pub struct DataflowResult {
    pub config: DataflowConfig,
    pub runs: Vec<Run>,
}

impl DataflowResult {
    pub fn run(&self, machine: usize, phase: Phase) -> Option<&Run> {
        self.runs.iter().find(|r| r.machine == machine && r.phase == phase)
    }
}

pub fn run(config: &DataflowConfig) -> DataflowResult {
    let mut runs = Vec::new();
    for (i, m) in config.machines.iter().enumerate() {
        for &phase in &config.phases {
            runs.push(Run {
                machine: i,
                phase,
                timing: time_layer(&layer(&config.model, phase), m),
            });
        }
    }
    DataflowResult {
        config: config.clone(),
        runs,
    }
}

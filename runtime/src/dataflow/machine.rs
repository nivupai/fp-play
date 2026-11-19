//! A toy fixed-function accelerator. Every constant is illustrative, not
//! a measured or named device.

/// The recorded ideal-interconnect input-to-output delay of FP Play's
/// baseline FP16 fused multiply–add (Nangate45 typical, OpenSTA, from the
/// M12 hardware-analysis artifact via `src/lib/showcase/hardware-summary.json`,
/// record `baseline-fma`), in nanoseconds. Used only as an **assumed**
/// cycle time for one machine variant: it is a combinational path delay
/// under stated IO constraints, not an achievable clock period, and no
/// frequency, pipelining or layout result is implied.
pub const RECORDED_FP16_FMA_DELAY_NS: f64 = 4.179512;

#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub name: String,
    /// Multiply–accumulate array: `rows × cols` MACs per cycle.
    pub rows: u64,
    pub cols: u64,
    /// Vector unit: element operations per cycle.
    pub lanes: u64,
    pub sram_bytes: u64,
    /// HBM bandwidth in bytes per cycle (rounded down).
    pub hbm_bytes_per_cycle: u64,
    /// Assumed cycle time, nanoseconds (only to convert cycles to time).
    pub cycle_ns: f64,
    #[cfg(feature = "planted-bugs")]
    #[doc(hidden)]
    pub planted: Option<super::DataflowMutation>,
}

impl Machine {
    /// A 128 × 128 BF16 array, 256 vector lanes, 16 MiB SRAM, 1.5 TB/s HBM,
    /// with the given cycle time.
    pub fn illustrative(name: &str, cycle_ns: f64) -> Machine {
        let hbm_bytes_per_second = 1.5e12;
        Machine {
            name: name.into(),
            rows: 128,
            cols: 128,
            lanes: 256,
            sram_bytes: 16 << 20,
            hbm_bytes_per_cycle: (hbm_bytes_per_second * cycle_ns * 1e-9).floor() as u64,
            cycle_ns,
            #[cfg(feature = "planted-bugs")]
            planted: None,
        }
    }

    /// An activation of `bytes` stays in SRAM if it fits in half of it.
    pub fn resident(&self, bytes: u64) -> bool {
        bytes <= self.sram_bytes / 2
    }

    /// Peak multiply–accumulates per cycle.
    pub fn peak_macs(&self) -> u64 {
        self.rows * self.cols
    }
}

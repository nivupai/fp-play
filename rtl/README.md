# Synthesizable FP16 add/subtract unit

Milestone 10 adds an original, combinational SystemVerilog binary16 add/subtract datapath and a separate Verilator simulation workflow. It uses public numerical semantics and independently developed logic. No employer designs, proprietary logic structures, or internal documentation are used. The existing software arithmetic and all 424 historical tests remain unchanged.

Milestone 11 adds [multiplication, true FMA, nonfused comparison, and generic synthesis checks](MULTIPLY_FMA.md). The sections below retain the Milestone 10 adder contract and validation record.

Milestone 12 adds [controlled synthesis, Nangate45 mapping, OpenSTA timing, and design-space analysis](../docs/HARDWARE_SYNTHESIS.md) with immutable baseline RTL.

## Interface and scope

Top module: `rtl/src/fp16_addsub.sv`, `fp16_addsub`.

| Port                                          | Width   | Meaning                                                        |
| --------------------------------------------- | ------- | -------------------------------------------------------------- |
| `a`, `b`                                      | 16 each | Raw binary16 operands, including signed zeros and NaN metadata |
| `subtract`                                    | 1       | 0: A+B; 1: A−B                                                 |
| `result`                                      | 16      | Raw nearest-even result                                        |
| `invalid`, `overflow`, `underflow`, `inexact` | 1 each  | Per-operation indicators, no sticky global state               |

There is no clock, reset, handshake, pipeline register, or valid signal. Outputs settle combinationally. No clock-cycle latency, throughput, delay, area, power, or device timing claim is made. The design is intentionally wide and unoptimized. This adder module does not perform multiplication/FMA; those separate Milestone 11 modules are documented above. HLS, transformer hardware, and area/power optimization are not implemented.

Additional **observation output ports** expose the logical stages below. They are ordinary synthesizable signals, not simulator-only floating-point hooks; an integrating design may leave them unconnected. There are no real-number operations, data-dependent unbounded loops, delays, latches, DPI numerical shortcuts, or inferred memories in the datapath. Fixed 41-iteration loops express a bounded priority encoder and sticky reduction. Verilator `--Wall` lint is strict/fatal, with no RTL warning waivers. This is not a netlist, place-and-route, timing, or formal equivalence certification; Milestone 10 did not run synthesis; Milestone 11 additionally synthesizes this adder within the nonfused comparison top.

## Exact fixed-width architecture

Every finite binary16 value is an integer times **2^-24**. Normals have integer significand `1024 + fraction`, subnormals/zeros have integer `fraction`. Use effective biased exponent 1 for raw exponent zero. Place each 11-bit significand at shift `effectiveExponent − 1` (0…29) in a **40-bit unsigned alignment vector**. The maximum input `2047 << 29` fits 40 bits; their sum fits **41 bits**.

This lossless global lattice is a deliberate alternative to a narrower right-shift/jam aligner. It preserves all cancellation information without needing an approximation of a discarded tail. Sticky is generated only at final rounding. The wider circuit makes software-to-hardware reasoning straightforward; it is not an optimization or a claim to reproduce any commercial datapath.

1. **Unpack/classify:** sign, raw exponent, fraction; distinguish zero, subnormal, normal, infinity, signaling/quiet NaN.
2. **Effective signs and compare:** B's arithmetic sign is `b.sign XOR subtract`; compare effective exponents and form their absolute difference.
3. **Align:** losslessly place both significands on the fixed 2^-24 grid. Specials bypass ordinary arithmetic.
4. **Arithmetic:** equal effective signs add widened magnitudes; opposing signs subtract the smaller magnitude from the larger. Select the correct result sign, including zero rules.
5. **Leading-zero normalization:** a bounded priority encoder finds the leading 1. A left-normalized 41-bit observation places that bit at position 40. For a nonzero leading index L, raw normal exponent is L−9 when L≥10; otherwise the result is subnormal. Discard `max(L−10,0)` low grid bits to retain eleven significant bits.
6. **GRS:** G is the first discarded bit, R the next, S is the OR of all remaining low bits. Retain every bit until this point. Exact subnormals discard nothing.
7. **Nearest-even:** increment iff `G && (R || S || retained[0])`. An exact tie keeps an even retained integer. A carry renormalizes the significand and advances the exponent.
8. **Pack:** encode signed zero, subnormal, normal, infinity, or NaN, and the four indicators. Finite exponent overflow returns signed infinity with overflow **and** inexact.

| Observation port                       | Width   | Interpretation                                                           |
| -------------------------------------- | ------- | ------------------------------------------------------------------------ |
| `effective_b_sign`                     | 1       | Effective arithmetic sign, not propagated NaN sign                       |
| `exponent_difference`                  | 6       | Absolute difference of effective biased exponents                        |
| `aligned_a`, `aligned_b`               | 40 each | Exact nonnegative operand magnitudes in units of 2^-24                   |
| `arithmetic_magnitude`                 | 41      | Exact sum/difference magnitude on that grid                              |
| `normalization_shift`                  | 6       | Left shift to put the leading 1 at bit 40; zero has shift 0              |
| `normalized_magnitude`                 | 41      | Left-normalized debug magnitude, zero if exact zero                      |
| `discard_count`                        | 6       | Number of low fixed-grid bits outside retained precision                 |
| `retained_significand`                 | 12      | Before rounding; valid finite values use at most eleven significant bits |
| `guard_bit`, `round_bit`, `sticky_bit` | 1 each  | Final rounding evidence                                                  |
| `increment`                            | 1       | Actual nearest-even increment decision                                   |
| `exponent_rounded`                     | 5       | Final biased exponent, 31 for finite overflow; special-path default 0    |
| `special_case`                         | 1       | Ordinary finite stages bypassed                                          |

Special paths give defined zero defaults to finite-stage observations; the exponent comparison and effective B sign still describe decoded raw operands. These zeros must not be portrayed as executed normalization stages.

The TypeScript reference aligns to a per-operation minimum exponent, whereas RTL always uses 2^-24. Consequently alignment vectors, normalization shifts, and discard counts need not have identical coordinates. The exact value, retained significand, GRS, rounding increment, final bits, and flags agree. The verifier checks exact magnitude by rescaling the reference's authoritative BigInt coefficient/exponent, not its Number projection.

## IEEE-style policies

- **Rounding:** round-to-nearest, ties-to-even only; normal/subnormal carries and overflow midpoint included. Subnormals are never flushed to zero.
- **Zeros:** same effective negative signs yield −0; exact opposite-sign cancellation yields +0. Subtraction flips B's effective arithmetic sign, including its zero sign.
- **NaNs:** first signaling, otherwise first quiet, original A/B order. Set quiet bit 9; retain original sign and low nine payload bits. B's propagated NaN sign is never flipped for subtraction. Any signaling input raises invalid. No difference from FP Play's add/sub policy is intended or observed.
- **Infinities:** opposing effective infinities return canonical positive quiet `0x7e00` with invalid; otherwise propagate the effective infinity without finite overflow/inexact flags.
- **Overflow/inexact:** overflow always implies inexact, even for an exact unbounded-exponent sum. Other finite inexactness comes from discarded nonzero GRS evidence.
- **Underflow:** output is always clear for this add/sub operation set. This is a mathematical property, not a missing subnormal implementation: every exact result is an integer multiple of 2^-24, and every nonzero result below 2^-14 fits the ten stored fraction bits exactly. Thus a tiny add/sub result cannot be inexact. The shared software policy requires tininess after precision-only rounding **and** inexactness. Conversion/multiplication can underflow, but they are not RTL operations here.

No traps, alternate rounding modes, sticky exception state, flush-to-zero, multiplication, FMA, or full IEEE 754 conformance claim are provided.

## Tools and commands

Local validation used **Verilator 5.052**, Apple clang/C++17, make, Node **22.9.0**, and the existing locked TypeScript dependency. The frontend and normal Vitest suite do not require Verilator. Install the tool separately, for example `brew install verilator` or the distribution's `verilator` package. See the [official installation guide](https://verilator.org/guide/latest/install.html). Verilator 5.x APIs are used; only 5.052 was executed locally, so other tool releases are not claimed tested.

From the repository root, after `npm ci`:

```sh
make -C rtl lint       # Strict SystemVerilog lint, including latch diagnostics
make -C rtl build      # Lint + C++ simulator, trace-enabled
make -C rtl test       # Complete fixed-seed 462,172-case comparison
make -C rtl waveform   # 28 directed cases + rtl/build/directed.vcd
# Override the executable if needed; Make exports this for accurate provenance.
make -C rtl VERILATOR=/path/to/verilator test
```

`rtl/build/` contains only generated local output and is ignored by Git, ESLint, and Prettier. The simulator protocol is five hex fields per vector: `a b subtract expectedBits expectedFlagMask`. Mask bits are **invalid=8, overflow=4, underflow=2, inexact=1**, distinct from SoftFloat's invalid=16. The testbench compares raw bits and all four indicators; it does not calculate its own floating-point expectations. Its observation file adds the documented internal signals.

The Node harness uses the unchanged `calculateFP16` TypeScript API via existing development-only transpilation. It keeps compact expected bits/flags, recomputing reference traces while comparing to avoid retaining hundreds of thousands of BigInt trace objects. It validates transaction correlation, record counts, rounding signals, and exact magnitude in addition to simulator comparisons. A failed comparison exits nonzero and writes `rtl/build/failures.json`, `failures.txt`, and bounded `first-failures.txt` plus `failures.vcd` for up to eight failed transactions. The first mismatch and reproducible raw patterns are printed. Directed-only runs use `directed-` prefixed vector/observation/summary/failure files, preserving the full corpus outputs. `--waveform` requires `--directed` for bounded capture; explicit simulator replay can trace a chosen file. Missing tools, empty inputs, malformed records, and protocol/count failures also fail the workflow.

C++ formatting uses the local `.clang-format` file (checked with clang-format 12.0.1); it is optional development tooling, not a frontend dependency. Generated Verilator/runtime headers are not vendored. The C++ build uses Verilator's own warning configuration rather than re-enabling unrelated runtime-header warnings; RTL `--Wall` remains unchanged and fatal.

## Verification corpus and actual results

Seed: **`0x46503136`** (1179660598). There are **462,172 completed transactions**:

| Corpus                                              | Cases   | Coverage                                                                                                                                                                   |
| --------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Independently specified directed vectors            | 28      | Exact/opposite signs, carry, exponent gaps, even/odd ties, guard/round/sticky, cancellation, signed zeros, normal/subnormal transitions, finite overflow, infinities, NaNs |
| All 65,536 encodings in four families               | 262,144 | A+0, A−0, A−A, A+sign-flipped-A; includes every signed zero, infinity, quiet/signaling NaN and payload                                                                     |
| 100,000 deterministic random pairs, both operations | 200,000 | Full-width raw operand pairs, signs/classes/exponent gaps                                                                                                                  |

**Observed: zero mismatches** in raw results and every exception flag. This includes **20,268 NaN outputs with strict sign/payload equality** against FP Play. **441,888 finite paths** additionally match GRS, retained significands, increment decisions, and exact arithmetic magnitude. Not every possible operand pair was exhausted; no formal proof or TestFloat conformance claim is made.

### Direct RTL-to-SoftFloat evidence

The optional existing Berkeley SoftFloat driver supplies an additional independent reference on **the same RTL transactions**, not merely on similar software inputs. Use the previously documented upstream revision **a0c6494cdc11865811dec815d5c0049fba9d82a8**, 8086-SSE specialization, nearest-even, after-rounding tininess. No SoftFloat source/binary is bundled.

```sh
# Build the external driver per the arithmetic documentation, then:
make -C rtl test
node rtl/scripts/softfloat.mjs /path/to/softfloat-driver
# Directed waveform outputs have a separate directed- prefix.
```

The script remaps invalid to SoftFloat's mask bit and compares all four flags strictly. **462,172 comparisons passed, zero mismatches**. All **20,268 NaN outputs** match quiet classification/flags; **65** select a different implementation-specific sign/payload in SoftFloat. FP Play's own NaN policy is separately matched strictly by RTL. See [the software verification/build instructions](../src/lib/arithmetic/MULTIPLY_FMA.md#verification-and-reproducible-counterexamples).

## Waveform inspection and artifacts

`make -C rtl waveform` creates an ordinary VCD containing operands, operation, outputs, classification, and internal/debug signals. Open it in a VCD viewer, for example `gtkwave rtl/build/directed.vcd`, if installed separately. Waveform generation and signal contents were checked; no GUI waveform viewer session is claimed. The trace records a settled zero baseline followed by each settled transaction. VCD timestamps are **observation indices**, not a clock, propagation-delay measurement, or claimed hardware latency.

Useful directed records include even tie `3c00 + 1000` (GRS 100, no increment), odd tie `3c01 + 1000` (GRS 100, increment), and tiny `3c00 + 0001` (GRS 001, inexact). Trace a single vector manually using the built binary:

```sh
# Result=3c02, flags=1 (inexact). All fields are hexadecimal.
printf '3c01 1000 0 3c02 1\n' > /tmp/fp-play-rtl-vector.txt
rtl/build/obj/fp16_sim /tmp/fp-play-rtl-vector.txt /tmp/fp-play-rtl-observation.txt /tmp/fp-play-rtl.vcd
```

The browser imports only the small **28-sample** `rtl/vectors/simulation.json`, generated from actual simulator observations. To refresh it intentionally:

```sh
make -C rtl build
node rtl/scripts/verify.mjs --export rtl/vectors/simulation.json
npm run format
npm test
```

That export runs the complete corpus, never substitutes software outputs for RTL fields, and records tool version, seed, counts, strict comparison policy, and SHA-256 hashes for RTL, testbench, harness, directed vectors, and reference sources. Source/record changes require regeneration; normal tests reject stale hashes. Repeating export on the same tools/sources is deterministic. The complete corpus and VCD are not bundled into the application.

Arithmetic Explorer's optional **Inspect precomputed FP16 RTL simulations** panel selects recorded examples independently of its custom operand form. It recomputes the current software reference and compares recorded RTL bits/flags, explicitly showing disagreement if either differs. It displays raw/decoded results, actual internal signals, and conceptual stage labels, including special-path bypasses and a GRS-sensitive tie. It does **not** execute Verilator, synthesize, or run arbitrary RTL in the browser. Long binary signals remain keyboard-scrollable; native example selection and status text supplement visuals.

## CI and limitations

`.github/workflows/rtl.yml` adds a separate read-only Ubuntu 24.04 job: install distribution Verilator, log its version, use Node 22.9.0/npm ci, lint/build/run the full corpus, then capture directed waveforms. Existing frontend CI and manual-only Pages deployment remain unchanged. Distribution tool versions can change and are recorded per run; only the local version above was executed during this milestone. Workflow syntax was checked with actionlint; the new workflow has not been run on GitHub because no branch was pushed. Local summaries/failures remain build outputs; no artifact/release publication is configured.

The module is designed with synthesizable fixed-width constructs and passes strict RTL lint; Milestone 10 synthesis/netlist validation, FPGA/ASIC implementation, formal equivalence, X/Z four-state behavior, timing closure, and all operand-pair exhaustion remain unperformed. Verilator runs a two-state model. Milestone 10 did not include multiplication/FMA RTL; see the Milestone 11 guide above for the implemented extension and synthesis evidence. No HLS, transformer datapath, area/power optimization, or silicon measurements are claimed. Existing software/UI features are preserved. No browser automation is available; the new panel has DOM/keyboard tests but no rendered-layout, theme, touch, or screen-reader verification claim. Nothing is pushed, merged, deployed, or published.

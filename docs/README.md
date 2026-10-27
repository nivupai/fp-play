# FP Play documentation

Start with the [technical case study](CASE_STUDY.md) for the complete story, or
[reproduce the project](REPRODUCIBILITY.md) for commands. The application is an
interactive laboratory; these documents specify what its results mean.

| Goal                                                                | Read                                                                                                            |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Explore existing controls and examples                              | [User guide](USER_GUIDE.md)                                                                                     |
| Understand motivation, architecture and measured findings           | [Technical case study](CASE_STUDY.md)                                                                           |
| Run fast frontend checks or advanced hardware tools                 | [Reproducibility](REPRODUCIBILITY.md)                                                                           |
| Inspect FP16 encodings, adjacency and ULP semantics                 | [Numerical API](../src/lib/numerics/README.md), [microscope](../src/lib/microscope/README.md)                   |
| Check exact arithmetic, FMA, flags and independent reference policy | [Arithmetic](../src/lib/arithmetic/README.md), [multiply/FMA](../src/lib/arithmetic/MULTIPLY_FMA.md)            |
| Understand BF16/FP32 accumulation and attention boundaries          | [Mixed precision](MIXED_PRECISION.md), [attention](ATTENTION_PRECISION.md)                                      |
| Reproduce actual RTL observations                                   | [Add/sub RTL](../rtl/README.md), [multiply/FMA RTL](../rtl/MULTIPLY_FMA.md)                                     |
| Interpret measured area/delay/depth                                 | [Synthesis methodology](HARDWARE_SYNTHESIS.md)                                                                  |
| Distinguish software traces, wires and logical diagrams             | [Datapath explorer](HARDWARE_DATAPATH.md)                                                                       |
| Submit a focused improvement                                        | [Contributing](../CONTRIBUTING.md), [conventions](../AGENTS.md)                                                 |
| Review license boundaries                                           | [Third-party notices](THIRD_PARTY.md), [project MIT license](../LICENSE)                                        |
| Review release status and manual publication steps                  | [Release procedure](RELEASING.md), [verification record](RELEASE_VERIFICATION.md), [changelog](../CHANGELOG.md) |

Numerical error estimates, bounded workloads, host transcendental approximations,
implementation-specific NaN policy, finite verification corpora, ideal-interconnect
timing and incomplete assistive-technology review remain explicit limitations.
No document turns a functional simulation into physical chip evidence.

## Actual interface captures

[Light desktop landing](images/research-landing-desktop-light.png) ·
[Dark mobile landing](images/research-landing-mobile-dark.png) ·
[Hardware comparison](images/research-hardware-desktop.png) ·
[Dark mobile hardware](images/research-hardware-mobile-dark.png).
See the verification record for browser/viewport details and remaining limits.

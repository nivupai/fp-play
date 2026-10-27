# Licensing and third-party attribution

FP Play's original software and documentation use the [MIT License](../LICENSE).
The original copyright notice is unchanged. That license does **not** relicense
external packages, cell libraries, datasets or development tools. No employer
hardware designs or internal Broadcom information are used.

## Browser distribution

- React, React DOM and their Scheduler dependency: MIT, copyright Meta Platforms,
  Inc. and affiliates. Exact installed versions are pinned in `package-lock.json`;
  upstream: [facebook/react](https://github.com/facebook/react).
- Tailwind CSS build output: MIT, copyright Tailwind Labs, Inc.; upstream:
  [tailwindlabs/tailwindcss](https://github.com/tailwindlabs/tailwindcss).
- FP Play uses system fonts and original SVG/CSS diagrams. It adds no external
  stock images, icon library, hosted font, plotting library or browser RTL runtime.
  Committed screenshots capture this project's actual interface.

Complete MIT notices copied from the installed upstream license files accompany
the static app in [`public/THIRD_PARTY_NOTICES.txt`](../public/THIRD_PARTY_NOTICES.txt).
Vite copies that file into `dist/`; release checks verify it and both VCD downloads.
Preserve upstream legal comments in generated bundles. Review license changes
whenever the lockfile changes. Developer packages (Vite, TypeScript, Vitest,
ESLint, Prettier and their transitive tools) retain their own package notices;
inspect installed `LICENSE` files and `npm ls --all`, not this list alone, for a
complete development dependency inventory.

## External reference and hardware tools

These are optional separately installed tools; their source/binaries are not
vendored or included in the browser distribution.

| Item                        | Attribution and authoritative terms                                                                                                                                                                                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Berkeley SoftFloat          | John R. Hauser / Regents of the University of California; pinned revision `a0c6494cdc11865811dec815d5c0049fba9d82a8`. Retain its [COPYING notice](https://github.com/ucb-bar/berkeley-softfloat-3/blob/a0c6494cdc11865811dec815d5c0049fba9d82a8/COPYING.txt) when building the external reference. FP Play's C driver is original integration code.     |
| Verilator                   | [Upstream licensing](https://github.com/verilator/verilator/blob/master/LICENSE); installed separately. Simulation outputs are not a redistribution of its tool executable.                                                                                                                                                                             |
| Yosys / ABC / OSS CAD Suite | Distinct upstream projects and bundled components with their own notices: [Yosys COPYING](https://github.com/YosysHQ/yosys/blob/main/COPYING), [ABC copyright](https://github.com/berkeley-abc/abc/blob/master/copyright.txt), [suite sources](https://github.com/YosysHQ/oss-cad-suite-build). Pin/check the selected archive and retain its licenses. |
| OpenSTA                     | The pinned source includes GNU GPL version 3 text; see [its LICENSE](https://github.com/The-OpenROAD-Project/OpenSTA/blob/150bc0dbd103cbc97c0f85f63c4c8338826121fd/LICENSE) and source headers. It is a separately built analysis tool, not MIT-licensed FP Play code.                                                                                  |
| CUDD                        | Regents of the University of Colorado and contributors; its separate redistribution notice includes attribution, disclaimer and non-endorsement conditions. See [the pinned source license](https://github.com/ivmai/cudd/blob/f54f533303640afd5dbe47a05ebeabb3066f2a25/LICENSE).                                                                       |

## Nangate45 library boundary

The analysis downloads public Nangate45 typical from OpenROAD-flow-scripts
revision `64939f3717ad7c694199ba04cb98924b98022868`, verifies its SHA-256 and keeps
it under ignored `rtl/build/analysis/library/`. The accompanying
[platform LICENSE](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/64939f3717ad7c694199ba04cb98924b98022868/flow/platforms/nangate45/LICENSE)
contains Apache 2.0 text, while the Liberty file retains Nangate copyright,
trademark and license-agreement notices. Preserve and review both the platform
license and original file notices; FP Play does not replace them with MIT.

The [platform README](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/64939f3717ad7c694199ba04cb98924b98022868/flow/platforms/nangate45/README.md)
identifies the research library and its non-manufacturability. FP Play commits
measurement numbers, provenance and original scripts, not Liberty models or
cell implementations. Mapped area and ideal-interconnect timing neither establish
physical silicon results nor grant manufacturing rights. See
[the exact library hash and conditions](HARDWARE_SYNTHESIS.md#library-and-tools).

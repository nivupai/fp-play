---
name: Bug or numerical discrepancy
about: Report a reproducible UI, numerical, RTL or artifact problem
---

## Reproduction

Version/commit, workspace/hash, exact steps and minimal input. For arithmetic,
include raw operand hex, original decimal text, operation, precision policy and
expected/actual result bits and flags. Retain signed zero and NaN metadata.

## Expected behavior and evidence

Explain the expected result and its independent reference/source. For randomized
or RTL failures, include seed, tool versions, mismatch row and source hashes.
Do not omit flag differences. Attach only bounded logs; no private/proprietary data.

## Environment

Node/npm or Verilator/Yosys/ABC/OpenSTA versions as relevant. For UI issues include
browser/OS, viewport, theme, keyboard/touch steps and an actual screenshot if useful.

## Checks performed

List commands, counts and failures. State any unverified behavior explicitly.

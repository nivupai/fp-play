@AGENTS.md

## Claude Code

- Runtime work lives in `runtime/`. The roadmap is `runtime/PLAN.md`. Work one milestone at a time.
- Before any runtime commit, run from `runtime/`: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`.
- Regenerate `runtime/results/` only with: `cargo run --release -- experiment --out results`
- If numerics change, regenerate `tests/vectors/exact-sums.txt` with: `python3 -I scripts/gen_sum_vectors.py`
- Never push, deploy, tag or release unless I ask.

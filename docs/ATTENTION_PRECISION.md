# Transformer Attention Precision Lab

Milestone 9 models one small, unmasked scaled dot-product attention head:

**Attention(Q, K, V) = softmax(QKᵀ / √d_k) V**

This is a deterministic numerical experiment, not a trained transformer, GPU kernel, benchmark, or accelerator implementation. There are no masks, dropout, positional embeddings, multiple heads, learned projections, training, backpropagation, RTL, or silicon cost estimates. All earlier engines/APIs and 390 historical tests are preserved.

## Shapes, input, and supported policies

`calculateAttention({ q, k, v }, mode, method)` takes rectangular Number matrices:

- Q: **1–4 queries × 1–8 features**.
- K: **1–8 keys × the same feature count**.
- V: **the same key count × 1–4 output features**.

Invalid shapes, ragged/sparse arrays, non-Number entries, unknown policy names, and unsupported softmax methods are rejected. Explicit Number NaN/infinities are allowed to investigate propagation. The text editor parses decimal/scientific text into binary64 first; it is not exact decimal arithmetic. Rows use newline/semicolon, columns comma/whitespace. `-0`, `Infinity`, `-Infinity`, `NaN` are documented tokens. Out-of-range decimal text must use an explicit infinity token. Invalid drafts preserve all applied matrices/results.

Only these five frozen named policies are supported. There is no arbitrary combination builder.

| Policy ID      | Q/K/V input | QK products and accumulation          | Scaling | Softmax working format | Stored probabilities | PV products and accumulation | Final output |
| -------------- | ----------- | ------------------------------------- | ------- | ---------------------- | -------------------- | ---------------------------- | ------------ |
| `fp16`         | FP16        | Mode A: FP16 product, FP16 add        | FP16    | FP16                   | FP16                 | A                            | FP16         |
| `fp16-wide`    | FP16        | Mode C: exact product inside FP32 FMA | FP32    | FP32                   | FP16                 | C                            | FP16         |
| `bf16-wide`    | BF16        | Mode D: exact product inside FP32 FMA | FP32    | FP32                   | BF16                 | D                            | BF16         |
| `fp32`         | FP32        | Mode E: FP32 product, FP32 add        | FP32    | FP32                   | FP32                 | E                            | FP32         |
| `fp16-softmax` | FP16        | C                                     | FP32    | FP16                   | FP16                 | C                            | FP16         |

All dot products are **sequential with +0 seeding**. Pairwise attention reduction is not offered. Modes refer to the [Milestone 8 contract](MIXED_PRECISION.md); notably C/D do not round/range-limit their product before accumulation. FP16-product/FP32-accumulator mode B remains available in the original dot lab but is not offered in attention. The FP32 reference deliberately uses separate products and sums, not implicit FMA or backend-dependent matrix reduction. Widening cannot recover input quantization loss.

```ts
import { calculateAttention, attentionValues } from './src/lib/numerics'

const input = {
  q: [[2048, 1, -2048, 0]],
  k: [
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  v: [[1], [0]],
}
const narrow = calculateAttention(input, 'fp16', 'stable')
const wide = calculateAttention(input, 'fp16-wide', 'stable')
attentionValues(narrow.scores) // [[0, 0]]
attentionValues(wide.scores) // [[1, 0]]
// Scaling divides by sqrt(4); the changed scores affect probabilities/output.
```

## Rounding boundaries and numerical architecture

Pure numerical code lives in `src/lib/numerics/attention.ts`; the attention-only division helper is `attentionDivision.ts`. Presentation/parsing and the twelve presets are under `src/lib/attention/`. Components consume actual typed events; no approximate parallel animation computes results.

1. **Input:** directly quantize binary64 Q/K/V to the selected input format, retaining signed zero. Existing BF16 direct conversion avoids binary32 double rounding. Number NaNs canonicalize; this interface does not accept raw NaN payload entry.
2. **Scores:** execute the existing exact BigInt dot engine for every Q/K row pair. Each actual multiply/add/FMA and its flags appears in the event log.
3. **Scaling:** compute `1 / Math.sqrt(d_k)` using binary64 host functions, quantize this constant to the scaling format, then multiply each score with the existing exact modeled multiply and one output rounding. This **multiplies by a rounded reciprocal**; it is not correctly rounded format-specific square root or direct exact division by √d_k.
4. **Softmax input:** quantize scaled logits to the working format, including the explicitly reduced `fp16-softmax` policy.
5. **Stable shift:** find the row maximum among decoded working-format values, subtract it via modeled exact addition of its negation, and round once to working precision. Naive softmax skips this stage.
6. **Exp:** evaluate **Math.exp on the decoded binary64 argument**, then directly quantize its binary64 result to working precision. This is an implementation-dependent binary64 approximation followed by a specified conversion, **not a claim of correctly rounded FP16/BF16/FP32 exponentiation**.
7. **Denominator:** add exponentials left to right, starting +0, rounding each addition to the working format.
8. **Normalize:** divide each stored exponential by the stored denominator. Finite operands form an exact BigInt rational; choose destination quantum, divide integers, compare twice the remainder with the denominator, and apply nearest-even once. No binary64 division models reduced-precision normalization. The helper is attention-specific; there is no new general division workspace.
9. **Probability:** quantize normalized values to the named probability storage format. There is **no subsequent renormalization**; quantized probabilities can sum slightly above/below one or lose small tails entirely.
10. **Aggregate:** use stored probabilities and quantized V with the named existing dot policy, recording every operation. **Output:** quantize the accumulator to the final storage format.

`AttentionEvent` is typed by stage/operation and carries coordinates, actual operands/result patterns, binary64 source where applicable, flags, tie/increment decisions, and a boundary annotation. Trace stages are conceptual computations, not physical cycles or scheduling/performance estimates. Result matrices retain `{ format, bits }`; `attentionValues` gives exact decoded Numbers for these finite formats.

## Flags, nonfinites, and failures

The arithmetic inherits nearest-even, gradual underflow, first-signaling/otherwise-first-quiet NaN propagation, and precision-only after-rounding tininess from the existing engine. Attention additionally records **finite conversion flags** at explicit boundaries, including input/probability/output conversion; the earlier dot API's flag semantics remain unchanged. The event/result union includes these modeled arithmetic and conversion flags. It is not processor sticky state.

The quotient helper returns a canonical positive quiet NaN for NaN operands, 0/0, or infinity/infinity; 0/0 and infinity/infinity raise invalid. Signed zero/finite/infinity division follows the documented arithmetic signs. A finite nonzero divided by zero returns signed infinity and a separate `divideByZero` indicator; it is not inserted into the historical four-flag type. In softmax, a zero denominator normally comes from all-zero exponentials and therefore causes 0/0 invalid, which is shown along with a zero-denominator diagnostic. Raw signaling/payload metadata is not an attention input contract; quotient NaN propagation is intentionally simpler than the general add/multiply/FMA policy.

Math.exp flags are **not** modeled as IEEE transcendental exception flags. Its event flags describe conversion of the host result only. Binary64 exp overflow/underflow-to-zero is separately diagnosed when the argument was finite. Consequently an already-infinite host exp result need not raise a conversion overflow flag; a finite exp(12) converted to FP16 does raise conversion overflow.

Stable softmax bounds finite shifted logits above by zero, preventing positive-exp overflow. It cannot repair input overflow, nonfinite scores, or infinity minus infinity. All-−Infinity or a +Infinity maximum can produce NaNs, retained without a fallback distribution. Naive softmax intentionally exposes exp overflow or all-negative exp underflow to a zero denominator. If finite exponentials overflow **only their sum**, division by infinity can return all-zero probabilities; that failure is diagnosed and retained too. No clipping, sanitizing, epsilon denominator, uniform fallback, or probability renormalization is applied. NaNs/infinities propagate through weighted-value arithmetic, including invalid zero × infinity. A signed-zero input survives its input event, although sequential +0 seeding can produce a +0 final result.

## Baselines and error interpretation

The interface always provides **stable modeled FP32** using the `fp32` policy, regardless of the selected method on either compared side. It also provides a separate **binary64 numerical baseline** computed with Number products/sums, Math.sqrt and Math.exp. Neither is exact attention or a high-precision mathematical reference. Comparing a naive result against stable FP32 includes both algorithm and precision changes; choose FP32 on the selected side to isolate naive/stable behavior.

An offline **100-digit Python Decimal reference** uses the original parsed binary64 inputs, converted exactly with `Decimal.from_float`, and high-precision dot products, square root, stable exponential normalization, and weighted sums. This is a finite-precision mathematical reference, not an exact transcendental answer, interval proof, or certification of correctly rounded modeled exp. It is not shipped in the browser. The committed verification record includes its output strings.

Error heatmaps show absolute and relative decoded-value differences against stable FP32 using binary64 subtraction/division; they are numerical estimates, not exact rational error claims. Relative error is undefined at a zero reference. Nonfinite operands produce an explicit **Undefined**, even when both sides are the same infinity/NaN. A separate signed left-minus-right final-output map shows output differences. Input error, score rounding, reciprocal approximation, exp approximation/quantization, denominator rounding, stored probability loss, and PV accumulation all contribute; error need not grow monotonically.

## Interactive controls and heatmaps

Use `#attention` from the landing page. Edit/apply Q/K/V together, select left/right policies and stable/naive method, and choose scores, scaled logits, probabilities, or output. Each compared quantity and its FP32/binary64 baseline share one symmetric magnitude scale. Left/right absolute errors share a scale; relative errors share another. Tints supplement full signed text, raw hex, and explicit NaN/Infinity/Undefined labels; color alone never conveys a difference.

Native left-cell buttons select exact decoded decimal/raw binary/classification. Any configuration can be selected on the left for complete stage inspection. Native previous/next/select controls navigate all computation boundaries. A probability-row select exposes weights, their displayed binary64 sum, and the stored normalization denominator. Tables and long values have keyboard-focusable scroll areas. System/light/dark themes reuse the existing tokens. Selecting a preset applies its matrices, policies, method, phenomenon, expected outcome, and verification explanation.

Twelve presets cover stable/naive behavior, target exp overflow, BF16 range, score accumulation, near ties, lost probability tails, large logits, product cancellation, scaling approximation, value aggregation, material output changes, and unrecoverable input loss. There are **28 explicit outcome assertions**, plus independent full-pipeline references for every preset under every supported policy/method, not only its default view.

## Verification and reproducible JSON

The full suite contains **424 tests**, preserving the earlier 390 unchanged. New coverage includes matrix bounds/sparse/ragged inputs, **9,000 independent rational quotient checks**, mixed-format midpoint/tininess regressions, every conceptual stage, all twelve preset expectations, stable/naive failures, finite normalization invariants for **250 deterministic rows**, maximum shapes in all five modes, and real-engine UI/keyboard integration. The independent quotient oracle compares exact rational distances against the historical test lattice; it does not reuse production normalization.

CI also replays **240 complete independently generated calculations** from `src/tests/reference/attention-vectors.json`, covering 24 inputs (12 presets plus 12 seeded matrices), five policies, both methods, all four result matrices, and unioned flags. NaNs compare quiet classification; non-NaN bits and all flags compare strictly. This fixture is development/test-only and is not imported by the application.

Run the deterministic full-trace exporter and optional independent verifier:

```sh
node scripts/run-attention.mjs /tmp/fp-play-attention.json
python3 scripts/verify-attention.py /tmp/fp-play-attention.json /tmp/attention-verification.json
# Optional third output regenerates CI vectors; review the diff before committing.
python3 scripts/verify-attention.py /tmp/fp-play-attention.json /tmp/attention-verification.json /tmp/attention-vectors.json
```

The exporter requires only existing project dependencies. The offline verifier requires Python 3 and NumPy (development only, not a runtime/npm dependency); it uses standard-library Fraction/Decimal otherwise. This run used **Node 22.9.0, Python 3.12.2, NumPy 1.26.4**, and seed **`0x6174746e`**. Two exports on the same runtime matched byte for byte. Node version is recorded; host exp implementations can vary across runtimes/platforms.

The exported JSON includes original Q/K/V, full policies, all computation events/operands/bits/rounding/flags, result matrices, FP32 and binary64 references, output errors, preset expectations, seed, and numerical-policy metadata. Special Numbers and negative zero use strings `NaN`, `Infinity`, `-Infinity`, `-0`; BigInt fields are decimal strings, as recorded in the schema metadata. Full traces are deliberately exported on demand rather than bundled as a large browser asset. No timestamps or measured-silicon claims are generated.

Actual independent results, recorded in [attention-verification.json](experiments/attention-verification.json):

| Check                                           | Actual comparisons                      | Acceptance                                                     |
| ----------------------------------------------- | --------------------------------------- | -------------------------------------------------------------- |
| Fraction full pipelines                         | 240 calculations / 3,010 matrix values  | Exact non-NaN bits; NaN classification                         |
| Fraction replay of arithmetic/conversion stages | 12,916 stage results and flag sets      | Exact non-NaN bits and all modeled flags                       |
| NumPy aligned FP32 pipelines                    | 48 calculations / 602 matrix values     | Exact non-NaN bits; NaN classification                         |
| Decimal(100) versus binary64 baseline           | 301 scores/logits/probabilities/outputs | abs(error) ≤ 5e-12 × max(1, abs(reference))                    |
| Preset outcomes                                 | 28 checks                               | Exact dyadic values / explicit finite-positive or NaN outcomes |

**Zero mismatches** were observed. Maximum binary64 absolute error against Decimal(100): **3.789746294556856e-14**. The Fraction oracle independently decodes, binary-searches exact representable-value distances, applies each boundary, and executes the full graph. Its exp is Python math.exp followed by independent quantization, deliberately aligned with the modeled host-exp policy rather than presented as exact exponentiation. Native NumPy uses explicit sequential float32 multiply/add, Python math.exp followed by float32 conversion, and native float32 division. It avoids BLAS, FMA, `np.exp(float32)`, and PyTorch fused attention, whose evaluation orders/transcendental implementations can differ. No PyTorch equality claim is made. Existing SoftFloat arithmetic coverage remains a separate guarantee for the reused engine, not a proof of transformer/exp conformance.

## Limits and visual verification

Only one bounded unmasked head, five named policies, sequential reductions, and nearest-even are implemented. No arbitrary configurations, training, kernels, runtime performance estimates, silicon measurements, or RTL are provided. Finite corpora and normalization tolerances are not full IEEE or framework conformance. Binary64-first parsing can lose input information before any modeled format conversion.

No browser automation tool is available in this environment. DOM tests verify controls, labels, keyboard-native selection, raw values, and actual numerical integration; static theme checks pass. No screenshots, rendered mobile/desktop layouts, touch/focus appearance, screen-reader announcements, or accessibility certification are claimed. Complete those manual checks before publication. Nothing is pushed, merged, deployed, or published by the scripts or this milestone.

Public references: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [ECMAScript Math.exp](https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math.exp), [Python Decimal](https://docs.python.org/3/library/decimal.html). Only public information and independently developed numerical models are used.

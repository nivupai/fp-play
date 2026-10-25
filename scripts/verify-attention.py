"""Development-only independent attention oracle: Fraction, NumPy float32, Decimal(100).
No production imports. Input JSON is from run-attention.mjs; no trained model/GPU needed.
"""

import json
import math
import struct
import sys
import platform
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import numpy as np

SPECS = {"FP16": (10, 5, 15, 16), "BF16": (7, 8, 127, 16), "FP32": (23, 8, 127, 32)}
POLICIES = {
    "fp16": ("FP16", "FP16", "FP16", False),
    "fp16-wide": ("FP16", "FP32", "FP32", True),
    "bf16-wide": ("BF16", "FP32", "FP32", True),
    "fp32": ("FP32", "FP32", "FP32", False),
    "fp16-softmax": ("FP16", "FP32", "FP16", True),
}


def num(x):
    return (
        {
            "NaN": float("nan"),
            "Infinity": float("inf"),
            "-Infinity": -float("inf"),
            "-0": -0.0,
        }.get(x, x)
        if isinstance(x, str)
        else x
    )


@lru_cache(maxsize=100000)
def decoded(fmt, bits):
    p, e, bias, width = SPECS[fmt]
    sign = bits // 2 ** (width - 1)
    exponent = (bits // 2**p) % 2**e
    fraction = bits % 2**p
    if exponent == 2**e - 1:
        return float("nan") if fraction else (-math.inf if sign else math.inf)
    n = (2**p + fraction) if exponent else fraction
    power = (exponent - bias if exponent else 1 - bias) - p
    return (-1 if sign else 1) * n * 2.0**power if n else (-0.0 if sign else 0.0)


def flags(invalid=False, overflow=False, underflow=False, inexact=False):
    return dict(
        invalid=invalid, overflow=overflow, underflow=underflow, inexact=inexact
    )


def quant(fmt, value, negative_zero=False):
    p, e, bias, width = SPECS[fmt]
    inf = (2**e - 1) * 2**p
    if isinstance(value, float) and not math.isfinite(value):
        return (
            inf + 2 ** (p - 1)
            if math.isnan(value)
            else inf + (2 ** (width - 1) if value < 0 else 0),
            flags(),
        )
    if isinstance(value, float) and value == 0:
        negative_zero = math.copysign(1, value) < 0
    exact = Fraction(value)
    negative = exact < 0 or (exact == 0 and negative_zero)
    n = abs(exact)
    lo, hi = 0, inf - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if Fraction(decoded(fmt, mid)) <= n:
            lo = mid
        else:
            hi = mid - 1
    lower = Fraction(decoded(fmt, lo))
    upper = (
        Fraction(2) ** (bias + 1) if lo == inf - 1 else Fraction(decoded(fmt, lo + 1))
    )
    bits = lo + int(n - lower > upper - n or (n - lower == upper - n and lo % 2 == 1))
    overflow = bits == inf
    inexact = overflow or Fraction(decoded(fmt, bits)) != n
    tiny = Fraction(2) ** (1 - bias) - Fraction(2) ** (1 - bias - p - 2)
    return bits + (2 ** (width - 1) if negative else 0), flags(
        overflow=overflow, inexact=inexact, underflow=n != 0 and n < tiny and inexact
    )


def value(fmt, bits):
    return {"format": fmt, "bits": bits}


def real(v):
    return decoded(v["format"], v["bits"])


def cast(fmt, x):
    return value(fmt, quant(fmt, x)[0])


def arithmetic(fmt, op, operands):
    vals = list(map(real, operands))
    if any(math.isnan(x) for x in vals):
        return quant(fmt, math.nan)
    a, b = vals[:2]
    if op == "subtract maximum":
        b = -b
        op = "add"
    if op == "divide":
        if (a == 0 and b == 0) or (math.isinf(a) and math.isinf(b)):
            return quant(fmt, math.nan)[0], flags(invalid=True)
        if b == 0:
            return quant(fmt, math.copysign(math.inf, a * math.copysign(1, b)))
        if math.isinf(a):
            return quant(fmt, math.copysign(math.inf, a * b))
        if math.isinf(b):
            return quant(fmt, math.copysign(0, a * b))
        return quant(
            fmt, Fraction(a) / Fraction(b), math.copysign(1, a) != math.copysign(1, b)
        )
    if op == "multiply" or op == "fma":
        if (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a)):
            return quant(fmt, math.nan)[0], flags(invalid=True)
        product_sign = math.copysign(1, a) != math.copysign(1, b)
        product = a * b if math.isinf(a) or math.isinf(b) else Fraction(a) * Fraction(b)
        if op == "multiply":
            return quant(fmt, product, product_sign)
        c = vals[2]
        if (
            isinstance(product, float)
            and math.isinf(product)
            and math.isinf(c)
            and (product < 0) != (c < 0)
        ):
            return quant(fmt, math.nan)[0], flags(invalid=True)
        if isinstance(product, float) and math.isinf(product):
            return quant(fmt, product)
        if math.isinf(c):
            return quant(fmt, c)
        return quant(
            fmt,
            product + Fraction(c),
            product == 0 and c == 0 and product_sign and math.copysign(1, c) < 0,
        )
    if math.isinf(a) and math.isinf(b) and (a < 0) != (b < 0):
        return quant(fmt, math.nan)[0], flags(invalid=True)
    if math.isinf(a) or math.isinf(b):
        return quant(fmt, a + b)
    return quant(
        fmt,
        Fraction(a) + Fraction(b),
        a == 0 and b == 0 and math.copysign(1, a) < 0 and math.copysign(1, b) < 0,
    )


def op(fmt, kind, *args):
    return value(fmt, arithmetic(fmt, kind, args)[0])


def host_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def simulate(inp, mode, method):
    inp_fmt, work, soft, fused = POLICIES[mode]
    q, k, v = [
        [[cast(inp_fmt, num(x)) for x in row] for row in inp[name]]
        for name in ["q", "k", "v"]
    ]

    def dot(a, b):
        acc = cast(work, 0)
        for x, y in zip(a, b):
            if fused:
                acc = op(work, "fma", x, y, acc)
            else:
                acc = op(work, "add", acc, op(inp_fmt, "multiply", x, y))
        return acc

    scores = [[dot(qr, kr) for kr in k] for qr in q]
    scale = cast(work, 1 / math.sqrt(len(q[0])))
    logits = [[op(work, "multiply", x, scale) for x in row] for row in scores]
    probs = []
    for row in logits:
        inputs = [cast(soft, real(x)) for x in row]
        maximum = (
            math.nan
            if any(math.isnan(real(x)) for x in inputs)
            else max(map(real, inputs))
        )
        shifted = (
            [op(soft, "subtract maximum", x, cast(soft, maximum)) for x in inputs]
            if method == "stable"
            else inputs
        )
        exps = [cast(soft, host_exp(real(x))) for x in shifted]
        denom = cast(soft, 0)
        for x in exps:
            denom = op(soft, "add", denom, x)
        probs.append([cast(inp_fmt, real(op(soft, "divide", x, denom))) for x in exps])
    output = [
        [cast(inp_fmt, real(dot(row, [vr[j] for vr in v]))) for j in range(len(v[0]))]
        for row in probs
    ]
    return dict(scores=scores, logits=logits, probabilities=probs, output=output)


def numpy_reference(inp, method):
    # Deliberately no matmul/BLAS/FMA: aligns with FP Play's separate FP32 operations.
    with np.errstate(all="ignore"):
        q, k, v = [
            np.array([[num(x) for x in row] for row in inp[name]], dtype=np.float32)
            for name in ["q", "k", "v"]
        ]

        def dot(a, b):
            acc = np.float32(0)
            for x, y in zip(a, b):
                acc = np.float32(acc + np.float32(x * y))
            return acc

        scores = np.array([[dot(a, b) for b in k] for a in q], dtype=np.float32)
        logits = np.float32(scores * np.float32(1 / math.sqrt(q.shape[1])))
        probs = []
        for row in logits:
            shifted = np.float32(row - np.max(row)) if method == "stable" else row
            exps = np.array([host_exp(float(x)) for x in shifted], dtype=np.float32)
            denom = np.float32(0)
            for x in exps:
                denom = np.float32(denom + x)
            probs.append(np.float32(exps / denom))
        probs = np.array(probs)
        output = np.array(
            [[dot(row, v[:, j]) for j in range(v.shape[1])] for row in probs],
            dtype=np.float32,
        )
        return dict(scores=scores, logits=logits, probabilities=probs, output=output)


def decimal_reference(inp):
    with localcontext() as ctx:
        ctx.prec = 100
        q, k, v = [
            [[Decimal.from_float(float(num(x))) for x in row] for row in inp[name]]
            for name in ["q", "k", "v"]
        ]
        scores = [
            [sum((x * y for x, y in zip(a, b)), Decimal(0)) for b in k] for a in q
        ]
        root = Decimal(len(q[0])).sqrt()
        logits = [[x / root for x in row] for row in scores]
        probs = []
        for row in logits:
            maximum = max(row)
            exps = [(x - maximum).exp() for x in row]
            denom = sum(exps)
            probs.append([x / denom for x in exps])
        output = [
            [
                sum((x * v[i][j] for i, x in enumerate(row)), Decimal(0))
                for j in range(len(v[0]))
            ]
            for row in probs
        ]
        return dict(scores=scores, logits=logits, probabilities=probs, output=output)


def matches(actual, fmt, bits):
    return (
        math.isnan(decoded(fmt, bits))
        if math.isnan(real(actual))
        else actual["bits"] == bits and actual["format"] == fmt
    )


def main():
    artifact = json.load(open(sys.argv[1], encoding="utf8"))
    counts = dict(
        calculations=0,
        matrixValues=0,
        stageResultsAndFlags=0,
        numpyCalculations=0,
        numpyValues=0,
        decimalValues=0,
        presetExpectations=0,
    )
    max_decimal_error = 0.0
    records = []
    vectors = []
    for case in artifact["records"]:
        high = decimal_reference(case["input"])
        for stage, rows in high.items():
            for i, row in enumerate(rows):
                for j, x in enumerate(row):
                    baseline = float(num(case["references"]["binary64"][stage][i][j]))
                    error = float(abs(Decimal.from_float(baseline) - x))
                    tolerance = 5e-12 * max(1, abs(float(x)))
                    assert error <= tolerance, (case["id"], stage, error, tolerance)
                    max_decimal_error = max(max_decimal_error, error)
                    counts["decimalValues"] += 1
        vector = dict(id=case["id"], input=case["input"], results=[])
        for result in case["results"]:
            expected = simulate(case["input"], result["mode"], result["method"])
            counts["calculations"] += 1
            for stage, rows in expected.items():
                for i, row in enumerate(rows):
                    for j, x in enumerate(row):
                        actual = result[stage][i][j]
                        assert matches(actual, x["format"], x["bits"]), (
                            case["id"],
                            result["mode"],
                            result["method"],
                            stage,
                            i,
                            j,
                            actual,
                            x,
                        )
                        counts["matrixValues"] += 1
            merged_flags = flags()
            for event in result["events"]:
                fmt = event["result"]["format"]
                if event["operation"] == "quantize":
                    bits, status = quant(fmt, num(event["before"]))
                elif event["operation"] == "Math.exp → quantize":
                    bits, status = quant(fmt, host_exp(real(event["operands"][0])))
                else:
                    bits, status = arithmetic(
                        fmt, event["operation"], event["operands"]
                    )
                assert matches(event["result"], fmt, bits), (
                    case["id"],
                    result["mode"],
                    event["stage"],
                    event["result"],
                    bits,
                )
                assert event["flags"] == status, (
                    case["id"],
                    result["mode"],
                    event["stage"],
                    event["flags"],
                    status,
                )
                for key, raised in status.items():
                    merged_flags[key] = merged_flags[key] or raised
                counts["stageResultsAndFlags"] += 1
            assert result["flags"] == merged_flags, (case["id"], "aggregate flags")
            vector["results"].append(
                dict(
                    mode=result["mode"],
                    method=result["method"],
                    stages={
                        name: [[x["bits"] for x in row] for row in rows]
                        for name, rows in expected.items()
                    },
                    flags=merged_flags,
                )
            )
            if result["mode"] == "fp32":
                native = numpy_reference(case["input"], result["method"])
                counts["numpyCalculations"] += 1
                for stage, rows in native.items():
                    for i, row in enumerate(rows):
                        for j, x in enumerate(row):
                            bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
                            assert matches(result[stage][i][j], "FP32", bits), (
                                case["id"],
                                stage,
                                x,
                                result[stage][i][j],
                            )
                            counts["numpyValues"] += 1
        for check in case.get("checks", []):
            result = next(
                r
                for r in case["results"]
                if r["mode"] == check["mode"] and r["method"] == check["method"]
            )
            x = real(result[check["stage"]][check["row"]][check["column"]])
            assert (
                x == check.get("value")
                if check["relation"] == "equal"
                else math.isnan(x)
                if check["relation"] == "nan"
                else math.isfinite(x) and x > 0
            ), (case["id"], check, x)
            counts["presetExpectations"] += 1
        vectors.append(vector)
        records.append(
            dict(
                id=case["id"],
                decimal100Output=[[str(x) for x in row] for row in high["output"]],
            )
        )
    summary = dict(
        schema=1,
        seed=artifact["seed"],
        python=platform.python_version(),
        numpy=np.__version__,
        decimalPrecision=100,
        counts=counts,
        mismatches=0,
        nanPolicy="Compare NaNs by classification, all modeled flags strictly; payloads/signs implementation-specific",
        fractionTolerance="Exact bits and flags; Python math.exp approximation checked after quantization",
        numpyPolicy="Separate sequential float32 products/additions, Python math.exp then float32, native float32 divide; no BLAS or FMA",
        decimalTolerance="Binary64 baseline vs Decimal(100): abs error <= 5e-12 * max(1, abs(reference))",
        maxBinary64AbsoluteErrorVsDecimal100=max_decimal_error,
        highPrecisionReferences=records,
    )
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf8").write(text)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w", encoding="utf8").write(
            json.dumps(
                dict(
                    schema=1,
                    seed=artifact["seed"],
                    source="Independent Python Fraction oracle; Math.exp host approximation independently quantized; see docs/ATTENTION_PRECISION.md",
                    cases=vectors,
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    print(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate independent reference vectors for fused dot-product chains.

A fused chain in accumulator format ACC over inputs in format IN:
acc = +0; for each k: acc = round_ACC(acc + a[k] * b[k]), the product exact
and the sum rounded once (round to nearest, ties to even, gradual underflow,
overflow to infinity). This is the semantics of runtime/src/ops.rs
``fused_chain_dot`` and of runtime/src/precision/mac.rs. This script shares
no code with the crate: values are Python ``Fraction``s, rounding uses
rational arithmetic. An exact zero sum is -0 only when both addends are -0
(the product of two zeros is -0 when exactly one factor is negative).
Inputs are finite; once the accumulator overflows it stays infinite (no
infinite inputs, so no infinity minus infinity). Only the standard library
is used, and the output is deterministic.

Usage: python3 -I scripts/gen_chain_vectors.py > tests/vectors/chain-dots.txt
Each line: ``<acc> <in> <a,a,...> <b,b,...> <expected raw>`` in lowercase hex.
"""

import random
from fractions import Fraction

FORMATS = {"fp16": (5, 10), "bf16": (8, 7), "fp32": (8, 23)}
CASES = 400
SEED = 20261119


def params(name):
    e, f = FORMATS[name]
    return e, f, (1 << (e - 1)) - 1


def decode(name, raw):
    """(Fraction value, negative) of a finite raw pattern."""
    e, f, bias = params(name)
    width = 1 + e + f
    negative = (raw >> (width - 1)) & 1 == 1
    field = (raw >> f) & ((1 << e) - 1)
    frac = raw & ((1 << f) - 1)
    assert field != (1 << e) - 1
    if field == 0:
        value = Fraction(frac) * Fraction(2) ** (1 - bias - f)
    else:
        value = Fraction(frac + (1 << f)) * Fraction(2) ** (field - bias - f)
    return (-value if negative else value), negative


def encode_round(name, x, negative_zero):
    """Round Fraction x to the nearest raw pattern, ties to even."""
    e, f, bias = params(name)
    width = 1 + e + f
    sign = 1 << (width - 1)
    if x == 0:
        return sign if negative_zero else 0
    negative = x < 0
    a = -x if negative else x
    emin = 1 - bias
    # Exponent of the leading bit.
    k = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** k > a:
        k -= 1
    q = max(k, emin) - f
    scaled = a / Fraction(2) ** q
    n = scaled.numerator // scaled.denominator
    rest = scaled - n
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    # n * 2**q, normalized.
    if n >= (1 << (f + 1)):
        n >>= 1
        q += 1
    if n < (1 << f):  # subnormal (q == emin - f)
        raw = n
    else:
        field = q + f + bias
        if field >= (1 << e) - 1:
            raw = ((1 << e) - 1) << f
        else:
            raw = (field << f) | (n - (1 << f))
    return raw | (sign if negative else 0)


def random_raw(name, rng):
    e, f, bias = params(name)
    width = 1 + e + f
    r = rng.random()
    if r < 0.05:
        field = 0  # zeros and subnormals
    elif r < 0.1:
        field = (1 << e) - 2  # near overflow
    else:
        # Mostly values near 1 so that sums interact and cancel.
        field = max(1, min((1 << e) - 2, bias + rng.randint(-6, 6)))
    frac = rng.getrandbits(f)
    if r < 0.02:
        frac = 0
    return (rng.getrandbits(1) << (width - 1)) | (field << f) | frac


def chain(acc_name, in_name, a, b):
    e, f, _ = params(acc_name)
    width = 1 + e + f
    acc = Fraction(0)
    acc_negative_zero = False
    acc_infinite = None  # None, or the sign of the infinity
    for x, y in zip(a, b):
        xv, xn = decode(in_name, x)
        yv, yn = decode(in_name, y)
        p = xv * yv
        p_negative_zero = p == 0 and (xn != yn)
        if acc_infinite is not None:
            continue
        s = acc + p
        raw = encode_round(acc_name, s, acc_negative_zero and p_negative_zero)
        field = (raw >> f) & ((1 << e) - 1)
        if field == (1 << e) - 1:
            acc_infinite = raw
            continue
        acc, acc_negative_zero = decode(acc_name, raw)
        acc_negative_zero = acc == 0 and acc_negative_zero
    if acc_infinite is not None:
        return acc_infinite
    return encode_round(acc_name, acc, acc_negative_zero)


def main():
    rng = random.Random(SEED)
    lines = []
    for acc_name in ("fp32", "bf16", "fp16"):
        for in_name in ("fp32", "bf16", "fp16"):
            for _ in range(CASES):
                n = rng.randint(1, 16)
                a = [random_raw(in_name, rng) for _ in range(n)]
                b = [random_raw(in_name, rng) for _ in range(n)]
                want = chain(acc_name, in_name, a, b)
                lines.append("%s %s %s %s %x" % (
                    acc_name, in_name,
                    ",".join("%x" % v for v in a),
                    ",".join("%x" % v for v in b),
                    want,
                ))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

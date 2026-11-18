#!/usr/bin/env python3
"""Generate independent reference vectors for the fixed shard tree.

The tree (runtime/src/shard_tree.rs): over S leaves, at level k = 0, 1, ...
node i absorbs node i + 2**k for every multiple i of 2**(k+1) with
i + 2**k < S; node 0 is the root. Each combine is one correctly rounded
addition in the accumulator format. This script shares no code with the
crate. Values are decoded to Python ``Fraction``s (or infinities / NaN), each
addition is exact and then rounded with rational arithmetic, round to nearest,
ties to even. An exact zero sum is -0 only when both addends are -0. NaN
results are the format's canonical quiet NaN, and +inf + -inf is NaN. Only the
standard library is used, and the output is deterministic.

Usage: python3 -I scripts/gen_tree_vectors.py > tests/vectors/tree-sums.txt
Each line: ``<acc>><out> <leaf,leaf,...> <expected raw>`` in lowercase hex,
where leaves are accumulator-format raw patterns, the tree runs in <acc>, and
the root is rounded once to <out>. With S = 1 and acc == out, the single leaf
is copied bit for bit.
"""

import random
import sys
from fractions import Fraction

FORMATS = {
    # name: (exponent bits, fraction bits)
    "fp16": (5, 10),
    "bf16": (8, 7),
    "fp32": (8, 23),
}
CANONICAL_NAN = {"fp16": 0x7E00, "bf16": 0x7FC0, "fp32": 0x7FC00000}
PAIRS = [("fp16", "fp16"), ("bf16", "bf16"), ("fp32", "fp32"), ("fp32", "bf16")]
SHARD_COUNTS = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100]
CASES_PER_PAIR = 1500
SEED = 20261117

NAN = ("nan",)


def params(name):
    e, f = FORMATS[name]
    return e, f, (1 << (e - 1)) - 1


def decode(name, raw):
    """("nan",), ("inf", negative) or ("num", Fraction, negative)."""
    e, f, bias = params(name)
    width = 1 + e + f
    negative = (raw >> (width - 1)) & 1 == 1
    field = (raw >> f) & ((1 << e) - 1)
    frac = raw & ((1 << f) - 1)
    if field == (1 << e) - 1:
        return NAN if frac else ("inf", negative)
    if field == 0:
        value = Fraction(frac) * Fraction(2) ** (1 - bias - f)
    else:
        value = Fraction(frac + (1 << f)) * Fraction(2) ** (field - bias - f)
    return ("num", -value if negative else value, negative)


def round_to(name, v):
    """Encode a decoded value in the format, round to nearest, ties to even."""
    e, f, bias = params(name)
    width = 1 + e + f
    sign_bit = 1 << (width - 1)
    if v[0] == "nan":
        return CANONICAL_NAN[name]
    if v[0] == "inf":
        return (sign_bit if v[1] else 0) | (((1 << e) - 1) << f)
    _, x, negative_zero = v
    if x == 0:
        return sign_bit if negative_zero else 0
    negative = x < 0
    a = -x if negative else x
    lead = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** lead > a:
        lead -= 1
    if Fraction(2) ** (lead + 1) <= a:
        lead += 1
    p = f + 1
    quantum = max(lead - (p - 1), (1 - bias) - f)
    scaled = a / Fraction(2) ** quantum
    n = scaled.numerator // scaled.denominator
    remainder = scaled - n
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 1 << p:
        n >>= 1
        quantum += 1
    s = sign_bit if negative else 0
    if n == 0:
        return s
    if n >= 1 << f:
        biased = quantum + f + bias
        if biased >= (1 << e) - 1:
            return s | (((1 << e) - 1) << f)
        return s | (biased << f) | (n & ((1 << f) - 1))
    return s | n


def add(name, a_raw, b_raw):
    """One correctly rounded addition of two raw patterns."""
    a, b = decode(name, a_raw), decode(name, b_raw)
    if a[0] == "nan" or b[0] == "nan":
        return CANONICAL_NAN[name]
    if a[0] == "inf" and b[0] == "inf":
        return CANONICAL_NAN[name] if a[1] != b[1] else round_to(name, a)
    if a[0] == "inf":
        return round_to(name, a)
    if b[0] == "inf":
        return round_to(name, b)
    total = a[1] + b[1]
    both_negative_zero = a[1] == 0 and a[2] and b[1] == 0 and b[2]
    return round_to(name, ("num", total, both_negative_zero))


def tree_sum(name, leaves):
    nodes = list(leaves)
    s = len(nodes)
    half = 1
    while half < s:
        for left in range(0, s, 2 * half):
            if left + half < s:
                nodes[left] = add(name, nodes[left], nodes[left + half])
        half *= 2
    return nodes[0]


def convert(src, dst, raw):
    if src == dst:
        return raw
    return round_to(dst, decode(src, raw))


def random_leaf(rng, name, style):
    e, f, bias = params(name)
    width = 1 + e + f
    sign = rng.getrandbits(1) << (width - 1)
    if style == "special" and rng.random() < 0.85:
        style = "narrow"  # special values stay sparse so results are not all NaN
    if style == "special":
        kind = rng.randrange(4)
        if kind == 0:
            return sign | (((1 << e) - 1) << f)  # infinity
        if kind == 1:
            return sign | (((1 << e) - 1) << f) | (rng.getrandbits(f) | 1)  # NaN, any payload
        if kind == 2:
            return sign  # signed zero
    if style == "large":
        field = (1 << e) - 2 - rng.randrange(2)  # near overflow
    elif style == "subnormal":
        field = 0
    elif style == "narrow":
        field = bias + rng.randint(-3, 3)  # cancellation and ties are common
    else:
        field = rng.randint(0, (1 << e) - 2)
    frac = rng.getrandbits(f)
    if style == "short":
        frac &= ~((1 << max(f - 3, 0)) - 1)  # few significant bits: exact ties
    return sign | (field << f) | frac


def main():
    rng = random.Random(SEED)
    out = sys.stdout
    styles = ("uniform", "narrow", "subnormal", "short", "large", "special")
    for acc, dst in PAIRS:
        for case in range(CASES_PER_PAIR):
            style = styles[case % len(styles)]
            count = SHARD_COUNTS[case % len(SHARD_COUNTS)]
            # Mostly one style, with a few leaves of a second for mixed magnitudes.
            leaves = [
                random_leaf(rng, acc, style if rng.random() < 0.9 else styles[rng.randrange(4)])
                for _ in range(count)
            ]
            if case % 11 == 0 and count >= 2:
                e, f, _ = params(acc)
                leaves[1] = leaves[0] ^ (1 << (e + f))  # exact cancellation in the first node
            if case % 97 == 0:
                e, f, _ = params(acc)
                leaves = [1 << (e + f)] * count  # all negative zero
            expected = convert(acc, dst, tree_sum(acc, leaves))
            out.write(f"{acc}>{dst} {','.join(format(r, 'x') for r in leaves)} {expected:x}\n")


if __name__ == "__main__":
    main()

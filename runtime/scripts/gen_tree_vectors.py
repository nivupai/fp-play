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


# Styles of a case, with weights. Most cases are finite sums that cover the
# hard parts of rounding: subnormals, exact ties, and cancellation. Near
# overflow and special inputs are kept rare so NaN results stay a few percent.
STYLES = [
    ("uniform", 6),  # exponents spread over a range that cannot overflow any S used here
    ("narrow", 5),  # exponents near 1.0: frequent carries, ties and partial cancellation
    ("subnormal", 4),  # subnormal leaves and sums crossing the normal boundary
    ("short", 4),  # few significant bits: exact halfway cases
    ("cancel", 3),  # sibling leaves that cancel exactly or almost exactly
    ("large", 1),  # near overflow: sums that overflow to infinity
    ("special", 1),  # one infinity, NaN or signed zero among finite leaves
]
STYLE_CYCLE = [name for name, weight in STYLES for _ in range(weight)]
FINITE_STYLES = ("uniform", "narrow", "subnormal", "short")


def random_leaf(rng, name, style):
    e, f, bias = params(name)
    width = 1 + e + f
    sign = rng.getrandbits(1) << (width - 1)
    max_field = (1 << e) - 2
    if style == "large":
        field = max_field - rng.randrange(2)
    elif style == "subnormal":
        field = 0 if rng.random() < 0.8 else 1
    elif style == "narrow":
        field = bias + rng.randint(-3, 3)
    elif style == "uniform":
        # 2**8 > 100 leaves, so a sum of S <= 100 of these cannot overflow.
        field = rng.randint(0, max_field - 8)
    else:  # "short"
        field = bias + rng.randint(-6, 6)
    frac = rng.getrandbits(f)
    if style == "short":
        frac &= ~((1 << max(f - 3, 0)) - 1)
    return sign | (field << f) | frac


def special_leaf(rng, name):
    e, f, _ = params(name)
    width = 1 + e + f
    sign = rng.getrandbits(1) << (width - 1)
    kind = rng.randrange(4)
    if kind == 0:
        return sign | (((1 << e) - 1) << f) | (rng.getrandbits(f) | 1)  # NaN, any payload
    if kind == 1:
        return sign | (((1 << e) - 1) << f)  # infinity
    return sign  # signed zero


def case_leaves(rng, name, style, count):
    e, f, _ = params(name)
    sign_bit = 1 << (e + f)
    if style == "cancel":
        leaves = [random_leaf(rng, name, "narrow") for _ in range(count)]
        for i in range(0, count - 1, 2):
            # The sibling is the negation, sometimes nudged by one ulp so a tiny residue survives.
            leaves[i + 1] = leaves[i] ^ sign_bit
            if rng.random() < 0.5 and leaves[i + 1] & ((1 << f) - 1):
                leaves[i + 1] -= 1
        return leaves
    if style == "large":
        # Mostly one sign, so overflow gives an infinity rather than inf - inf.
        negative = rng.getrandbits(1)
        leaves = [random_leaf(rng, name, "large") for _ in range(count)]
        return [(r & ~sign_bit) | (sign_bit if negative else 0) for r in leaves]
    # Mostly one style, with a few leaves of another for mixed magnitudes.
    leaves = [
        random_leaf(rng, name, style if style in FINITE_STYLES and rng.random() < 0.9 else rng.choice(FINITE_STYLES))
        for _ in range(count)
    ]
    if style == "special":
        leaves[rng.randrange(count)] = special_leaf(rng, name)
    return leaves


def main():
    rng = random.Random(SEED)
    out = sys.stdout
    for acc, dst in PAIRS:
        e, f, _ = params(acc)
        for case in range(CASES_PER_PAIR):
            style = STYLE_CYCLE[case % len(STYLE_CYCLE)]
            # Drawn, not cycled, so every style meets every shard count.
            count = rng.choice(SHARD_COUNTS)
            leaves = case_leaves(rng, acc, style, count)
            if case % 97 == 0:
                leaves = [1 << (e + f)] * count  # all negative zero
            expected = convert(acc, dst, tree_sum(acc, leaves))
            out.write(f"{acc}>{dst} {','.join(format(r, 'x') for r in leaves)} {expected:x}\n")


if __name__ == "__main__":
    main()

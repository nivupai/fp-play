#!/usr/bin/env python3
"""Generate independent reference vectors for exact multi-term sums.

The Rust crate rounds an exact fixed-point integer. This script shares no code
with it: values are decoded to Python ``Fraction``s, summed exactly, and
rounded to the destination format with rational arithmetic. Only the standard
library is used; the output is deterministic for a given seed.

Usage: python3 scripts/gen_sum_vectors.py > tests/vectors/exact-sums.txt
Each line: ``<format> <raw,raw,...> <expected raw>`` in lowercase hex.
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
CASES_PER_FORMAT = 6000
SEED = 20261115


def params(name):
    e, f = FORMATS[name]
    bias = (1 << (e - 1)) - 1
    return e, f, bias


def decode(name, raw):
    """Exact Fraction for a finite raw pattern, plus its sign bit."""
    e, f, bias = params(name)
    width = 1 + e + f
    negative = (raw >> (width - 1)) & 1 == 1
    field = (raw >> f) & ((1 << e) - 1)
    frac = raw & ((1 << f) - 1)
    assert field != (1 << e) - 1, "only finite inputs are generated"
    if field == 0:
        value = Fraction(frac) * Fraction(2) ** (1 - bias - f)
    else:
        value = Fraction(frac + (1 << f)) * Fraction(2) ** (field - bias - f)
    return (-value if negative else value), negative


def round_to(name, x, negative_zero):
    """Round a Fraction to the format with round-to-nearest, ties-to-even."""
    e, f, bias = params(name)
    width = 1 + e + f
    sign = 1 << (width - 1)
    if x == 0:
        return sign if negative_zero else 0
    negative = x < 0
    a = -x if negative else x
    # floor(log2(a)) by integer bit lengths, then correct by one if needed.
    lead = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** lead > a:
        lead -= 1
    if Fraction(2) ** (lead + 1) <= a:
        lead += 1
    p = f + 1
    min_quantum = (1 - bias) - f
    quantum = max(lead - (p - 1), min_quantum)
    scaled = a / Fraction(2) ** quantum
    n = scaled.numerator // scaled.denominator
    remainder = scaled - n
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 1 << p:
        n >>= 1
        quantum += 1
    s = sign if negative else 0
    if n == 0:
        return s
    if n >= 1 << f:
        biased = quantum + f + bias
        if biased >= (1 << e) - 1:
            return s | (((1 << e) - 1) << f)
        return s | (biased << f) | (n & ((1 << f) - 1))
    return s | n


def random_finite(rng, name, style):
    e, f, bias = params(name)
    width = 1 + e + f
    while True:
        if style == "subnormal":
            field = 0
        elif style == "narrow":
            # exponents clustered near 1.0 so cancellation and ties are common
            field = bias + rng.randint(-3, 3)
        else:
            field = rng.randint(0, (1 << e) - 2)
        frac = rng.getrandbits(f)
        if style == "short":
            frac &= ~((1 << max(f - 3, 0)) - 1)  # few significant bits → exact ties
        sign = rng.getrandbits(1)
        raw = (sign << (width - 1)) | (field << f) | frac
        return raw


def main():
    rng = random.Random(SEED)
    out = sys.stdout
    for name in FORMATS:
        for case in range(CASES_PER_FORMAT):
            style = ("uniform", "narrow", "subnormal", "short")[case % 4]
            count = rng.randint(1, 12)
            raws = [random_finite(rng, name, style) for _ in range(count)]
            if case % 7 == 0 and count >= 2:
                # exact cancellation of the first term
                e, f, _ = params(name)
                raws[1] = raws[0] ^ (1 << (e + f))
            if case % 97 == 0:
                e, f, _ = params(name)
                raws = [1 << (e + f)] * count  # all negative zero
            values = [decode(name, r) for r in raws]
            total = sum((v for v, _ in values), Fraction(0))
            all_negative_zero = all(v == 0 and neg for v, neg in values)
            expected = round_to(name, total, all_negative_zero)
            out.write(f"{name} {','.join(format(r, 'x') for r in raws)} {expected:x}\n")


if __name__ == "__main__":
    main()

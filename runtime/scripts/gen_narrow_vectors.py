#!/usr/bin/env python3
"""Independent reference vectors for FP8 (E4M3, E5M2) and INT8 with scales.

Standard library only; shares no code with the Rust crate. Values are exact
`fractions.Fraction`s. Rounding is done by brute force: the nearest of every
finite value of the format (ties to the even pattern, or the even integer),
saturating to the largest magnitude.

Output (tests/vectors/narrow.txt), one record per line:

  d <format> <raw hex> <value>       every one of the 256 FP8 patterns
                                     (value: exact fraction, inf, -inf, nan)
  q <format> <w hex> <s hex> <want>  Q(w / s) for binary32 w and s: an FP8
                                     pattern in hex or an INT8 integer
  dot <format> <k> <w hex>... <x hex>... <y hex>
                                     one output of a scaled product (see
                                     precision/narrow.rs ScaledMatrix::apply)

Usage: python3 -I scripts/gen_narrow_vectors.py > tests/vectors/narrow.txt
"""

import random
import struct
from fractions import Fraction

FORMATS = {"fp8-e4m3": (4, 3, 7), "fp8-e5m2": (5, 2, 15)}
QMAX = {"fp8-e4m3": Fraction(448), "fp8-e5m2": Fraction(57344), "int8": Fraction(127)}


def fp8_value(name, raw):
    e_bits, m_bits, bias = FORMATS[name]
    sign = -1 if raw & 0x80 else 1
    field = (raw >> m_bits) & ((1 << e_bits) - 1)
    frac = raw & ((1 << m_bits) - 1)
    if name == "fp8-e4m3" and field == 15 and frac == 7:
        return "nan"
    if name == "fp8-e5m2" and field == 31:
        if frac == 0:
            return "inf" if sign > 0 else "-inf"
        return "nan"
    if field == 0:
        mag = Fraction(frac, 1 << m_bits) * Fraction(2) ** (1 - bias)
    else:
        mag = (1 + Fraction(frac, 1 << m_bits)) * Fraction(2) ** (field - bias)
    return sign * mag


def finite_table(name):
    """(value, raw) for every finite non-NaN pattern except -0."""
    out = []
    for raw in range(256):
        v = fp8_value(name, raw)
        if isinstance(v, str) or (v == 0 and raw & 0x80):
            continue
        out.append((v, raw))
    return out


TABLES = {n: finite_table(n) for n in FORMATS}


def nearest_fp8(name, x, negative):
    """Brute-force round to nearest, ties to the even pattern, saturating."""
    best = None
    for v, raw in TABLES[name]:
        d = abs(x - v)
        key = (d, raw & 1)
        if best is None or key < best[0]:
            best = (key, v, raw)
    v, raw = best[1], best[2]
    if v == 0:
        return 0x80 if negative else 0x00
    return raw


def nearest_int8(x):
    lo = x.numerator // x.denominator
    frac = x - lo
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and lo % 2 == 1):
        n = lo + 1
    else:
        n = lo
    return max(-127, min(127, n))


def f32_value(bits):
    sign = -1 if bits >> 31 else 1
    field = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    assert field != 0xFF
    if field == 0:
        return sign * Fraction(frac) * Fraction(2) ** -149
    return sign * (Fraction(frac) + (1 << 23)) * Fraction(2) ** (field - 150)


def f32_bits(x):
    """Round a Fraction to binary32 (nearest, ties to even); returns bits."""
    if x == 0:
        return 0
    sign = 0x80000000 if x < 0 else 0
    m = abs(x)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    q = max(e, -126) - 23
    scaled = m / Fraction(2) ** q
    n = scaled.numerator // scaled.denominator
    r = scaled - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    v = n * Fraction(2) ** q
    if v >= Fraction(2) ** 128 - Fraction(2) ** 103:
        return sign | 0x7F800000
    if v < Fraction(2) ** -126:
        return sign | int(v / Fraction(2) ** -149)
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    frac = v / Fraction(2) ** e - 1
    return sign | ((e + 127) << 23) | int(frac * (1 << 23))


def r32(x):
    return f32_value(f32_bits(x))


def quantize(name, w, s):
    """Q(w / s) as an exact value, and the record field."""
    x = w / s
    neg = x < 0 or (x == 0 and w < 0)
    if name == "int8":
        n = nearest_int8(x)
        return Fraction(n), str(n)
    raw = nearest_fp8(name, x, neg)
    v = fp8_value(name, raw)
    return v, "%02x" % raw


def scale(name, xs):
    a = max(abs(v) for v in xs)
    return Fraction(1) if a == 0 else r32(a / QMAX[name])


def dot(name, w, x):
    sw, sx = scale(name, w), scale(name, x)
    qw = [quantize(name, v, sw)[0] for v in w]
    qx = [quantize(name, v, sx)[0] for v in x]
    if name == "int8":
        acc = r32(sum(a * b for a, b in zip(qw, qx)))
    else:
        acc = Fraction(0)
        for a, b in zip(qw, qx):
            acc = r32(acc + a * b)
    return f32_bits(r32(r32(acc * sw) * sx))


def bits_of(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def random_f32(rng, lo, hi):
    sign = rng.getrandbits(1) << 31
    e = rng.randint(lo, hi)
    return sign | ((e + 127) << 23) | rng.getrandbits(23)


def main():
    rng = random.Random(0x6E61_7272)
    lines = []
    for name in FORMATS:
        for raw in range(256):
            v = fp8_value(name, raw)
            text = v if isinstance(v, str) else (str(v.numerator) if v.denominator == 1 else "%d/%d" % (v.numerator, v.denominator))
            if v == 0 and not isinstance(v, str) and raw & 0x80:
                text = "-0"
            lines.append("d %s %02x %s" % (name, raw, text))
    names = list(FORMATS) + ["int8"]
    for name in names:
        # Random quotients across the range, below it and above it.
        for _ in range(1500):
            wb = random_f32(rng, -30, 20)
            sb = random_f32(rng, -25, 10) & 0x7FFFFFFF
            w, s = f32_value(wb), f32_value(sb)
            lines.append("q %s %08x %08x %s" % (name, wb, sb, quantize(name, w, s)[1]))
        # Realistic: s = fl32(absmax / Qmax) of a random row, w from the row.
        for _ in range(300):
            row = [f32_value(random_f32(rng, -8, 2)) for _ in range(8)]
            s = scale(name, row)
            sb = f32_bits(s)
            for w in row:
                lines.append("q %s %08x %08x %s" % (name, f32_bits(w), sb, quantize(name, w, s)[1]))
        # Exact ties and their binary32 neighbours under random power-of-two scales.
        grid = [v for v, _ in TABLES[name]] if name != "int8" else [Fraction(n) for n in range(-127, 128)]
        grid = sorted(set(grid))
        for _ in range(400):
            i = rng.randrange(len(grid) - 1)
            mid = (grid[i] + grid[i + 1]) / 2
            s = Fraction(2) ** rng.randint(-12, 4)
            wb = f32_bits(mid * s)
            for b in (wb, wb + 1, wb - 1):
                if (b & 0x7FFFFFFF) == 0 or b < 0:
                    continue
                lines.append("q %s %08x %08x %s" % (name, b, f32_bits(s), quantize(name, f32_value(b), s)[1]))
        # Scaled products of one output.
        for k in (1, 4, 16, 64):
            for _ in range(40):
                w = [f32_value(random_f32(rng, -6, 1)) for _ in range(k)]
                x = [f32_value(random_f32(rng, -10, 4)) for _ in range(k)]
                if rng.random() < 0.2:
                    x[rng.randrange(k)] = Fraction(0)
                y = dot(name, w, x)
                lines.append(
                    "dot %s %d %s %s %08x"
                    % (name, k, " ".join("%08x" % f32_bits(v) for v in w), " ".join("%08x" % f32_bits(v) for v in x), y)
                )
    print("# FP8/INT8 reference vectors: scripts/gen_narrow_vectors.py (Fraction, brute-force nearest)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

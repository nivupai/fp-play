#!/usr/bin/env python3
"""Generate reference vectors for the R3 decoder (runtime/src/decoder).

A stdlib-only reimplementation of the reduced decoder
(``DecoderConfig::REFERENCE``): deterministic BF16 weights, RMSNorm, RoPE,
grouped-query attention, SwiGLU, tied embeddings, greedy decoding over one
static batch. It shares no code with the crate.

Arithmetic. binary32 values are Python floats holding binary32 values.
Each binary32 ``+ - * /`` and ``sqrt`` is computed in binary64 and rounded
once to binary32 (``f32``, via ``struct``). That equals the correctly
rounded binary32 result because binary64 has at least 2*24 + 2 significant
bits (double rounding is then innocuous for these five operations). BF16
rounding is done separately with integer arithmetic (``round_bits``).
Every sum runs in the order the crate documents (``forward.rs``).

The transcendental functions (``exp``, ``sin_cos``, ``ln``) mirror the
crate's binary64 algorithms operation for operation; they are not
independent, and the crate tests them against the host library instead.

Usage: python3 -I scripts/gen_decoder_vectors.py > tests/vectors/decoder.txt
"""

import math
import struct
import sys

MASK64 = (1 << 64) - 1

# DecoderConfig::REFERENCE
CONFIG = dict(
    hidden=32,
    layers=2,
    q_heads=4,
    kv_heads=2,
    head_dim=8,
    ffn=48,
    vocab=64,
    rope_base=10000.0,
    rms_eps=1e-5,
    seed=0x00DEC0DE00F1,
    logit_scale=1.0,
)
PROMPT_SEED = 0x5EED_0003
PROMPTS = 4
PROMPT_LEN = 5
STEPS = 6


# ---------------------------------------------------------------- rounding


def f32(x):
    """Round a binary64 value to binary32 (nearest, ties to even)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def round_bits(x, precision, emin, emax):
    """Round the exact value of float x to a binary format with `precision`
    significant bits, minimum normal exponent `emin` and maximum exponent
    `emax`; nearest, ties to even. Returns a float."""
    if x != x or x == 0 or math.isinf(x):
        return x
    negative = x < 0
    a = -x if negative else x
    _, e = math.frexp(a)  # a = m * 2**e with 0.5 <= m < 1
    q = max(e - 1, emin) - (precision - 1)  # exponent of the last kept bit
    scaled = math.ldexp(a, -q)  # exact: a power-of-two scaling within range
    n = math.floor(scaled)
    rest = scaled - n  # exact
    if rest > 0.5 or (rest == 0.5 and n % 2 == 1):
        n += 1
    r = math.ldexp(n, q)
    if r >= math.ldexp(1.0, emax + 1):
        r = math.inf
    return -r if negative else r


def bf16(x):
    return round_bits(x, 8, -126, 127)


def bf16_bits(x):
    """BF16 bit pattern of a value already representable in BF16."""
    return f32_bits(x) >> 16


# ------------------------------------------------- deterministic binary64 math


def from_bits64(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


LOG2_E = from_bits64(0x3FF71547652B82FE)
LN2_HI = from_bits64(0x3FE62E42FEE00000)
LN2_LO = from_bits64(0x3DEA39EF35793C76)
TWO_OVER_PI = from_bits64(0x3FE45F306DC9C883)
PIO2_1 = from_bits64(0x3FF921FB54400000)
PIO2_2 = from_bits64(0x3DD0B4611A600000)
PIO2_3 = from_bits64(0x3BA3198A2E037073)
EXP_TERMS = 14
TRIG_TERMS = 10
SQRT_2 = from_bits64(0x3FF6A09E667F3BCD)
LN_2 = from_bits64(0x3FE62E42FEFA39EF)
LN_TERMS = 14


def inv_factorial(n):
    return 1.0 / float(math.factorial(n))


def exp64(x):
    assert abs(x) <= 700.0
    k = math.floor(x * LOG2_E + 0.5)
    r = (x - k * LN2_HI) - k * LN2_LO
    p = inv_factorial(EXP_TERMS - 1)
    for n in range(EXP_TERMS - 2, -1, -1):
        p = p * r + inv_factorial(n)
    return p * math.ldexp(1.0, k)


def sin_cos(a):
    assert 0.0 <= a <= 524288.0
    n = math.floor(a * TWO_OVER_PI + 0.5)
    r = ((a - n * PIO2_1) - n * PIO2_2) - n * PIO2_3
    r2 = r * r
    s = c = 0.0
    for k in range(TRIG_TERMS - 1, -1, -1):
        sign = 1.0 if k % 2 == 0 else -1.0
        s = s * r2 + sign * inv_factorial(2 * k + 1)
        c = c * r2 + sign * inv_factorial(2 * k)
    s = r * s
    q = n % 4
    return [(s, c), (c, -s), (-s, -c), (-c, s)][q]


def ln64(x):
    """Mirror of runtime/src/serving/detmath.rs (normal positive inputs)."""
    assert x > 0 and x >= 2.2250738585072014e-308
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    x = from_bits64((bits & ((1 << 52) - 1)) | (1023 << 52))
    if x >= SQRT_2:
        x /= 2.0
        exponent += 1
    s = (x - 1.0) / (x + 1.0)
    s2 = s * s
    poly = 0.0
    for k in range(LN_TERMS - 1, -1, -1):
        poly = poly * s2 + 1.0 / float(2 * k + 1)
    return float(exponent) * LN_2 + 2.0 * s * poly


def exp_f32(x):
    if x > 100.0:
        return math.inf
    if x < -110.0:
        return 0.0
    return f32(exp64(x))


# ---------------------------------------------------------------- weights


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def unit(self):
        return float(self.next_u64() >> 11) / float(1 << 53)

    def normalish(self):
        total = 0.0
        for _ in range(12):
            total += self.unit()
        return total - 6.0

    def below(self, n):
        return self.next_u64() % n


def tensor(seed, index, n, scale, offset):
    rng = SplitMix64(seed ^ ((index * 0x9E3779B97F4A7C15) & MASK64))
    return [bf16(offset + scale * rng.normalish()) for _ in range(n)]


def matrix(seed, index, rows, cols):
    """Row-major list of rows; scale 1/sqrt(cols)."""
    flat = tensor(seed, index, rows * cols, 1.0 / math.sqrt(float(cols)), 0.0)
    return [flat[o * cols:(o + 1) * cols] for o in range(rows)]


def weights(c):
    s, h = c["seed"], c["hidden"]
    kvw = c["kv_heads"] * c["head_dim"]
    flat = tensor(s, 0, c["vocab"] * h, 1.0 / math.sqrt(float(h)), 0.0)
    embed = [flat[v * h:(v + 1) * h] for v in range(c["vocab"])]
    layers = []
    for l in range(c["layers"]):
        i = 16 + 16 * l
        layers.append(dict(
            attn_norm=tensor(s, i, h, 0.1, 1.0),
            wq=matrix(s, i + 1, c["q_heads"] * c["head_dim"], h),
            wk=matrix(s, i + 2, kvw, h),
            wv=matrix(s, i + 3, kvw, h),
            wo=matrix(s, i + 4, h, c["q_heads"] * c["head_dim"]),
            mlp_norm=tensor(s, i + 5, h, 0.1, 1.0),
            w_gate=matrix(s, i + 6, c["ffn"], h),
            w_up=matrix(s, i + 7, c["ffn"], h),
            w_down=matrix(s, i + 8, h, c["ffn"]),
        ))
    return dict(embed=embed, layers=layers, final_norm=tensor(s, 1, h, 0.1, 1.0))


def checksum(w):
    """FNV-1a over every weight's BF16 bits, in generation order."""
    h = 0xCBF29CE484222325

    def eat(values):
        nonlocal h
        for v in values:
            for b in bf16_bits(v).to_bytes(2, "little"):
                h = ((h ^ b) * 0x100000001B3) & MASK64

    for row in w["embed"]:
        eat(row)
    for l in w["layers"]:
        eat(l["attn_norm"])
        for m in ("wq", "wk", "wv", "wo"):
            for row in l[m]:
                eat(row)
        eat(l["mlp_norm"])
        for m in ("w_gate", "w_up", "w_down"):
            for row in l[m]:
                eat(row)
    eat(w["final_norm"])
    return h


# ---------------------------------------------------------------- kernels


def dot_range(row, x, ks):
    """Sequential binary32 sum over ks from +0."""
    acc = 0.0
    for k in ks:
        acc = f32(acc + f32(row[k] * x[k]))
    return acc


def matvec(m, x):
    ks = range(len(x))
    return [dot_range(row, x, ks) for row in m]


def rms_norm(x, gain, eps):
    ss = 0.0
    for v in x:
        ss = f32(ss + f32(v * v))
    ms = f32(ss / float(len(x)))
    inv = f32(1.0 / f32(math.sqrt(f32(ms + eps))))
    return [f32(f32(v * inv) * g) for v, g in zip(x, gain)]


class Rope:
    def __init__(self, c):
        self.half = c["head_dim"] // 2
        lnb = ln64(c["rope_base"])
        self.theta = [exp64(-((float(2 * i) / float(c["head_dim"])) * lnb)) for i in range(self.half)]
        self.cache = {}

    def table(self, pos):
        if pos not in self.cache:
            cs = [sin_cos(float(pos) * t) for t in self.theta]
            self.cache[pos] = ([f32(c) for _, c in cs], [f32(s) for s, _ in cs])
        return self.cache[pos]

    def apply(self, x, pos, d):
        cos, sin = self.table(pos)
        out = list(x)
        for base in range(0, len(x), d):
            for i in range(self.half):
                a, b = x[base + i], x[base + i + self.half]
                out[base + i] = f32(f32(a * cos[i]) - f32(b * sin[i]))
                out[base + i + self.half] = f32(f32(b * cos[i]) + f32(a * sin[i]))
        return out


def top2(logits):
    best = 0
    for i, v in enumerate(logits):
        if v > logits[best]:
            best = i
    second = max(v for i, v in enumerate(logits) if i != best)
    return best, logits[best], second


# ------------------------------------------------- reductions (row-parallel)


class Unsharded:
    """P = 1: full sequential sum per output, rounded once to BF16 (both
    tested policies output BF16; from binary32 through the BF16 or binary32
    accumulator it is one rounding either way)."""

    def project(self, m, xs):
        return [[bf16(v) for v in matvec(m, x)] for x in xs]


# ---------------------------------------------------------------- forward


def step(c, w, rope, kv, tokens, reduce):
    """tokens: list of (seq, pos, token, want_logits). Returns logits rows."""
    h, d = c["hidden"], c["head_dim"]
    group = c["q_heads"] // c["kv_heads"]
    scale = f32(1.0 / f32(math.sqrt(float(d))))
    xs = [list(w["embed"][tok]) for (_, _, tok, _) in tokens]
    for li, layer in enumerate(w["layers"]):
        qs = []
        for i, (seq, pos, _, _) in enumerate(tokens):
            n = rms_norm(xs[i], layer["attn_norm"], c["rms_eps"])
            q = rope.apply(matvec(layer["wq"], n), pos, d)
            k = rope.apply(matvec(layer["wk"], n), pos, d)
            v = matvec(layer["wv"], n)
            kv.setdefault((seq, li), []).append(([bf16(a) for a in k], [bf16(a) for a in v]))
            assert len(kv[(seq, li)]) == pos + 1
            qs.append(q)
        attn = []
        for i, (seq, pos, _, _) in enumerate(tokens):
            out = [0.0] * h
            cache = kv[(seq, li)]
            for head in range(c["q_heads"]):
                kvh = head // group
                q = qs[i][head * d:(head + 1) * d]
                scores = []
                for j in range(pos + 1):
                    kj = cache[j][0][kvh * d:(kvh + 1) * d]
                    s = 0.0
                    for a, b in zip(q, kj):
                        s = f32(s + f32(a * b))
                    scores.append(f32(s * scale))
                mx = max(scores)
                total = 0.0
                es = []
                for s in scores:
                    e = exp_f32(f32(s - mx))
                    es.append(e)
                    total = f32(total + e)
                for j, e in enumerate(es):
                    p = f32(e / total)
                    vj = cache[j][1][kvh * d:(kvh + 1) * d]
                    for t in range(d):
                        out[head * d + t] = f32(out[head * d + t] + f32(p * vj[t]))
            attn.append(out)
        ys = reduce.project(layer["wo"], attn)
        xs = [[f32(a + b) for a, b in zip(x, y)] for x, y in zip(xs, ys)]
        acts = []
        for i in range(len(tokens)):
            n = rms_norm(xs[i], layer["mlp_norm"], c["rms_eps"])
            gate = matvec(layer["w_gate"], n)
            up = matvec(layer["w_up"], n)
            acts.append([f32(f32(g / f32(1.0 + exp_f32(-g))) * u) for g, u in zip(gate, up)])
        ys = reduce.project(layer["w_down"], acts)
        xs = [[f32(a + b) for a, b in zip(x, y)] for x, y in zip(xs, ys)]
    rows = []
    for i, (seq, pos, _, want) in enumerate(tokens):
        if want:
            n = rms_norm(xs[i], w["final_norm"], c["rms_eps"])
            rows.append([f32(z * c["logit_scale"]) for z in matvec(w["embed"], n)])
    return rows


def prompts(seed, count, length, vocab):
    rng = SplitMix64(seed)
    return [[rng.below(vocab) for _ in range(length)] for _ in range(count)]


def generate(c, w, ps, steps, reduce):
    rope = Rope(c)
    kv = {}
    gens = [[] for _ in ps]
    work = [(i, pos, t, pos + 1 == len(p)) for i, p in enumerate(ps) for pos, t in enumerate(p)]
    for s in range(steps):
        rows = step(c, w, rope, kv, work, reduce)
        for i, row in enumerate(rows):
            gens[i].append(row)
        work = [(i, len(ps[i]) + s, top2(gens[i][-1])[0], True) for i in range(len(ps))]
    return gens


def main():
    c = CONFIG
    w = weights(c)
    ps = prompts(PROMPT_SEED, PROMPTS, PROMPT_LEN, c["vocab"])
    out = sys.stdout
    out.write("# R3 decoder reference vectors from scripts/gen_decoder_vectors.py.\n")
    out.write("# config REFERENCE; prompts from seed %#x; %d steps.\n" % (PROMPT_SEED, STEPS))
    out.write("weights %016x\n" % checksum(w))
    for i, p in enumerate(ps):
        out.write("prompt %d %s\n" % (i, " ".join(str(t) for t in p)))
    runs = [("p1", Unsharded())]
    for name, reduce in runs:
        gens = generate(c, w, ps, STEPS, reduce)
        for i, rows in enumerate(gens):
            out.write("run %s seq %d tokens %s\n" % (name, i, " ".join(str(top2(r)[0]) for r in rows)))
            for s, row in enumerate(rows):
                out.write("run %s seq %d step %d logits %s\n" % (name, i, s, " ".join("%08x" % f32_bits(z) for z in row)))


if __name__ == "__main__":
    main()

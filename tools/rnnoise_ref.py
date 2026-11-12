"""Independent numpy reference of the RNNoise v0.1 algorithm (Valin, MMSP 2018) for validating the
Rust plugin `vi_rnnoise`. Our code, written from the published algorithm description and
constants; weights from models/rnnoise (tools/make_rnnoise.py). float32 arithmetic where the
Rust engine uses f32.

    python tools/rnnoise_ref.py models/rnnoise in48k.f32 out.f32
"""
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt

F = np.float32
Array = npt.NDArray[Any]
F32Array = npt.NDArray[np.float32]
C64Array = npt.NDArray[np.complex64]
Scalar = float | np.floating[Any]


class Layer(TypedDict):
    W: F32Array            # [inputs][gates * neurons]
    b: F32Array
    act: str
    n: int                 # neurons
    R: NotRequired[F32Array]   # GRU recurrent weights [neurons][3 * neurons]

FRAME = 480
WIN = 960
FREQ = 481
NB = 22
CEPS_MEM = 8
NDC = 6
NFEAT = NB + 3 * NDC + 2
PMIN, PMAX, PFRAME = 60, 768, 960
PBUF = PMAX + PFRAME
EBAND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]

HALF = np.array([np.sin(.5 * np.pi * np.sin(.5 * np.pi * (i + .5) / FRAME) ** 2) for i in range(FRAME)], F)
WINDOW = np.concatenate([HALF, HALF[::-1]])
DCT = np.array([[np.cos((i + .5) * j * np.pi / NB) * (np.sqrt(.5) if j == 0 else 1) for j in range(NB)] for i in range(NB)], F)


def band_energy(X: C64Array) -> F32Array:
    p = (X.real ** 2 + X.imag ** 2).astype(F)
    s = np.zeros(NB, F)
    for i in range(NB - 1):
        a, n = EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2
        fr = (np.arange(n) / n).astype(F)
        s[i] += np.sum((1 - fr) * p[a:a + n])
        s[i + 1] += np.sum(fr * p[a:a + n])
    s[0] *= 2
    s[-1] *= 2
    return s


def band_corr(X: C64Array, P: C64Array) -> F32Array:
    p = (X.real * P.real + X.imag * P.imag).astype(F)
    s = np.zeros(NB, F)
    for i in range(NB - 1):
        a, n = EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2
        fr = (np.arange(n) / n).astype(F)
        s[i] += np.sum((1 - fr) * p[a:a + n])
        s[i + 1] += np.sum(fr * p[a:a + n])
    s[0] *= 2
    s[-1] *= 2
    return s


def interp_gain(b: F32Array) -> F32Array:
    g = np.zeros(FREQ, F)
    for i in range(NB - 1):
        a, n = EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2
        fr = (np.arange(n) / n).astype(F)
        g[a:a + n] = (1 - fr) * b[i] + fr * b[i + 1]
    return g


def dct(x: F32Array) -> F32Array:
    return (x @ DCT * F(np.sqrt(2. / 22))).astype(F)


def fwd(x: F32Array) -> C64Array:
    return (np.fft.rfft(x.astype(np.float64)) / WIN).astype(np.complex64)


def inv(X: C64Array) -> F32Array:
    return (np.fft.irfft(X.astype(np.complex128), WIN) * WIN).astype(F)


def autocorr(x: F32Array, lag: int) -> F32Array:
    n = len(x)
    return np.array([np.dot(x[:n - k], x[k:]) for k in range(lag + 1)], F)


def lpc(ac: F32Array, p: int) -> F32Array:
    a = np.zeros(p, F)
    err = ac[0]
    if ac[0] != 0:
        for i in range(p):
            rr = F(0)
            for j in range(i):
                rr += a[j] * ac[i - j]
            rr += ac[i + 1]
            r = -rr / err
            a[i] = r
            for j in range((i + 1) >> 1):
                t1, t2 = a[j], a[i - 1 - j]
                a[j] = t1 + r * t2
                a[i - 1 - j] = t2 + r * t1
            err = err - r * r * err
            if err < F(.001) * ac[0]:
                break
    return a


def pitch_downsample(x: F32Array) -> F32Array:
    n = len(x) >> 1
    xl = np.zeros(n, F)
    xl[1:] = .5 * (.5 * (x[1:2 * n - 1:2] + x[3:2 * n + 1:2]) + x[2:2 * n:2])
    xl[0] = .5 * (.5 * x[1] + x[0])
    ac = autocorr(xl, 4)
    ac[0] *= F(1.0001)
    for i in range(1, 5):
        ac[i] -= ac[i] * (F(.008) * i) * (F(.008) * i)
    a = lpc(ac, 4)
    tmp = F(1)
    for i in range(4):
        tmp = F(.9) * tmp
        a[i] = a[i] * tmp
    c1 = F(.8)
    l2 = np.array([a[0] + F(.8), a[1] + c1 * a[0], a[2] + c1 * a[1], a[3] + c1 * a[2], c1 * a[3]], F)
    y = np.zeros(n, F)
    mem = np.zeros(5, F)
    for i in range(n):
        s = xl[i] + l2[0] * mem[0] + l2[1] * mem[1] + l2[2] * mem[2] + l2[3] * mem[3] + l2[4] * mem[4]
        mem[1:] = mem[:4].copy()
        mem[0] = xl[i]
        y[i] = s
    return y


def find_best_pitch(xc: F32Array, y: F32Array, ln: int, maxp: int) -> list[int]:
    syy = F(1)
    bn, bd, bp = [F(-1), F(-1)], [F(0), F(0)], [0, 1]
    syy = F(1) + np.sum(y[:ln] * y[:ln], dtype=F)
    for i in range(maxp):
        if xc[i] > 0:
            x16 = xc[i] * F(1e-12)
            num = x16 * x16
            if num * bd[1] > bn[1] * syy:
                if num * bd[0] > bn[0] * syy:
                    bn[1], bd[1], bp[1] = bn[0], bd[0], bp[0]
                    bn[0], bd[0], bp[0] = num, syy, i
                else:
                    bn[1], bd[1], bp[1] = num, syy, i
        syy += y[i + ln] * y[i + ln] - y[i] * y[i]
        syy = max(F(1), syy)
    return bp


def pitch_search(xl: F32Array, y: F32Array, ln: int, maxp: int) -> int:
    lag = ln + maxp
    x4 = xl[0:ln >> 1:2][:ln >> 2]
    y4 = y[0:lag >> 1:2][:lag >> 2]
    xc = np.array([np.dot(x4, y4[i:i + (ln >> 2)]) for i in range(maxp >> 2)], F)
    bp = find_best_pitch(xc, y4, ln >> 2, maxp >> 2)
    xc2 = np.zeros(maxp >> 1, F)
    for i in range(maxp >> 1):
        if abs(i - 2 * bp[0]) > 2 and abs(i - 2 * bp[1]) > 2:
            continue
        xc2[i] = max(F(-1), np.dot(xl[:ln >> 1], y[i:i + (ln >> 1)]))
    bp = find_best_pitch(xc2, y, ln >> 1, maxp >> 1)
    off = 0
    if 0 < bp[0] < (maxp >> 1) - 1:
        a, b, c = xc2[bp[0] - 1], xc2[bp[0]], xc2[bp[0] + 1]
        if (c - a) > F(.7) * (b - a):
            off = 1
        elif (a - c) > F(.7) * (b - c):
            off = -1
    return 2 * bp[0] - off


SECOND = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2]


def gain_(xy: Scalar, xx: Scalar, yy: Scalar) -> np.float32:
    return F(xy / np.sqrt(F(1) + xx * yy))


def remove_doubling(xfull: F32Array, maxp: int, minp: int, N: int, T0in: int, prevp: int, prevg: Scalar) -> tuple[int, np.float32]:
    minp0 = minp
    maxp //= 2
    minp //= 2
    T0in //= 2
    prevp //= 2
    N //= 2
    x = xfull  # index base: x[maxp + k]
    b = maxp
    if T0in >= maxp:
        T0in = maxp - 1
    T = T0 = T0in
    xx = np.dot(x[b:b + N], x[b:b + N])
    xy = np.dot(x[b:b + N], x[b - T0:b - T0 + N])
    yyl = np.zeros(maxp + 1, F)
    yyl[0] = xx
    yy = xx
    for i in range(1, maxp + 1):
        yy = yy + x[b - i] * x[b - i] - x[b + N - i] * x[b + N - i]
        yyl[i] = max(F(0), yy)
    yy = yyl[T0]
    bxy, byy = xy, yy
    g = g0 = gain_(xy, xx, yy)
    for k in range(2, 16):
        T1 = (2 * T0 + k) // (2 * k)
        if T1 < minp:
            break
        if k == 2:
            T1b = T0 if T1 + T0 > maxp else T0 + T1
        else:
            T1b = (2 * SECOND[k] * T0 + k) // (2 * k)
        xy = np.dot(x[b:b + N], x[b - T1:b - T1 + N])
        xy2 = np.dot(x[b:b + N], x[b - T1b:b - T1b + N])
        xy = F(.5) * (xy + xy2)
        yy = F(.5) * (yyl[T1] + yyl[T1b])
        g1 = gain_(xy, xx, yy)
        if abs(T1 - prevp) <= 1:
            cont = prevg
        elif abs(T1 - prevp) <= 2 and 5 * k * k < T0:
            cont = F(.5) * prevg
        else:
            cont = F(0)
        thresh = max(F(.3), F(.7) * g0 - cont)
        if T1 < 3 * minp:
            thresh = max(F(.4), F(.85) * g0 - cont)
        elif T1 < 2 * minp:
            thresh = max(F(.5), F(.9) * g0 - cont)
        if g1 > thresh:
            bxy, byy, T, g = xy, yy, T1, g1
    bxy = max(F(0), bxy)
    pg = F(1) if byy <= bxy else F(bxy / (byy + 1))
    xc = [np.dot(x[b:b + N], x[b - (T + k - 1):b - (T + k - 1) + N]) for k in range(3)]
    if (xc[2] - xc[0]) > F(.7) * (xc[1] - xc[0]):
        off = 1
    elif (xc[0] - xc[2]) > F(.7) * (xc[1] - xc[2]):
        off = -1
    else:
        off = 0
    pg = min(pg, g)
    T0out = max(2 * T + off, minp0)
    return T0out, pg


def sig(x: Array) -> F32Array:
    return (1 / (1 + np.exp(-x))).astype(F)


ACTIVATIONS: dict[str, Callable[[Array], Array]] = {"tanh": np.tanh, "sigmoid": sig, "relu": lambda v: np.maximum(v, 0)}


class RNNoise:
    def __init__(self, pkg: str | Path) -> None:
        pkg = Path(pkg)
        man = json.loads((pkg / "manifest.json").read_text())
        w = np.fromfile(pkg / "weights.bin", "<f4")
        self.L: dict[str, Layer] = {}
        for e in man["config"]["layers"]:
            t = {k: w[v["offset"]:v["offset"] + v["size"]] for k, v in e["tensors"].items()}
            g = 3 if "recurrent_weights" in t else 1
            ent: Layer = {"W": t["weights"].reshape(e["inputs"], g * e["neurons"]), "b": t["bias"], "act": e["activation"], "n": e["neurons"]}
            if g == 3:
                ent["R"] = t["recurrent_weights"].reshape(e["neurons"], 3 * e["neurons"])
            self.L[e["name"]] = ent
        self.scale: float = man["config"].get("input_scale", 32768.0)
        self.reset()

    def reset(self) -> None:
        self.analysis = np.zeros(FRAME, F)
        self.ceps = np.zeros((CEPS_MEM, NB), F)
        self.memid = 0
        self.synth = np.zeros(FRAME, F)
        self.pbuf = np.zeros(PBUF, F)
        self.last_gain = F(0)
        self.last_period = 0
        self.hp: list[float] = [0.0, 0.0]
        self.lastg = np.zeros(NB, F)
        self.h = {k: np.zeros(v["n"], F) for k, v in self.L.items() if "R" in v}

    def act(self, x: Array, a: str) -> F32Array:
        return ACTIVATIONS[a](x).astype(F)

    def dense(self, name: str, x: Array) -> F32Array:
        L = self.L[name]
        return self.act(x @ L["W"] + L["b"], L["act"])

    def gru(self, name: str, x: Array) -> F32Array:
        L = self.L[name]
        n, h = L["n"], self.h[name]
        gx = x @ L["W"] + L["b"]
        z = sig(gx[:n] + h @ L["R"][:, :n])
        r = sig(gx[n:2 * n] + h @ L["R"][:, n:2 * n])
        c = self.act(gx[2 * n:] + (r * h) @ L["R"][:, 2 * n:], L["act"])
        self.h[name] = (z * h + (1 - z) * c).astype(F)
        return self.h[name]

    def rnn(self, f: F32Array) -> tuple[F32Array, F32Array]:
        d = self.dense("input_dense", f)
        v = self.gru("vad_gru", d)
        vad = self.dense("vad_output", v)
        n = self.gru("noise_gru", np.concatenate([d, v, f]))
        dn = self.gru("denoise_gru", np.concatenate([v, n, f]))
        return self.dense("denoise_output", dn), vad

    def frame(self, inp: F32Array) -> F32Array:
        # high-pass biquad (b = [-2, 1], a = [-1.99599, 0.99600]) in double, stored as float
        x = np.zeros(FRAME, F)
        m0, m1 = self.hp
        for i in range(FRAME):
            xi = float(inp[i])
            yi = xi + m0
            m0 = m1 + (-2.0 * xi - (-1.99599) * yi)
            m1 = (1.0 * xi - 0.99600 * yi)
            m0, m1 = float(F(m0)), float(F(m1))
            x[i] = yi
        self.hp = [m0, m1]
        # analysis
        buf = np.concatenate([self.analysis, x])
        self.analysis = x.copy()
        X = fwd(buf * WINDOW)
        Ex = band_energy(X)
        self.pbuf = np.concatenate([self.pbuf[FRAME:], x])
        pl = pitch_downsample(self.pbuf)
        pi = pitch_search(pl[PMAX >> 1:], pl, PFRAME, PMAX - 3 * PMIN)
        pi = PMAX - pi
        pi, g = remove_doubling(pl, PMAX, PMIN, PFRAME, pi, self.last_period, self.last_gain)
        self.last_period, self.last_gain = pi, g
        p = self.pbuf[PBUF - WIN - pi:PBUF - pi]
        P = fwd(p * WINDOW)
        Ep = band_energy(P)
        Exp = band_corr(X, P) / np.sqrt(F(.001) + Ex * Ep)
        feat = np.zeros(NFEAT, F)
        t = dct(Exp.astype(F))
        feat[NB + 2 * NDC:NB + 3 * NDC] = t[:NDC]
        feat[NB + 2 * NDC] -= F(1.3)
        feat[NB + 2 * NDC + 1] -= F(.9)
        feat[NB + 3 * NDC] = F(.01) * (pi - 300)
        Ly = np.zeros(NB, F)
        logmax, follow, E = F(-2), F(-2), F(0)
        for i in range(NB):
            v = F(np.log10(F(1e-2) + Ex[i]))
            v = max(logmax - 7, max(follow - F(1.5), v))
            Ly[i] = v
            logmax = max(logmax, v)
            follow = max(follow - F(1.5), v)
            E += Ex[i]
        if E < F(0.04):
            Y = X  # silence: keep spectrum, state untouched
        else:
            c = dct(Ly)
            c[0] -= 12
            c[1] -= 4
            self.ceps[self.memid] = c
            c1 = self.ceps[(self.memid - 1) % CEPS_MEM]
            c2 = self.ceps[(self.memid - 2) % CEPS_MEM]
            c0 = c
            feat[:NB] = c
            feat[:NDC] = c0[:NDC] + c1[:NDC] + c2[:NDC]
            feat[NB:NB + NDC] = c0[:NDC] - c2[:NDC]
            feat[NB + NDC:NB + 2 * NDC] = c0[:NDC] - 2 * c1[:NDC] + c2[:NDC]
            self.memid = (self.memid + 1) % CEPS_MEM
            sv = F(0)
            for i in range(CEPS_MEM):
                d = np.sum((self.ceps[i][None, :] - self.ceps) ** 2, axis=1, dtype=F)
                d[i] = F(1e15)
                sv += min(F(1e15), np.min(d))
            feat[NB + 3 * NDC + 1] = sv / CEPS_MEM - F(2.1)
            gains, vad = self.rnn(feat)
            # pitch filter
            r = (np.sqrt(np.clip(np.where(Exp > gains, F(1), (Exp ** 2) * (1 - gains ** 2) / (F(.001) + gains ** 2 * (1 - Exp ** 2))), 0, 1)) * np.sqrt(Ex / (F(1e-8) + Ep))).astype(F)
            rf = interp_gain(r)
            X = (X + rf * P).astype(np.complex64)
            newE = band_energy(X)
            nf = interp_gain(np.sqrt(Ex / (F(1e-8) + newE)).astype(F))
            X = (X * nf).astype(np.complex64)
            gains = np.maximum(gains, F(.6) * self.lastg).astype(F)
            self.lastg = gains
            Y = (X * interp_gain(gains)).astype(np.complex64)
        y = inv(Y) * WINDOW
        out: F32Array = y[:FRAME] + self.synth
        self.synth = y[FRAME:].copy()
        return out

    def process(self, x: Array) -> F32Array:
        x = np.concatenate([np.asarray(x, F) * F(self.scale), np.zeros((-len(x)) % FRAME, F)])
        out = np.zeros(len(x), F)
        for i in range(0, len(x), FRAME):
            out[i:i + FRAME] = self.frame(x[i:i + FRAME])
        return out / F(self.scale)


if __name__ == "__main__":
    import time
    r = RNNoise(sys.argv[1])
    x = np.fromfile(sys.argv[2], "<f4")
    t = time.time()
    y = r.process(x)
    print(f"rnnoise ref: {len(x)} samples in {time.time() - t:.1f}s")
    y.astype("<f4").tofile(sys.argv[3])

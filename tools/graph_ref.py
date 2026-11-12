"""Reference executor for vi model packages (models/FORMAT.md) — our own numpy code.

Independent second implementation of the `graph` engine (graph ops + pipeline DSP) used to
validate the Rust engine, and by the converters (constant folding, shape checks).

    python tools/graph_ref.py models/<name> in.wav|in.f32 out.f32 [--T N]
    from tools.graph_ref import Package; p = Package("models/gtcrn"); y = p.process(x)
"""
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = np.float32
Array = npt.NDArray[Any]          # any dtype (inputs, folded constants)
F32Array = npt.NDArray[np.float32]
F64Array = npt.NDArray[np.float64]
Attrs = dict[str, Any]            # op attributes as in graph.json / ONNX: ints, floats, strings and lists of them
Json = dict[str, Any]             # parsed graph.json / manifest.json (models/FORMAT.md)
Inputs = Sequence[Array | None]   # op inputs (None = omitted optional input)

# ------------------------------------------------------------------ dims


def dim(v: int | float | str | np.integer[Any], T: int) -> int:
    """int or affine string in T ("40*T+40") -> int."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        return int(v)
    s = v.replace(" ", "")
    tot = 0
    for sign, term in re.findall(r"([+-]?)([^+-]+)", s):
        k = -1 if sign == "-" else 1
        if term == "T":
            tot += k * T
        elif term.endswith("*T"):
            tot += k * int(term[:-2]) * T
        elif term.startswith("T*"):
            tot += k * int(term[2:]) * T
        else:
            tot += k * int(term)
    return tot


def dims(v: Sequence[int | float | str], T: int) -> list[int]:
    return [dim(x, T) for x in v]


# ------------------------------------------------------------------ ops


def _axes(a: Sequence[int], nd: int) -> list[int]:
    return [x % nd for x in a]


def _sig(x: Array) -> F32Array:
    return (1.0 / (1.0 + np.exp(-x))).astype(F32)


def conv(x: Array, w: Array, b: Array | None, a: Attrs) -> F32Array:
    """ONNX Conv, N-C-spatial, 1-D or 2-D, direct im2col."""
    nd = w.ndim - 2
    ks = list(w.shape[2:])
    st = a.get("strides", [1] * nd)
    dl = a.get("dilations", [1] * nd)
    pads = a.get("pads", [0] * (2 * nd))
    g = a.get("group", 1)
    padw = [(0, 0), (0, 0)] + [(pads[i], pads[i + nd]) for i in range(nd)]
    x = np.pad(x, padw)
    N, C = x.shape[:2]
    sp = x.shape[2:]
    out_sp = [(sp[i] - dl[i] * (ks[i] - 1) - 1) // st[i] + 1 for i in range(nd)]
    co = w.shape[0]
    cig = C // g
    cog = co // g
    y = np.zeros([N, co] + out_sp, F32)
    if nd == 1:
        x = x[:, :, None, :]
        w4 = w[:, :, None, :]
        ks4, st4, dl4, osp = [1] + ks, [1] + st, [1] + dl, [1] + out_sp
        y4 = y[:, :, None, :]
    else:
        w4, ks4, st4, dl4, osp, y4 = w, ks, st, dl, out_sp, y
    for gi in range(g):
        xs = x[:, gi * cig:(gi + 1) * cig]
        cols = np.empty([N, cig, ks4[0], ks4[1], osp[0], osp[1]], F32)
        for i in range(ks4[0]):
            for j in range(ks4[1]):
                h0, w0 = i * dl4[0], j * dl4[1]
                cols[:, :, i, j] = xs[:, :, h0:h0 + st4[0] * (osp[0] - 1) + 1:st4[0], w0:w0 + st4[1] * (osp[1] - 1) + 1:st4[1]]
        wm = w4[gi * cog:(gi + 1) * cog].reshape(cog, -1)
        cm = cols.reshape(N, cig * ks4[0] * ks4[1], osp[0] * osp[1])
        y4[:, gi * cog:(gi + 1) * cog] = np.einsum("ok,nkp->nop", wm, cm).reshape(N, cog, osp[0], osp[1])
    if b is not None:
        y += b.reshape([1, co] + [1] * nd)
    return y.astype(F32)


def conv_transpose(x: Array, w: Array, b: Array | None, a: Attrs) -> F32Array:
    nd = w.ndim - 2
    ks = list(w.shape[2:])
    st = a.get("strides", [1] * nd)
    dl = a.get("dilations", [1] * nd)
    pads = a.get("pads", [0] * (2 * nd))
    opad = a.get("output_padding", [0] * nd)
    g = a.get("group", 1)
    N, C = x.shape[:2]
    sp = list(x.shape[2:])
    cog = w.shape[1]
    co = cog * g
    cig = C // g
    full = [(sp[i] - 1) * st[i] + dl[i] * (ks[i] - 1) + 1 + opad[i] for i in range(nd)]
    if nd == 1:
        x4, w4, ks4, st4, dl4, sp4, full4 = x[:, :, None], w[:, :, None], [1] + ks, [1] + st, [1] + dl, [1] + sp, [1] + full
    else:
        x4, w4, ks4, st4, dl4, sp4, full4 = x, w, ks, st, dl, sp, full
    y = np.zeros([N, co] + full4, F32)
    for gi in range(g):
        xs = x4[:, gi * cig:(gi + 1) * cig]
        ws = w4[gi * cig:(gi + 1) * cig]  # [cig, cog, kh, kw]
        for i in range(ks4[0]):
            for j in range(ks4[1]):
                contrib = np.einsum("nchw,co->nohw", xs, ws[:, :, i, j])
                h0, w0 = i * dl4[0], j * dl4[1]
                y[:, gi * cog:(gi + 1) * cog, h0:h0 + st4[0] * (sp4[0] - 1) + 1:st4[0], w0:w0 + st4[1] * (sp4[1] - 1) + 1:st4[1]] += contrib
    if nd == 1:
        y = y[:, :, 0]
    sl = [slice(None), slice(None)] + [slice(pads[i], full[i] - pads[i + nd]) for i in range(nd)]
    y = y[tuple(sl)]
    if b is not None:
        y = y + b.reshape([1, co] + [1] * nd)
    return np.ascontiguousarray(y, F32)


def _rnn(ins: Inputs, a: Attrs, lstm: bool) -> list[F32Array]:
    X, W, R = _need(ins, 0, "X"), _need(ins, 1, "W"), _need(ins, 2, "R")
    B = ins[3] if len(ins) > 3 else None
    h0 = ins[5] if len(ins) > 5 else None
    c0 = ins[6] if len(ins) > 6 else None
    H = a["hidden_size"]
    lbr = a.get("linear_before_reset", 0)
    D = 2 if a.get("direction", "forward") == "bidirectional" else 1
    S, Bt, _ = X.shape
    G = 4 if lstm else 3
    Y = np.zeros((S, D, Bt, H), F32)
    Yh = np.zeros((D, Bt, H), F32)
    Yc = np.zeros((D, Bt, H), F32)
    for d in range(D):
        Wd, Rd = W[d], R[d]
        Bv = B[d] if B is not None else np.zeros(2 * G * H, F32)
        Wb, Rb = Bv[:G * H], Bv[G * H:]
        h = h0[d].copy() if h0 is not None else np.zeros((Bt, H), F32)
        c = c0[d].copy() if c0 is not None else np.zeros((Bt, H), F32)
        steps = range(S) if d == 0 else range(S - 1, -1, -1)
        for t in steps:
            gx = X[t] @ Wd.T + Wb
            if lstm:
                g = gx + h @ Rd.T + Rb  # i, o, f, c
                i, o, f, cc = _sig(g[:, :H]), _sig(g[:, H:2 * H]), _sig(g[:, 2 * H:3 * H]), np.tanh(g[:, 3 * H:])
                c = (f * c + i * cc).astype(F32)
                h = (o * np.tanh(c)).astype(F32)
            else:
                z = _sig(gx[:, :H] + h @ Rd[:H].T + Rb[:H])
                r = _sig(gx[:, H:2 * H] + h @ Rd[H:2 * H].T + Rb[H:2 * H])
                if lbr:
                    hh = np.tanh(gx[:, 2 * H:] + r * (h @ Rd[2 * H:].T + Rb[2 * H:]))
                else:
                    hh = np.tanh(gx[:, 2 * H:] + (r * h) @ Rd[2 * H:].T + Rb[2 * H:])
                h = ((1 - z) * hh + z * h).astype(F32)
            Y[t, d] = h
        Yh[d], Yc[d] = h, c
    return [Y, Yh, Yc] if lstm else [Y, Yh]


def gru(ins: Inputs, a: Attrs) -> list[F32Array]:
    return _rnn(ins, a, False)


def lstm(ins: Inputs, a: Attrs) -> list[F32Array]:
    return _rnn(ins, a, True)


def _slice(x: Array, a: Attrs) -> Array:
    axes = a.get("axes", list(range(len(a["starts"]))))
    steps = a.get("steps", [1] * len(axes))
    for s, e, ax, st in zip(a["starts"], a["ends"], axes, steps):
        ax %= x.ndim
        idx = list(range(*slice(int(s), int(e), int(st)).indices(x.shape[ax])))  # explicit indices (negative steps)
        x = np.take(x, idx, axis=ax)
    return x


def _reshape(x: Array, shape: Sequence[int]) -> Array:
    shape = list(shape)
    for i, s in enumerate(shape):
        if s == 0:
            shape[i] = x.shape[i]
    return x.reshape(shape)


def _pad(x: Array, a: Attrs) -> Array:
    p = a["pads"]
    nd = x.ndim
    pw = [(p[i], p[i + nd]) for i in range(nd)]
    mode = a.get("mode", "constant")
    if any(v < 0 for v in p):  # negative pads crop
        sl = tuple(slice(max(0, -pw[i][0]), x.shape[i] - max(0, -pw[i][1])) for i in range(nd))
        x = x[sl]
        pw = [(max(0, b), max(0, e)) for b, e in pw]
    if mode == "constant":
        return np.pad(x, pw, constant_values=a.get("value", 0.0))
    if mode == "reflect":
        return np.pad(x, pw, mode="reflect")
    if mode == "edge":
        return np.pad(x, pw, mode="edge")
    raise KeyError(mode)


def _scatter_nd(data: Array, idx: Array, upd: Array) -> Array:
    out = data.copy()
    idx = idx.astype(np.int64)
    k = idx.shape[-1]
    flat = idx.reshape(-1, k)
    u = upd.reshape((flat.shape[0],) + data.shape[k:])
    for i, ix in enumerate(flat):
        out[tuple(ix)] = u[i]
    return out


def channel_norm(x: Array, g: Array, b: Array, a: Attrs) -> F32Array:
    ax = a.get("axis", 1) % x.ndim
    m = x.mean(axis=ax, keepdims=True, dtype=F32)
    d = x - m
    v = (d * d).mean(axis=ax, keepdims=True, dtype=F32)
    sd = np.sqrt(v + F32(a.get("epsilon", 1e-5)))
    shp = [1] * x.ndim
    shp[ax] = x.shape[ax]
    return (g.reshape(shp) * d / sd + b.reshape(shp)).astype(F32)


def layer_norm(x: Array, g: Array, b: Array | None, a: Attrs) -> F32Array:
    ax = a.get("axis", -1) % x.ndim
    axes = tuple(range(ax, x.ndim))
    m = x.mean(axis=axes, keepdims=True)
    d = x - m
    v = (d * d).mean(axis=axes, keepdims=True)
    y = d / np.sqrt(v + F32(a.get("epsilon", 1e-5))) * g
    if b is not None:
        y = y + b
    return y.astype(F32)


def _need(ins: Inputs, i: int, what: str) -> Array:
    """A required op input."""
    v = ins[i] if i < len(ins) else None
    if v is None:
        raise ValueError(f"missing input {i} ({what})")
    return v


def _opt(ins: Inputs, i: int) -> Array | None:
    """An optional op input (absent or omitted)."""
    return ins[i] if i < len(ins) else None


UNARY: dict[str, Callable[[Array], Array]] = {
    "Relu": lambda v: np.maximum(v, 0), "Sigmoid": _sig, "Tanh": np.tanh, "Sqrt": np.sqrt, "Abs": np.abs,
    "Exp": np.exp, "Log": np.log, "Neg": np.negative, "Reciprocal": lambda v: 1 / v, "Identity": lambda v: v.copy(),
    "Cast": lambda v: v.astype(F32),
    "Softplus": lambda v: np.log1p(np.exp(v))}
BINARY: dict[str, Callable[[Array, Array], Array]] = {
    "Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide, "Pow": np.power,
    "Max": np.maximum, "Min": np.minimum}
REDUCE: dict[str, Callable[..., Any]] = {"ReduceMean": np.mean, "ReduceSum": np.sum, "ReduceMax": np.max}


def run_op(op: str, ins: Inputs, a: Attrs) -> list[Array]:
    """Execute one op. ins: list of np arrays (None = omitted). Returns list of outputs."""
    if op == "Concat":
        return [np.concatenate([v for v in ins if v is not None and v.size > 0] or ins, axis=a["axis"]).astype(F32)]
    if op == "Einsum":  # reference only (converters lower Einsum to MatMul/Transpose/Reshape)
        return [np.einsum(a["equation"], *(_need(ins, i, "operand") for i in range(len(ins)))).astype(F32)]
    x = _need(ins, 0, "X")
    if op in UNARY:
        return [UNARY[op](x).astype(F32)]
    if op in BINARY:
        return [BINARY[op](x, _need(ins, 1, "B")).astype(F32)]
    if op == "LeakyRelu":
        al = F32(a.get("alpha", 0.01))
        return [(np.maximum(x, 0) + al * np.minimum(x, 0)).astype(F32)]
    if op == "PRelu":
        return [(np.maximum(x, 0) + _need(ins, 1, "slope") * np.minimum(x, 0)).astype(F32)]
    if op == "Elu":
        al = a.get("alpha", 1.0)
        return [np.where(x > 0, x, al * (np.exp(x) - 1)).astype(F32)]
    if op == "HardSigmoid":
        return [np.clip(a.get("alpha", 0.2) * x + a.get("beta", 0.5), 0, 1).astype(F32)]
    if op == "Clip":
        return [np.clip(x, a.get("min", -np.inf), a.get("max", np.inf)).astype(F32)]
    if op == "MatMul":
        return [np.matmul(x, _need(ins, 1, "B")).astype(F32)]
    if op == "Gemm":
        A = x.T if a.get("transA", 0) else x
        B = _need(ins, 1, "B").T if a.get("transB", 0) else _need(ins, 1, "B")
        y = a.get("alpha", 1.0) * (A @ B)
        c = _opt(ins, 2)
        if c is not None:
            y = y + a.get("beta", 1.0) * c
        return [y.astype(F32)]
    if op in ("Conv", "ConvTranspose", "StreamConv"):
        fn = conv if op != "ConvTranspose" else conv_transpose
        cl = a.get("channels_last", 0)
        xx = np.ascontiguousarray(np.swapaxes(x, 1, 2)) if cl else x
        y = fn(xx, _need(ins, 1, "W"), _opt(ins, 2), a)
        return [np.ascontiguousarray(np.swapaxes(y, 1, 2)) if cl else y]
    if op == "GRU":
        return list(gru(ins, a))
    if op == "LSTM":
        return list(lstm(ins, a))
    if op == "BatchNormalization":
        sc, bb, mu, var = (_need(ins, i, "scale, B, mean, var") for i in range(1, 5))
        shp = [1, -1] + [1] * (x.ndim - 2)
        return [((x - mu.reshape(shp)) / np.sqrt(var.reshape(shp) + a.get("epsilon", 1e-5)) * sc.reshape(shp) + bb.reshape(shp)).astype(F32)]
    if op == "ChannelNorm":
        return [channel_norm(x, _need(ins, 1, "scale"), _need(ins, 2, "B"), a)]
    if op == "LayerNormalization":
        return [layer_norm(x, _need(ins, 1, "scale"), _opt(ins, 2), a)]
    if op == "InstanceNormalization":
        axes = tuple(range(2, x.ndim))
        m = x.mean(axis=axes, keepdims=True)
        v = ((x - m) ** 2).mean(axis=axes, keepdims=True)
        shp = [1, -1] + [1] * (x.ndim - 2)
        return [((x - m) / np.sqrt(v + a.get("epsilon", 1e-5)) * _need(ins, 1, "scale").reshape(shp) + _need(ins, 2, "B").reshape(shp)).astype(F32)]
    if op in REDUCE:
        rax = a.get("axes")
        kd = bool(a.get("keepdims", 1))
        return [np.asarray(REDUCE[op](x, axis=tuple(_axes(rax, x.ndim)) if rax is not None else None, keepdims=kd), F32)]
    if op == "Split":
        ax = a.get("axis", 0)
        sp = a["split"]
        return [s.copy() for s in np.split(x, np.cumsum(sp)[:-1], axis=ax)]
    if op == "Slice":
        return [_slice(x, a).copy()]
    if op == "Gather":
        return [np.take(x, _need(ins, 1, "indices").astype(np.int64), axis=a.get("axis", 0)).astype(F32)]
    if op == "ScatterND":
        return [_scatter_nd(x, _need(ins, 1, "indices"), _need(ins, 2, "updates"))]
    if op == "Reshape":
        return [_reshape(x, a["shape"]).copy()]
    if op == "Transpose":
        return [np.ascontiguousarray(np.transpose(x, a.get("perm")))]
    if op == "Squeeze":
        sax = a.get("axes")
        return [np.squeeze(x, axis=tuple(_axes(sax, x.ndim)) if sax is not None else None).copy()]
    if op == "Unsqueeze":
        y = x
        nd = x.ndim + len(a["axes"])
        for ax in sorted(_axes(a["axes"], nd)):
            y = np.expand_dims(y, ax)
        return [y.copy()]
    if op == "Flatten":
        ax = a.get("axis", 1) % max(x.ndim, 1)
        return [x.reshape(int(np.prod(x.shape[:ax])), -1).copy()]
    if op == "Pad":
        return [_pad(x, a).astype(F32)]
    if op == "Expand":
        return [(x * np.ones(a["shape"], F32)).astype(F32)]
    if op == "Tile":
        return [np.tile(x, a["repeats"]).astype(F32)]
    if op == "MaxPool":
        k: list[int] = a["kernel_shape"]
        st: list[int] = a.get("strides", k)
        pads = a.get("pads", [0] * (2 * len(k)))
        nd = len(k)
        xp = np.pad(x, [(0, 0), (0, 0)] + [(pads[i], pads[i + nd]) for i in range(nd)], constant_values=-np.inf)
        osz = [(xp.shape[2 + i] - k[i]) // st[i] + 1 for i in range(nd)]
        if nd == 1:
            y = np.stack([xp[:, :, o * st[0]:o * st[0] + k[0]].max(-1) for o in range(osz[0])], -1)
        else:
            y = np.empty(list(x.shape[:2]) + osz, F32)
            for i in range(osz[0]):
                for j in range(osz[1]):
                    y[:, :, i, j] = xp[:, :, i * st[0]:i * st[0] + k[0], j * st[1]:j * st[1] + k[1]].max(axis=(2, 3))
        return [y.astype(F32)]
    if op == "Rfft":
        X = np.fft.rfft(x.astype(np.float64), axis=-1)
        return [np.stack([X.real, X.imag], -1).astype(F32)]
    if op == "Irfft":
        Z = x[..., 0].astype(np.float64) + 1j * x[..., 1]
        return [np.fft.irfft(Z, n=a["n"], axis=-1).astype(F32)]
    raise NotImplementedError(op)


# ------------------------------------------------------------------ graph

INT_ATTRS = ("shape", "starts", "ends")


class Graph:
    def __init__(self, g: Json, weights: F32Array, T: int = 1) -> None:
        self.g, self.T = g, T
        self.consts: dict[str, Array] = {}
        for name, d in g["initializers"].items():
            shp = dims(d["shape"], T)
            if "value" in d:
                v = np.array(d["value"], F32).reshape(shp)
            else:
                n = int(np.prod(shp)) if shp else 1
                v = weights[d["offset"]:d["offset"] + n].reshape(shp)
            self.consts[name] = v
        self.inputs: dict[str, list[int]] = {i["name"]: dims(i["shape"], T) for i in g["inputs"]}
        self.states: list[tuple[str, str]] = [(s["in"], s["out"]) for s in g.get("states", [])]
        self.nodes: list[tuple[str, list[str], list[str], Attrs]] = []
        for n in g["nodes"]:
            a = dict(n.get("attrs", {}))
            for k in INT_ATTRS:
                if k in a and isinstance(a[k], list):
                    a[k] = dims(a[k], T)
            self.nodes.append((n["op"], n["inputs"], n["outputs"], a))
        self.state_vals: dict[str, Array] = {}
        self.reset()

    def reset(self) -> None:
        self.state_vals = {si: np.zeros(self.inputs[si], F32) for si, _ in self.states}

    def run(self, feeds: dict[str, Array], keep: bool = False) -> dict[str, Array]:
        env = dict(self.consts)
        env.update(self.state_vals)
        env.update({k: np.asarray(v, F32).reshape(self.inputs[k]) for k, v in feeds.items()})
        for op, ins, outs, a in self.nodes:
            vals = run_op(op, [env[i] if i else None for i in ins], a)
            for o, v in zip(outs, vals):
                if o:
                    env[o] = v
        for si, so in self.states:
            assert env[so].shape == env[si].shape, (si, so, env[so].shape, env[si].shape)
            self.state_vals[si] = env[so]
        if keep:
            return env
        return {o["name"]: env[o["name"]] for o in self.g["outputs"]}


# ------------------------------------------------------------------ pipeline DSP


def window(spec: str | Json | None, n: int, consts: dict[str, Array] | None = None) -> F64Array:
    if spec in (None, "none", "rect"):
        return np.ones(n, np.float64)
    if isinstance(spec, dict):
        if "values" in spec:
            return np.array(spec["values"], np.float64)
        if consts is None:
            raise ValueError(f"window {spec}: a tensor window needs the graph's constants")
        return consts[spec["tensor"]].astype(np.float64).reshape(-1)
    k = np.arange(n)
    if spec == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    if spec == "hann_sym":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    if spec == "sqrt_hann":
        return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * k / n))
    if spec == "sqrt_hann_sym":
        return np.sqrt(np.maximum(0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1)), 0))
    if spec == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    if spec == "vorbis":
        return np.sin(np.pi / 2 * np.sin(np.pi * (k + 0.5) / n) ** 2)
    raise ValueError(spec)


class Package:
    # set only for framed front-ends ("frames", "stft") / framed back-ends ("ola", "istft")
    flen: int
    nfft: int
    wa: F64Array
    hist: F64Array
    blen: int
    ws: F64Array
    norm: F64Array
    ola: F64Array

    def __init__(self, path: str | Path, T: int | None = None) -> None:
        path = Path(path)
        self.man: Json = json.loads((path / "manifest.json").read_text())
        g = json.loads((path / self.man.get("graph", "graph.json")).read_text())
        w = np.fromfile(path / self.man.get("weights", "weights.bin"), "<f4") if (path / self.man.get("weights", "weights.bin")).exists() else np.zeros(0, F32)
        p: Json = self.man["pipeline"]
        self.p = p
        cfg = self.man.get("config", {})
        self.T: int = T or cfg.get("frames_per_call") or p.get("frames_per_call", 1)
        self.graph = Graph(g, w, self.T)
        self.hop: int = p["hop"]
        self.fs = self.T * self.hop
        self.front: Json = p["front"]
        self.back: Json = p.get("back", {"type": "waveform"})
        c = self.graph.consts
        ft = self.front["type"]
        if ft in ("frames", "stft"):
            self.flen = self.front.get("frame_len", self.front.get("n_fft"))
            self.nfft = self.front.get("n_fft", self.flen)
            self.wa = window(self.front.get("window"), self.flen, c)
        bt = self.back["type"]
        if bt in ("ola", "istft"):
            self.blen = self.back.get("frame_len", getattr(self, "nfft", None))
            self.ws = window(self.back.get("window"), self.blen, c)
            self.norm = np.ones(self.blen)
            if self.back.get("ola_norm", "none") == "wola":
                wa = self.wa if len(self.wa) == self.blen else np.ones(self.blen)
                env = np.zeros(self.blen)
                prod = wa * self.ws
                for s in range(-self.blen, self.blen, self.hop):
                    lo, hi = max(0, s), min(self.blen, s + self.blen)
                    if lo < hi:
                        env[lo:hi] += prod[lo - s:hi - s]
                self.norm = env
        self.dsp_latency = 0
        if ft in ("frames", "stft"):
            self.dsp_latency = self.flen - self.hop
        if bt == "istft":
            self.dsp_latency += self.back.get("mask_delay", 0) * self.hop
        self.latency: int = self.dsp_latency + p.get("graph_latency", 0)
        self.reset()

    def reset(self) -> None:
        self.graph.reset()
        if hasattr(self, "flen"):
            self.hist = np.zeros(self.flen, np.float64)
        if hasattr(self, "blen"):
            self.ola = np.zeros(self.blen + self.hop, np.float64)
        md = self.back.get("mask_delay", 0)
        # the noisy spectra the masks apply to, mask_delay frames back; silence before the first (as the engine)
        self.specq: list[F32Array] = [np.zeros((self.nfft // 2 + 1, 2), F32) for _ in range(md)]

    def _front(self, x: F64Array) -> tuple[F32Array, F32Array | None]:
        ft = self.front["type"]
        if ft == "time":
            return x.astype(F32), None
        T, hop, L = self.T, self.hop, self.flen
        frames = np.zeros((T, L))
        for t in range(T):
            self.hist = np.concatenate([self.hist[hop:], x[t * hop:(t + 1) * hop]])
            frames[t] = self.hist * self.wa
        if ft == "frames":
            return frames.astype(F32), None
        buf = np.zeros((T, self.nfft))
        buf[:, :L] = frames
        X = np.fft.rfft(buf, axis=-1)
        spec = np.stack([X.real, X.imag], -1).astype(F32)
        feat = self.front.get("feature", "complex")
        eps = self.front.get("log_eps", 1e-12)
        mag = np.abs(X)
        pw = mag ** 2
        if feat == "complex":
            return spec, spec
        base = mag if "magnitude" in feat else pw
        if "filterbank" in self.front:
            base = base @ self.graph.consts[self.front["filterbank"]].astype(np.float64)
        fl = np.maximum(base, eps) if self.front.get("log_floor") == "max" else base + eps
        if feat in ("magnitude", "power"):
            v = base
        elif feat in ("log_power", "log_magnitude"):
            v = np.log(fl)
        elif feat in ("log10_power", "log10_magnitude"):
            v = np.log10(fl)
        elif feat == "log1p_magnitude":
            v = np.log1p(base)
        else:
            raise ValueError(f"front-end feature {feat!r}")
        return v.astype(F32), spec

    def _back(self, y: Array, spec: F32Array | None) -> F64Array:
        bt = self.back["type"]
        if bt == "waveform":
            return y.reshape(-1)
        T, hop = self.T, self.hop
        out = np.zeros(T * hop)
        if bt == "ola":
            fr = y.reshape(T, self.blen).astype(np.float64)
        else:
            src = self.back.get("source", "complex_spectrum")
            F = self.nfft // 2 + 1
            fr = np.zeros((T, self.blen))
            for t in range(T):
                if src == "complex_spectrum":
                    Z = y.reshape(T, F, 2)[t]
                    Zc = Z[:, 0].astype(np.float64) + 1j * Z[:, 1]
                else:
                    if spec is None:
                        raise ValueError(f"back-end source {src!r} needs an stft front-end")
                    self.specq.append(spec[t])
                    s = self.specq.pop(0)
                    sc = s[:, 0].astype(np.float64) + 1j * s[:, 1]
                    if src == "mag_mask":
                        Zc = sc * y.reshape(T, F)[t]
                    else:
                        m = y.reshape(T, F, 2)[t]
                        Zc = sc * (m[:, 0].astype(np.float64) + 1j * m[:, 1])
                fr[t] = np.fft.irfft(Zc, n=self.nfft)[:self.blen]
        for t in range(T):
            self.ola[:self.blen] += fr[t] * self.ws
            out[t * hop:(t + 1) * hop] = self.ola[:hop] / self.norm[:hop]
            self.ola = np.concatenate([self.ola[hop:], np.zeros(hop)])
        return out

    def process(self, x: Array) -> F32Array:
        """Whole signal (any length; zero-padded to a multiple of frame_size). Output delayed by self.latency."""
        n0 = len(x)
        x = np.concatenate([np.asarray(x, np.float64), np.zeros((-n0) % self.fs)])
        out = np.zeros(len(x))
        pin: str = self.p["input"]
        pout: str = self.p["output"]
        for i in range(0, len(x), self.fs):
            feat, spec = self._front(x[i:i + self.fs])
            y = self.graph.run({pin: feat})[pout]
            out[i:i + self.fs] = self._back(y, spec)
        return out.astype(F32)


def read_audio(p: str | Path) -> F32Array:
    if str(p).endswith(".f32"):
        return np.fromfile(p, "<f4")
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32")
    mono: F32Array = x if x.ndim == 1 else x[:, 0]
    return mono


if __name__ == "__main__":
    import time
    args = sys.argv[1:]
    T = int(args[args.index("--T") + 1]) if "--T" in args else None
    pk = Package(args[0], T)
    x = read_audio(args[1])
    t = time.time()
    y = pk.process(x)
    print(f"{args[0]}: T={pk.T} frame={pk.fs} latency={pk.latency} {len(x)} samples in {time.time()-t:.1f}s")
    y.astype("<f4").tofile(args[2])

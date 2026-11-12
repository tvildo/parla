"""ONNX -> vi model package (models/FORMAT.md). Our own code: reads the ONNX protobuf with
tools/onnx_pb.py (no `onnx`/`onnxruntime`), never executes third-party code.

    python tools/onnx2graph.py model.onnx out_dir --name NAME --pipeline pipeline.json \
        [--state in:out ...] [--input mix] [--output enh] [--manifest extra.json]

Passes:
  1. Constant / initializer collection (all numeric dtypes -> f32 at the end).
  2. Static-input lowering: Reshape.shape, Slice.starts/ends/axes/steps, Pad.pads/value,
     Squeeze/Unsqueeze.axes, ReduceX.axes, Split.split, Clip.min/max, Expand.shape, Tile.repeats
     become attributes (must be constant).
  3. Constant folding: every node whose inputs are all constant is evaluated with
     tools/graph_ref.py ops (plus Shape/ConstantOfShape/Range/Equal/Where/Cast/... folding-only ops).
     Shapes are known by executing the graph on zero inputs with graph_ref (static shapes).
  4. BatchNormalization folding into the preceding Conv / ConvTranspose (single consumer).
  5. Dead-node elimination; unused initializers dropped.
"""
import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_ref import Array, Attrs, F32Array, Json, run_op  # noqa: E402
from onnx_pb import Node, load  # noqa: E402


class GraphNode(TypedDict):
    """one node of graph.json (models/FORMAT.md)"""
    op: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: NotRequired[Attrs]


States = list[tuple[str, str]]   # (state input, state output) pairs
Converted = tuple[list[GraphNode], dict[str, F32Array], list[Json], list[Json], dict[str, Array]]

BIG = 2 ** 31 - 1
SUPPORTED = {"Relu", "Sigmoid", "Tanh", "Sqrt", "Abs", "Exp", "Log", "Neg", "Reciprocal", "Softplus", "LeakyRelu", "Elu",
             "HardSigmoid", "Clip", "Identity", "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "PRelu", "MatMul", "Gemm",
             "Conv", "ConvTranspose", "GRU", "LSTM", "BatchNormalization", "LayerNormalization", "InstanceNormalization",
             "ReduceMean", "ReduceSum", "ReduceMax", "Concat", "Split", "Slice", "Gather", "ScatterND", "Reshape",
             "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Pad", "Expand", "Tile", "Cast", "MaxPool"}


def ints(a: Any) -> list[int]:
    return [int(min(max(v, -BIG), BIG)) for v in np.asarray(a).reshape(-1).tolist()]


def fold_only(op: str, ins: Sequence[Any], a: Attrs, shape_of: Callable[[int], tuple[int, ...]]) -> list[Array] | None:
    """Ops that only appear in shape arithmetic (must be constant-folded)."""
    if op == "Shape":
        return [np.array(shape_of(0), np.int64)]
    if op == "Size":
        return [np.array(int(np.prod(shape_of(0))), np.int64)]
    if op == "ConstantOfShape":
        v = a.get("value", np.zeros(1, np.float32))
        return [np.full(ints(ins[0]), np.asarray(v).reshape(-1)[0], np.asarray(v).dtype)]
    if op == "Range":
        return [np.arange(np.asarray(ins[0]).item(), np.asarray(ins[1]).item(), np.asarray(ins[2]).item())]
    if op == "Equal":
        return [np.equal(ins[0], ins[1])]
    if op == "Less":
        return [np.less(ins[0], ins[1])]
    if op == "Greater":
        return [np.greater(ins[0], ins[1])]
    if op == "Not":
        return [np.logical_not(ins[0])]
    if op == "And":
        return [np.logical_and(ins[0], ins[1])]
    if op == "Where":
        return [np.where(ins[0], ins[1], ins[2])]
    if op == "Cast":
        to = a.get("to", 1)
        dt = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 11: np.float64}.get(to, np.float32)
        return [np.asarray(ins[0]).astype(dt)]
    if op in ("Floor", "Ceil"):
        return [getattr(np, op.lower())(ins[0])]
    return None


def static_dims(name: str, dims: list[int | str] | None) -> list[int]:
    """An ONNX input's shape with symbolic sizes (batch, time) set to 1."""
    if dims is None:
        raise ValueError(f"graph input {name} has no static rank")
    return [d if isinstance(d, int) else 1 for d in dims]


def _need_x(arr: list[Array | None]) -> Array:
    if not arr or arr[0] is None:
        raise ValueError("missing input 0")
    return arr[0]


def lower(n: Node, const: dict[str, Array]) -> tuple[str, list[str], Attrs]:
    """ONNX node -> our node (op, inputs, attrs); moves static tensor inputs into attributes."""
    op, ins, a = n["op"], list(n["inputs"]), dict(n["attrs"])

    def c(i: int) -> Array | None:
        if i >= len(ins) or not ins[i]:
            return None
        if ins[i] not in const:
            raise ValueError(f"{op} {n['name']}: input {i} ({ins[i]}) must be constant")
        return const[ins[i]]

    if op == "Slice" and len(ins) > 1:
        a["starts"], a["ends"] = ints(c(1)), ints(c(2))
        if c(3) is not None:
            a["axes"] = ints(c(3))
        if c(4) is not None:
            a["steps"] = ints(c(4))
        ins = ins[:1]
    elif op == "Reshape" and len(ins) > 1:
        a["shape"] = ints(c(1))
        ins = ins[:1]
    elif op == "Pad" and len(ins) > 1:
        a["pads"] = ints(c(1))
        if c(2) is not None:
            a["value"] = float(np.asarray(c(2)).reshape(-1)[0])
        ins = ins[:1]
    elif op in ("Squeeze", "Unsqueeze", "ReduceMean", "ReduceSum", "ReduceMax") and len(ins) > 1:
        if c(1) is not None:
            a["axes"] = ints(c(1))
        ins = ins[:1]
    elif op == "Split" and len(ins) > 1:
        if c(1) is not None:
            a["split"] = ints(c(1))
        ins = ins[:1]
    elif op == "Clip" and len(ins) > 1:
        if c(1) is not None:
            a["min"] = float(np.asarray(c(1)).reshape(-1)[0])
        if c(2) is not None:
            a["max"] = float(np.asarray(c(2)).reshape(-1)[0])
        ins = ins[:1]
    elif op == "Expand":
        a["shape"] = ints(c(1))
        ins = ins[:1]
    elif op == "Tile":
        a["repeats"] = ints(c(1))
        ins = ins[:1]
    elif op == "Dropout":
        op, ins = "Identity", ins[:1]
    for k in ("momentum", "training_mode"):
        a.pop(k, None)
    if op in ("LSTM", "GRU") and "activations" in a:
        d = 2 if a.get("direction", "forward") == "bidirectional" else 1
        default = (["Sigmoid", "Tanh", "Tanh"] if op == "LSTM" else ["Sigmoid", "Tanh"]) * d
        if [x.capitalize() if x.islower() else x for x in a["activations"]] != default:
            raise ValueError(f"{op}: non-default activations {a['activations']}")
        del a["activations"]
        a.pop("activation_alpha", None)
        a.pop("activation_beta", None)
    if a.get("auto_pad") == "VALID":
        a.pop("auto_pad")
        a.pop("pads", None)
    if op == "MaxPool":
        if a.get("ceil_mode", 0) or a.get("storage_order", 0) or any(d != 1 for d in a.get("dilations", [1])):
            raise ValueError("MaxPool: ceil_mode/storage_order/dilations not supported")
        a.pop("ceil_mode", None)
    if op in ("Gemm", "Conv", "ConvTranspose", "Split", "MaxPool") and a.get("auto_pad", "NOTSET") not in ("NOTSET",):
        raise ValueError(f"{op}: auto_pad {a['auto_pad']} not supported")
    a.pop("auto_pad", None) if a.get("auto_pad") == "NOTSET" else None
    if op == "Cast":
        if a.get("to", 1) != 1:
            raise ValueError(f"runtime Cast to non-float ({n['name']})")
        a = {}
    # numpy arrays in attrs -> lists
    for k, v in list(a.items()):
        if isinstance(v, np.ndarray):
            a[k] = v.reshape(-1).tolist()
    return op, ins, a


def convert(onnx_path: str | Path, states: States, pin: str, pout: str, stream_rnn: bool = False,
            stream_time: int | None = None, pins: list[str] | None = None, pouts: list[str] | None = None) -> Converted:
    m = load(onnx_path)
    g = m["graph"]
    const = {k: np.asarray(v) for k, v in g["initializers"].items()}
    inputs: dict[str, list[int]] = {name: static_dims(name, dims) for name, _, dims in g["inputs"]}
    vals: dict[str, Array] = {k: np.zeros(v, np.float32) for k, v in inputs.items()}   # zero-input execution for shapes
    nodes: list[GraphNode] = []
    folded = 0
    for n in g["nodes"]:
        if n["op"] == "Constant":
            v = n["attrs"].get("value")
            if v is None:
                for k in ("value_float", "value_int", "value_floats", "value_ints"):
                    if k in n["attrs"]:
                        v = np.array(n["attrs"][k])
            const[n["outputs"][0]] = np.asarray(v)
            continue
        live_ins = [i for i in n["inputs"] if i]
        if all(i in const for i in live_ins):
            arr = [const[i] if i else None for i in n["inputs"]]
            r = fold_only(n["op"], arr, n["attrs"], lambda k: np.asarray(arr[k]).shape)
            if r is None:
                op, ins, a = lower(n, const)
                r = run_op(op, [const[i].astype(np.float32) if i else None for i in ins], a)
            for o, v in zip(n["outputs"], r):
                const[o] = np.asarray(v)
            folded += 1
            continue
        if n["op"] in ("Shape", "Size"):
            src = vals[n["inputs"][0]]
            const[n["outputs"][0]] = np.array(src.shape if n["op"] == "Shape" else src.size, np.int64)
            folded += 1
            continue
        op, ins, a = lower(n, const)
        if stream_rnn and op in ("GRU", "LSTM"):
            # whole-sequence RNN -> streaming: initial state becomes a state input, Y_h (Y_c) the state output
            d = 2 if a.get("direction", "forward") == "bidirectional" else 1
            xin = vals[ins[0]] if ins[0] in vals else const[ins[0]]
            shp = [d, xin.shape[1], a["hidden_size"]]
            ins = ins + [""] * (7 - len(ins))
            outs = list(n["outputs"]) + [""] * (3 - len(n["outputs"]))
            for si, (slot, osl) in enumerate([(5, 1), (6, 2)] if op == "LSTM" else [(5, 1)]):
                sname = f"{n['name'] or n['outputs'][0]}_state{si}_in"
                if not outs[osl]:
                    outs[osl] = f"{n['name'] or n['outputs'][0]}_state{si}_out"
                ins[slot] = sname
                inputs[sname] = shp
                vals[sname] = np.zeros(shp, np.float32)
                states.append((sname, outs[osl]))
            n = n.copy()
            n["outputs"] = outs
            while ins and not ins[-1]:
                ins.pop()
        def emit(op2: str, ins2: list[str], outs2: list[str], a2: Attrs) -> None:
            arr2 = [(vals[i] if i in vals else const[i].astype(np.float32)) if i else None for i in ins2]
            r2 = run_op(op2, arr2, a2)
            for o2, v2 in zip(outs2, r2):
                if o2:
                    vals[o2] = v2
            nodes.append({"op": op2, "name": outs2[0], "inputs": ins2, "outputs": outs2, "attrs": a2})

        if op == "Einsum":
            eq = a.get("equation", "").replace(" ", "")
            if eq != "btgi,gih->btgh" or ins[1] not in const:
                raise NotImplementedError(f"Einsum {eq}")
            b_, t_, g_, i_ = vals[ins[0]].shape
            h_ = const[ins[1]].shape[2]
            o = n["outputs"][0]
            emit("Transpose", [ins[0]], [o + "_t"], {"perm": [2, 0, 1, 3]})
            emit("Reshape", [o + "_t"], [o + "_r"], {"shape": [g_, b_ * t_, i_]})
            emit("MatMul", [o + "_r", ins[1]], [o + "_m"], {})
            emit("Reshape", [o + "_m"], [o + "_r2"], {"shape": [g_, b_, t_, h_]})
            emit("Transpose", [o + "_r2"], [o], {"perm": [1, 2, 0, 3]})
            continue
        if stream_time is not None and op == "Pad" and a.get("mode", "constant") == "constant":
            # causal padding on the time axis of a whole-sequence model -> streaming cache state
            x0 = vals[ins[0]]
            ndim = x0.ndim
            pads = list(a["pads"])
            ta = stream_time
            pb, pe = pads[ta], pads[ta + ndim]
            if pb > 0 and pe == 0 and a.get("value", 0.0) == 0.0:
                o = n["outputs"][0]
                sname = o + "_tcache_in"
                shp = [int(v) for v in x0.shape]
                shp[ta] = pb
                inputs[sname] = shp
                vals[sname] = np.zeros(shp, np.float32)
                rest = list(pads)
                rest[ta] = 0
                cat = o if not any(rest) else o + "_cat"
                emit("Concat", [sname, ins[0]], [cat], {"axis": ta})
                emit("Slice", [cat], [o + "_tcache_out"], {"starts": [-pb], "ends": [BIG], "axes": [ta]})
                states.append((sname, o + "_tcache_out"))
                if any(rest):
                    emit("Pad", [cat], [o], {**a, "pads": rest})
                continue
            if pe > 0:
                raise NotImplementedError(f"Pad with future time padding ({n['name']}) needs a lookahead delay")
        if op not in SUPPORTED:
            raise NotImplementedError(f"op {op} ({n['name']}) not supported by the graph engine")
        if op == "Cast":
            op = "Identity"
        arr = [(vals[i] if i in vals else const[i].astype(np.float32)) if i else None for i in ins]
        if op == "Split" and "split" not in a:
            parts = len(n["outputs"])
            a["split"] = [_need_x(arr).shape[a.get("axis", 0)] // parts] * parts
        r = run_op(op, arr, a)
        for o, v in zip(n["outputs"], r):
            if o:
                vals[o] = v
        nodes.append({"op": op, "name": n["name"], "inputs": ins, "outputs": [o for o in n["outputs"]], "attrs": a})

    # ---- BatchNorm folding
    prod = {o: k for k, nd in enumerate(nodes) for o in nd["outputs"]}
    uses: dict[str, int] = {}
    for nd in nodes:
        for i in nd["inputs"]:
            uses[i] = uses.get(i, 0) + 1
    drop: set[int] = set()
    n_bn = 0
    for ki, nd in enumerate(nodes):
        if nd["op"] != "BatchNormalization":
            continue
        x = nd["inputs"][0]
        pk = prod.get(x)
        if pk is None or uses.get(x) != 1 or nodes[pk]["op"] not in ("Conv", "ConvTranspose") or nodes[pk]["inputs"][1] not in const:
            continue
        sc, bb, mu, var = (const[i].astype(np.float64) for i in nd["inputs"][1:5])
        s = sc / np.sqrt(var + nd["attrs"].get("epsilon", 1e-5))
        cv = nodes[pk]
        w = const[cv["inputs"][1]].astype(np.float64)
        co = len(s)
        b = const[cv["inputs"][2]].astype(np.float64) if len(cv["inputs"]) > 2 and cv["inputs"][2] else np.zeros(co)
        if cv["op"] == "Conv":
            w2 = w * s.reshape([-1] + [1] * (w.ndim - 1))
        else:
            gr = cv["attrs"].get("group", 1)
            cog = w.shape[1]
            w2 = w.copy()
            cig = w.shape[0] // gr
            for gi in range(gr):
                w2[gi * cig:(gi + 1) * cig] *= s[gi * cog:(gi + 1) * cog].reshape([1, -1] + [1] * (w.ndim - 2))
        b2 = (b - mu) * s + bb
        wn, bn = cv["inputs"][1] + "_bn", nd["name"] + "_bias"
        const[wn], const[bn] = w2.astype(np.float32), b2.astype(np.float32)
        cv["inputs"] = [cv["inputs"][0], wn, bn]
        cv["outputs"] = nd["outputs"]
        drop.add(ki)
        n_bn += 1
    nodes = [nd for ki, nd in enumerate(nodes) if ki not in drop]

    # ---- stride-1 ConvTranspose -> Conv (flipped kernel, in/out swapped per group, pads' = d*(k-1) - p)
    n_ct = 0
    for nd in nodes:
        if nd["op"] != "ConvTranspose" or nd["inputs"][1] not in const:
            continue
        a = nd["attrs"]
        w = const[nd["inputs"][1]]
        nds = w.ndim - 2
        st = a.get("strides", [1] * nds)
        op_ = a.get("output_padding", [0] * nds)
        dl = a.get("dilations", [1] * nds)
        pads = a.get("pads", [0] * (2 * nds))
        ks = list(w.shape[2:])
        newp = [dl[i % nds] * (ks[i % nds] - 1) - pads[i] for i in range(2 * nds)]
        if any(v != 1 for v in st) or any(op_) or "output_shape" in a or any(v < 0 for v in newp):
            continue
        gr = a.get("group", 1)
        cin, cog = w.shape[0], w.shape[1]
        cig = cin // gr
        wf = np.flip(w, axis=tuple(range(2, w.ndim)))
        wc = np.concatenate([np.swapaxes(wf[gi * cig:(gi + 1) * cig], 0, 1) for gi in range(gr)], axis=0)  # [gr*cog, cig, k..]
        wn = nd["inputs"][1] + "_conv"
        const[wn] = np.ascontiguousarray(wc)
        nd["op"] = "Conv"
        nd["inputs"] = [nd["inputs"][0], wn] + nd["inputs"][2:]
        nd["attrs"] = {"kernel_shape": ks, "strides": [1] * nds, "dilations": dl, "pads": newp, "group": gr}
        n_ct += 1
    if n_ct:
        print(f"  {n_ct} stride-1 ConvTranspose -> Conv")

    # ---- dead node elimination
    need = set(pouts or [pout]) | {o for _, o in states}
    keep = []
    for nd in reversed(nodes):
        if any(o in need for o in nd["outputs"]):
            keep.append(nd)
            need |= {i for i in nd["inputs"] if i}
    nodes = keep[::-1]
    used = {i for nd in nodes for i in nd["inputs"] if i}
    inits = {k: v.astype(np.float32) for k, v in const.items() if k in used}
    keep_in = set(pins or [pin])
    ginputs = [{"name": k, "shape": v} for k, v in inputs.items() if k in keep_in or k in dict(states)]
    gstates = [{"in": i, "out": o, "shape": inputs[i]} for i, o in states]
    for i, o in states:
        assert list(vals[o].shape) == inputs[i], (i, o, vals[o].shape, inputs[i])
    print(f"{onnx_path}: {len(g['nodes'])} onnx nodes -> {len(nodes)} graph nodes ({folded} folded, {n_bn} BatchNorm folded), "
          f"{sum(v.size for v in inits.values())} weights")
    return nodes, inits, ginputs, gstates, vals


def prefixed(nodes: list[GraphNode], inits: dict[str, F32Array], ginputs: list[Json], gstates: list[Json], pfx: str,
             keep: set[str]) -> tuple[list[GraphNode], dict[str, F32Array], list[Json], list[Json]]:
    """One converted stage with every tensor name prefixed (except `keep`), to combine several stages in one graph."""
    def ren(t: str) -> str:
        return t if (not t or t in keep) else pfx + t
    out: list[GraphNode] = []
    for n in nodes:
        r = n.copy()
        r["name"], r["inputs"], r["outputs"] = pfx + n["name"], [ren(i) for i in n["inputs"]], [ren(o) for o in n["outputs"]]
        out.append(r)
    return (out, {pfx + k: v for k, v in inits.items()}, [{**g, "name": ren(g["name"])} for g in ginputs],
            [{**s, "in": ren(s["in"]), "out": ren(s["out"])} for s in gstates])


def write_package(out_dir: str | Path, name: str, nodes: list[GraphNode], inits: dict[str, F32Array], ginputs: list[Json],
                  gstates: list[Json], pout: str, manifest: Json) -> Json:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_json: dict[str, Json] = {}
    data: list[F32Array] = []
    off = 0
    for k, v in inits.items():
        if v.size <= 4:
            init_json[k] = {"shape": list(v.shape), "value": [float(x) for x in v.reshape(-1)]}
        else:
            init_json[k] = {"offset": off, "shape": list(v.shape)}
            data.append(v.reshape(-1))
            off += v.size
    graph = {"format": "vi-graph", "version": 1, "name": name, "inputs": ginputs, "outputs": [{"name": pout}],
             "states": gstates, "initializers": init_json, "nodes": nodes}
    (out_dir / "graph.json").write_text(json.dumps(graph, separators=(",", ":")))
    w = np.concatenate(data).astype("<f4") if data else np.zeros(0, "<f4")
    w.tofile(out_dir / "weights.bin")
    manifest = dict(manifest)
    manifest.update({"name": name, "engine": "graph", "graph": "graph.json", "weights": "weights.bin"})
    manifest["sha256"] = {f: hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in ("graph.json", "weights.bin")}
    manifest["package_bytes"] = sum((out_dir / f).stat().st_size for f in ("graph.json", "weights.bin"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", required=True, help="JSON with display_name, license, source, version, upstream_sha256, pipeline ...")
    ap.add_argument("--state", action="append", default=[], help="in:out state pair")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    p = man["pipeline"]
    states = [tuple(s.split(":")) for s in a.state]
    nodes, inits, gi, gs, _ = convert(a.onnx, states, p["input"], p["output"])
    write_package(a.out, a.name, nodes, inits, gi, gs, p["output"], man)


if __name__ == "__main__":
    main()

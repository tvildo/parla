"""Converter check: interpret the ORIGINAL ONNX graph node by node with our ops (graph_ref.run_op
after static-input lowering; no folding / BN fusion / DCE) in streaming mode, and compare with the
converted package run by graph_ref.Package on the same audio. Our code only.

    python tools/check_onnx.py model.onnx models/<pkg> in.f32 [--frames N]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_ref import Package, run_op  # noqa: E402
from onnx2graph import fold_only, lower  # noqa: E402
from onnx_pb import load  # noqa: E402


def interp(g, env):
    """Execute ONNX graph dict `g` (onnx_pb) node by node with our ops; env holds inputs/states."""
    env = {**{k: np.asarray(v) for k, v in g["initializers"].items()}, **env}
    for n in g["nodes"]:
        if n["op"] == "Constant":
            env[n["outputs"][0]] = np.asarray(n["attrs"]["value"])
            continue
        arr = [env[i] if i else None for i in n["inputs"]]
        r = fold_only(n["op"], arr, n["attrs"], lambda k: np.asarray(arr[k]).shape)
        if r is None:
            op, ins, a = lower(n, env)
            if op == "Cast":
                op, a = "Identity", {}
            if op == "Split" and "split" not in a:
                k = len(n["outputs"])
                a["split"] = [np.asarray(env[ins[0]]).shape[a.get("axis", 0)] // k] * k
            r = run_op(op, [np.asarray(env[i], np.float32) if i else None for i in ins], a)
        for o, v in zip(n["outputs"], r):
            env[o] = v
    return env


def main():
    onnx_path, pkg_dir, wav = sys.argv[1:4]
    nf = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 200
    pk = Package(pkg_dir)
    g = load(onnx_path)["graph"]
    states = [(s["in"], s["out"]) for s in json.loads((Path(pkg_dir) / "graph.json").read_text())["states"]]
    pin, pout = pk.p["input"], pk.p["output"]
    shapes = {n: [d if isinstance(d, int) else 1 for d in dims] for n, _, dims in g["inputs"]}
    st = {i: np.zeros(shapes[i], np.float32) for i, _ in states}
    x = np.fromfile(wav, "<f4")[: nf * pk.fs]
    out_ref = np.zeros(len(x), np.float32)
    maxd = 0.0
    for f in range(nf):
        seg = x[f * pk.fs:(f + 1) * pk.fs]
        feat, spec = pk._front(seg.astype(np.float64))
        y_pkg = pk.graph.run({pin: feat})[pout]
        env = {k: np.asarray(v) for k, v in g["initializers"].items()}
        env.update(st)
        env[pin] = feat.reshape(shapes[pin])
        for n in g["nodes"]:
            if n["op"] == "Constant":
                env[n["outputs"][0]] = np.asarray(n["attrs"]["value"])
                continue
            arr = [env[i] if i else None for i in n["inputs"]]
            r = fold_only(n["op"], arr, n["attrs"], lambda k: np.asarray(arr[k]).shape)
            if r is None:
                op, ins, a = lower(n, env)
                if op == "Cast":
                    op, a = "Identity", {}
                r = run_op(op, [np.asarray(env[i], np.float32) if i else None for i in ins], a)
            for o, v in zip(n["outputs"], r):
                env[o] = v
        for i, o in states:
            st[i] = env[o].astype(np.float32)
        d = np.abs(env[pout].reshape(-1) - y_pkg.reshape(-1)).max()
        maxd = max(maxd, d)
        out_ref[f * pk.fs:(f + 1) * pk.fs] = pk._back(env[pout].astype(np.float32), spec)
    ref_mag = np.abs(env[pout]).max()
    print(f"{onnx_path}: {nf} frames, max|original ONNX (our ops) - package| over graph outputs = {maxd:.3e} (output scale {ref_mag:.3e})")


if __name__ == "__main__":
    main()

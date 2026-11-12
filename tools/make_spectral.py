"""Generate the `spectral` baseline package: classic single-channel STFT noise suppression written
as a vi graph (elementwise ops + streaming states) — no neural net, no special engine kind.

    python tools/make_spectral.py [out_dir]   (default models/spectral)

Per frame (P = |X|^2 noisy power, 257 bins, 16 kHz, n_fft 512 / hop 256, sqrt-Hann):
  S    = a_s * S' + (1 - a_s) * P                                 recursive smoothing
  Pmin = min(S, g * Pmin' + (1-g)/(1-b) * (S - b * S'))           continuous minimum tracking
                                                                  (Doblinger 1995 style)
  N    = bias * Pmin + eps                                        noise PSD estimate
  xi   = a_dd * A' / N + (1 - a_dd) * max(P / N - 1, 0)           decision-directed a-priori SNR
                                                                  (Ephraim-Malah 1984), A = G^2 P
  G    = max(xi / (1 + xi), G_min)                                Wiener gain, floor -15 dB
  A    = G^2 * P
Back-end: mag_mask on the noisy STFT, sqrt-Hann synthesis, OLA. Latency 256 samples.
"""
import hashlib
import json
import sys
from pathlib import Path

from typing import Any

import numpy as np

A_S, GAMMA, BETA, BIAS, A_DD, G_MIN, EPS = 0.7, 0.998, 0.96, 1.5, 0.98, 10 ** (-15 / 20), 1e-10
F = 257


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/spectral")
    out.mkdir(parents=True, exist_ok=True)
    c = {"a_s": A_S, "one_m_a_s": 1 - A_S, "g": GAMMA, "k": (1 - GAMMA) / (1 - BETA), "b": BETA, "bias": BIAS,
         "eps": EPS, "a_dd": A_DD, "one_m_a_dd": 1 - A_DD, "one": 1.0, "zero": 0.0, "gmin": G_MIN}
    inits = {k: {"shape": [], "value": [v]} for k, v in c.items()}
    N: list[dict[str, Any]] = []

    def node(op: str, ins: list[str], out_name: str) -> str:
        N.append({"op": op, "name": out_name, "inputs": ins, "outputs": [out_name]})
        return out_name

    s1 = node("Mul", ["a_s", "S_in"], "s1")
    s2 = node("Mul", ["one_m_a_s", "P"], "s2")
    S = node("Add", [s1, s2], "S_out")
    bS = node("Mul", ["b", "S_in"], "bS")
    d = node("Sub", [S, bS], "d")
    kd = node("Mul", ["k", d], "kd")
    gp = node("Mul", ["g", "Pmin_in"], "gp")
    upd = node("Add", [gp, kd], "upd")
    pmin = node("Min", [S, upd], "Pmin_out")
    nb = node("Mul", ["bias", pmin], "nb")
    Nn = node("Add", [nb, "eps"], "noise")
    r1 = node("Div", ["A_in", Nn], "r1")
    t1 = node("Mul", ["a_dd", r1], "t1")
    post = node("Div", ["P", Nn], "post")
    pm1 = node("Sub", [post, "one"], "pm1")
    pr = node("Max", [pm1, "zero"], "pr")
    t2 = node("Mul", ["one_m_a_dd", pr], "t2")
    xi = node("Add", [t1, t2], "xi")
    den = node("Add", [xi, "one"], "den")
    g = node("Div", [xi, den], "g_raw")
    G = node("Max", [g, "gmin"], "gain")
    G2 = node("Mul", [G, G], "g2")
    node("Mul", [G2, "P"], "A_out")
    graph = {"format": "vi-graph", "version": 1, "name": "spectral",
             "inputs": [{"name": "P", "shape": [1, F]}, {"name": "S_in", "shape": [1, F]},
                        {"name": "Pmin_in", "shape": [1, F]}, {"name": "A_in", "shape": [1, F]}],
             "outputs": [{"name": "gain"}],
             "states": [{"in": "S_in", "out": "S_out", "shape": [1, F]}, {"in": "Pmin_in", "out": "Pmin_out", "shape": [1, F]},
                        {"in": "A_in", "out": "A_out", "shape": [1, F]}],
             "initializers": inits, "nodes": N}
    (out / "graph.json").write_text(json.dumps(graph, indent=0))
    np.zeros(0, "<f4").tofile(out / "weights.bin")
    man = {
        "name": "spectral", "display_name": "Spectral Wiener (classic DSP baseline, 16 kHz)", "engine": "graph",
        "graph": "graph.json", "weights": "weights.bin",
        "sample_rate": 16000, "frame_size": 256, "latency": 256, "local_only": False,
        "license": "MIT (this repository; own implementation)",
        "source": "tools/make_spectral.py (min-statistics noise PSD via continuous minimum tracking, decision-directed a-priori SNR, Wiener gain)",
        "version": "1", "params": c,
        "pipeline": {"sample_rate": 16000, "hop": 256, "frames_per_call": 1, "input": "P", "output": "gain", "graph_latency": 0,
                     "front": {"type": "stft", "n_fft": 512, "window": "sqrt_hann", "feature": "power"},
                     "back": {"type": "istft", "source": "mag_mask", "window": "sqrt_hann"}},
    }
    man["sha256"] = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("graph.json", "weights.bin")}
    man["package_bytes"] = sum((out / f).stat().st_size for f in ("graph.json", "weights.bin"))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}: {len(N)} nodes")


if __name__ == "__main__":
    main()

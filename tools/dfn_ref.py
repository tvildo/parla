"""Independent numpy reference of the DeepFilterNet3 streaming DSP (our code, from the paper and the
published constants in manifest.config) around the converted graph (graph_ref.Graph). Used to
validate the Rust plugin `vi_dfn`.

    python tools/dfn_ref.py models/deepfilternet3 in48k.f32 out.f32
"""
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_ref import Array, F32Array, Graph, Json  # noqa: E402

C64Array = npt.NDArray[np.complex64]

F = np.float32


class DFN:
    def __init__(self, pkg: str | Path) -> None:
        pkg = Path(pkg)
        man = json.loads((pkg / "manifest.json").read_text())
        self.c: Json = man["config"]
        c = self.c
        self.g = Graph(json.loads((pkg / man["graph"]).read_text()), np.fromfile(pkg / man["weights"], "<f4"))
        self.n: int = c["fft_size"]
        self.hop: int = c["hop_size"]
        self.F = self.n // 2 + 1
        h = self.n // 2
        i = np.arange(self.n)
        self.win = np.sin(.5 * np.pi * np.sin(.5 * np.pi * (i + .5) / h) ** 2).astype(F)
        self.wnorm = F(1.0 / (self.n ** 2 / (2 * self.hop)))
        self.erb: list[int] = c["erb_widths"]
        self.alpha = F(c["norm_alpha"])
        self.reset()

    def reset(self) -> None:
        c = self.c
        self.g.reset()
        self.amem = np.zeros(self.n - self.hop, F)
        self.smem = np.zeros(self.n - self.hop, F)
        self.mn = np.linspace(c["mean_norm_init"][0], c["mean_norm_init"][1], c["nb_erb"]).astype(F)
        self.un = np.linspace(c["unit_norm_init"][0], c["unit_norm_init"][1], c["nb_df"]).astype(F)
        self.ry: list[C64Array] = [np.zeros(self.F, np.complex64) for _ in range(c["df_order"] + c["conv_lookahead"])]
        self.rx: list[C64Array] = [np.zeros(self.F, np.complex64) for _ in range(max(c["df_order"], c["conv_lookahead"]))]
        self.skip = 0

    def band_gain(self, X: C64Array, g: Array) -> C64Array:
        out = X.copy()
        k = 0
        for b, w in enumerate(self.erb):
            out[k:k + w] *= g[b]
            k += w
        return out

    def frame(self, x: Array) -> F32Array:
        c = self.c
        if np.mean(x.astype(F) ** 2, dtype=F) < 1e-7:
            self.skip += 1
        else:
            self.skip = 0
        if self.skip > 5:
            return np.zeros(self.hop, F)
        buf = np.concatenate([self.amem, x]).astype(F) * self.win
        self.amem = np.concatenate([self.amem[self.hop:], x])[-(self.n - self.hop):].astype(F)
        X = (np.fft.rfft(buf.astype(np.float64)) * self.wnorm).astype(np.complex64)
        self.ry = self.ry[1:] + [X.copy()]
        self.rx = self.rx[1:] + [X.copy()]
        # ERB features
        p = (X.real ** 2 + X.imag ** 2).astype(F)
        e = np.zeros(c["nb_erb"], F)
        k = 0
        for b, w in enumerate(self.erb):
            e[b] = np.sum(p[k:k + w] * F(1.0 / w), dtype=F)
            k += w
        e = (np.log10(e + F(1e-10)) * F(10)).astype(F)
        self.mn = (e * (1 - self.alpha) + self.mn * self.alpha).astype(F)
        fe = ((e - self.mn) / F(40)).astype(F)
        # complex features
        nd = c["nb_df"]
        Xd = X[:nd]
        self.un = (np.abs(Xd).astype(F) * (1 - self.alpha) + self.un * self.alpha).astype(F)
        fs = (Xd / np.sqrt(self.un)).astype(np.complex64)
        feat_spec = np.stack([fs.real, fs.imag]).astype(F).reshape(1, 2, 1, nd)
        o = self.g.run({"feat_erb": fe.reshape(1, 1, 1, -1), "feat_spec": feat_spec}, keep=True)
        lsnr = float(o["lsnr"].reshape(-1)[0])
        if lsnr < c["min_db_thresh"]:
            gains: Array | None = np.zeros(c["nb_erb"], F)
            df = False
        elif lsnr > c["max_db_erb_thresh"]:
            gains, df = None, False
        elif lsnr > c["max_db_df_thresh"]:
            gains, df = o["m"].reshape(-1), False
        else:
            gains, df = o["m"].reshape(-1), True
        idx = c["df_order"] - 1
        if gains is not None:
            self.ry[idx] = self.band_gain(self.ry[idx], gains)
        out = self.ry[idx].copy()
        if df:
            co = o["coefs"].reshape(nd, c["df_order"], 2)
            acc = np.zeros(nd, np.complex64)
            for k in range(c["df_order"]):
                acc += self.rx[k][:nd] * (co[:, k, 0] + 1j * co[:, k, 1]).astype(np.complex64)
            out[:nd] = acc
        y: F32Array = (np.fft.irfft(out.astype(np.complex128), self.n) * self.n).astype(F) * self.win
        res = y[:self.hop] + self.smem[:self.hop]
        sm = np.concatenate([self.smem[self.hop:], np.zeros(self.hop, F)])
        sm[: self.n - self.hop] += y[self.hop:]
        self.smem = sm[: self.n - self.hop].astype(F)
        return res.astype(F)

    def process(self, x: Array) -> F32Array:
        x = np.concatenate([np.asarray(x, F), np.zeros((-len(x)) % self.hop, F)])
        out = np.zeros(len(x), F)
        for i in range(0, len(x), self.hop):
            out[i:i + self.hop] = self.frame(x[i:i + self.hop])
        return out


if __name__ == "__main__":
    import time
    d = DFN(sys.argv[1])
    x = np.fromfile(sys.argv[2], "<f4")
    t = time.time()
    y = d.process(x)
    print(f"dfn ref: {len(x)} samples in {time.time() - t:.1f}s")
    y.astype("<f4").tofile(sys.argv[3])

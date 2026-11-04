"""DFN3 streaming-conversion check: the ORIGINAL whole-sequence ONNX graphs (enc -> erb_dec, df_dec)
interpreted with our ops over S frames of random-but-smooth features vs the composed streaming
package (graph_ref.Graph) run frame by frame. Validates time caches + GRU state conversion.

    python tools/check_dfn_stream.py oss_dl/dfn models/deepfilternet3 [S]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_onnx import interp  # noqa: E402
from graph_ref import Graph  # noqa: E402
from onnx_pb import load  # noqa: E402

src, pkg = Path(sys.argv[1]) / "onnx/tmp/export", Path(sys.argv[2])
S = int(sys.argv[3]) if len(sys.argv) > 3 else 12
rng = np.random.default_rng(0)
erb = (rng.standard_normal((1, 1, S, 32)) * 0.3).astype(np.float32)
spec = (rng.standard_normal((1, 2, S, 96)) * 0.5).astype(np.float32)
e = interp(load(src / "enc.onnx")["graph"], {"feat_erb": erb, "feat_spec": spec})
m_ref = interp(load(src / "erb_dec.onnx")["graph"], {k: e[k] for k in ("emb", "e3", "e2", "e1", "e0")})["m"]
c_ref = interp(load(src / "df_dec.onnx")["graph"], {"emb": e["emb"], "c0": e["c0"]})["coefs"]
man = json.loads((pkg / "manifest.json").read_text())
g = Graph(json.loads((pkg / "graph.json").read_text()), np.fromfile(pkg / "weights.bin", "<f4"))
dm = dc = dl = 0.0
for t in range(S):
    o = g.run({"feat_erb": erb[:, :, t:t + 1], "feat_spec": spec[:, :, t:t + 1]}, keep=True)
    dm = max(dm, np.abs(o["m"].reshape(-1) - m_ref[0, 0, t].reshape(-1)).max())
    dc = max(dc, np.abs(o["coefs"].reshape(-1) - c_ref[0, t].reshape(-1)).max())
    dl = max(dl, abs(float(o["lsnr"].reshape(-1)[0]) - float(e["lsnr"][0, t, 0])))
print(f"dfn stream vs whole-sequence ({S} frames): max|dm| {dm:.2e}  max|dcoefs| {dc:.2e} (scale {np.abs(c_ref).max():.2f})  max|dlsnr| {dl:.2e} dB")

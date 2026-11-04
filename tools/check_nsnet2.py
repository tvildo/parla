"""NSNet2 converter + streamify check: independent offline implementation of the published
pipeline constants (numpy STFT, ORIGINAL whole-sequence ONNX interpreted with our ops over all
frames at once, gain clip, windowed OLA) vs the streaming package (graph_ref). Our code only.

    python tools/check_nsnet2.py oss_dl/nsnet2 models/nsnet2 in.f32 [--seconds S]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_onnx import interp  # noqa: E402
from graph_ref import Package  # noqa: E402
from onnx_pb import load  # noqa: E402


def main():
    src, pkg, wav = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sec = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 3.0
    x = np.fromfile(wav, "<f4")[: int(sec * 16000)].astype(np.float64)
    N, H = 320, 160
    w = np.sqrt(np.hanning(N))
    nfr = len(x) // H
    buf = np.zeros(N)
    spec = []
    for i in range(nfr):
        buf = np.concatenate([buf[H:], x[i * H:(i + 1) * H]])
        spec.append(np.fft.rfft(buf * w))
    S = np.array(spec)                                     # [T, 161]
    feat = np.log10(np.maximum(np.abs(S) ** 2, 1e-12)).astype(np.float32)[None]
    g = load(src / "nsnet2-20ms-baseline.onnx")["graph"]
    gain = interp(g, {"input": feat})["output"][0]
    gain = np.clip(gain, 1e-4, 1.0)
    y = np.zeros(nfr * H + N)
    for i in range(nfr):
        y[i * H:i * H + N] += w * np.fft.irfft(S[i] * gain[i], N)
    ref = y[:nfr * H]                                       # frame i ends at sample (i+1)*H -> our output delayed by N-H
    out = Package(pkg).process(x.astype(np.float32))[:nfr * H].astype(np.float64)
    d = np.abs(out[:len(ref)] - ref[:len(out)]).max()
    print(f"nsnet2: {nfr} frames, max|offline whole-sequence pipeline - streaming package| = {d:.3e} (peak {np.abs(ref).max():.3f})")


if __name__ == "__main__":
    main()

"""DTLN converter check: independent streaming implementation of the published DTLN pipeline
(numpy rfft/irfft + the two ORIGINAL ONNX graphs interpreted node by node with our ops) vs the
composed package run by graph_ref.Package. Our code only.

    python tools/check_dtln.py oss_dl/dtln models/dtln in.f32 [--frames N]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_onnx import interp  # noqa: E402
from graph_ref import Package  # noqa: E402
from onnx_pb import load  # noqa: E402


def main() -> None:
    src, pkg, wav = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    nf = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 300
    g1 = load(src / "pretrained_model_model_1.onnx")["graph"]
    g2 = load(src / "pretrained_model_model_2.onnx")["graph"]
    x = np.fromfile(wav, "<f4")[: nf * 128]
    inb, outb = np.zeros(512, np.float32), np.zeros(512, np.float32)
    s1, s2 = np.zeros((1, 2, 128, 2), np.float32), np.zeros((1, 2, 128, 2), np.float32)
    ref = np.zeros(len(x), np.float32)
    for i in range(nf):
        inb = np.concatenate([inb[128:], x[i * 128:(i + 1) * 128]])
        X = np.fft.rfft(inb)
        e1 = interp(g1, {"input_2": np.abs(X).astype(np.float32).reshape(1, 1, -1), "input_3": s1})
        s1 = e1["tf_op_layer_stack_2"]
        blk = np.fft.irfft(X * e1["activation_2"].reshape(-1)).astype(np.float32)
        e2 = interp(g2, {"input_4": blk.reshape(1, 1, -1), "input_5": s2})
        s2 = e2["tf_op_layer_stack_5"]
        outb = np.concatenate([outb[128:], np.zeros(128, np.float32)]) + e2["conv1d_3"].reshape(-1)
        ref[i * 128:(i + 1) * 128] = outb[:128]
    y = Package(pkg).process(x)[: len(x)]
    # our package emits the OLA head one hop later in the buffer order? both: out = first hop after add
    d = np.abs(y - ref).max()
    print(f"dtln: {nf} frames, max|independent pipeline - package| = {d:.3e} (signal peak {np.abs(ref).max():.3f})")


if __name__ == "__main__":
    main()

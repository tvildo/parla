"""NSNet2 (DNS Challenge baseline, Braun et al. 2021) -> vi package.

    python tools/make_nsnet2.py oss_dl/nsnet2 models/nsnet2

The published ONNX runs a whole utterance (GRU initial state = zeros); `stream_rnn` turns both
GRUs' initial_h into streaming state inputs (Y_h -> state output) so the graph runs one 10 ms
frame per call. Glue: Clip(gain, -80 dB, 1) as in the published enhancer constants.
Front-end (published constants): 16 kHz, 20 ms sqrt(symmetric Hann) window, hop 10 ms, n_fft 320,
feature log10(max(|X|^2, 1e-12)); back-end: gain on the noisy STFT, same window, overlap-add.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx2graph import convert, write_package  # noqa: E402


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    onnx = src / "nsnet2-20ms-baseline.onnx"
    states = []
    nodes, inits, gi, gs, vals = convert(str(onnx), states, "input", "output", stream_rnn=True)
    nodes.append({"op": "Clip", "name": "mingain", "inputs": ["output"], "outputs": ["gain"], "attrs": {"min": 1e-4, "max": 1.0}})
    man = json.loads(Path("tools/specs/nsnet2.json").read_text())
    man["upstream_sha256"] = {onnx.name: hashlib.sha256(onnx.read_bytes()).hexdigest()}
    m = write_package(out, "nsnet2", nodes, inits, gi, gs, "gain", man)
    print(f"nsnet2: {len(nodes)} nodes, states {[(s['in'], s['shape']) for s in gs]}, {m['package_bytes']} bytes")


if __name__ == "__main__":
    main()

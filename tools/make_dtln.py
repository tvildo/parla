"""DTLN (Westhausen & Meyer 2020) -> one vi package: both published ONNX stages + our glue ops.

    python tools/make_dtln.py oss_dl/dtln models/dtln

Graph (per 512-sample frame, hop 128, no analysis window — as in the published real-time script):
  spec = Rfft(frame); mag = sqrt(sum(spec^2)); mask = stage1(mag) (2 LSTMs + dense + sigmoid)
  y = Irfft(spec * mask); out = stage2(y) (learned conv encoder, normalization, 2 LSTMs, mask, conv decoder)
Back-end: overlap-add of the 512-sample output frames at hop 128 (latency 384 samples = 24 ms).
Only weights + graph structure are taken from the ONNX files (our reader); nothing of theirs is run.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx2graph import GraphNode, convert, prefixed, write_package  # noqa: E402


def main() -> None:
    src, out = Path(sys.argv[1]), sys.argv[2]
    m1 = src / "pretrained_model_model_1.onnx"
    m2 = src / "pretrained_model_model_2.onnx"
    n1, i1, g1, s1, _ = convert(str(m1), [("input_3", "tf_op_layer_stack_2")], "input_2", "activation_2")
    n2, i2, g2, s2, _ = convert(str(m2), [("input_5", "tf_op_layer_stack_5")], "input_4", "conv1d_3")
    n1, i1, g1, s1 = prefixed(n1, i1, g1, s1, "s1/", set())
    n2, i2, g2, s2 = prefixed(n2, i2, g2, s2, "s2/", set())
    glue_a: list[GraphNode] = [
        {"op": "Rfft", "name": "rfft", "inputs": ["frame"], "outputs": ["spec"]},                       # [1,257,2]
        {"op": "Mul", "name": "sq", "inputs": ["spec", "spec"], "outputs": ["spec_sq"]},
        {"op": "ReduceSum", "name": "pow", "inputs": ["spec_sq"], "outputs": ["power"], "attrs": {"axes": [-1], "keepdims": 0}},
        {"op": "Sqrt", "name": "mag", "inputs": ["power"], "outputs": ["mag"]},
        {"op": "Reshape", "name": "to_s1", "inputs": ["mag"], "outputs": ["s1/input_2"], "attrs": {"shape": [1, 1, 257]}},
    ]
    glue_b: list[GraphNode] = [
        {"op": "Reshape", "name": "mask_r", "inputs": ["s1/activation_2"], "outputs": ["mask"], "attrs": {"shape": [1, 257, 1]}},
        {"op": "Mul", "name": "apply_mask", "inputs": ["spec", "mask"], "outputs": ["spec_m"]},
        {"op": "Irfft", "name": "irfft", "inputs": ["spec_m"], "outputs": ["y"], "attrs": {"n": 512}},
        {"op": "Reshape", "name": "to_s2", "inputs": ["y"], "outputs": ["s2/input_4"], "attrs": {"shape": [1, 1, 512]}},
    ]
    glue_c: list[GraphNode] = [{"op": "Reshape", "name": "out", "inputs": ["s2/conv1d_3"], "outputs": ["out_frame"], "attrs": {"shape": [1, 512]}}]
    nodes = glue_a + n1 + glue_b + n2 + glue_c
    inits = {**i1, **i2}
    ginputs = [{"name": "frame", "shape": [1, 512]}] + [g for g in g1 + g2 if g["name"] in {s["in"] for s in s1 + s2}]
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (m1, m2)}
    man = json.loads(Path("tools/specs/dtln.json").read_text())
    man["upstream_sha256"] = sha
    m = write_package(out, "dtln", nodes, inits, ginputs, s1 + s2, "out_frame", man)
    print(f"dtln: {len(nodes)} nodes, {sum(v.size for v in inits.values())} weights, {m['package_bytes']} bytes")


if __name__ == "__main__":
    main()

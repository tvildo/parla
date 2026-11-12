"""Minimal ONNX protobuf reader (our own code; no `onnx` / `onnxruntime` dependency).

Decodes ModelProto -> {opset, graph}; graph = {nodes, initializers, inputs, outputs, value_info}.
Only what the converters need: nodes (op_type, inputs, outputs, name, attributes),
initializers / Constant tensors as numpy arrays, input/output names + static shapes.

    m = load("model.onnx");  g = m["graph"]
    g["nodes"][i] = {"op": "Conv", "name": ..., "inputs": [...], "outputs": [...], "attrs": {k: v}}
    g["initializers"][name] = np.ndarray
    g["inputs"] = [(name, elem_type, [dims (int | str)])]
"""
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
Attrs = dict[str, Any]   # attribute values as in the file: float, int, str, tensor, graph, or lists of them
Dim = int | str          # a static size or a symbolic name ("?" when absent)
ValueInfo = tuple[str, int, list[Dim] | None]   # name, element type, dims (None: unknown rank)
Value = int | bytes      # one protobuf field: varint as int; length-delimited and fixed-size as bytes


class Node(TypedDict):
    op: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: Attrs
    domain: str


class Graph(TypedDict):
    name: str
    nodes: list[Node]
    initializers: dict[str, Array]
    inputs: list[ValueInfo]
    outputs: list[ValueInfo]
    value_info: dict[str, tuple[int, list[Dim] | None]]


class Model(TypedDict):
    opset: dict[str, int]
    producer: str
    ir_version: int
    graph: Graph


DTYPES: dict[int, type[np.generic]] = {
    1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64,
    9: np.bool_, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(b: bytes) -> Iterator[tuple[int, int, Value]]:
    """Yield (field_number, wire_type, value) for one message; value = int (varint) or bytes."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        v: Value
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield f, wt, v


def _int(v: Value) -> int:
    if not isinstance(v, int):
        raise ValueError("malformed ONNX: a number field holds bytes")
    return v


def _bytes(v: Value) -> bytes:
    if not isinstance(v, bytes):
        raise ValueError("malformed ONNX: a bytes field holds a number")
    return v


def _sint64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


def _packed_varints(v: Value, wt: int) -> list[int]:
    if wt == 0:
        return [_sint64(_int(v))]
    b = _bytes(v)
    out: list[int] = []
    i = 0
    while i < len(b):
        x, i = _varint(b, i)
        out.append(_sint64(x))
    return out


def _floats(v: Value, wt: int, code: str, size: int) -> list[float]:
    """A packed (wire type 2) or single fixed-size float / double field."""
    b = _bytes(v)
    return list(struct.unpack(f"<{len(b) // size}{code}", b)) if wt == 2 else [struct.unpack(f"<{code}", b)[0]]


def tensor(b: bytes) -> tuple[str, Array]:
    dims: list[int] = []
    dt, name = 1, ""
    raw: bytes | None = None
    lists: dict[int, list[float]] = {4: [], 10: []}
    ilists: dict[int, list[int]] = {5: [], 7: [], 11: []}
    for f, wt, v in fields(b):
        if f == 1:
            dims += _packed_varints(v, wt)
        elif f == 2:
            dt = _int(v)
        elif f == 8:
            name = _bytes(v).decode()
        elif f == 9:
            raw = _bytes(v)
        elif f == 4:
            lists[4] += _floats(v, wt, "f", 4)
        elif f == 10:
            lists[10] += _floats(v, wt, "d", 8)
        elif f in (5, 7, 11):
            ilists[f] += _packed_varints(v, wt)
        elif f == 14 and v == 1:
            raise ValueError(f"external data not supported ({name})")
    npdt = DTYPES[dt]
    a: Array
    if raw is not None:
        a = np.frombuffer(raw, npdt).copy()
    elif dt == 1:
        a = np.array(lists[4], np.float32)
    elif dt == 11:
        a = np.array(lists[10], np.float64)
    elif dt == 7:
        a = np.array(ilists[7], np.int64)
    elif dt in (13, 12):
        a = np.array(ilists[11], npdt)
    elif dt == 10:
        a = np.array(ilists[5], np.uint16).view(np.float16)
    else:
        a = np.array(ilists[5], npdt)
    return name, a.reshape(dims) if dims else a.reshape(())


def attribute(b: bytes) -> tuple[str, Any]:
    name, typ = "", 0
    val: Any = None
    floats: list[float] = []
    ints: list[int] = []
    strings: list[str] = []
    tensors: list[Array] = []
    for f, wt, v in fields(b):
        if f == 1:
            name = _bytes(v).decode()
        elif f == 2:
            val = struct.unpack("<f", _bytes(v))[0]
        elif f == 3:
            val = _sint64(_int(v))
        elif f == 4:
            val = _bytes(v).decode(errors="replace")
        elif f == 5:
            val = tensor(_bytes(v))[1]
        elif f == 6:
            val = graph(_bytes(v))
        elif f == 7:
            floats += _floats(v, wt, "f", 4)
        elif f == 8:
            ints += _packed_varints(v, wt)
        elif f == 9:
            strings.append(_bytes(v).decode(errors="replace"))
        elif f == 10:
            tensors.append(tensor(_bytes(v))[1])
        elif f == 20:
            typ = _int(v)
    if typ == 6 or (val is None and floats):
        val = floats
    elif typ == 7 or (val is None and ints):
        val = ints
    elif typ == 8 or (val is None and strings):
        val = strings
    elif typ == 9:
        val = tensors
    elif val is None and typ in (6, 7, 8):
        val = []
    return name, val


def node(b: bytes) -> Node:
    n: Node = {"op": "", "name": "", "inputs": [], "outputs": [], "attrs": {}, "domain": ""}
    for f, wt, v in fields(b):
        if f == 1:
            n["inputs"].append(_bytes(v).decode())
        elif f == 2:
            n["outputs"].append(_bytes(v).decode())
        elif f == 3:
            n["name"] = _bytes(v).decode()
        elif f == 4:
            n["op"] = _bytes(v).decode()
        elif f == 5:
            k, a = attribute(_bytes(v))
            n["attrs"][k] = a
        elif f == 7:
            n["domain"] = _bytes(v).decode()
    return n


def value_info(b: bytes) -> ValueInfo:
    name, et = "", 0
    dims: list[Dim] | None = None
    for f, wt, v in fields(b):
        if f == 1:
            name = _bytes(v).decode()
        elif f == 2:
            for f2, _, v2 in fields(_bytes(v)):
                if f2 == 1:  # tensor_type
                    for f3, _, v3 in fields(_bytes(v2)):
                        if f3 == 1:
                            et = _int(v3)
                        elif f3 == 2:
                            dims = []
                            for f4, _, v4 in fields(_bytes(v3)):
                                if f4 == 1:
                                    d: Dim = "?"
                                    for f5, _, v5 in fields(_bytes(v4)):
                                        d = _int(v5) if f5 == 1 else _bytes(v5).decode()
                                    dims.append(d)
    return name, et, dims


def graph(b: bytes) -> Graph:
    g: Graph = {"name": "", "nodes": [], "initializers": {}, "inputs": [], "outputs": [], "value_info": {}}
    for f, wt, v in fields(b):
        if f == 1:
            g["nodes"].append(node(_bytes(v)))
        elif f == 2:
            g["name"] = _bytes(v).decode()
        elif f == 5:
            k, a = tensor(_bytes(v))
            g["initializers"][k] = a
        elif f == 11:
            g["inputs"].append(value_info(_bytes(v)))
        elif f == 12:
            g["outputs"].append(value_info(_bytes(v)))
        elif f == 13:
            k, et, d = value_info(_bytes(v))
            g["value_info"][k] = (et, d)
    # graph inputs that are initializers are not real inputs
    g["inputs"] = [x for x in g["inputs"] if x[0] not in g["initializers"]]
    return g


def load(path: str | Path) -> Model:
    b = Path(path).read_bytes()
    opset: dict[str, int] = {}
    producer, ir_version = "", 0
    g: Graph | None = None
    for f, wt, v in fields(b):
        if f == 1:
            ir_version = _int(v)
        elif f == 2:
            producer = _bytes(v).decode()
        elif f == 7:
            g = graph(_bytes(v))
        elif f == 8:
            dom, ver = "", 0
            for f2, _, v2 in fields(_bytes(v)):
                if f2 == 1:
                    dom = _bytes(v2).decode()
                elif f2 == 2:
                    ver = _int(v2)
            opset[dom] = ver
    if g is None:
        raise ValueError(f"{path}: no graph in this ONNX file")
    return {"opset": opset, "producer": producer, "ir_version": ir_version, "graph": g}


if __name__ == "__main__":
    from collections import Counter
    m = load(sys.argv[1])
    g = m["graph"]
    print("opset", m["opset"], "producer", m["producer"])
    print("inputs", g["inputs"])
    print("outputs", g["outputs"])
    print("initializers", len(g["initializers"]), sum(a.size for a in g["initializers"].values()), "values")
    print(Counter(n["op"] for n in g["nodes"]))
    if "-v" in sys.argv:
        for n in g["nodes"]:
            print(n["op"], n["inputs"], "->", n["outputs"], {k: (v if not hasattr(v, "shape") else f"T{list(v.shape)}") for k, v in n["attrs"].items()})

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

import numpy as np

DTYPES = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64,
          9: np.bool_, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(b):
    """Yield (field_number, wire_type, value) for one message; value = int (varint/fixed) or bytes."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
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


def _sint64(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _packed_varints(v, wt):
    if wt == 0:
        return [_sint64(v)]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(_sint64(x))
    return out


def tensor(b):
    dims, dt, name, raw = [], 1, "", None
    lists = {4: [], 5: [], 7: [], 10: [], 11: []}
    for f, wt, v in fields(b):
        if f == 1:
            dims += _packed_varints(v, wt)
        elif f == 2:
            dt = v
        elif f == 8:
            name = v.decode()
        elif f == 9:
            raw = bytes(v)
        elif f == 4:
            lists[4] += list(struct.unpack(f"<{len(v)//4}f", v)) if wt == 2 else [struct.unpack("<f", v)[0]]
        elif f == 10:
            lists[10] += list(struct.unpack(f"<{len(v)//8}d", v)) if wt == 2 else [struct.unpack("<d", v)[0]]
        elif f in (5, 7, 11):
            lists[f] += _packed_varints(v, wt)
        elif f == 14 and v == 1:
            raise ValueError(f"external data not supported ({name})")
    npdt = DTYPES[dt]
    if raw is not None:
        a = np.frombuffer(raw, npdt).copy()
    elif dt == 1:
        a = np.array(lists[4], np.float32)
    elif dt == 11:
        a = np.array(lists[10], np.float64)
    elif dt == 7:
        a = np.array(lists[7], np.int64)
    elif dt in (13, 12):
        a = np.array(lists[11], npdt)
    elif dt == 10:
        a = np.array(lists[5], np.uint16).view(np.float16)
    else:
        a = np.array(lists[5], npdt)
    return name, a.reshape(dims) if dims else a.reshape(())


def attribute(b):
    name, val, typ = "", None, 0
    floats, ints, strings, tensors = [], [], [], []
    for f, wt, v in fields(b):
        if f == 1:
            name = v.decode()
        elif f == 2:
            val = struct.unpack("<f", v)[0]
        elif f == 3:
            val = _sint64(v)
        elif f == 4:
            val = bytes(v).decode(errors="replace")
        elif f == 5:
            val = tensor(v)[1]
        elif f == 6:
            val = graph(v)
        elif f == 7:
            floats += list(struct.unpack(f"<{len(v)//4}f", v)) if wt == 2 else [struct.unpack("<f", v)[0]]
        elif f == 8:
            ints += _packed_varints(v, wt)
        elif f == 9:
            strings.append(bytes(v).decode(errors="replace"))
        elif f == 10:
            tensors.append(tensor(v)[1])
        elif f == 20:
            typ = v
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


def node(b):
    n = {"op": "", "name": "", "inputs": [], "outputs": [], "attrs": {}, "domain": ""}
    for f, wt, v in fields(b):
        if f == 1:
            n["inputs"].append(v.decode())
        elif f == 2:
            n["outputs"].append(v.decode())
        elif f == 3:
            n["name"] = v.decode()
        elif f == 4:
            n["op"] = v.decode()
        elif f == 5:
            k, a = attribute(v)
            n["attrs"][k] = a
        elif f == 7:
            n["domain"] = v.decode()
    return n


def value_info(b):
    name, et, dims = "", 0, None
    for f, wt, v in fields(b):
        if f == 1:
            name = v.decode()
        elif f == 2:
            for f2, _, v2 in fields(v):
                if f2 == 1:  # tensor_type
                    for f3, _, v3 in fields(v2):
                        if f3 == 1:
                            et = v3
                        elif f3 == 2:
                            dims = []
                            for f4, _, v4 in fields(v3):
                                if f4 == 1:
                                    d = "?"
                                    for f5, _, v5 in fields(v4):
                                        d = v5 if f5 == 1 else v5.decode()
                                    dims.append(d)
    return name, et, dims


def graph(b):
    g = {"name": "", "nodes": [], "initializers": {}, "inputs": [], "outputs": [], "value_info": {}}
    for f, wt, v in fields(b):
        if f == 1:
            g["nodes"].append(node(v))
        elif f == 2:
            g["name"] = v.decode()
        elif f == 5:
            k, a = tensor(v)
            g["initializers"][k] = a
        elif f == 11:
            g["inputs"].append(value_info(v))
        elif f == 12:
            g["outputs"].append(value_info(v))
        elif f == 13:
            k, et, d = value_info(v)
            g["value_info"][k] = (et, d)
    # graph inputs that are initializers are not real inputs
    g["inputs"] = [x for x in g["inputs"] if x[0] not in g["initializers"]]
    return g


def load(path):
    b = open(path, "rb").read()
    m = {"opset": {}, "producer": "", "ir_version": 0}
    for f, wt, v in fields(b):
        if f == 1:
            m["ir_version"] = v
        elif f == 2:
            m["producer"] = v.decode()
        elif f == 7:
            m["graph"] = graph(v)
        elif f == 8:
            dom, ver = "", 0
            for f2, _, v2 in fields(v):
                if f2 == 1:
                    dom = v2.decode()
                elif f2 == 2:
                    ver = v2
            m["opset"][dom] = ver
    return m


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

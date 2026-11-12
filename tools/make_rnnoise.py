"""RNNoise v0.1.1 weights (src/rnn_data.c int8 arrays, scale 1/256) -> models/rnnoise package.

    python tools/make_rnnoise.py oss_dl/rnnoise models/rnnoise

weights.bin = raw f32 in the order of LAYERS (row-major as stored: weights[in][out] with gate
blocks z|r|h for GRUs, i.e. W[j*3N + gate*N + i]). Only the published weight data is used;
the engine (rust/vi_rnnoise) and reference (tools/rnnoise_ref.py) implement the algorithm.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from typing import Any

import numpy as np
import numpy.typing as npt

LAYERS = [  # name, arrays, (nb_inputs, nb_neurons, activation)
    ("input_dense", ["input_dense_weights", "input_dense_bias"], (42, 24, "tanh")),
    ("vad_gru", ["vad_gru_weights", "vad_gru_recurrent_weights", "vad_gru_bias"], (24, 24, "relu")),
    ("noise_gru", ["noise_gru_weights", "noise_gru_recurrent_weights", "noise_gru_bias"], (90, 48, "relu")),
    ("denoise_gru", ["denoise_gru_weights", "denoise_gru_recurrent_weights", "denoise_gru_bias"], (114, 96, "relu")),
    ("denoise_output", ["denoise_output_weights", "denoise_output_bias"], (96, 22, "sigmoid")),
    ("vad_output", ["vad_output_weights", "vad_output_bias"], (24, 1, "sigmoid")),
]


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = (src / "src_rnn_data.c").read_text()
    arrays = {m.group(1): np.array([int(v) for v in m.group(3).replace("\n", " ").split(",") if v.strip()], np.float32)
              for m in re.finditer(r"static const rnn_weight (\w+)\[(\d+)\] = \{([^}]*)\}", text)}
    data: list[npt.NDArray[np.float32]] = []
    layout: list[dict[str, Any]] = []
    off = 0
    for name, arrs, (ni, nn, act) in LAYERS:
        g = 3 if "gru" in name else 1
        exp = {"weights": ni * nn * g, "recurrent_weights": nn * nn * g, "bias": nn * g}
        ent: dict[str, Any] = {"name": name, "inputs": ni, "neurons": nn, "activation": act, "tensors": {}}
        for a in arrs:
            kind = a[len(name) + 1:]
            v = arrays[a]
            assert v.size == exp[kind], (a, v.size, exp[kind])
            ent["tensors"][kind] = {"offset": off, "size": int(v.size)}
            data.append(v / 256.0)
            off += v.size
        layout.append(ent)
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate(data).astype("<f4").tofile(out / "weights.bin")
    up = {f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in ("src_rnn_data.c", "src_denoise.c", "src_pitch.c")}
    man: dict[str, Any] = {
        "name": "rnnoise", "display_name": "RNNoise v0.1.1 (48 kHz, Bark-band GRU + pitch filter)", "engine": "rnnoise",
        "weights": "weights.bin", "sample_rate": 48000, "frame_size": 480, "latency": 480, "local_only": False,
        "license": "BSD-3-Clause (Copyright (c) 2017 Mozilla, 2018 Gregor Richards; COPYING)",
        "source": "https://github.com/xiph/rnnoise/blob/6cbfd53eb348a8d394e0757b4025c6ded34eb2b6/src/rnn_data.c",
        "version": "tag v0.1.1 = 6cbfd53eb348a8d394e0757b4025c6ded34eb2b6 (model of Valin, 'A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement', MMSP 2018)",
        "upstream_sha256": {"src/rnn_data.c": up["src_rnn_data.c"]},
        "published_metrics": "paper: PESQ (P.862) gains over noisy at 0-20 dB SNR on their own test mixes; third-party VCTK-DEMAND: PESQ ~2.3, STOI ~0.92 (commonly reported)",
        "notes": "Custom-DSP plugin `rnnoise` (rust/vi_rnnoise): 960-pt FFT with Vorbis power-complementary window, 22 Bark-ish bands (eband5ms), 42 features (cepstrum of log band energies + deltas, pitch correlation DCT, pitch period, spectral variability), pitch analysis (2x decimation + LPC whitening, coarse/fine xcorr search, doubling removal), comb pitch filter, band-gain interpolation, gain smoothing, OLA. Input/output scaled to the int16 range as the model expects.",
        "config": {"layers": layout, "input_scale": 32768.0},
    }
    man["sha256"] = {"weights.bin": hashlib.sha256((out / "weights.bin").read_bytes()).hexdigest()}
    man["package_bytes"] = (out / "weights.bin").stat().st_size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"rnnoise: {off} weights, layers {[(e['name'], e['inputs'], e['neurons']) for e in layout]}")


if __name__ == "__main__":
    main()

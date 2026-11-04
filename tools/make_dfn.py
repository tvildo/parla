"""DeepFilterNet3 (Schroeter et al., Interspeech 2023) -> one vi package for the `dfn` plugin.

    python tools/make_dfn.py oss_dl/dfn models/deepfilternet3

The three published ONNX graphs (enc, erb_dec, df_dec; whole-sequence exports) are converted with
our reader into streaming form (stream_time: causal time Pads -> cache states; stream_rnn: GRU
states) and composed into one graph: inputs feat_erb [1,1,1,32], feat_spec [1,2,1,96];
outputs m (ERB gains [1,1,1,32]), coefs (deep-filter coefficients [1,1,96,10]), lsnr.
The DSP (analysis/synthesis, ERB features + mean norm, complex unit-norm features, gain
application with 2-frame conv lookahead, order-5 deep filtering with 2-frame lookahead, LSNR
stage gating) is our implementation in rust/vi_dfn, configured by manifest.config.
"""
import configparser
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx2graph import convert, write_package  # noqa: E402


def erb_fb(sr, fft_size, nb_bands, min_nb_freqs):
    """ERB band widths (bins) — published formula (freq2erb = 9.265 ln(1 + f / (24.7*9.265)))."""
    f2e = lambda f: 9.265 * math.log1p(f / (24.7 * 9.265))  # noqa: E731
    e2f = lambda e: 24.7 * 9.265 * (math.exp(e / 9.265) - 1)  # noqa: E731
    lo, hi = f2e(0), f2e(sr // 2)
    step = (hi - lo) / nb_bands
    fw = sr / fft_size
    erb, prev, over = [], 0, 0
    for i in range(1, nb_bands + 1):
        fb = int(round(e2f(lo + i * step) / fw + 1e-9)) if False else int(math.floor(e2f(lo + i * step) / fw + 0.5))
        nb = fb - prev - over
        if nb < min_nb_freqs:
            over = min_nb_freqs - nb
            nb = min_nb_freqs
        else:
            over = 0
        erb.append(nb)
        prev = fb
    erb[-1] += 1
    too = sum(erb) - (fft_size // 2 + 1)
    if too > 0:
        erb[-1] -= too
    assert sum(erb) == fft_size // 2 + 1
    return erb


def norm_alpha(sr, hop, tau):
    a0 = math.exp(-(hop / sr) / tau)
    a, prec = 1.0, 3
    while a >= 1.0:
        a = round(a0 * 10 ** prec) / 10 ** prec
        prec += 1
    return a


def prefixed(nodes, inits, gin, gst, pfx, keep):
    ren = lambda t: t if (not t or t in keep) else pfx + t  # noqa: E731
    nodes = [{**n, "name": pfx + n["name"], "inputs": [ren(i) for i in n["inputs"]], "outputs": [ren(o) for o in n["outputs"]]} for n in nodes]
    return nodes, {pfx + k: v for k, v in inits.items()}, [{**g, "name": ren(g["name"])} for g in gin], [{**s, "in": ren(s["in"]), "out": ren(s["out"])} for s in gst]


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    d = src / "onnx/tmp/export"
    cfg = configparser.ConfigParser()
    cfg.read(d / "config.ini")
    df = cfg["df"]
    enc_outs = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    n1, i1, g1, s1, _ = convert(str(d / "enc.onnx"), [], "feat_erb", "lsnr", stream_rnn=True, stream_time=2,
                                pins=["feat_erb", "feat_spec"], pouts=enc_outs)
    n2, i2, g2, s2, _ = convert(str(d / "erb_dec.onnx"), [], "emb", "m", stream_rnn=True, stream_time=2,
                                pins=["emb", "e3", "e2", "e1", "e0"], pouts=["m"])
    n3, i3, g3, s3, _ = convert(str(d / "df_dec.onnx"), [], "emb", "coefs", stream_rnn=True, stream_time=2,
                                pins=["emb", "c0"], pouts=["coefs"])
    # encoder keeps its names (inputs feat_erb/feat_spec, outputs e*/emb/c0/lsnr); decoders read them
    n1, i1, g1, s1 = prefixed(n1, i1, g1, s1, "enc/", set(enc_outs + ["feat_erb", "feat_spec"]))
    n2, i2, g2, s2 = prefixed(n2, i2, g2, s2, "erb/", set(["emb", "e0", "e1", "e2", "e3", "m"]))
    n3, i3, g3, s3 = prefixed(n3, i3, g3, s3, "df/", set(["emb", "c0", "coefs"]))
    nodes = n1 + n2 + n3
    states = s1 + s2 + s3
    sin = {s["in"] for s in states}
    ginputs = [g for g in g1 if g["name"] in ("feat_erb", "feat_spec")] + [g for g in g1 + g2 + g3 if g["name"] in sin]
    sr, fft, hop = int(df["sr"]), int(df["fft_size"]), int(df["hop_size"])
    erb = erb_fb(sr, fft, int(df["nb_erb"]), int(df["min_nb_erb_freqs"]))
    up = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in [src / "models_DeepFilterNet3_onnx.tar.gz"]}
    man = {
        "display_name": "DeepFilterNet3 (48 kHz, ERB gains + deep filtering)",
        "license": "MIT OR Apache-2.0 (Copyright Hendrik Schroeter; LICENSE-MIT / LICENSE-APACHE)",
        "source": "https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3_onnx.tar.gz",
        "version": "git d375b2d8309e0935d165700c91da9de862a99c31 (2024-10-17), models/DeepFilterNet3_onnx.tar.gz (enc/erb_dec/df_dec ONNX + config.ini, exported 2023-05-23)",
        "upstream_sha256": up,
        "paper": "Schroeter, Maier, Escalante-B., Rosenkranz, DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement, Interspeech 2023",
        "published_metrics": "VoiceBank-DEMAND test: PESQ 3.17, CSIG 4.34, CBAK 3.61, COVL 3.77, STOI 0.944 (README table, DeepFilterNet3)",
        "local_only": False, "sample_rate": sr, "frame_size": hop, "latency": hop * (1 + int(cfg["deepfilternet"]["conv_lookahead"])),
        "notes": "Custom-DSP plugin `dfn` (rust/vi_dfn): Vorbis-window STFT 960/480 (analysis norm 1/960), ERB features (32 bands, 10 log10 band power, exponential mean norm, /40), complex features on 96 bins (exponential unit norm), ERB gains applied to the frame 2 hops back (conv lookahead), order-5 deep filtering over noisy frames t-4..t for bins < 96, LSNR stage gating available but disabled by config (thresholds +-1000 dB: gains + DF on every frame, as the paper model), OLA. Graph = 3 ONNX stages streamed (time caches + GRU states) by tools/make_dfn.py.",
        "config": {
            "sr": sr, "fft_size": fft, "hop_size": hop, "nb_erb": int(df["nb_erb"]), "nb_df": int(df["nb_df"]),
            "df_order": int(df["df_order"]), "df_lookahead": int(df["df_lookahead"]),
            "conv_lookahead": int(cfg["deepfilternet"]["conv_lookahead"]), "erb_widths": erb,
            "norm_alpha": norm_alpha(sr, hop, float(df["norm_tau"])), "mean_norm_init": [-60.0, -90.0], "unit_norm_init": [0.001, 0.0001],
            # LSNR stage gating OFF (= the paper's offline model, which always applies gains + DF). The
            # runtime defaults of the original streaming engine (-10 / 30 / 20 dB) cost ~0.9 PESQ on VBD here:
            # in 20..30 dB frames they apply ERB gains only, but the DFN3 ERB gains below 4.8 kHz are
            # untrained (mask loss 0, DF always overwrites those bins in training).
            "min_db_thresh": -1000.0, "max_db_erb_thresh": 1000.0, "max_db_df_thresh": 1000.0,
            "inputs": ["feat_erb", "feat_spec"], "outputs": {"gains": "m", "coefs": "coefs", "lsnr": "lsnr"},
        },
    }
    m = write_package(out, "deepfilternet3", nodes, {**i1, **i2, **i3}, ginputs, states, "lsnr", man)
    g = json.loads((Path(out) / "graph.json").read_text())
    g["outputs"] = [{"name": "m"}, {"name": "coefs"}, {"name": "lsnr"}]
    (Path(out) / "graph.json").write_text(json.dumps(g, separators=(",", ":")))
    m["engine"] = "dfn"
    m["sha256"]["graph.json"] = hashlib.sha256((Path(out) / "graph.json").read_bytes()).hexdigest()
    (Path(out) / "manifest.json").write_text(json.dumps(m, indent=1))
    json.dump({k: m[k] for k in ("display_name", "license", "source", "version", "upstream_sha256", "paper", "published_metrics", "notes")},
              open("tools/specs/deepfilternet3.json", "w"), indent=1)
    print(f"deepfilternet3: {len(nodes)} nodes, {len(states)} states, erb {erb}, alpha {man['config']['norm_alpha']}, {m['package_bytes']} bytes")


if __name__ == "__main__":
    main()

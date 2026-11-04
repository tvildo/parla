# vi model package format (engine kind `graph`)

A model package is **pure data**: a declarative manifest + an operator graph + raw weights.
Comparing models = switching packages; no per-model code anywhere (Rust engine, wasm, JS host).
All DSP (framing, windows, STFT/iSTFT, filterbanks, features, masks, OLA, resampling in the host
worklet) is implemented once in our Rust engine (`rust/vi_rs/src/{graph,ops,dsp,pipeline}.rs`)
and mirrored by our independent numpy reference executor `tools/graph_ref.py`.

## Hard rules (provenance / "weights only")

- From third-party models we take **only weights and the architecture/graph description**
  (ONNX graph, MIL text, published constants such as filterbank matrices or windows).
  **Never run their code**: no onnxruntime, no official PyTorch/TF inference, export scripts or
  model classes, no third-party DSP code in the inference or benchmark path.
- Reading weight containers is allowed: ONNX protobuf via our own `tools/onnx_pb.py`;
  PyTorch checkpoints only via `torch.load(weights_only=True)` (tensors only).
- Every denoised output in every table comes from **our engine** (Rust native, wasm in Node,
  or the Chrome AudioWorklet). Validation = our `graph_ref.py` vs Rust native vs wasm, and
  published metric scores as a sanity check. Scoring tools (pesq, pystoi, DNSMOS) are evaluation-only.
- Each manifest records `license`, `source` (exact URL), `version` (commit / release) and the
  sha256 of every upstream file used. Only permissively licensed weights (MIT/BSD/Apache) belong here.

## Budgets (design constraints)

- **Size**: one wasm with every engine kind (passthrough, graph + ops + dsp + pipeline, dfn, rnnoise)
  around 250 KB raw / 100 KB gzip (simd build; `vi_kernels` at opt-level 3, everything else
  opt-level z, binaryen `wasm-opt -Oz`).
  Weights are separate package files.
- **Speed**: load-time compilation to a flat plan (constant folding, StreamConv ring-buffer fusion,
  activation fusion, Toeplitz convs, arena memory planning, pre-packed weights); zero allocation and
  no JSON/hash lookups in `process`. The generic `graph` engine runs within a few percent of a
  hand-written engine for the same network when batched (T=12), +16-20 % frame by frame (T=1).

## Browser model store

`web/vi_store.js` (Cache API `vi-models-v1`, keys `vi-store/<name>/<file>`): `registry()` (index.json
+ manifests), `pull(name)` (fetch package files, verify `sha256`, cache), `addFiles(files)` (local
manifest + files), `list()`, `get(name)`, `remove(name)`. The host is model-agnostic: it builds
the engine config with `engineConfig(manifest, graphJson)` (`web/vi_engine.js`) and hands the
weights blob to the same wasm. Works offline once pulled.

## Files

```
models/<name>/manifest.json   provenance + pipeline (+ "graph"/"weights" file names)
models/<name>/graph.json      operator graph (this spec)
models/<name>/weights.bin     raw little-endian f32, no header; tensors addressed by float offset
```
`models/index.json` lists package names (`{"models": [...]}`); the web model store, `vi_cli`,
`run_node.mjs` and `bench.py` discover models through it.

### manifest.json

| Field | Meaning |
|---|---|
| `name`, `display_name` | id (directory name) / UI label |
| `engine` | engine kind = **plugin**: `graph` for packages in this format (plugin crate `vi_graph`); other plugins: `passthrough` (shell), `dfn`, `rnnoise` (custom-DSP plugins). The browser store loads the bundle or the plugin's own wasm module (`plugins/vi_<plugin>_<build>.wasm`). |
| `graph`, `weights` | file names relative to the manifest (`graph.json`, `weights.bin`) |
| `sha256` | `{file: hex}` of the package files (the browser store verifies them) |
| `sample_rate`, `frame_size`, `latency` | informational; the engine reports authoritative values |
| `license`, `source`, `version`, `upstream_sha256`, `notes` | provenance |
| `pipeline` | front-end / back-end spec (below) |
| `config` | optional engine overrides, e.g. `{"frames_per_call": 12}` |

Hosts build the engine config as `{"pipeline": manifest.pipeline, "graph": <graph.json>, ...manifest.config}`
and pass `weights.bin` as the weight blob: `engine_create("graph", config_json, weights)`.

### pipeline

```jsonc
"pipeline": {
  "sample_rate": 16000,
  "hop": 256,               // new input samples per model frame
  "frames_per_call": 1,     // T: model frames per graph execution (graph dims may use "T")
  "input": "mix",           // graph input fed by the front-end
  "output": "enh",          // graph output consumed by the back-end
  "graph_latency": 0,       // algorithmic delay inside the graph, samples (lookahead)
  "front": {...},
  "back": {...}
}
```
The engine processes `frame_size = T * hop` samples per call. Reported latency =
DSP latency (`frame_len - hop` for framed front-ends, `+ mask_delay*hop`) + `graph_latency`.

**Front-ends** (write the graph input; its element count must equal the produced buffer;
layouts are canonical, converters insert Transpose/Reshape nodes when a model wants another):

| type | params | graph input layout |
|---|---|---|
| `time` | – | `[T*hop]` new samples |
| `frames` | `frame_len`, `window` | `[T][frame_len]`: last `frame_len` samples per frame (zero history at start), times window |
| `stft` | `n_fft`, `frame_len` (≤ n_fft, zero-padded right; default n_fft), `window`, `feature`, `log_eps`, `filterbank` (tensor `[n_bins][n_bands]`, applied to magnitude/power features), `center: false` | complex: `[T][F][2]` (re, im), F = n_fft/2+1; real features: `[T][F]` or `[T][n_bands]` |

`feature`: `complex` | `magnitude` | `power` | `log_power` (ln(p + eps)) | `log10_power` |
`log_magnitude` | `log10_magnitude` | `log1p_magnitude`. Forward FFT is unnormalized:
`X[k] = sum_n w[n] x[n] e^{-2 pi i k n / N}`.

**Back-ends**:

| type | params | graph output |
|---|---|---|
| `waveform` | – | `[T*hop]` output samples |
| `raw` | – | any (≤ T*hop values): copied to the start of the output frame, rest zero — analysis/scorer models (e.g. `dnsmos-p835`: frame 144160, 3 raw scores) |
| `ola` | `frame_len`, `window` (synthesis), `ola_norm` | `[T][frame_len]` time frames, windowed and overlap-added at `hop` |
| `istft` | `source`: `complex_spectrum` \| `mag_mask` \| `complex_mask`; `window` (synthesis); `mask_delay` (frames); `ola_norm` | `[T][F][2]` spectrum or complex mask, `[T][F]` real mask |

`istft` requires an `stft` front-end; masks multiply the *noisy* STFT frame (delayed by
`mask_delay` frames). Inverse: `x[n] = (1/N) sum_k Y[k] e^{+2 pi i k n / N}` (Hermitian,
imag of DC/Nyquist ignored), times synthesis window, overlap-add. `ola_norm`: `none` (default) or
`wola` = divide by the periodic steady-state sum of `w_analysis * w_synthesis` (like torch.istft).
Output sample emitted for a frame = the oldest `hop` samples of the OLA buffer (DSP latency
`frame_len - hop`).

`window`: `"none"`, `"hann"` (periodic, = torch.hann_window), `"hann_sym"`, `"sqrt_hann"`
(periodic), `"hamming"` (periodic), `"vorbis"`, `{"tensor": "<initializer>"}` or `{"values": [...]}`.

### graph.json

```jsonc
{
  "format": "vi-graph", "version": 1,
  "name": "gtcrn",
  "inputs":  [{"name": "mix", "shape": [1, 257, 1, 2]}],
  "outputs": [{"name": "enh"}],
  "states":  [{"in": "conv_cache", "out": "conv_cache_out", "shape": [2, 1, 16, 16, 33]}],
  "initializers": {"w0": {"offset": 0, "shape": [16, 9, 1, 5]}, "eps": {"shape": [], "value": [1e-12]}},
  "nodes": [{"op": "Conv", "name": "conv0", "inputs": ["x", "w0", "b0"], "outputs": ["y"],
             "attrs": {"kernel_shape": [1, 5], "strides": [1, 2], "pads": [0, 2, 0, 2], "dilations": [1, 1], "group": 1}}]
}
```

- **Tensors** are dense f32, row-major. Integer-valued constants (indices, shapes) are stored as
  f32 too (exact below 2^24). `""` = omitted optional input.
- **Dims** are integers or affine strings in the symbol `T` (frames per call): `"T"`, `"40*T"`,
  `"40*T+40"`, `"2*T-1"`. Allowed in input/state shapes and integer attributes (`shape`, slice
  `starts/ends`). All other shapes are inferred at load by each op's shape function; the engine
  validates them and refuses unknown ops/attrs or inconsistent shapes.
- **States**: `in` is a graph input, `out` a graph tensor of the same shape. Zero at start/reset;
  after every execution `in <- out` (buffer swap, no copy): the cache I/O of streaming ONNX
  exports.
- **Nodes** are in topological order. Inputs that are all constant are folded at load.
- **Initializers**: `offset` (in floats) into weights.bin + `shape`, or inline `value` (small).

### Op set (semantics = ONNX opset 17 unless noted; f32 only; batch-free shapes allowed)

Inputs that ONNX passes as tensors but that must be static here are **attributes**:
`Reshape.shape`, `Slice.starts/ends/axes/steps`, `Pad.pads/value`, `Squeeze/Unsqueeze.axes`,
`ReduceX.axes`, `Split.split`, `Clip.min/max`, `Expand.shape`, `Tile.repeats`.

| Group | Ops |
|---|---|
| Linear | `MatMul` (numpy broadcasting), `Gemm` (alpha, beta, transA, transB) |
| Conv | `Conv` (1-D/2-D, group, dilations, strides, pads, bias), `ConvTranspose` (1-D/2-D, group, dilations, strides, pads, output_padding) |
| Recurrent | `LSTM`, `GRU` (forward direction, batch 1; `linear_before_reset`; inputs X, W, R, B, seq_lens (ignored, must be empty), initial_h, initial_c; outputs Y, Y_h, Y_c) |
| Norm | `BatchNormalization` (inference; folded into the preceding conv by converters where possible), `LayerNormalization` (axis, epsilon), `InstanceNormalization`, **`ChannelNorm`** (custom: inputs x, gamma, beta; attrs axis, epsilon; `y = gamma*(x-mean)/sqrt(mean((x-mean)^2)+eps) + beta`, reduced over `axis`, gamma/beta broadcast along `axis`) |
| Activation | `Relu`, `LeakyRelu`, `PRelu`, `Sigmoid`, `Tanh`, `Elu`, `Softplus`, `HardSigmoid`, `Clip` |
| Elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Max`, `Min` (2 inputs, broadcasting), `Sqrt`, `Abs`, `Exp`, `Log`, `Neg`, `Reciprocal`, `Identity`, `Cast` (no-op) |
| Reduce | `ReduceMean`, `ReduceSum`, `ReduceMax` (axes, keepdims) |
| Shape | `Concat`, `Split`, `Slice`, `Gather` (constant indices), `ScatterND` (constant indices), `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Flatten`, `Pad` (constant/reflect/edge), `Expand`, `Tile` |
| Spectral (custom) | **`Rfft`** (last axis n -> `[..., n/2+1, 2]`, unnormalized), **`Irfft`** (attr `n`; `[..., n/2+1, 2]` -> `[..., n]`, 1/n) |

Extension: add an op = shape/prepare function + kernel in `src/engine/vi_graph/src/ops.rs` (registry
`prepare_node`) and the same op in `tools/graph_ref.py`; document it here.

## Converters (tools/)

- `tools/onnx2graph.py` — ONNX (our protobuf reader) -> package; constant folding with
  `graph_ref` ops, static-input-to-attribute lowering, BatchNorm folding, shape check.
- `tools/graph_ref.py` — numpy reference executor of packages (graph + pipeline); used for
  validation (`graph_ref` vs Rust native vs wasm).

## Custom-DSP plugin packages (`engine` != `graph`)

Models whose DSP cannot be expressed with the declarative stages ship as packages for a dedicated
plugin crate; the network part still uses the graph format where possible.

| engine | crate | package | custom stages (ours, from the papers + published constants) |
|---|---|---|---|
| `rnnoise` | `src/engine/vi_rnnoise` | `manifest.json` (`config.layers`: name/inputs/neurons/activation + tensor offsets, `input_scale`) + `weights.bin` (int8 / 256 as f32) | 960-pt Vorbis-window FFT (1/N), 22 eband5ms bands, 42 features (log-band cepstrum + deltas, pitch-correlation DCT, pitch period, spectral variability), pitch analysis (2x decimation + LPC, coarse/fine xcorr, doubling removal), GRU stack (dense + 3 GRU + 2 dense, inline), comb pitch filter, gain smoothing, OLA; int16-range scaling |
| `dfn` | `src/engine/vi_dfn` (+ `vi_graph`) | `manifest.json` (`config`: sr, fft/hop, nb_erb, nb_df, df_order, df/conv lookahead, `erb_widths`, `norm_alpha`, norm inits, LSNR thresholds, graph I/O names) + `graph.json` (3 ONNX stages streamed) + `weights.bin` | Vorbis STFT 960/480 (1/960), ERB features (10 log10 band power, EMA mean norm, /40), complex unit-norm features, LSNR stage gating (disabled in the shipped config: thresholds +-1000 dB, gains + DF every frame as the paper model), ERB gains on frame t-conv_lookahead, order-5 deep filtering over noisy frames, OLA |

Analysis/scorer packages (e.g. a DNSMOS scorer) can be graph packages with a `raw` back-end; they are not
listed in `index.json`.

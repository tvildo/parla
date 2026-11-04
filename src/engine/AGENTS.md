# src/engine/ — the Parla engine: std-only pluggable noise suppression (native + wasm)

No third-party crates, no wasm-bindgen. Model packages are data (`models/FORMAT.md`); the engine is model-agnostic.

Cargo **workspace** (`src/engine/Cargo.toml`, shared `target/`, profiles at the root):

| Crate | What |
|---|---|
| `vi_kernels` | SIMD math, **always opt-level 3** (`[profile.release.package.vi_kernels]`): `kernels!` per backend (GEMM `gemm`/`gemm_ld`/`gemm_s`, relu, leaky, ChannelNorm, depthwise k=3, add), `util` (broadcast binary ops, Cephes expf/sigmoid/tanh, gather, madd, small mat-vec, GRU/LSTM gates), `Backend` (scalar / avx2+fma runtime / simd128 compile-time) + `call!` dispatch macro. Relaxed-simd build uses `f32x4_relaxed_madd`. |
| `vi_audio` | shared audio shell: `vad` (voice activity on the denoised output: 10 ms level vs adaptive floor + 9 dB, combined with a model's own VAD, 30 ms attack / 250 ms hangover), `fft` (mixed-radix Stockham FFT, rFFT/irFFT, windows), `stages` (`FrontEnd` time/frames/stft + features + filterbank; `BackEnd` waveform/ola/istft spectrum/mag mask/complex mask, mask delay, WOLA; `mix`), `resample` (Rust port of `js/resampler.ts`, for native/Node hosts), `json` (minimal parser + `parse_num`). |
| `vi_engine` | contract: `NoiseEngine` (hosts; + `set_stages(n)`/`process_stage(s, in, out)` = one frame split into parts for realtime smoothing, bit-exact == `process_frame`; `speech_prob` = the model's own VAD or -1), `ModelCore` (model block), `driver::Pipeline` (FrontEnd -> ModelCore -> BackEnd from a manifest pipeline spec), `passthrough`. |
| `vi_graph` | plugin `graph`: package loader, symbolic dims, constant folding, StreamConv/activation fusion, arena planning, state swap, cost-balanced `set_stages`; `ops.rs` (op registry + kernels); `GraphCore: ModelCore`. |
| `vi_rnnoise` | plugin `rnnoise`: RNNoise v0.1, our implementation (bands, pitch search, 42 features, GRU net, pitch comb filter, gain smoothing, OLA). Validated vs `tools/rnnoise_ref.py` (78 dB). |
| `vi_dfn` | plugin `dfn`: DeepFilterNet3 streaming DSP (ERB features + mean norm, complex unit norm, LSNR stage gating (off in the package config: DFN3's ERB gains below 4.8 kHz are untrained), ERB gains with conv lookahead, order-5 deep filtering, OLA) around a `vi_graph` graph (3 ONNX stages). Validated vs `tools/dfn_ref.py` (129-134 dB). |
| `vi_rs` | top crate: plugin registry with **stable kind ids** (`KINDS` = passthrough 0, (1 reserved), graph 2, dfn 3, rnnoise 4; features `graph`, `rnnoise`, `dfn`, **default = all three**), `create_engine`, `Manifest`; `host.rs` (**`Host`: the layer every host uses** — engine + on/off/strength mix (`set_mix(strength, ramp_ms)`: g·denoised + (1-g)·input delayed by exactly the engine latency, linear ramp; g = 1 bit-exact engine output, g = 0 bit-exact delayed input) + VAD on the denoised signal + frame parts + a live-instance count); `wasm_api.rs` (C ABI over `Host`, 0 imports — any wasm runtime); `bin/vi_cli.rs` (`run`/`all`/`batch`, `--frame --scalar --reps --T --profile --stages N`); `stages_test.rs` (every package: 2/3/4 parts bit-exact vs unstaged). |
| `vi_otel` / `vi_otel_wasm` | std-only OpenTelemetry subset (metrics + traces, allocation-free recording, OTLP/HTTP JSON); a separate opt-in wasm module. See `vi_otel/AGENTS.md`. |

ModelCore lives in `vi_engine` and the registry in `vi_rs` to avoid a dependency cycle: plugins depend on vi_engine,
the registry depends on the plugins.

`js/` is the browser side of the engine (TypeScript, ES modules; Node runs it as is, browsers get it from
`python build.py ts` as JavaScript with the types erased, lines unchanged): `vi_engine.ts` (hand-written glue over the
C ABI, its exports typed; works in Node, the main thread and AudioWorkletGlobalScope), `worklet.ts` (the `vi-engine`
AudioWorklet: resampling, frame FIFOs, lanes with crossfaded switching, frame parts, voice events, optional telemetry;
its port protocol is the `WorkletIn` / `WorkletOut` types every host posts against), `resampler.ts`, `vi_store.ts`
(Cache API model store), `otel_export.ts`. The extension and the SDK bundle `worklet` into one classic script
(`tools/worklet_bundle.ts`).

### Writing a plugin
1. New crate `vi_<name>` depending on `vi_engine` (+ `vi_audio`, `vi_kernels`), add it to the workspace members.
2. Standard DSP (framing/STFT/masks/OLA expressible in the pipeline spec): implement `ModelCore` and return
   `Box::new(Pipeline::new(pipeline_spec, t, Box::new(core), "<name>")?)`. Custom DSP (RNNoise band features,
   DeepFilterNet ERB/deep filtering): implement `NoiseEngine` directly, reuse `vi_audio::stages` / `fft` for the
   standard parts and document the custom stages in the crate docs + models/FORMAT.md.
3. `vi_rs/Cargo.toml`: optional dep + feature `<name>` (add to `default` for the bundle); `vi_rs/src/lib.rs`:
   `kind_available` + a `#[cfg(feature)]` match arm in `create_engine` (a new kind id appended to `KINDS`, also in
   `js/vi_engine.ts`).
4. Optional: `set_stages`/`process_stage` (split a long frame for realtime smoothing; add it to `stages_test`),
   `speech_prob` (if the model has a VAD output).
5. `build.py` `WASM_VARIANTS` + `tools/wasm_index.py`; manifest `engine: "<name>"`.
Prefer a model package (models/FORMAT.md) over a new kind: the `spectral` baseline is a 23-node graph package.

## Build / run

```sh
python build.py native     # vi_cli -> src/engine/target/release/
python build.py wasm       # 16 modules (5 variants x simd/relaxed/nosimd + vi_otel) -> build/wasm/, wasm-opt -Oz
python build.py regress    # every model package, bit-exact vs tests/golden.json (per platform)
cd src/engine && cargo test --release && cargo clippy --release -- -D clippy::correctness
src/engine/target/release/vi_cli run models/gtcrn/manifest.json in_16k.f32 out.f32 --frame --reps 1
```
Wasm builds: non-kernel code at opt-level z, LTO off, `vi_kernels` O3, `--remap-path-prefix=<src/engine>=/vi`
(no checkout path in the binary), one target dir per variant (RUSTFLAGS differ). Linux builds are byte-reproducible;
Windows vs Linux differ only in panic-location strings. Regression goldens are bit-exact **per platform**: std
exp/sin/cos come from the platform libm (MSVC CRT vs glibc, ~1 ulp); a new platform: check numerically, then
`regress --record`. The regression input is generated (`build.py test_signal`), no recording.

## Numbers

| Module (simd) | raw | gzip |
|---|---|---|
| bundle `vi_rs_simd.wasm` (passthrough + graph + dfn + rnnoise) | 258,776 | — |
| `vi_graph_simd.wasm` | 227,901 | — |
| `vi_dfn_simd.wasm` (includes the graph interpreter) | 218,129 | — |
| `vi_rnnoise_simd.wasm` | 64,432 | — |
| `vi_core_simd.wasm` (shell + passthrough) | 30,528 | — |

Native i9-11900K AVX2 (`build.py regress` run, ms of compute per s of audio): DeepFilterNet3 39, NSNet2 43,
RNNoise 12. GTCRN wasm simd ~194 us per 10 ms of audio. In Chrome every model runs in real time with 0 underruns;
frame parts (2 by default) keep the worst callback at ~1 ms.
Profiling: `vi_cli run <manifest> in.f32 out.f32 --profile [--T n]` (per-op us/frame).

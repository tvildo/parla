// vi_engine.ts — hand-written glue for vi_rs wasm (no wasm-bindgen). ES module; works in
// Node, on the main thread and inside an AudioWorkletGlobalScope (synchronous instantiation).
//
//   const rt = new ViRuntime(wasm)                     // the wasm file's bytes or a WebAssembly.Module
//   const eng = rt.create('graph', configJson, weightsArrayBuffer)
//   eng.info -> {sampleRate, frameSize, latency, inCh, outCh, simd}
//   eng.process(Float32Array in, Float32Array out)     // length = k * frameSize
//   eng.processFrame()                                  // zero-copy: fill eng.inFrame, read eng.outFrame
//   eng.reset(); eng.setScalar(true); eng.destroy()
//   eng.setStages(n) -> parts; eng.processStage(s)     // one frame (inFrame -> outFrame) spread over
//                                                       // `parts` calls, s = 0..parts-1 (realtime smoothing)
//   eng.voice() -> {prob, speaking, levelDb}            // voice activity of the denoised output
//   eng.setMix(strength 0..1, rampMs = 20)              // noise cancellation off (0) .. full (1), ramped
//   eng.getMix() -> {gain, target}

export const KINDS = { passthrough: 0, graph: 2, dfn: 3, rnnoise: 4 } as const;   // stable C-ABI ids (1 is reserved)
export type EngineKind = keyof typeof KINDS;

/** a model package's manifest.json (the fields this glue reads; models/FORMAT.md has the rest) */
export interface Manifest {
  name: string;
  engine: EngineKind;
  config?: Record<string, unknown>;
  graph?: string;
  weights?: string;
  pipeline?: unknown;
  [key: string]: unknown;
}

/** what engine_info reports */
export interface EngineInfo { sampleRate: number; frameSize: number; latency: number; inCh: number; outCh: number; simd: boolean }
export interface Voice { prob: number; speaking: boolean; levelDb: number }

/** the vi_rs wasm module's C ABI (optional ones: older or per-plugin builds lack them) */
interface Exports {
  memory: WebAssembly.Memory;
  simd_build(): number;
  kind_available_c?(kind: number): number;
  vi_alloc(n: number): number;
  vi_free(p: number, n: number): void;
  last_error(p: number, cap: number): number;
  engine_create(kind: number, cfg: number, cfgLen: number, w: number, wLen: number): number;
  engine_info(h: number, p: number): void;
  engine_process(h: number, input: number, output: number, frames: number): void;
  engine_set_stages?(h: number, n: number): number;
  engine_process_stage(h: number, input: number, output: number, stage: number): void;
  engine_set_mix?(h: number, strength: number, rampMs: number): void;
  engine_get_mix?(h: number, p: number): void;
  engine_voice?(h: number, p: number): void;
  engine_reset(h: number): void;
  engine_set_scalar(h: number, on: number): void;
  engine_destroy(h: number): void;
}

// Engine config for a manifest: model packages (engine "graph") get the pipeline and the parsed
// graph.json merged in; other kinds use manifest.config verbatim. Model-agnostic: no per-model code.
export function engineConfig(manifest: Manifest, graphJson: string | object | null = null): Record<string, unknown> {
  const cfg: Record<string, unknown> = { ...(manifest.config || {}) };
  if (manifest.graph) {
    if (!graphJson) throw new Error(`${manifest.name}: graph.json required`);
    cfg.pipeline = manifest.pipeline;
    cfg.graph = typeof graphJson === 'string' ? JSON.parse(graphJson) : graphJson;
  }
  return cfg;
}

// AudioWorkletGlobalScope may lack TextEncoder/TextDecoder -> tiny fallbacks
const enc: { encode(s: string): Uint8Array } = typeof TextEncoder !== 'undefined' ? new TextEncoder()
  : { encode: (s) => Uint8Array.from(unescape(encodeURIComponent(s)), (c) => c.charCodeAt(0)) };
const dec: { decode(b: Uint8Array): string } = typeof TextDecoder !== 'undefined' ? new TextDecoder()
  : { decode: (b) => String.fromCharCode(...b) };
const kindId = (kind: EngineKind | number): number | undefined => (typeof kind === 'number' ? kind : KINDS[kind]);

export class ViRuntime {
  readonly instance: WebAssembly.Instance;
  /** @internal the module's exports */
  readonly x: Exports;
  readonly simdBuild: boolean;
  /** module: a WebAssembly.Module, or the wasm file's bytes (compiled here). Hosts that post to an
   *  AudioWorklet send bytes: Chrome 154 drops a message carrying a Module there (a silent messageerror
   *  in the worklet; 156 delivers it). */
  constructor(module: WebAssembly.Module | BufferSource) {
    if (!(module instanceof WebAssembly.Module)) module = new WebAssembly.Module(module);
    this.instance = new WebAssembly.Instance(module, {});
    this.x = this.instance.exports as unknown as Exports;
    this.simdBuild = this.x.simd_build() === 1;
  }
  /** is the plugin for this engine kind compiled into this module? (bundle vs per-plugin wasm) */
  kindAvailable(kind: EngineKind | number): boolean {
    const kid = kindId(kind);
    if (kid === undefined) return false;
    return this.x.kind_available_c ? this.x.kind_available_c(kid) === 1 : kid <= 2;
  }
  get mem(): ArrayBuffer { return this.x.memory.buffer; }
  /** @internal */
  _copyIn(bytes: ArrayBuffer | ArrayBufferView): [number, number] {
    const n = bytes.byteLength;
    const p = this.x.vi_alloc(n);
    const src = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, n) : new Uint8Array(bytes, 0, n);
    new Uint8Array(this.mem, p, n).set(src);
    return [p, n];
  }
  create(kind: EngineKind | number, config: string | object = '', weights: ArrayBuffer | ArrayBufferView | null = null): ViEngine {
    const kid = kindId(kind);
    if (kid === undefined) throw new Error(`unknown engine kind ${kind}`);
    const cfg = enc.encode(typeof config === 'string' ? config : JSON.stringify(config || {}));
    const [cp, cn] = this._copyIn(cfg);
    let wp = 0, wn = 0;
    if (weights && weights.byteLength) [wp, wn] = this._copyIn(weights);
    const h = this.x.engine_create(kid, cp, cn, wp, wn);
    this.x.vi_free(cp, cn);
    if (wn) this.x.vi_free(wp, wn);
    if (!h) {
      const ep = this.x.vi_alloc(512);
      const n = this.x.last_error(ep, 512);
      const msg = dec.decode(new Uint8Array(this.mem, ep, n).slice());
      this.x.vi_free(ep, 512);
      throw new Error(`engine_create(${kind}) failed: ${msg}`);
    }
    return new ViEngine(this, h);
  }
}

interface Views { buf: ArrayBuffer; inF: Float32Array; outF: Float32Array; voice: Float32Array }

export class ViEngine {
  readonly rt: ViRuntime;
  /** @internal the engine's handle (0 once destroyed) */
  h: number;
  info!: EngineInfo;
  /** parts each frame is spread over (setStages) */
  parts: number;
  private _infoPtr: number;
  private _fp: number;
  private _fo: number;
  private _vp: number;
  private _bufCap: number;
  private _bp: number;
  private _bo: number;
  private _views: Views | null;
  private _voice: Voice;

  constructor(rt: ViRuntime, h: number) {
    this.rt = rt; this.h = h;
    this._infoPtr = rt.x.vi_alloc(32);
    this._readInfo();
    // zero-copy single-frame buffers inside wasm memory
    this._fp = rt.x.vi_alloc(this.info.frameSize * 4);
    this._fo = rt.x.vi_alloc(this.info.frameSize * 4);
    this._vp = rt.x.vi_alloc(12);
    this._bufCap = 0; this._bp = 0; this._bo = 0;
    this._views = null;
    this.parts = 1;
    this._voice = { prob: 0, speaking: false, levelDb: -120 };
  }
  // typed-array views into wasm memory, re-created only when memory grew (no per-call allocation)
  private _v(): Views {
    const m = this.rt.mem;
    if (!this._views || this._views.buf !== m) {
      const fs = this.info.frameSize;
      this._views = { buf: m, inF: new Float32Array(m, this._fp, fs), outF: new Float32Array(m, this._fo, fs), voice: new Float32Array(m, this._vp, 3) };
    }
    return this._views;
  }
  private _readInfo(): void {
    this.rt.x.engine_info(this.h, this._infoPtr);
    const v = new Int32Array(this.rt.mem, this._infoPtr, 8);
    this.info = { sampleRate: v[0], frameSize: v[1], latency: v[2], inCh: v[3], outCh: v[4], simd: v[5] === 1 };
  }
  // views must be re-created if memory grew (buffer detached)
  get inFrame(): Float32Array { return this._v().inF; }
  get outFrame(): Float32Array { return this._v().outF; }
  processFrame(): void { this.rt.x.engine_process(this.h, this._fp, this._fo, 1); }
  /** split each frame into up to n parts (1 if the module or engine can't); returns the count */
  setStages(n: number): number {
    this.parts = this.rt.x.engine_set_stages ? this.rt.x.engine_set_stages(this.h, n) : 1;
    return this.parts;
  }
  /** part s of the frame in inFrame; outFrame is complete after the last part */
  processStage(s: number): void {
    if (this.parts > 1) this.rt.x.engine_process_stage(this.h, this._fp, this._fo, s);
    else if (s === 0) this.processFrame();
  }
  /** noise cancellation strength 0 (off: input delayed by the engine latency) .. 1 (full), ramped */
  setMix(strength: number, rampMs = 20): void {
    if (this.rt.x.engine_set_mix) this.rt.x.engine_set_mix(this.h, strength, rampMs);
    else if (strength < 1) throw new Error('this wasm module has no engine_set_mix (rebuild)');
  }
  getMix(): { gain: number; target: number } {
    if (!this.rt.x.engine_get_mix) return { gain: 1, target: 1 };
    const p = this._vp;   // reuse the 12-byte voice scratch
    this.rt.x.engine_get_mix(this.h, p);
    const v = new Float32Array(this.rt.mem, p, 2);
    return { gain: v[0], target: v[1] };
  }
  /** voice activity of the denoised output (same object, updated in place) */
  voice(): Voice {
    const o = this._voice;
    if (!this.rt.x.engine_voice) return o;
    this.rt.x.engine_voice(this.h, this._vp);
    const v = this._v().voice;
    o.prob = v[0]; o.speaking = v[1] > 0.5; o.levelDb = v[2];
    return o;
  }
  process(input: Float32Array, output: Float32Array): void {
    const n = input.length, fs = this.info.frameSize;
    if (n % fs) throw new Error(`length ${n} not a multiple of frame size ${fs}`);
    if (n > this._bufCap) {
      if (this._bufCap) { this.rt.x.vi_free(this._bp, this._bufCap * 4); this.rt.x.vi_free(this._bo, this._bufCap * 4); }
      this._bp = this.rt.x.vi_alloc(n * 4); this._bo = this.rt.x.vi_alloc(n * 4); this._bufCap = n;
    }
    new Float32Array(this.rt.mem, this._bp, n).set(input);
    this.rt.x.engine_process(this.h, this._bp, this._bo, n / fs);
    output.set(new Float32Array(this.rt.mem, this._bo, n));
  }
  reset(): void { this.rt.x.engine_reset(this.h); }
  setScalar(on: boolean): void { this.rt.x.engine_set_scalar(this.h, on ? 1 : 0); this._readInfo(); }
  destroy(): void {
    const x = this.rt.x;
    x.engine_destroy(this.h);
    x.vi_free(this._infoPtr, 32); x.vi_free(this._fp, this.info.frameSize * 4); x.vi_free(this._fo, this.info.frameSize * 4); x.vi_free(this._vp, 12);
    if (this._bufCap) { x.vi_free(this._bp, this._bufCap * 4); x.vi_free(this._bo, this._bufCap * 4); }
    this.h = 0;
  }
}

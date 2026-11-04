// worklet.ts — AudioWorklet host for vi_rs engines. ALL DSP runs here, client-side, in wasm.
//
// 'vi-engine' processor (1 in, 1 out, mono, context rate e.g. 48 kHz):
//   per render quantum: input -> Resampler(ctx -> engine rate) -> frame FIFO -> engine_process
//   (all complete frames, normally 1-2 at 40 samples) -> Resampler(engine -> ctx rate) -> output FIFO.
//   Dry signal goes through the same resampling chain, delayed by the engine latency at the
//   On/off and strength (wet) live in the engine's wasm host layer (engine_set_mix): ramped, the dry part is
//   the input delayed by exactly the engine latency, so toggling never clicks or jumps in time.
//   Several engines ("lanes") may coexist; switching crossfades between lanes (~20 ms).
// 'vi-tap' processor: records its input (for the post-WebRTC signal), passes audio through.
//
// Port messages (main -> worklet):
//   {type:'init', module: the wasm file's bytes (or a WebAssembly.Module; bytes are safer: Chrome 154
//        drops a message carrying a Module posted to an AudioWorklet)}
//   {type:'add', id, kind, config, weights: ArrayBuffer|null, scalar, minFrames, spread, module?, moduleKey?}
//        spread: max engine parts per frame (0/1 = whole frame per callback); frames of >= 2 quanta
//        are split into min(spread, quanta per frame) parts, one per callback (see Lane)
//        module: optional per-plugin wasm (bytes or Module; else the bundle from 'init')
//        minFrames: engine frames per wasm call (1 = minimum latency; >1 batches -> fewer weight
//        passes per frame at +(minFrames-1)*frame_size latency)
//   {type:'select', id} | {type:'remove', id}
//   {type:'params', wet (0..1), bypass (bool), rampMs (default 20)}  noise cancellation strength / off
//   {type:'record', seconds}   start recording input + output (Float32 @ ctx rate)
//   {type:'dump'}              -> {type:'dump', input, output, startFrame, stats}
//   {type:'stats'}             -> {type:'stats', ...}
//   {type:'bench', id, frames} -> {type:'bench', ...} in-worklet microbenchmark on a scratch engine
//                                 instance (Date.now() over the whole run; blocks the audio thread
//                                 briefly, so run it before audio starts)
//   {type:'otel', module, resource} telemetry on (opt-in): vi_otel.wasm instance, per-lane instruments
//   {type:'otelSnapshot'}      -> {type:'otelSnapshot', metrics, spans} (OTLP JSON bytes or null)
//   {type:'otelRecord', kind, name, unit, bounds, attrs, value}   main-thread values (load times, quality)
//   {type:'voice', on}         -> {type:'voice', prob, speaking, levelDb, inDb, outDb} ~19 times/s (active lane's
//                                 voice activity on the denoised output: animated mic etc.; inDb / outDb = RMS dBFS
//                                 of the input and of the output over the same ~53 ms: what was removed)
//   {type:'close'}             every engine freed, the processor ends (the host drops the node)
// An exception in the audio path (a wasm trap) never mutes the output: from then on the input passes through and one
// {type:'error', message, fatal: true} tells the host.
//
// Telemetry (when on): per lane, attributes from 'add' `labels` (vi.model, vi.build, vi.engine); every
// ~1 s of audio the active lane records window aggregates (engine cost per 10 ms of audio, worklet load,
// in/out level, suppression, speech share, per-part engine cost, max callback time) and counts (quanta,
// overruns = one callback over its 2.67 ms, deadline misses = 4 callbacks (~10.7 ms, a typical device
// buffer) over their summed budget, underruns); every callback's time goes into a histogram. Recording is lock-free and
// allocation-free; instruments are created at setup ('otel' / 'add'), snapshots in the message handler.
//
// NOTE (Chrome 156): AudioWorkletGlobalScope has no performance.now(), TextEncoder or
// SharedArrayBuffer. Per-quantum times use Date.now() (1 ms ticks): the AVERAGE over many quanta
// is unbiased (random phase vs the ms tick), percentiles/max are only ms-resolution.
import { ViRuntime } from './vi_engine.ts';
import type { ViEngine, EngineInfo, EngineKind } from './vi_engine.ts';
import { Resampler } from './resampler.ts';
import { otelBind } from './otel_export.ts';
import type { OtelExports, OtelHandle, Attrs, InstrumentKind } from './otel_export.ts';

type WasmSource = WebAssembly.Module | ArrayBuffer;
/** messages to the 'vi-engine' processor (the list above) */
export type WorkletIn =
  // module: null in the extension's isolated mode (its worker compiles the engine itself, `build` names the file)
  | { type: 'init'; module: WasmSource | null; build?: string }
  | { type: 'add'; id: string; kind: EngineKind | number; config: string | object; weights: ArrayBuffer | null; scalar?: boolean;
      minFrames?: number; spread?: number; module?: WasmSource; moduleKey?: string; labels?: Attrs }
  | { type: 'select'; id: string }
  | { type: 'remove'; id: string }
  | { type: 'params'; wet?: number; bypass?: boolean; rampMs?: number }
  | { type: 'record'; seconds: number }
  | { type: 'dump' }
  | { type: 'stats' }
  | { type: 'resetStats' }
  | { type: 'bench'; id: string; frames?: number }
  | { type: 'voice'; on: boolean }
  | { type: 'otel'; module: WebAssembly.Module; resource?: Attrs }
  | { type: 'otelRecord'; kind: InstrumentKind; name: string; unit?: string; bounds?: number[]; attrs?: Attrs; value: number }
  | { type: 'otelSnapshot' }
  | { type: 'close' };
export type LaneInfo = EngineInfo & { parts: number; resamplerDelayMs: number };
/** what stats() reports ({type:'stats'} answers with these fields) */
export interface WorkletStats {
  count: number; avgUs: number; engAvgUs: number; engUsPerFrame: number; frames: number; maxUs: number; p95Us: number; p99Us: number;
  overruns: number; slow2ms?: number; deadlineMisses: number; blocks: number; parts: number; budgetUs: number; havePerf: boolean;
  underruns: number; startupCtx?: number; latencyCtx: number; sampleRate: number;
}
export interface BenchResult {
  id: string; frames: number; frameSize: number; sampleRate: number; usPerFrame?: number; usPerFrameBlock?: number; passes: string;
}
/** messages from the 'vi-engine' processor */
export type WorkletOut =
  | { type: 'ready'; simdBuild: boolean; havePerf: boolean }
  // the extension's isolated-mode worker is listening (engine_worker.ts)
  | { type: 'up' }
  | { type: 'added'; id: string; info: LaneInfo; latencyCtx: number }
  | { type: 'dump'; input: Float32Array; output: Float32Array; startFrame: number; stats: WorkletStats }
  | ({ type: 'stats' } & WorkletStats)
  | ({ type: 'bench' } & BenchResult)
  | { type: 'voice'; prob: number; speaking: boolean; levelDb: number; inDb: number; outDb: number }
  | { type: 'otelSnapshot'; metrics: Uint8Array | null; spans: Uint8Array | null }
  | { type: 'error'; message: string; fatal?: boolean };

const COST_BOUNDS = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 6000];
const LOAD_BOUNDS = [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150];
const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : () => Date.now();
const HAVE_PERF = typeof performance !== 'undefined' && !!performance.now;

class Fifo {
  b: Float32Array;
  n = 0;
  over?: number;
  constructor(cap: number) { this.b = new Float32Array(cap); }
  push(x: Float32Array) {
    if (this.n + x.length > this.b.length) { this.over = (this.over || 0) + 1; this.n = 0; }
    this.b.set(x, this.n); this.n += x.length;
  }
  consume(k: number) { this.b.copyWithin(0, k, this.n); this.n -= k; }
}

type Recorder = (v: number) => void;
interface LaneTelemetry {
  quanta: Recorder; overruns: Recorder; underruns: Recorder; misses: Recorder; blocks: Recorder; load: Recorder; cost: Recorder;
  quantum: Recorder; quantumMax: Recorder; speech: Recorder; stage: Recorder[]; inLvl: Recorder; outLvl: Recorder; supp: Recorder;
  latency: Recorder; underrunsSeen: number;
}

class Lane {
  id: string; eng: ViEngine; sr: number; fs: number; down: Resampler; upW: Resampler; inF: Fifo; wetF: Fifo; maxFrames: number;
  inBuf: Float32Array; outBuf: Float32Array; started: boolean; minFrames: number; cushion: number; latencyCtx: number;
  kind: EngineKind | number; config: string | object; weights: ArrayBuffer | null; scalar: boolean; rt: ViRuntime;
  parts: number; stage: number; stageMs: Float64Array; stageN: Float64Array; info: LaneInfo;
  framesDone?: number; underruns?: number; startupCtx?: number; labels: Attrs = {}; tm?: LaneTelemetry;
  constructor(rt: ViRuntime, id: string, kind: EngineKind | number, config: string | object, weights: ArrayBuffer | null, scalar: boolean,
    ctxRate: number, minFrames = 1, spread = 0) {
    this.id = id;
    this.eng = rt.create(kind, config, weights);
    if (scalar) this.eng.setScalar(true);
    const { sampleRate: sr, frameSize: fs, latency } = this.eng.info;
    this.sr = sr; this.fs = fs;
    this.down = new Resampler(ctxRate, sr);
    this.upW = new Resampler(sr, ctxRate);
    this.inF = new Fifo(fs * 64);
    this.wetF = new Fifo(16384);   // engine output = final output (on/off/strength mix is in the engine)
    this.maxFrames = 32;
    this.inBuf = new Float32Array(fs * this.maxFrames); this.outBuf = new Float32Array(fs * this.maxFrames);
    this.started = false;
    this.minFrames = Math.max(1, Math.min(minFrames | 0, this.maxFrames));
    this.cushion = 128 + Math.ceil(this.minFrames * fs * ctxRate / sr) + 8;    // ctx samples kept buffered: batch + quantum
    this.latencyCtx = (this.down.delaySeconds + this.upW.delaySeconds) * ctxRate + latency * ctxRate / sr;
    this.kind = kind; this.config = config; this.weights = weights; this.scalar = scalar; this.rt = rt;
    // Realtime smoothing: a frame spanning >= 2 render quanta (e.g. 480 samples = 3.75 quanta) is
    // split into that many engine parts (engine_set_stages), one per callback, instead of the whole
    // frame in one callback and nothing in the next ones. Costs (parts - 1) quanta of latency.
    const qpf = Math.floor(fs * ctxRate / sr / 128);
    this.parts = (spread >= 2 && this.minFrames === 1 && qpf >= 2) ? this.eng.setStages(Math.min(qpf, spread)) : 1;
    this.stage = -1;                                  // next part of the frame in flight; -1 = none
    this.stageMs = new Float64Array(this.parts); this.stageN = new Float64Array(this.parts);
    this.cushion += (this.parts - 1) * 128;
    this.latencyCtx += (this.parts - 1) * 128;
    this.info = { ...this.eng.info, parts: this.parts, resamplerDelayMs: 1e3 * (this.down.delaySeconds + this.upW.delaySeconds) };
  }
  push(x: Float32Array): number {
    const d = this.down.process(x);
    this.inF.push(d);
    if (this.parts > 1) return this.pushStaged();
    const nfr = Math.min(Math.floor(this.inF.n / this.fs), this.maxFrames);
    if (nfr < this.minFrames) return 0;
    const n = nfr * this.fs;
    const t0 = now();
    const inp = this.inF.b.subarray(0, n);
    const out = this.outBuf.subarray(0, n);
    this.eng.process(inp, out);
    const dt = now() - t0;
    this.framesDone = (this.framesDone || 0) + nfr;
    this.wetF.push(this.upW.process(out));
    this.inF.consume(n);
    return dt;
  }
  // one part per callback; if a whole new frame is already waiting (stall, throttled tab), finish
  // the frame in flight now so the lane never falls behind
  pushStaged(): number {
    const t0 = now();
    if (this.stage >= 0) {
      if (this.inF.n >= this.fs) while (this.stage >= 0) this.step();
      else this.step();
    }
    if (this.stage < 0 && this.inF.n >= this.fs) {
      this.eng.inFrame.set(this.inF.b.subarray(0, this.fs));   // stays put in wasm memory until done
      this.inF.consume(this.fs);
      this.stage = 0;
      this.step();
    }
    return now() - t0;
  }
  step(): void {
    const s = this.stage, t0 = now();
    this.eng.processStage(s);
    this.stageMs[s] += now() - t0; this.stageN[s]++;
    if (++this.stage < this.parts) return;
    this.stage = -1;
    this.framesDone = (this.framesDone || 0) + 1;
    this.wetF.push(this.upW.process(this.eng.outFrame));
  }
  pull(out: Float32Array, k: number): boolean {
    if (!this.started) {
      if (this.wetF.n < this.cushion) { out.fill(0); return false; }
      this.started = true;
      this.startupCtx = this.wetF.n;
    }
    if (this.wetF.n < k) { this.underruns = (this.underruns || 0) + 1; out.fill(0); return false; }
    out.set(this.wetF.b.subarray(0, k));
    this.wetF.consume(k);
    return true;
  }
}

interface Stats { n: number; sum: number; max: number; over: number; engSum: number; frames: number; hist: Uint32Array; slow2?: number; miss?: number; blocks?: number }

class ViEngineProcessor extends AudioWorkletProcessor {
  rt: ViRuntime | null; rts?: Map<string | undefined, ViRuntime>; lanes: Map<string, Lane>; active: string | null; fadeFrom: string | null; fade: number;
  wet: number; bypass: boolean; rampMs = 20; w: Float32Array; w2: Float32Array; st!: Stats;
  rec: { input: Float32Array; output: Float32Array; n: number; startFrame: number } | null;
  otel: OtelHandle | null; otelInst: Map<string, Recorder>;
  win: { n: number; ms: number; eng: number; over: number; inE: number; outE: number; k: number; max: number; miss: number; blocks: number; speech: number };
  blk: { n: number; ms: number }; voiceOn: boolean; voiceTick: number; vIn: number; vOut: number; vK: number; zeros?: Float32Array;
  closed = false; failed = false;
  constructor() {
    super();
    this.rt = null; this.lanes = new Map(); this.active = null; this.fadeFrom = null; this.fade = 1;
    this.wet = 1; this.bypass = false;
    this.w = new Float32Array(128); this.w2 = new Float32Array(128);
    this.resetStats();
    this.rec = null;
    this.otel = null; this.otelInst = new Map();
    this.win = { n: 0, ms: 0, eng: 0, over: 0, inE: 0, outE: 0, k: 0, max: 0, miss: 0, blocks: 0, speech: 0 };
    this.blk = { n: 0, ms: 0 };                     // current block of 4 callbacks (deadline misses)
    this.voiceOn = false; this.voiceTick = 0; this.vIn = 0; this.vOut = 0; this.vK = 0;
    this.port.onmessage = (e: MessageEvent<WorkletIn>) => this.onMsg(e.data);
  }
  post(m: WorkletOut, transfer: Transferable[] = []) { this.port.postMessage(m, transfer); }
  mixTarget() { return this.bypass ? 0 : this.wet; }
  resetStats() { this.st = { n: 0, sum: 0, max: 0, over: 0, engSum: 0, frames: 0, hist: new Uint32Array(400) }; }
  onMsg(m: WorkletIn) {
    try {
      if (m.type === 'init') {
        if (!m.module) throw new Error('init without the engine module');
        this.rt = new ViRuntime(m.module);
        this.post({ type: 'ready', simdBuild: this.rt.simdBuild, havePerf: HAVE_PERF });
      }
      else if (m.type === 'add') {
        // per-plugin wasm: the lane gets its own module (one runtime per module key)
        let rt = this.rt;
        if (m.module) {
          this.rts = this.rts || new Map();
          if (!this.rts.has(m.moduleKey)) this.rts.set(m.moduleKey, new ViRuntime(m.module));
          rt = this.rts.get(m.moduleKey)!;
        }
        if (!rt) throw new Error('add before init');
        const lane = new Lane(rt, m.id, m.kind, m.config, m.weights, !!m.scalar, sampleRate, m.minFrames || 1, (m.spread ?? 0) | 0);
        lane.labels = m.labels || { 'vi.model': m.id };
        lane.eng.setMix(this.mixTarget(), 0);
        this.lanes.set(m.id, lane);
        if (this.otel) this.otelLane(lane);
        this.post({ type: 'added', id: m.id, info: lane.info, latencyCtx: lane.latencyCtx });
      } else if (m.type === 'select') {
        if (this.active && this.active !== m.id) { this.fadeFrom = this.active; this.fade = 0; }
        this.active = m.id; this.resetStats();
      } else if (m.type === 'remove') {
        const l = this.lanes.get(m.id); if (l) { l.eng.destroy(); this.lanes.delete(m.id); }
        if (this.active === m.id) this.active = null;
      } else if (m.type === 'params') {
        // noise cancellation on/off + strength: applied inside every lane's engine (wasm host layer:
        // ramped, dry = input delayed by the engine latency), so a later switch keeps the setting
        if (m.wet !== undefined) this.wet = m.wet;
        if (m.bypass !== undefined) this.bypass = m.bypass;
        this.rampMs = m.rampMs ?? 20;
        for (const l of this.lanes.values()) l.eng.setMix(this.mixTarget(), this.rampMs);
      }
      else if (m.type === 'record') {
        const n = Math.ceil(m.seconds * sampleRate);
        this.rec = { input: new Float32Array(n), output: new Float32Array(n), n: 0, startFrame: -1 };
      } else if (m.type === 'dump') {
        const r = this.rec || { input: new Float32Array(0), output: new Float32Array(0), n: 0, startFrame: -1 };
        const input = r.input.slice(0, r.n), output = r.output.slice(0, r.n);
        this.post({ type: 'dump', input, output, startFrame: r.startFrame, stats: this.stats() }, [input.buffer, output.buffer]);
        this.rec = null;
      } else if (m.type === 'stats') this.post({ type: 'stats', ...this.stats() });
      else if (m.type === 'resetStats') this.resetStats();
      else if (m.type === 'bench') this.post({ type: 'bench', ...this.bench(m.id, m.frames || 2000) });
      else if (m.type === 'voice') this.voiceOn = !!m.on;
      else if (m.type === 'close') {
        // free every engine now (wasm memory, model weights): the node may outlive this until the context closes
        for (const l of this.lanes.values()) { try { l.eng.destroy(); } catch { /* already */ } }
        this.lanes.clear(); this.active = null; this.fadeFrom = null; this.rt = null; this.rts = undefined; this.rec = null;
        this.closed = true;
      }
      else if (m.type === 'otel') this.otelSetup(m);
      else if (m.type === 'otelRecord') this.otelRecord(m);
      else if (m.type === 'otelSnapshot') {
        const s = this.otel ? (this.otel.tick(Date.now()), this.otel.snapshot()) : { metrics: null, spans: null };
        this.post({ type: 'otelSnapshot', ...s }, [s.metrics, s.spans].filter((b) => b !== null).map((b) => b.buffer));
      }
    } catch (err) { this.post({ type: 'error', message: String((err as Error | null)?.stack || err) }); }
  }
  // ---- telemetry (setup-time code: may allocate; the audio path only calls the bound recorders)
  otelSetup(m: Extract<WorkletIn, { type: 'otel' }>) {
    if (this.otel) return;
    const ex = new WebAssembly.Instance(m.module, {}).exports as unknown as OtelExports;
    this.otel = otelBind(ex);
    this.otel.seed();
    this.otel.tick(Date.now());
    this.otel.resource({ 'service.name': 'vi-web', ...m.resource });
    this.otel.enable(true);
    for (const lane of this.lanes.values()) this.otelLane(lane);
  }
  otelLane(lane: Lane) {
    const o = this.otel!, a = lane.labels;
    lane.tm = {
      quanta: o.instrument('counter', 'vi.worklet.quanta', '{quantum}', [], a),
      overruns: o.instrument('counter', 'vi.worklet.overruns', '{quantum}', [], a),
      underruns: o.instrument('counter', 'vi.worklet.underruns', '{quantum}', [], a),
      misses: o.instrument('counter', 'vi.worklet.deadline_misses', '{block}', [], a),
      blocks: o.instrument('counter', 'vi.worklet.blocks', '{block}', [], a),
      load: o.instrument('histogram', 'vi.worklet.load', '%', LOAD_BOUNDS, a),
      cost: o.instrument('histogram', 'vi.engine.cost', 'us', COST_BOUNDS, a),
      // every callback (Date.now ticks: 0, 1, 2, 3 ms ...): the tail that the window averages hide
      quantum: o.instrument('histogram', 'vi.worklet.quantum', 'ms', [0.5, 1.5, 2.5, 3.5, 5, 10, 20, 50], a),
      quantumMax: o.instrument('gauge', 'vi.worklet.quantum.max', 'ms', [], a),
      speech: o.instrument('histogram', 'vi.audio.speech', '%', [1, 10, 25, 50, 75, 90, 99], a),
      stage: Array.from({ length: lane.parts }, (_, s) =>
        o.instrument('histogram', 'vi.engine.stage.cost', 'us', COST_BOUNDS, { ...a, 'vi.stage': `${s + 1}/${lane.parts}` })),
      inLvl: o.instrument('histogram', 'vi.audio.input.level', 'dBFS', [-80, -70, -60, -50, -40, -30, -20, -10, 0], a),
      outLvl: o.instrument('histogram', 'vi.audio.output.level', 'dBFS', [-80, -70, -60, -50, -40, -30, -20, -10, 0], a),
      supp: o.instrument('histogram', 'vi.audio.suppression', 'dB', [0, 1, 3, 6, 10, 15, 20, 30, 40], a),
      latency: o.instrument('gauge', 'vi.engine.latency', 'ms', [], a),
      underrunsSeen: 0,
    };
    lane.tm.latency(1e3 * lane.latencyCtx / sampleRate);
  }
  otelRecord(m: Extract<WorkletIn, { type: 'otelRecord' }>) {
    if (!this.otel) return;
    const key = `${m.kind}|${m.name}|${JSON.stringify(m.attrs || {})}`;
    let rec = this.otelInst.get(key);
    if (!rec) { rec = this.otel.instrument(m.kind, m.name, m.unit || '', m.bounds || [], m.attrs || null); this.otelInst.set(key, rec); }
    rec(m.value);
  }
  // audio thread: close a ~1 s window into the active lane's instruments (lock-free records only)
  otelWindow(lane: Lane) {
    const w = this.win, t = lane.tm;
    if (t && w.n) {
      this.otel!.tick(Date.now());
      const audioS = w.n * 128 / sampleRate, budgetMs = w.n * 1e3 * 128 / sampleRate;
      t.quanta(w.n); t.overruns(w.over);
      const u = lane.underruns || 0; t.underruns(u - t.underrunsSeen); t.underrunsSeen = u;
      t.load(100 * w.ms / budgetMs);
      t.cost(1e3 * w.eng / (audioS * 100));          // engine us per 10 ms of audio
      t.quantumMax(w.max); t.misses(w.miss); t.blocks(w.blocks);
      t.speech(100 * w.speech / w.n);
      for (let s = 0; s < lane.parts; s++) {
        if (lane.stageN[s]) t.stage[s](1e3 * lane.stageMs[s] / lane.stageN[s]);   // us per part call
        lane.stageMs[s] = 0; lane.stageN[s] = 0;
      }
      if (w.k) {
        const li = 10 * Math.log10(w.inE / w.k + 1e-12), lo = 10 * Math.log10(w.outE / w.k + 1e-12);
        if (li > -70) { t.inLvl(li); t.outLvl(lo); t.supp(li - lo); }
      }
      t.latency(1e3 * lane.latencyCtx / sampleRate);
    }
    w.n = 0; w.ms = 0; w.eng = 0; w.over = 0; w.inE = 0; w.outE = 0; w.k = 0;
    w.max = 0; w.miss = 0; w.blocks = 0; w.speech = 0;
  }
  bench(id: string, frames: number): BenchResult {
    const l = this.lanes.get(id);
    if (!l) throw new Error(`bench: no lane ${id}`);
    const e = l.rt.create(l.kind, l.config, l.weights);
    if (l.scalar) e.setScalar(true);
    const fs = e.info.frameSize, x = new Float32Array(fs * frames), y = new Float32Array(fs * frames);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.01) * 0.3 + (Math.random() - 0.5) * 0.1;
    const r: { frame?: number; block?: number } = {}, passes: string[] = [];
    for (const mode of ['frame', 'block', 'frame', 'block', 'frame'] as const) {   // early passes = JIT warm-up (Liftoff -> TurboFan)
      e.reset();
      const t0 = Date.now();
      if (mode === 'frame') for (let i = 0; i < frames; i++) { e.inFrame.set(x.subarray(i * fs, (i + 1) * fs)); e.processFrame(); y.set(e.outFrame, i * fs); }
      else e.process(x, y);
      r[mode] = Math.min(r[mode] ?? Infinity, 1e3 * (Date.now() - t0) / frames);   // best pass (others: JIT tier-up, noise)
      passes.push(`${mode}:${r[mode]!.toFixed(1)}`);
    }
    e.destroy();
    return { id, frames, frameSize: fs, sampleRate: e.info.sampleRate, usPerFrame: r.frame, usPerFrameBlock: r.block, passes: passes.join(' ') };
  }
  stats(): WorkletStats {
    const s = this.st; let acc = 0, p95 = 0, p99 = 0;
    for (let i = 0; i < s.hist.length; i++) { acc += s.hist[i]; if (!p95 && acc >= 0.95 * s.n) p95 = (i + 1) * 10; if (!p99 && acc >= 0.99 * s.n) p99 = (i + 1) * 10; }
    const l = this.active ? this.lanes.get(this.active) : undefined;
    return { count: s.n, avgUs: s.n ? 1e3 * s.sum / s.n : 0, engAvgUs: s.n ? 1e3 * s.engSum / s.n : 0, engUsPerFrame: s.frames ? 1e3 * s.engSum / s.frames : 0, frames: s.frames, maxUs: 1e3 * s.max, p95Us: p95, p99Us: p99,
      overruns: s.over, slow2ms: s.slow2, deadlineMisses: s.miss || 0, blocks: s.blocks || 0, parts: l ? l.parts : 1, budgetUs: 1e6 * 128 / sampleRate, havePerf: HAVE_PERF, underruns: l ? (l.underruns || 0) : 0,
      startupCtx: l ? l.startupCtx : 0, latencyCtx: l ? l.latencyCtx : 0, sampleRate };
  }
  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    if (this.closed) return false;
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0] && outputs[0][0];
    if (!out) return true;
    if (!this.failed) {
      try { return this.run(inp, out); } catch (err) {
        // the engine trapped: the plain input from now on (never silence), the host is told once
        this.failed = true;
        this.post({ type: 'error', message: `engine stopped: ${String((err as Error | null)?.message || err)}`, fatal: true });
      }
    }
    if (inp) out.set(inp.subarray(0, out.length)); else out.fill(0);
    return true;
  }
  run(inp: Float32Array | undefined, out: Float32Array): boolean {
    const t0 = now();
    const k = out.length;
    const x = inp || this.zeros || (this.zeros = new Float32Array(k));
    let eng = 0;
    const lane = this.active ? this.lanes.get(this.active) : undefined;
    if (lane) {
      const f0 = lane.framesDone || 0;
      eng += lane.push(x);
      this.st.frames += (lane.framesDone || 0) - f0;
      lane.pull(out, k);
      // switching models: 20 ms crossfade between the two lanes (two engines: a host-side job)
      const from = this.fadeFrom ? this.lanes.get(this.fadeFrom) : undefined;
      if (from && this.fade < 1) {
        eng += from.push(x); from.pull(this.w2, k);
        const step = 1 / (0.02 * sampleRate);
        for (let i = 0; i < k; i++) {
          const g = Math.min(1, this.fade + i * step);
          out[i] = g * out[i] + (1 - g) * this.w2[i];
        }
        this.fade += k * step;
        if (this.fade >= 1) this.fadeFrom = null;
      }
    } else out.set(x.subarray(0, k));
    if (this.rec && this.rec.n + k <= this.rec.input.length) {
      if (this.rec.startFrame < 0) this.rec.startFrame = currentFrame;
      this.rec.input.set(x.subarray(0, k), this.rec.n); this.rec.output.set(out, this.rec.n); this.rec.n += k;
    }
    const dt = now() - t0, s = this.st;
    s.n++; s.sum += dt; s.engSum += eng; if (dt > s.max) s.max = dt;
    const over = dt * 1e3 > 1e6 * k / sampleRate;
    if (over) s.over++;
    // deadline view of a ~10.7 ms device buffer: 4 callbacks against their summed budget
    const b = this.blk; b.n++; b.ms += dt;
    let miss = false, blockDone = false;
    if (b.n === 4) { miss = b.ms * 1e3 > 4e6 * k / sampleRate; s.miss = (s.miss || 0) + +miss; s.blocks = (s.blocks || 0) + 1; b.n = 0; b.ms = 0; blockDone = true; }
    let speaking = false;
    if (lane) {
      const v = lane.eng.voice();
      speaking = v.speaking;
      if (this.voiceOn) {
        let ie = 0, oe = 0;
        for (let i = 0; i < k; i++) { ie += x[i] * x[i]; oe += out[i] * out[i]; }
        this.vIn += ie; this.vOut += oe; this.vK += k;
      }
      if (this.voiceOn && ++this.voiceTick >= 20) {
        const db = (e: number) => (e > 0 ? Math.max(-120, 10 * Math.log10(e / this.vK)) : -120);
        const inDb = db(this.vIn), outDb = db(this.vOut);
        this.voiceTick = 0; this.vIn = 0; this.vOut = 0; this.vK = 0;
        this.post({ type: 'voice', prob: v.prob, speaking: v.speaking, levelDb: v.levelDb, inDb, outDb });
      }
    }
    if (this.otel && lane) {
      const w = this.win;
      w.n++; w.ms += dt; w.eng += eng; if (over) w.over++;
      if (dt > w.max) w.max = dt;
      if (blockDone) { w.blocks++; if (miss) w.miss++; }
      if (speaking) w.speech++;
      if (lane.tm) lane.tm.quantum(dt);
      for (let i = 0; i < k; i++) { w.inE += x[i] * x[i]; w.outE += out[i] * out[i]; }
      w.k += k;
      if (w.n >= 375) this.otelWindow(lane);                 // 375 x 128 = 1 s at 48 kHz
    }
    if (dt >= 2) s.slow2 = (s.slow2 || 0) + 1;
    s.hist[Math.min(s.hist.length - 1, Math.floor(dt * 100))]++;   // 10 us bins
    return true;
  }
}

class ViTapProcessor extends AudioWorkletProcessor {
  rec: { buf: Float32Array; n: number; startFrame: number } | null;
  constructor() {
    super(); this.rec = null;
    this.port.onmessage = (e: MessageEvent<{ type: string; seconds: number }>) => {
      const m = e.data;
      if (m.type === 'record') this.rec = { buf: new Float32Array(Math.ceil(m.seconds * sampleRate)), n: 0, startFrame: -1 };
      else if (m.type === 'dump') {
        const r = this.rec || { buf: new Float32Array(0), n: 0, startFrame: -1 };
        const data = r.buf.slice(0, r.n);
        this.port.postMessage({ type: 'dump', data, startFrame: r.startFrame }, [data.buffer]);
        this.rec = null;
      }
    };
  }
  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const x = inputs[0] && inputs[0][0];
    const o = outputs[0] && outputs[0][0];
    if (x && o) o.set(x);
    const r = this.rec;
    if (r && x && r.n + x.length <= r.buf.length) {
      if (r.startFrame < 0) r.startFrame = currentFrame;
      r.buf.set(x, r.n); r.n += x.length;
    }
    return true;
  }
}

registerProcessor('vi-engine', ViEngineProcessor);
registerProcessor('vi-tap', ViTapProcessor);

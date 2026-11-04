// inject.ts — runs in the PAGE (MAIN world) at document_start: wraps navigator.mediaDevices.getUserMedia so
// any web calling app gets noise-suppressed microphone audio without changing its code.
//
//   getUserMedia({audio, video}) -> the real microphone (browser noise suppression off: ours replaces it;
//   echo cancellation / gain control as the site asked) -> the vi engine -> a MediaStreamTrack that looks
//   like the microphone (label, settings, constraints, stop() and 'ended' follow the real one) + the
//   original video tracks.
//
// Two ways to run the engine (both run the same worklet code, src/engine/js/worklet.ts, bundled):
//   page      the engine's AudioWorklet + wasm in the page's own audio graph (lowest latency). Needs the
//             site's CSP to allow WebAssembly ('unsafe-eval' / 'wasm-unsafe-eval'): Meet, Zoom, Discord do.
//   isolated  sites that forbid WebAssembly (e.g. Teams): the page runs only a tiny JS pump worklet;
//             audio goes over a MessagePort to a worker in an extension-origin frame (engine_frame.html,
//             the extension's CSP) and back, + a few ms of buffering.
// 'auto' tries page, then isolated. If neither can start, the site gets the original microphone: a
// failure here must never break a call.
//
// Live control: the popup / policy change settings -> bridge pushes a new config -> strength and on/off
// apply to running streams (ramped in the engine, no clicks), a model change crossfades.
import type { WorkletIn, WorkletOut, WorkletStats } from '../../engine/js/worklet.ts';
import type { Manifest } from '../../engine/js/vi_engine.ts';
import type { PageConfig, PageToBridge, BridgeToPage, TabStatus, TelemetryEvent, Mode } from './messages.ts';

(() => {
  if ((window as { __viExtension?: unknown }).__viExtension) return;
  const md = navigator.mediaDevices;
  if (!md || typeof md.getUserMedia !== 'function') return;
  Object.defineProperty(window, '__viExtension', { value: { version: 1 } });

  const origGUM = md.getUserMedia.bind(md);
  const PROBE = {
    simd: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11],
    relaxed: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11],
  };
  const valid = (b: number[]) => { try { return WebAssembly.validate(new Uint8Array(b)); } catch { return false; } };
  const BUILD = valid(PROBE.relaxed) ? 'relaxed' : valid(PROBE.simd) ? 'simd' : 'nosimd';

  // ---- bridge channel ----
  // The bridge (extension side) can go away while a call runs: an extension update or reload invalidates
  // it. The old bridge then says 'lost' (if it still can); the new extension re-injects a bridge, whose
  // hello replaces the port. So the hello listener stays for the page's lifetime.
  let port: MessagePort | null = null, seq = 0, cfg: PageConfig | null = null, portOk: (p: MessagePort) => void = () => {};
  const pending = new Map<number, { res: (v: any) => void; rej: (e: Error) => void }>();
  let portReady = new Promise<MessagePort>((res) => { portOk = res; });
  window.addEventListener('message', (e: MessageEvent) => {
    if (e.source !== window || !e.data || e.data.type !== 'vi-bridge-hello' || !e.ports[0]) return;
    e.stopImmediatePropagation();
    const again = !!port || sessions.size > 0;
    if (port) { try { port.close(); } catch { /* gone */ } lost(); }   // the old bridge died without a word
    for (const p of pending.values()) p.rej(new Error('extension reloaded'));
    pending.clear();
    const p = (port = e.ports[0]);
    p.onmessage = (ev: MessageEvent<BridgeToPage>) => { if (p === port) onBridge(ev); };   // a replaced bridge is ignored
    portOk(p);
    if (again) reconnected();
  }, true);
  type Rpc = Extract<PageToBridge, { id: number }>;
  type RpcBody<T extends Rpc['type']> = Omit<Extract<Rpc, { type: T }>, 'id' | 'type'>;
  function rpc(type: 'config'): Promise<PageConfig>;
  function rpc(type: 'fetch', extra: RpcBody<'fetch'> & { as: 'text' }): Promise<string>;
  function rpc(type: 'fetch', extra: RpcBody<'fetch'> & { as: 'buffer' }): Promise<ArrayBuffer>;
  function rpc(type: 'frame', extra: RpcBody<'frame'>, transfer: Transferable[]): Promise<true>;
  async function rpc(type: Rpc['type'], extra: object = {}, transfer: Transferable[] = []): Promise<unknown> {
    const p = await portReady;
    const id = ++seq;
    return new Promise((res, rej) => { pending.set(id, { res, rej }); p.postMessage({ id, type, ...extra }, transfer); });
  }
  const tell = (m: Exclude<PageToBridge, { id: number }>) => { if (port) port.postMessage(m); };
  // a report for the extension's anonymous technical reports (only when they are on; the service worker filters it)
  const telemetry = (ev: TelemetryEvent) => { if (cfg && cfg.telemetry) tell({ type: 'telemetry', ev }); };
  function onBridge(e: MessageEvent<BridgeToPage>) {
    const m = e.data || {};
    if ('id' in m) {
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      return m.error ? p.rej(new Error(m.error)) : p.res(m.result);
    }
    if (m.type === 'config') { const c = (cfg = m.config); for (const s of sessions) s.apply(c).catch((err) => report({ error: err.message })); }
    if (m.type === 'lost') lost();
    if (m.type === 'meter') meterOn = !!m.on;
  }
  // a popup shows the live meter: forward the engine's levels while it is open
  let meterOn = false;
  /** the extension went away: running sessions keep going (page mode: the engine lives in the page;
   *  isolated mode: its worker died with the extension, so the plain microphone until it is back) */
  function lost() {
    log('extension reloaded: keeping the call going');
    if (port) { try { port.close(); } catch { /* gone */ } }
    port = null;
    meterOn = false;
    portReady = new Promise<MessagePort>((res) => { portOk = res; });
    cfg = null; wasmModule = null; packages.clear();
    for (const s of sessions) if (s.mode === 'isolated') s.raw();
  }
  /** a new bridge after an update: fresh config for everyone, engines brought back where they died */
  async function reconnected() {
    log('extension back');
    cfg = null;
    let c: PageConfig;
    try { c = await config(); } catch (e) { return log(`extension back, no config: ${(e as Error).message}`); }
    for (const s of sessions) {
      (s.mode === null ? s.revive(c) : s.apply(c)).catch((err) => report({ error: err.message }));
    }
    report();
  }
  // one line per decision in the page console (support: "why is it (not) working on this site")
  const log = (msg: string) => console.info(`[vi] ${msg}`);
  const timeout = <T>(p: Promise<T>, ms: number, what: string): Promise<T> =>
    Promise.race([p, new Promise<never>((_, rej) => setTimeout(() => rej(new Error(`${what}: timeout`)), ms))]);
  async function config(): Promise<PageConfig> {
    if (!cfg) cfg = await timeout(rpc('config'), 6000, 'extension config');   // first run after install: policy read ~2 s
    return cfg;
  }

  // ---- status for the badge / popup ----
  const sessions = new Set<Session>();
  let last: TabStatus = {};
  function report(extra: TabStatus = {}) {
    last = { ...last, ...extra, active: [...sessions].filter((s) => s.live).length, build: BUILD };
    tell({ type: 'status', ...last });
  }

  // ---- shared pieces ----
  let ctx: AudioContext | null = null;
  function audioContext(): AudioContext {
    if (!ctx || ctx.state === 'closed') {
      const c = (ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' }));
      const say = () => log(`audio ${c.state} (${c.sampleRate} Hz${c.baseLatency ? `, ${Math.round(c.baseLatency * 1000)} ms` : ''})`);
      say();
      c.addEventListener('statechange', say);
    }
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    return ctx;
  }
  // Autoplay rules (Firefox's in particular) start an AudioContext only during a click / key press. The page's
  // getUserMedia usually runs in one, but the permission prompt outlasts it: the context is created at the call (before
  // any await), and if it is still suspended, the next click or key press on the page resumes it. A suspended context
  // means a silent microphone: recordings and calls would carry nothing.
  for (const type of ['pointerdown', 'keydown']) {
    addEventListener(type, () => { if (ctx && ctx.state === 'suspended') ctx.resume().catch(() => {}); }, { capture: true });
  }
  const worklets = new WeakMap<AudioContext, Set<string>>();   // ctx -> Set of loaded module names
  async function loadWorklet(c: AudioContext, path: string) {
    let done = worklets.get(c);
    if (!done) worklets.set(c, (done = new Set()));
    if (done.has(path)) return;
    const { base } = await config();
    try {
      await c.audioWorklet.addModule(base + path);   // extension URLs are exempt from most page CSPs
    } catch {
      // some pages refuse it: the same code from a blob (allowed where CSP lists blob:)
      const url = URL.createObjectURL(new Blob([await rpc('fetch', { path, as: 'text' })], { type: 'text/javascript' }));
      try { await c.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    }
    done.add(path);
  }
  let wasmModule: Promise<ArrayBuffer> | null = null;
  // the wasm's bytes (compiled once here only to learn whether the site's CSP allows WebAssembly): the
  // worklet gets bytes, since Chrome 154 drops a message carrying a WebAssembly.Module to an AudioWorklet
  async function pageWasm(): Promise<ArrayBuffer> {
    if (!wasmModule) wasmModule = (async () => {
      const b = await rpc('fetch', { path: `engine/vi_rs_${BUILD}.wasm`, as: 'buffer' });
      await WebAssembly.compile(b);
      return b;
    })();
    try { return await wasmModule; } catch (e) { wasmModule = null; throw e; }
  }
  interface Package { manifest: Manifest; config: string }
  const packages = new Map<string, Promise<Package>>();
  function modelPackage(name: string): Promise<Package> {
    if (!packages.has(name)) {
      packages.set(name, (async () => {
        const m: Manifest = JSON.parse(await rpc('fetch', { path: `models/${name}/manifest.json`, as: 'text' }));
        const graph = m.graph ? JSON.parse(await rpc('fetch', { path: `models/${name}/${m.graph}`, as: 'text' })) : null;
        const config: Record<string, unknown> = { ...(m.config || {}) };
        if (graph) { config.pipeline = m.pipeline; config.graph = graph; }
        return { manifest: m, config: JSON.stringify(config) };
      })().catch((e) => { packages.delete(name); throw e; }));
    }
    return packages.get(name)!;
  }
  const weights = async (name: string, m: Manifest): Promise<ArrayBuffer | null> => (m.weights ? rpc('fetch', { path: `models/${name}/${m.weights}`, as: 'buffer' }) : null);

  /** Engine control link: the page worklet's port, or the isolated worker's control port. Same protocol
   *  (src/engine/js/worklet.ts header). */
  type Waiter = { res: (m: any) => void; rej: (e: Error) => void };
  class Link {
    p: MessagePort; waiters = new Map<string, Waiter>(); onEvent: (m: WorkletOut) => void = () => {};
    constructor(p: MessagePort) {
      this.p = p;
      p.onmessage = (e: MessageEvent<WorkletOut>) => {
        const m = e.data;
        const w = this.waiters.get(m.type);
        if (w) { this.waiters.delete(m.type); w.res(m); }
        if (m.type === 'error') {
          for (const x of this.waiters.values()) x.rej(new Error(m.message));
          this.waiters.clear();
        }
        this.onEvent(m);
      };
    }
    post(m: WorkletIn, t: Transferable[] = []) { this.p.postMessage(m, t); }
    wait<K extends WorkletOut['type']>(type: K, ms = 10000): Promise<Extract<WorkletOut, { type: K }>> {
      return timeout(new Promise((res, rej) => this.waiters.set(type, { res, rej })), ms, `engine ${type}`);
    }
  }

  // ---- one processed microphone stream ----
  // source (the real microphone) -> engine node -> wet gain -> dest, and source -> dry gain -> dest; dest's
  // track is what the site gets, for the whole call. The engine can be removed (raw(): dry gain 1, the plain
  // microphone, e.g. its worker died with an extension update) and brought back (crossfade dry -> wet once
  // the new engine's output has started: no gap the length of its latency).
  type EngineMode = 'page' | 'isolated';
  class Session {
    src: MediaStreamTrack; live = true; model: string | null = null; mode: EngineMode | null = null; users = 1;
    engine: AudioWorkletNode | null = null; link: Link | null = null; module: ArrayBuffer | null = null;
    source: MediaStreamAudioSourceNode | null = null; dest: MediaStreamAudioDestinationNode | null = null;
    dry: GainNode | null = null; wet: GainNode | null = null; out: MediaStreamTrack | null = null;
    t0 = 0; vt = 0; switches = 0; loadMs?: number; latencyMs = 0; strength?: number; offMs = 0; offSince = 0;
    constructor(src: MediaStreamTrack) { this.src = src; }

    /** one more track of the page uses this session (sites like Meet open the same microphone twice) */
    acquire() { this.users++; }
    release() { if (--this.users <= 0) this.stop(); }

    async start(c: PageConfig): Promise<MediaStreamTrack> {
      const order: EngineMode[] = c.mode === 'page' ? ['page'] : c.mode === 'isolated' ? ['isolated'] : ['page', 'isolated'];
      let lastErr: unknown = null;
      for (const mode of order) {
        try { await this[mode](); this.mode = mode; break; } catch (e) { lastErr = e; this.unplug(); }
      }
      if (!this.mode) throw lastErr || new Error('no engine mode available');
      const link = this.link!;
      link.onEvent = (m) => this.event(m);
      link.post({ type: 'init', module: this.module || null, build: BUILD });
      await link.wait('ready');
      await this.apply(c);
      link.post({ type: 'voice', on: true });
      this.t0 = Date.now();
      this.wire();   // audio through the engine only once its model is loaded (after a reload: raw until then)
      return this.out!;
    }

    /** the engine is gone (extension reloaded) or failed: the plain microphone, same track */
    raw() {
      if (!this.live || !this.source) return;
      this.unplug();
      this.mode = null; this.model = null;
      const dry = this.dry!, t = dry.context.currentTime;
      dry.gain.cancelScheduledValues(t);
      dry.gain.setValueAtTime(1, t);
      report({ mode: null, error: null });
    }

    /** engine back after a reload (in place of raw()) */
    async revive(c: PageConfig) {
      if (!this.live) return;
      try {
        await timeout(this.start(c), 25000, 'engine restart');
        log(`on again: ${this.model}, ${this.mode} mode`);
        report();
      } catch (e) {
        this.raw();
        log(`could not restart, plain microphone: ${(e as Error).message}`);
      }
    }

    /** engine in the page's audio graph */
    async page() {
      const c = audioContext();
      this.module = await pageWasm();          // throws where the site's CSP forbids WebAssembly
      await loadWorklet(c, 'engine/worklet.bundle.js');
      const node = new AudioWorkletNode(c, 'vi-engine', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit' });
      this.link = new Link(node.port);
      this.engine = node;
    }

    /** engine in an extension-origin worker; the page only pumps audio */
    async isolated() {
      const c = audioContext();
      this.module = null;
      await loadWorklet(c, 'pump_worklet.js');
      const ctl = new MessageChannel(), audio = new MessageChannel();
      await rpc('frame', { sampleRate: c.sampleRate, build: BUILD }, [ctl.port2, audio.port2]);
      this.link = new Link(ctl.port1);
      await this.link.wait('up', 8000);
      const node = new AudioWorkletNode(c, 'vi-pump', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit' });
      node.port.postMessage({ type: 'port' }, [audio.port1]);
      node.port.onmessage = (e: MessageEvent) => { if (e.data && e.data.type === 'pump') report({ pumpUnderruns: e.data.underruns }); };
      this.engine = node;
    }

    wire() {
      const c = audioContext();
      if (!this.source) {
        this.source = c.createMediaStreamSource(new MediaStream([this.src]));
        this.dest = c.createMediaStreamDestination();
        this.dry = c.createGain();
        this.dry.gain.value = 0;
        this.source.connect(this.dry).connect(this.dest);
        this.out = this.dest.stream.getAudioTracks()[0];
      }
      const dry = this.dry!, wet = (this.wet = c.createGain());
      this.source.connect(this.engine!).connect(wet).connect(this.dest!);
      if (dry.gain.value > 0) {    // back from raw(): fade over once the engine's output has started
        const t = c.currentTime + 0.25, fade = 0.03;
        wet.gain.setValueAtTime(0, c.currentTime);
        wet.gain.setValueAtTime(0, t);
        wet.gain.linearRampToValueAtTime(1, t + fade);
        dry.gain.setValueAtTime(1, t);
        dry.gain.linearRampToValueAtTime(0, t + fade);
      }
    }

    private applying: Promise<void> = Promise.resolve();
    /** settings into the engine, one change at a time (two quick quality changes must not interleave) */
    apply(c: PageConfig): Promise<void> {
      const run = this.applying.then(() => this.applyNow(c));
      this.applying = run.catch(() => {});
      return run;
    }
    private async applyNow(c: PageConfig) {
      const link = this.link;
      if (!link || !this.live) return;
      if (c.model && c.model !== this.model) {
        const tLoad = Date.now();
        if (this.model) this.switches = (this.switches || 0) + 1;
        const pkg = await modelPackage(c.model);
        const w = await weights(c.model, pkg.manifest);
        const id = c.model;
        link.post({ type: 'add', id, kind: pkg.manifest.engine, config: pkg.config, weights: w, scalar: false, minFrames: 1, spread: 2,
          labels: { 'vi.model': c.model, 'vi.build': BUILD, 'vi.engine': pkg.manifest.engine, 'vi.host': 'extension' } }, w ? [w] : []);
        const added = await link.wait('added', 20000);
        link.post({ type: 'select', id });   // ~20 ms crossfade from the previous lane
        const old = this.model;
        if (old) setTimeout(() => { try { link.post({ type: 'remove', id: old }); } catch { /* closed */ } }, 200);
        this.model = c.model;
        this.loadMs = Date.now() - tLoad;
        // + the pump's jitter buffer in isolated mode (pump_worklet.ts TARGET = 3 quanta)
        this.latencyMs = (1e3 * added.latencyCtx) / 48000 + (this.mode === 'isolated' ? (3 * 128 * 1e3) / 48000 : 0);
      }
      link.post({ type: 'params', wet: c.strength, bypass: !c.enabled, rampMs: 30 });
      // time with noise suppression off, for the call summary
      const now = Date.now();
      if (this.offSince && c.enabled) { this.offMs = (this.offMs || 0) + now - this.offSince; this.offSince = 0; }
      if (!this.offSince && !c.enabled) this.offSince = now;
      this.strength = c.strength;
      report({ mode: this.mode, model: this.model ?? undefined, enabled: c.enabled, strength: c.strength, latencyMs: this.latencyMs, error: null });
    }

    event(m: WorkletOut) {
      if (m.type === 'voice') {
        const now = Date.now();
        if (now - (this.vt || 0) > 400) { this.vt = now; report({ speaking: m.speaking, levelDb: Math.round(m.levelDb) }); }
        if (meterOn) tell({ type: 'meter', inDb: m.inDb, outDb: m.outDb, speaking: m.speaking });
      } else if (m.type === 'error') { report({ error: m.message }); telemetry({ t: 'error', code: 'engine', msg: m.message, mode: this.mode, model: this.model }); }
    }

    /** remove the engine node and its link (source and dest stay) */
    unplug() {
      if (this.engine) { try { this.source?.disconnect(this.engine); } catch { /* not connected */ } }
      for (const n of [this.engine, this.wet]) { if (n) { try { n.disconnect(); } catch { /* already */ } } }
      this.wet = null;
      this.engine = null;
      if (this.link) { try { this.link.post({ type: 'close' }); this.link.p.close(); } catch { /* closed */ } }
      this.link = null;
    }

    teardown() {
      this.unplug();
      for (const n of [this.source, this.dry, this.dest]) { if (n) { try { n.disconnect(); } catch { /* already */ } } }
      this.source = this.dry = this.dest = null;
      try { this.out?.stop(); } catch { /* stopped */ }
      this.out = null;
    }

    /** hang-up: with reports on, the engine's stats go into the call summary first (at most 300 ms) */
    stop() {
      if (!this.live) return;
      this.live = false;
      sessions.delete(this);
      try { this.src.stop(); } catch { /* stopped */ }
      const link = this.link;
      const done = (st: WorkletStats | null) => { this.teardown(); report(); this.summary(st); idle(); };
      if (link && cfg && cfg.telemetry && this.t0) { link.post({ type: 'stats' }); link.wait('stats', 300).then(done, () => done(null)); }
      else done(null);
    }

    /** one report per call (telemetry.ts keeps only known fields; nothing about the page or the audio) */
    summary(st: WorkletStats | null) {
      if (!cfg || !cfg.telemetry || !this.t0) return;
      const ms = Date.now() - this.t0;
      const off = (this.offMs || 0) + (this.offSince ? Date.now() - this.offSince : 0);
      const min = ms / 60000;
      telemetry({
        t: 'call', mode: this.mode, model: this.model, strength: Math.round((this.strength ?? 1) * 10) * 10,
        dur: min < 1 ? '<1m' : min < 5 ? '<5m' : min < 15 ? '<15m' : min < 60 ? '<60m' : '>=60m',
        lat: Math.round(this.latencyMs || 0), load: this.loadMs, switches: this.switches || 0, offShare: ms ? (100 * off) / ms : 0,
        // engine time per 10 ms of audio (a callback is 128 samples = 2.67 ms at 48 kHz), the worst callbacks, gaps
        cost: st ? st.engAvgUs * 3.75 : undefined, p95: st ? st.p95Us : undefined, under: st ? st.underruns : undefined,
        over: st ? st.overruns : undefined, pumpUnder: last.pumpUnderruns,
      });
    }
  }

  /** sessions between getUserMedia and their engine being up: the audio context must not close under them */
  let starting = 0;
  /** no call left in this page: close the audio context (a new call opens a new one) */
  function idle() {
    if (ctx && starting === 0 && ![...sessions].some((s) => s.live)) { ctx.close().catch(() => {}); ctx = null; }
  }

  /** make the processed track behave like the microphone it replaces. Each track the page holds counts as one user
   *  of the session (clones too: a page may clone and stop the original); the last one stopped ends the session. */
  function disguise(out: MediaStreamTrack, src: MediaStreamTrack, session: Session): MediaStreamTrack {
    const own = (name: string, value: unknown) => Object.defineProperty(out, name, { value, configurable: true });
    const cloneOut = out.clone.bind(out);
    own('clone', () => { session.acquire(); return disguise(cloneOut(), src, session); });
    Object.defineProperty(out, 'label', { get: () => src.label, configurable: true });
    own('getSettings', () => ({ ...src.getSettings(), noiseSuppression: true }));
    own('getCapabilities', () => src.getCapabilities());
    own('getConstraints', () => src.getConstraints());
    own('applyConstraints', (c?: MediaTrackConstraints) => src.applyConstraints(c && typeof c === 'object' ? { ...c, noiseSuppression: false } : c));
    const stopOut = out.stop.bind(out);
    let done = false;
    own('stop', () => { if (!done) { done = true; session.release(); } stopOut(); });
    src.addEventListener('ended', () => { done = true; session.stop(); stopOut(); out.dispatchEvent(new Event('ended')); });
    return out;
  }

  async function getUserMedia(constraints?: MediaStreamConstraints): Promise<MediaStream> {
    if (!constraints || !constraints.audio) return origGUM(constraints);
    if (!cfg || cfg.enabled) audioContext();   // now, while the page's click still allows audio to start
    starting++;
    try { return await wrap(constraints as MediaStreamConstraints & { audio: true | MediaTrackConstraints }); } finally { starting--; idle(); }
  }

  async function wrap(constraints: MediaStreamConstraints & { audio: true | MediaTrackConstraints }): Promise<MediaStream> {
    let c: PageConfig;
    try { c = await config(); } catch (e) { log(`not applied: ${(e as Error).message}`); return origGUM(constraints); }
    if (!c.enabled) { log('off for this site'); return origGUM(constraints); }
    const audio: MediaTrackConstraints = constraints.audio === true ? {} : { ...constraints.audio };
    // what the site asked for (browsers default to on): restored if our engine can't start
    const siteNs = audio.noiseSuppression ?? true;
    audio.noiseSuppression = false;   // ours replaces the browser's; echo cancellation / AGC stay as asked
    const stream = await origGUM({ ...constraints, audio });
    const src = stream.getAudioTracks()[0];
    if (!src) return stream;
    // the same microphone already processed in this page (Meet opens it twice: preview + call): share that
    // session; processing and counting it twice would double the hours
    const same = (a: MediaTrackSettings, b: MediaTrackSettings) => (['deviceId', 'echoCancellation', 'autoGainControl'] as const).every((k) => a[k] === b[k]);
    const shared = [...sessions].find((x) => x.live && x.out?.readyState === 'live' && x.src.readyState === 'live' && same(x.src.getSettings(), src.getSettings()));
    if (shared) {
      shared.acquire();
      src.stop();
      log(`same microphone again: sharing the running session (${shared.users} tracks)`);
      return new MediaStream([disguise(shared.out!.clone(), shared.src, shared), ...stream.getVideoTracks()]);
    }
    const s = new Session(src);
    try {
      const out = await timeout(s.start(c), 25000, 'engine start');
      sessions.add(s);
      if (cfg && cfg !== c) await s.apply(cfg);   // settings changed while the engine was starting
      report();
      log(`on: ${s.model}, ${s.mode} mode`);
      return new MediaStream([disguise(out.clone(), src, s), ...stream.getVideoTracks()]);
    } catch (e) {
      s.live = false;
      s.teardown();
      const msg = (e as Error).message;
      report({ error: `noise suppression could not start here: ${msg}`, mode: null });
      telemetry({ t: 'error', code: 'start', msg, model: c.model });
      log(`could not start, using the plain microphone: ${msg}`);
      src.applyConstraints({ ...audio, noiseSuppression: siteNs }).catch(() => {});
      return stream;   // the call goes on with the plain microphone
    }
  }

  // keep the method's shape (name, length) for sites that inspect it
  Object.defineProperty(getUserMedia, 'name', { value: 'getUserMedia' });
  md.getUserMedia = getUserMedia;
})();

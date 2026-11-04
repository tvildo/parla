// parla.ts — Parla noise suppression for a web app's own microphone (browser, ES module).
//
//   import { Parla } from './parla/parla.js';
//   const parla = await Parla.load({ base: '/static/parla/' });
//   const ns = await parla.process(micTrack);     // micTrack from getUserMedia (noiseSuppression: false)
//   pc.addTrack(ns.track, stream);                // the cleaned microphone
//   ns.setEnabled(false);                         // live on/off, ramped (no click)
//   ns.setInput(otherMic);                        // another device, same output track
//   ns.stop();                                    // hang-up (the input track is the app's: not stopped)
//
// Errors never cut the microphone: when the engine can't start, `process()` throws and the app keeps its
// own track.
//
// What runs where: the wasm is compiled once on the main thread; each processed track gets its own
// AudioWorkletNode running the engine (engine/worklet.bundle.js, the same worklet as the Parla extension); all on the user's device. Assets under `base`: engine/ (worklet + 3 wasm builds) and
// models/<model>/ (the model package). Serve them with your app; the page's CSP needs
// script-src 'wasm-unsafe-eval' (WebAssembly) and the worklet from 'self'.
import { engineConfig } from './vi_engine.ts';
import type { Manifest } from './vi_engine.ts';
import type { WorkletIn, WorkletOut } from '../../engine/js/worklet.ts';
import { ASSETS } from './assets.ts';

export const VERSION = '0.1.0';

// relaxed SIMD > SIMD > none, by validating tiny probe modules (the same probes as the web host)
const PROBE = {
  simd: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11],
  relaxed: [0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11],
};
const valid = (b: number[]) => { try { return WebAssembly.validate(new Uint8Array(b)); } catch { return false; } };
/** the wasm build this browser runs, null without WebAssembly */
export const BUILD: 'relaxed' | 'simd' | 'nosimd' | null = typeof WebAssembly === 'undefined' ? null : valid(PROBE.relaxed) ? 'relaxed' : valid(PROBE.simd) ? 'simd' : 'nosimd';

export type ParlaErrorCode = 'unsupported' | 'load' | 'engine';
export class ParlaError extends Error {
  /** 'unsupported' (no WebAssembly / AudioWorklet) | 'load' | 'engine' */
  code: ParlaErrorCode;
  constructor(code: ParlaErrorCode, message: string) { super(message); this.name = 'ParlaError'; this.code = code; }
}

const errText = (e: unknown) => (e as Error | null)?.message || String(e);
const timeout = <T>(p: Promise<T>, ms: number, what: string): Promise<T> => Promise.race([p, new Promise<never>((_, rej) => setTimeout(() => rej(new ParlaError('engine', `${what}: timeout`)), ms))]);

async function fetchOk(url: URL, what: string): Promise<Response> {
  let r: Response;
  try { r = await fetch(url); } catch (e) { throw new ParlaError('load', `${what}: ${errText(e)}`); }
  if (!r.ok) throw new ParlaError('load', `${what}: HTTP ${r.status}`);
  return r;
}

type At = (path: string) => URL;
interface ModelPackage { manifest: Manifest; config: string; weights: ArrayBuffer | null }
export interface LoadOptions {
  /** where engine/ and models/ are served; omit it with a bundler (the npm package's own files are used) */
  base?: string;
  /** "deepfilternet3" (default) | "rnnoise" | "gtcrn" */
  model?: string;
}
export interface ProcessOptions { enabled?: boolean; gain?: number; strength?: number }
export interface VoiceEvent { speaking: boolean; prob: number; levelDb: number }

export class Parla {
  readonly model: string;
  private at: At;
  private bytes: ArrayBuffer;
  private ctx: AudioContext | null = null;
  private workletLoaded: Promise<void> | null = null;
  private pkg: Promise<ModelPackage> | null = null;
  /** @internal */
  streams = new Set<ParlaStream>();
  private closed = false;
  /** Compiles the engine. Model files are fetched on the first process().
   *  base: where the engine's files are served (engine/, models/). Omit it with the npm package and a bundler: the
   *  package's own files are used (the bundler emits them as assets).
   */
  static async load({ base, model = 'deepfilternet3' }: LoadOptions = {}): Promise<Parla> {
    if (!BUILD || typeof AudioWorkletNode === 'undefined') throw new ParlaError('unsupported', 'this browser has no WebAssembly or AudioWorklet');
    const root = base ? new URL(base.endsWith('/') ? base : `${base}/`, globalThis.location?.href) : null;
    if (!root && !ASSETS) throw new ParlaError('load', 'no engine files: pass `base` (where engine/ and models/ are served)');
    const at: At = (path) => {
      if (root) return new URL(path, root);
      const u = ASSETS![path];
      if (!u) throw new ParlaError('load', `${path}: not in this package`);
      return u;
    };
    const wasm = at(`engine/vi_rs_${BUILD}.wasm`);
    // the bytes are kept: each engine's worklet compiles its own copy (a Module posted to an AudioWorklet
    // never arrives in Chrome 154: a silent messageerror there)
    const bytes = await (await fetchOk(wasm, 'engine')).arrayBuffer();
    const magic = new Uint8Array(bytes, 0, Math.min(4, bytes.byteLength));
    if (magic.length < 4 || magic[0] !== 0 || magic[1] !== 0x61 || magic[2] !== 0x73 || magic[3] !== 0x6d) {
      throw new ParlaError('load', `engine: ${wasm} is not WebAssembly (a wrong \`base\`? the server answered something else)`);
    }
    // compiled here only to fail early where the page can't run WebAssembly (e.g. a CSP without 'wasm-unsafe-eval')
    try { await WebAssembly.compile(bytes); }
    catch (e) { throw new ParlaError('unsupported', `engine: ${errText(e)}`); }
    return new Parla(at, bytes, model);
  }

  private constructor(at: At, bytes: ArrayBuffer, model: string) {
    this.at = at; this.bytes = bytes; this.model = model;
  }

  /** @internal */
  wasmBytes(): ArrayBuffer { return this.bytes; }

  private audioContext(): AudioContext {
    if (!this.ctx || this.ctx.state === 'closed') {
      this.ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
      this.workletLoaded = null;
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    return this.ctx;
  }

  /** fetch the model now (the first process() then doesn't wait for its download) */
  async preload(): Promise<void> { await this.modelPackage(); }

  /** the model package (manifest + graph + weights), fetched once */
  private async modelPackage(): Promise<ModelPackage> {
    if (!this.pkg) {
      this.pkg = (async () => {
        const file = (f: string) => fetchOk(this.at(`models/${this.model}/${f}`), 'model');
        try {
          const manifest: Manifest = await (await file('manifest.json')).json();
          const graph = manifest.graph ? await (await file(manifest.graph)).text() : null;
          const weights = manifest.weights ? await (await file(manifest.weights)).arrayBuffer() : null;
          return { manifest, config: JSON.stringify(engineConfig(manifest, graph)), weights };
        } catch (e) {
          // not JSON (an HTML page for a wrong `base`), a broken package: the model's files, not the engine
          throw e instanceof ParlaError ? e : new ParlaError('load', `model ${this.model}: ${errText(e)}`);
        }
      })();
      this.pkg.catch(() => { this.pkg = null; });
    }
    return this.pkg;
  }

  /** A cleaned copy of `track` (a microphone track). Throws ParlaError when the engine can't start: the app then keeps using `track` itself.
   *  gain: input level (1 = as is); strength: how much of the cleaned signal (1 = all; less keeps some of the room,
   *  mixed in the engine) */
  async process(track: MediaStreamTrack, { enabled = true, gain = 1, strength = 1 }: ProcessOptions = {}): Promise<ParlaStream> {
    if (this.closed) throw new ParlaError('engine', 'this Parla was closed');
    const ctx = this.audioContext();
    // close() while this runs: stop where we are (every await below may span it)
    const live = () => { if (this.closed || ctx.state === 'closed') throw new ParlaError('engine', 'this Parla was closed'); };
    // a context the browser keeps suspended (no user gesture yet) would make the cleaned track silent
    if (ctx.state !== 'running') await Promise.race([ctx.resume(), new Promise((res) => setTimeout(res, 1500))]).catch(() => {});
    if (ctx.state !== 'running') throw new ParlaError('engine', 'the browser has not let audio start on this page yet (no click so far)');
    if (!this.workletLoaded) this.workletLoaded = ctx.audioWorklet.addModule(String(this.at('engine/worklet.bundle.js')));
    try { await this.workletLoaded; } catch (e) { this.workletLoaded = null; throw new ParlaError('load', `worklet: ${errText(e)}`); }
    live();
    const pkg = await this.modelPackage();
    live();
    let s: ParlaStream | null = null;
    try {
      s = new ParlaStream(this, ctx, track, enabled, gain, strength);
      this.streams.add(s);   // close() from here on stops it too
      await s.start(pkg);
      live();
    } catch (e) {
      s?.stop();
      throw e instanceof ParlaError ? e : new ParlaError('engine', errText(e));
    }
    return s;
  }

  /** every stream stopped, the audio context closed; process() refuses from then on (load a new Parla) */
  close(): void {
    this.closed = true;
    for (const s of [...this.streams]) s.stop();
    if (this.ctx) this.ctx.close().catch(() => {});
    this.ctx = null; this.workletLoaded = null;
  }
}

type Waiter = { res: (m: any) => void; rej: (e: unknown) => void };

export class ParlaStream {
  /** the cleaned microphone: hand this to the RTCPeerConnection */
  readonly track: MediaStreamTrack;
  input: MediaStreamTrack;
  enabled: boolean;
  strength: number;
  /** the engine's delay */
  latencyMs: number;
  /** speech on the cleaned output, ~19 times a second */
  onVoice: ((voice: VoiceEvent) => void) | null;
  /** the engine stopped while the stream ran (or the input track ended): the output now carries the plain input, or
   *  silence for an ended input. Switch back to your own track (or process() again). */
  onError: ((error: ParlaError) => void) | null = null;
  /** set once onError fired */
  error: ParlaError | null = null;
  private parla: Parla;
  private ctx: AudioContext;
  private live: boolean;
  private waiters: Map<string, Waiter>;
  private node: AudioWorkletNode;
  private dest: MediaStreamAudioDestinationNode;
  private source: MediaStreamAudioSourceNode | null;
  private gain: GainNode;
  /** @internal */
  constructor(parla: Parla, ctx: AudioContext, input: MediaStreamTrack, enabled: boolean, gain = 1, strength = 1) {
    this.parla = parla; this.ctx = ctx; this.input = input; this.enabled = enabled; this.strength = strength;
    this.latencyMs = 0; this.live = true; this.waiters = new Map();
    this.onVoice = null;
    this.node = new AudioWorkletNode(ctx, 'vi-engine', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit' });
    this.node.port.onmessage = (e: MessageEvent<WorkletOut>) => this.event(e.data);
    // the browser stopped the processor (it threw before our own guard ran, or ran out of memory)
    this.node.onprocessorerror = () => this.fail(new ParlaError('engine', 'the audio engine stopped'));
    this.dest = ctx.createMediaStreamDestination();
    this.track = this.dest.stream.getAudioTracks()[0];
    this.source = null;
    // input level before the engine (a microphone sensitivity setting); applies in bypass too
    this.gain = ctx.createGain();
    this.gain.gain.value = gain;
    this.gain.connect(this.node);
  }

  private post(m: WorkletIn, transfer: Transferable[] = []) { this.node.port.postMessage(m, transfer); }
  /** the worklet's answer `type`, or a ParlaError naming the step that never answered */
  private wait<K extends WorkletOut['type']>(type: K, ms: number): Promise<Extract<WorkletOut, { type: K }>> {
    return timeout(new Promise((res, rej) => this.waiters.set(type, { res, rej })), ms, `engine ${type}`);
  }

  /** @internal */
  async start(pkg: ModelPackage): Promise<void> {
    this.post({ type: 'init', module: this.parla.wasmBytes().slice(0) });
    await this.wait('ready', 10000);
    const id = this.parla.model;
    // the weights are copied: the package is kept for the next stream
    const weights = pkg.weights ? pkg.weights.slice(0) : null;
    this.post({ type: 'add', id, kind: pkg.manifest.engine, config: pkg.config, weights, scalar: false, minFrames: 1, spread: 2,
      labels: { 'vi.model': id, 'vi.build': BUILD, 'vi.engine': pkg.manifest.engine, 'vi.host': 'sdk' } }, weights ? [weights] : []);
    const added = await this.wait('added', 20000);   // the model's weights into the engine
    this.latencyMs = (1e3 * added.latencyCtx) / this.ctx.sampleRate;
    this.post({ type: 'select', id });
    this.mix(0);
    this.setInput(this.input);
    this.node.connect(this.dest);
    this.post({ type: 'voice', on: true });
  }

  private event(m: WorkletOut) {
    if (!m) return;
    const w = this.waiters.get(m.type);
    if (w) { this.waiters.delete(m.type); w.res(m); }
    if (m.type === 'error') {
      for (const x of this.waiters.values()) x.rej(new ParlaError('engine', m.message));
      this.waiters.clear();
      if (m.fatal) this.fail(new ParlaError('engine', m.message));
    } else if (m.type === 'voice') {
      if (this.onVoice) this.onVoice({ speaking: !!m.speaking, prob: m.prob, levelDb: m.levelDb });
    }
  }

  // the microphone went away (unplugged, permission revoked): the output would be silent from here
  private inputEnded = () => this.fail(new ParlaError('engine', 'the input track ended'));

  private fail(err: ParlaError) {
    if (!this.live || this.error) return;
    this.error = err;
    this.onError?.(err);
  }

  private mix(rampMs: number) { this.post({ type: 'params', wet: this.strength, bypass: !this.enabled, rampMs }); }

  /** noise suppression's strength, 0..1 (the cleaned part of the mix; the rest is the input, delayed to match),
   *  ramped in the engine */
  setStrength(s: number): void { this.strength = Math.min(1, Math.max(0, +s || 0)); if (this.live) this.mix(60); }

  /** noise suppression on / off while the stream runs (ramped in the engine: no click, same delay) */
  setEnabled(on: boolean): void { this.enabled = !!on; if (this.live) this.mix(30); }

  /** input level (1 = as is), ramped over 30 ms */
  setGain(g: number): void { if (this.live) this.gain.gain.setTargetAtTime(g, this.ctx.currentTime, 0.01); }

  /** another microphone into the same output track (a device change mid-call) */
  setInput(track: MediaStreamTrack): void {
    if (this.source) { try { this.source.disconnect(); } catch { /* gone */ } }
    this.input.removeEventListener('ended', this.inputEnded);
    track.addEventListener('ended', this.inputEnded);
    this.input = track;
    this.source = this.ctx.createMediaStreamSource(new MediaStream([track]));
    this.source.connect(this.gain);
  }

  /** release the engine and the output track (the input track belongs to the app) */
  stop(): void {
    if (!this.live) return;
    this.live = false;
    this.input.removeEventListener('ended', this.inputEnded);
    this.parla.streams.delete(this);
    for (const n of [this.source, this.gain, this.node, this.dest]) { try { n && n.disconnect(); } catch { /* already */ } }
    try { this.post({ type: 'close' }); this.node.port.close(); } catch { /* closed */ }
    this.track.stop();
  }
}

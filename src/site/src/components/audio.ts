// The demo's audio around the SDK: the microphone, the cleaned track, level meters and an A/B recording. The engine
// itself is the SDK (the npm package @tvildo/parla), the same code as the extension.
import { Parla, type ParlaStream } from "@tvildo/parla";

/** quality levels as users see them (never model names): translation keys demo.q.<label> / demo.q.<detail> */
export const QUALITIES = [
  { model: "deepfilternet3", label: "best", detail: "fullband" },
  { model: "rnnoise", label: "light", detail: "fullband" },
  { model: "gtcrn", label: "lightest", detail: "wideband" },
] as const;
export type Model = (typeof QUALITIES)[number]["model"];

export class Session {
  mic: MediaStream;
  ctx: AudioContext;
  inMeter: AnalyserNode;
  outMeter: AnalyserNode;
  private outSrc: MediaStreamAudioSourceNode | null = null;
  private parlas = new Map<Model, Promise<Parla>>();
  ns: ParlaStream | null = null;
  model: Model | null = null;
  /** noise suppression strength 0..1 (0 = the original voice, 1 = fully cleaned) */
  strength = 1;
  onVoice: (speaking: boolean) => void = () => {};

  private constructor(mic: MediaStream) {
    this.mic = mic;
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.inMeter = meter(this.ctx);
    this.outMeter = meter(this.ctx);
    this.ctx.createMediaStreamSource(mic).connect(this.inMeter);
  }

  /** asks for the microphone (call it from a click: browsers start audio only after a user gesture) */
  static async start(): Promise<Session> {
    const mic = await navigator.mediaDevices.getUserMedia({
      audio: { noiseSuppression: false, echoCancellation: true, autoGainControl: true, channelCount: 1 },
    });
    return new Session(mic);
  }

  private parla(model: Model): Promise<Parla> {
    let p = this.parlas.get(model);
    if (!p) {
      p = Parla.load({ model });
      p.catch(() => this.parlas.delete(model));
      this.parlas.set(model, p);
    }
    return p;
  }

  /** (re)start the engine with a quality level; the previous one stops once the new one runs */
  async use(model: Model): Promise<number> {
    const parla = await this.parla(model);
    const ns = await parla.process(this.mic.getAudioTracks()[0], { strength: this.strength });
    const old = this.ns;
    this.ns = ns;
    this.model = model;
    ns.onVoice = (v) => this.onVoice(v.speaking);
    if (this.outSrc) this.outSrc.disconnect();
    this.outSrc = this.ctx.createMediaStreamSource(new MediaStream([ns.track]));
    this.outSrc.connect(this.outMeter);
    old?.stop();
    return ns.latencyMs;
  }

  setStrength(s: number) {
    this.strength = s;
    this.ns?.setStrength(s);
  }

  /** the same seconds of the original microphone and the cleaned voice, recorded side by side; onTick gets the
   *  seconds elapsed (about 10 times a second) */
  async record(seconds: number, onTick: (elapsed: number) => void = () => {}): Promise<{ original: string; cleaned: string }> {
    if (!this.ns) throw new Error("not running");
    const take = (stream: MediaStream) => {
      const r = new MediaRecorder(stream);
      const parts: Blob[] = [];
      r.ondataavailable = (e) => parts.push(e.data);
      const done = new Promise<string>((res) => { r.onstop = () => res(URL.createObjectURL(new Blob(parts, { type: r.mimeType }))); });
      return { r, done };
    };
    const a = take(this.mic), b = take(new MediaStream([this.ns.track]));
    a.r.start(); b.r.start();
    const t0 = performance.now();
    await new Promise<void>((res) => {
      const iv = setInterval(() => {
        const el = (performance.now() - t0) / 1000;
        onTick(Math.min(el, seconds));
        if (el >= seconds) { clearInterval(iv); res(); }
      }, 100);
    });
    a.r.stop(); b.r.stop();
    return { original: await a.done, cleaned: await b.done };
  }

  stop() {
    this.ns?.stop();
    this.ns = null;
    for (const t of this.mic.getTracks()) t.stop();
    void this.ctx.close();
    for (const p of this.parlas.values()) p.then((x) => x.close()).catch(() => {});
  }
}

function meter(ctx: AudioContext): AnalyserNode {
  const a = ctx.createAnalyser();
  a.fftSize = 1024;
  a.smoothingTimeConstant = 0.6;
  return a;
}

/** level of an analyser now, RMS dBFS (-120 for silence) */
export function levelDb(a: AnalyserNode, buf: Float32Array<ArrayBuffer>): number {
  a.getFloatTimeDomainData(buf);
  let s = 0;
  for (const x of buf) s += x * x;
  return Math.max(-120, 10 * Math.log10(s / buf.length + 1e-12));
}

/** a dBFS level on the meter's scale, 0..1 (-70..-10 dBFS mapped linearly: the extension popup's scale, where the
 *  cleaned output of a noisy room still shows) */
export const meterScale = (db: number) => Math.min(1, Math.max(0, (db + 70) / 60));

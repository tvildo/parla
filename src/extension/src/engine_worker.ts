// engine_worker.ts — isolated mode, extension side: runs the SAME engine host code as the page mode
// (engine/worklet.bundle.js = web/worklet.js bundled) in a worker, under the extension's CSP. The worklet
// globals it uses are shimmed: AudioWorkletProcessor (its `port` = the control port), registerProcessor,
// sampleRate, currentFrame. The page's pump sends each 128-sample quantum; the processor's process()
// runs on it and the result goes straight back, so the page's audio clock drives the engine.
type Processor = { port: { onmessage: ((e: { data: unknown }) => void) | null }; process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean };
// the worklet globals the bundle expects, on this worker's global object
const scope = self as unknown as { currentFrame: number; sampleRate: number; registerProcessor: unknown; AudioWorkletProcessor: unknown };
const registry: Record<string, new () => Processor> = {};
let ctl: MessagePort | null = null;
scope.currentFrame = 0;
scope.sampleRate = 48000;
scope.registerProcessor = (name: string, cls: new () => Processor) => { registry[name] = cls; };
scope.AudioWorkletProcessor = class {
  port: { onmessage: null; postMessage: (m: unknown, t?: Transferable[]) => void };
  constructor() {
    const shim = {
      onmessage: null,
      postMessage: (m: unknown, t?: Transferable[]) => ctl && ctl.postMessage(m, t || []),
    };
    this.port = shim;
  }
};
importScripts('engine/worklet.bundle.js');

self.onmessage = (e: MessageEvent) => {
  if (!e.data || e.data.type !== 'start') return;
  const [ctlPort, audioPort] = e.ports;
  ctl = ctlPort;
  scope.sampleRate = e.data.sampleRate || 48000;
  const BUILDS = ['relaxed', 'simd', 'nosimd'];   // from the page: only a known engine file
  const build = BUILDS.includes(e.data.build) ? e.data.build : 'simd';
  const proc = new registry['vi-engine']();
  const deliver = (m: unknown) => proc.port.onmessage && proc.port.onmessage({ data: m });

  // control: 'init' without a module -> this worker compiles the engine itself (the page could not);
  // messages are handled in order, after that compile
  let chain = Promise.resolve();
  let modP: Promise<WebAssembly.Module> | null = null;
  ctlPort.onmessage = (ev: MessageEvent) => {
    const m = ev.data;
    if (m && m.type === 'close') { ctlPort.close(); audioPort.close(); self.postMessage('closed'); self.close(); return; }
    chain = chain.then(async () => {
      if (m && m.type === 'init' && !m.module) {
        modP = modP || fetch(`engine/vi_rs_${BUILDS.includes(m.build) ? m.build : build}.wasm`).then((r) => r.arrayBuffer()).then((b) => WebAssembly.compile(b));
        deliver({ ...m, module: await modP });
      } else deliver(m);
    }).catch((err) => ctlPort.postMessage({ type: 'error', message: String((err && err.message) || err) }));
  };

  const outBuf = [new Float32Array(128)];
  audioPort.onmessage = (ev: MessageEvent<Float32Array>) => {
    const x = ev.data;
    const out = outBuf[0];
    proc.process([[x]], [[out]]);
    scope.currentFrame += 128;
    x.set(out);                       // reuse the incoming buffer for the reply
    audioPort.postMessage(x, [x.buffer]);
  };
  ctlPort.postMessage({ type: 'up' });
};

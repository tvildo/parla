// engine_frame.ts — isolated mode: one engine worker per processed microphone stream. Only the embedding
// page (through the extension's bridge script) can start one; the ports it hands over are the only link.
// Any page can embed this frame (it is web-accessible), so at most a few workers per frame: a hostile page can
// use its own CPU, never pile up workers in the extension's process.
const MAX_WORKERS = 8;
let workers = 0;
window.addEventListener('message', (e) => {
  if (e.source !== window.parent || !e.data || e.data.type !== 'vi-engine' || e.ports.length !== 2) return;
  if (workers >= MAX_WORKERS) { for (const p of e.ports) p.close(); return; }
  workers++;
  const w = new Worker('engine_worker.js');
  w.addEventListener('message', (m: MessageEvent) => { if (m.data === 'closed') workers--; });
  w.postMessage({ type: 'start', sampleRate: e.data.sampleRate, build: e.data.build }, [...e.ports]);
});

// Node test of src/engine/js/otel_export.ts against the real wasm C ABI (size_probe built with --features otel):
//   node src/engine/vi_otel/js/test_otel_export.ts [path/to/otel.wasm] [collector base url]
// With a collector URL the payloads are also POSTed for real (e.g. http://<windows-box>:4318).
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { otelBind, createOtelExporter, browserResource } from '../../js/otel_export.ts';
import type { OtelExports } from '../../js/otel_export.ts';

const here = fileURLToPath(new URL('.', import.meta.url));
const wasmPath = process.argv[2] || here + '../size_probe/out/otel.wasm';
if (!existsSync(wasmPath)) { console.log('skip: no wasm at', wasmPath); process.exit(0); }
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
// the probe module's own test exports on top of the vi_otel C ABI
const ex = instance.exports as unknown as OtelExports & { probe_record(us: number, db: number): void; probe_span(n: number): void };
const o = otelBind(ex);

o.tick(); o.seed();
// opt-in default off: nothing comes out before enable
o.snapshot();
assert.deepEqual(o.snapshot(), { metrics: null, spans: null });
o.resource({ 'service.name': 'vi-web-test', 'service.version': '0.1.0', ...browserResource({ 'vi.simd': true, 'audio.sample_rate': 48000, 'audio.base_latency': 0.0053 }) });
o.enable(true);
for (let i = 0; i < 375; i++) ex.probe_record(80 + (i % 50) * 10, -30 + (i % 7));
ex.probe_span(1); ex.probe_span(2);
const jitter = o.instrument('histogram', 'vi.webrtc.jitter', 'ms', [5, 10, 20, 40, 80]);
const lost = o.instrument('counter', 'vi.webrtc.packets_lost', '{packet}');
jitter(12); jitter(3); lost(4); lost(1);

const s = o.snapshot();
// OTLP JSON, read loosely: the assertions below are the check
const dec = (b: Uint8Array | null): any => JSON.parse(new TextDecoder().decode(b!));
const m = dec(s.metrics), t = dec(s.spans);
const res = Object.fromEntries(m.resourceMetrics[0].resource.attributes.map((a: any) => [a.key, a.value]));
assert.deepEqual(res['service.name'], { stringValue: 'vi-web-test' });
assert.deepEqual(res['audio.sample_rate'], { intValue: '48000' });
assert.deepEqual(res['vi.simd'], { boolValue: true });
assert.deepEqual(res['telemetry.sdk.name'], { stringValue: 'vi_otel' });
const metrics = Object.fromEntries(m.resourceMetrics[0].scopeMetrics.flatMap((sm: any) => sm.metrics.map((x: any) => [x.name, x])));
const dur = metrics['vi.engine.process.duration'].histogram.dataPoints[0];
assert.equal(dur.count, '375');
assert.equal(dur.bucketCounts.reduce((a: number, b: string) => a + +b, 0), 375);
assert.equal(metrics['vi.webrtc.jitter'].histogram.dataPoints[0].bucketCounts.join(), '1,0,1,0,0,0');
assert.equal(metrics['vi.webrtc.packets_lost'].sum.dataPoints[0].asDouble, 5);
assert.equal(t.resourceSpans[0].scopeSpans[0].spans.length, 2);
assert.match(t.resourceSpans[0].scopeSpans[0].spans[0].traceId, /^[0-9a-f]{32}$/);
assert.equal(o.snapshot().spans, null, 'spans are drained');

// small buffer path: negative return -> grow and retry
const tiny = ex.otel_buf_alloc(16);
assert.ok(ex.otel_snapshot_metrics(tiny, 16) < -1000);
ex.otel_buf_free(tiny, 16);

// exporter: retry/backoff, metrics supersede, non-retryable drop
const calls: [string, number, boolean | undefined][] = [];
let fail = 1;
const fakeFetch = (async (url: string, init: RequestInit) => {
  calls.push([url, (init.body as Uint8Array).length, init.keepalive]);
  if (fail-- > 0) return { status: 503 };
  return { status: 200 };
}) as unknown as typeof fetch;
ex.probe_span(3);
const errs: string[] = [];
const exp = createOtelExporter({ endpoint: 'http://collector:4318/', snapshot: o.snapshot, enabled: true, intervalMs: 1e9, fetch: fakeFetch, onError: (e) => errs.push((e as Error).message) });
await exp.flush();
assert.equal(errs[0], 'otlp http 503');
assert.equal(exp.queued, 2); // metrics + traces kept for retry
await exp.flush(); // inside backoff window: snapshot taken, metrics superseded, nothing sent
assert.equal(exp.queued, 2);
assert.equal(calls.length, 1);
await new Promise((r) => setTimeout(r, 1100));
await exp.flush();
assert.equal(exp.queued, 0);
assert.deepEqual(calls.slice(1).map((c) => c[0]), ['http://collector:4318/v1/traces', 'http://collector:4318/v1/metrics']);
exp.setEnabled(false);

// optional: real collector
const url = process.argv[3];
if (url) {
  ex.probe_span(4);
  const logFetch = (async (u: string, i: RequestInit) => { const r = await fetch(u, i); console.log('POST', u, (i.body as Uint8Array).length, 'B ->', r.status, await r.clone().text()); return r; }) as typeof fetch;
  const real = createOtelExporter({ endpoint: url, snapshot: o.snapshot, enabled: true, intervalMs: 1e9, fetch: logFetch, onError: (e) => console.log('error', (e as Error).message) });
  await real.flush();
  real.setEnabled(false);
}
console.log('otel_export ok:', s.metrics!.length, 'B metrics,', s.spans!.length, 'B spans');

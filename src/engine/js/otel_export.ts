// vi_otel browser transport (no deps). Two halves, so the wasm side can live in an AudioWorklet
// (no fetch there) while sending happens on the main thread:
//   otelBind(wasmExports)      -> wasm-side handle: tick/seed/enable/ratio/resource/snapshot
//   createOtelExporter(opts)   -> main-thread sender: interval batching, opt-in, session
//                                 sampling, retry with backoff, keepalive/sendBeacon on pagehide
// Payloads are OTLP/HTTP JSON produced by Rust (otel_snapshot_*), POSTed to
// `${endpoint}/v1/metrics` and `${endpoint}/v1/traces`.
// AudioWorkletGlobalScope (Chrome) has no TextEncoder: attribute/resource strings are ASCII-ish,
// so a minimal UTF-8 encoder suffices there.
const enc: { encode(s: string): Uint8Array } = globalThis.TextEncoder ? new TextEncoder() : { encode(s: string) {
  const out: number[] = [];
  for (const ch of s) {
    let c = ch.codePointAt(0)!;
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return Uint8Array.from(out);
} };

/** the vi_otel wasm module's C ABI (plugins/vi_otel.wasm) */
export interface OtelExports {
  memory: WebAssembly.Memory;
  otel_buf_alloc(n: number): number;
  otel_buf_free(p: number, n: number): void;
  otel_set_time_ms(ms: number): void;
  otel_seed(a: number, b: number): void;
  otel_set_enabled(on: number): void;
  otel_set_trace_ratio(r: number): void;
  otel_resource_bool(k: number, kl: number, v: number): void;
  otel_resource_num(k: number, kl: number, v: number, isInt: number): void;
  otel_resource_str(k: number, kl: number, v: number, vl: number): void;
  otel_instrument(kind: number, n: number, nl: number, u: number, ul: number, b: number, bl: number): number;
  otel_instrument_attrs?(kind: number, n: number, nl: number, u: number, ul: number, b: number, bl: number, a: number, al: number): number;
  otel_record(id: number, v: number): void;
  otel_snapshot_metrics(p: number, cap: number): number;
  otel_snapshot_spans(p: number, cap: number): number;
}
export type AttrValue = string | number | boolean | null | undefined;
export type Attrs = Record<string, AttrValue>;
export interface Snapshot { metrics: Uint8Array | null; spans: Uint8Array | null }
export type InstrumentKind = 'counter' | 'updown' | 'gauge' | 'histogram';

export function otelBind(ex: OtelExports) {
  let cap = 1 << 16, ptr = ex.otel_buf_alloc(cap);
  const put = (s: unknown): [number, number] => {
    const b = enc.encode(String(s)), n = b.length, p = ex.otel_buf_alloc(n);
    new Uint8Array(ex.memory.buffer, p, n).set(b);
    return [p, n];
  };
  const snap = (fn: (p: number, cap: number) => number): Uint8Array | null => {
    for (;;) {
      const n = fn(ptr, cap);
      if (n >= 0) return n ? new Uint8Array(ex.memory.buffer, ptr, n).slice() : null;
      ex.otel_buf_free(ptr, cap);
      cap = Math.max(-n, cap * 2);
      ptr = ex.otel_buf_alloc(cap);
    }
  };
  return {
    tick: (ms = Date.now()) => ex.otel_set_time_ms(ms),
    seed() {
      const a = new Uint32Array(2);
      if (globalThis.crypto?.getRandomValues as unknown) crypto.getRandomValues(a);
      else { a[0] = Math.random() * 2 ** 32; a[1] = Date.now(); }
      ex.otel_seed(a[0], a[1]);
    },
    enable: (on: boolean) => ex.otel_set_enabled(on ? 1 : 0),
    ratio: (r: number) => ex.otel_set_trace_ratio(r),
    // values: string | number (integers -> intValue) | boolean; null/undefined skipped
    resource(attrs: Attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null) continue;
        const [kp, kl] = put(k);
        if (typeof v === 'boolean') ex.otel_resource_bool(kp, kl, v ? 1 : 0);
        else if (typeof v === 'number') ex.otel_resource_num(kp, kl, v, Number.isInteger(v) ? 1 : 0);
        else { const [vp, vl] = put(v); ex.otel_resource_str(kp, kl, vp, vl); ex.otel_buf_free(vp, vl); }
        ex.otel_buf_free(kp, kl);
      }
    },
    // JS-side instruments. kind: 'counter' | 'updown' | 'gauge' | 'histogram'. `attrs` binds the
    // handle to one attribute set (e.g. {'vi.model': 'gtcrn', 'vi.build': 'simd'}): setup time;
    // the returned record(v) is lock-free and allocation-free (safe on the audio thread).
    instrument(kind: InstrumentKind, name: string, unit = '', bounds: number[] = [], attrs: Attrs | null = null): (v: number) => void {
      const [np, nl] = put(name), [up, ul] = put(unit);
      const bp = ex.otel_buf_alloc(bounds.length * 8 || 8);
      new Float64Array(ex.memory.buffer, bp, bounds.length).set(bounds);
      const lines = attrs ? Object.entries(attrs).filter(([, v]) => v != null).map(([k, v]) => `${k}=${v}`).join('\n') : '';
      const [ap, al] = put(lines);
      const k = ['counter', 'updown', 'gauge', 'histogram'].indexOf(kind);
      const id = ex.otel_instrument_attrs ? ex.otel_instrument_attrs(k, np, nl, up, ul, bp, bounds.length, ap, al)
        : ex.otel_instrument(k, np, nl, up, ul, bp, bounds.length);
      ex.otel_buf_free(np, nl); ex.otel_buf_free(up, ul); ex.otel_buf_free(ap, al); ex.otel_buf_free(bp, bounds.length * 8 || 8);
      return (v) => ex.otel_record(id, v);
    },
    snapshot: (): Snapshot => ({ metrics: snap(ex.otel_snapshot_metrics), spans: snap(ex.otel_snapshot_spans) }),
  };
}
export type OtelHandle = ReturnType<typeof otelBind>;

/** Chrome's User-Agent Client Hints and device memory (not in every browser, not in TypeScript's DOM library) */
interface NavigatorHints {
  userAgentData?: { brands?: { brand: string; version: string }[]; platform?: string; mobile?: boolean };
  hardwareConcurrency?: number;
  deviceMemory?: number;
}

// Coarse, non-identifying environment attributes (no UA string, no ids).
export function browserResource(extra: Attrs = {}): Attrs {
  const n: NavigatorHints = globalThis.navigator || {}, ua = n.userAgentData;
  const brand = ua?.brands?.find((b) => !/Not.A.Brand|Chromium/i.test(b.brand)) || ua?.brands?.[0];
  return {
    'browser.name': brand?.brand, 'browser.version': brand?.version, 'os.type': ua?.platform,
    'browser.mobile': ua?.mobile, 'host.cpu.count': n.hardwareConcurrency, 'host.memory.gb': n.deviceMemory,
    ...extra,
  };
}

export interface ExporterOptions {
  endpoint: string;
  snapshot: () => Snapshot | null | Promise<Snapshot | null>;
  intervalMs?: number;
  enabled?: boolean;
  /** fraction of sessions that export */
  sampleRate?: number;
  headers?: Record<string, string>;
  maxQueue?: number;
  onError?: (e: unknown) => void;
  fetch?: typeof fetch;
}

// opts: { endpoint, snapshot: () => {metrics, spans} | Promise, intervalMs = 15000, enabled = false,
//   sampleRate = 1 (fraction of sessions that export), headers = {}, maxQueue = 16, onError, fetch }
export function createOtelExporter(o: ExporterOptions) {
  const f = o.fetch || globalThis.fetch.bind(globalThis);
  const base = o.endpoint.replace(/\/+$/, '');
  const inSample = Math.random() < (o.sampleRate ?? 1);
  const q: { p: string; b: Uint8Array }[] = [];
  let on = false, timer: ReturnType<typeof setInterval> | undefined, backoff = 0, nextTry = 0, busy = false;
  const hdr = { 'Content-Type': 'application/json', ...o.headers };

  const enqueue = (s: Snapshot | null) => {
    if (!s) return;
    if (s.metrics) { // cumulative: a newer metrics snapshot supersedes queued ones
      for (let i = q.length - 1; i >= 0; i--) if (q[i].p === '/v1/metrics') q.splice(i, 1);
      q.push({ p: '/v1/metrics', b: s.metrics });
    }
    if (s.spans) q.push({ p: '/v1/traces', b: s.spans });
    while (q.length > (o.maxQueue || 16)) q.shift();
  };
  async function flush() {
    if (!on || busy) return;
    busy = true;
    try {
      enqueue(await o.snapshot());
      if (Date.now() < nextTry) return;
      while (q.length) {
        const r = await f(base + q[0].p, { method: 'POST', headers: hdr, body: q[0].b as BodyInit });
        if (r.status === 429 || r.status === 408 || r.status >= 500) throw new Error('otlp http ' + r.status);
        q.shift(); // success, or a non-retryable 4xx: drop
        backoff = 0;
      }
    } catch (e) {
      backoff = backoff ? Math.min(backoff * 2, 300000) : 1000; // 1 s, 2 s, ... 5 min (x0.5..1 jitter)
      nextTry = Date.now() + backoff * (0.5 + Math.random() / 2);
      o.onError?.(e);
    } finally { busy = false; }
  }
  function beacon() { // page is going away: send what we have without waiting
    if (!on) return;
    try { const s = o.snapshot(); if (s && !(s instanceof Promise)) enqueue(s); } catch {}
    let budget = 60000; // keepalive bodies share a 64 KiB budget
    for (const it of q.splice(0)) {
      if (it.b.length <= budget) {
        budget -= it.b.length;
        try { f(base + it.p, { method: 'POST', headers: hdr, body: it.b as BodyInit, keepalive: true }).catch(() => {}); continue; } catch {}
      }
      globalThis.navigator?.sendBeacon?.(base + it.p, new Blob([it.b as BlobPart], { type: 'application/json' }));
    }
  }
  const onHide = () => beacon();
  const onVis = () => { if (document.visibilityState === 'hidden') beacon(); };
  function setEnabled(v: boolean | undefined) {
    v = !!v && inSample;
    if (v === on) return;
    on = v;
    if (on) {
      timer = setInterval(flush, o.intervalMs || 15000);
      globalThis.addEventListener?.('pagehide', onHide);
      globalThis.document?.addEventListener('visibilitychange', onVis);
    } else {
      clearInterval(timer);
      globalThis.removeEventListener?.('pagehide', onHide);
      globalThis.document?.removeEventListener('visibilitychange', onVis);
      q.length = 0;
    }
  }
  setEnabled(o.enabled);
  return { setEnabled, flush, beacon, get enabled() { return on; }, get queued() { return q.length; }, sampled: inSample };
}

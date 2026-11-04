// worker.ts — the receiver of the extension's reports (a Cloudflare Worker at https://t.parla.cc/v1/e).
//
// POST /v1/e with a JSON array of up to 20 events (<= 8 KB). Every event is checked again here (the extension does
// the same; this side doesn't trust any client): only known fields, numbers clamped, words from fixed lists, strings
// cut. Each accepted event becomes one structured log line (kept at most 7 days) and, with the AE binding, one
// Analytics Engine data point without page addresses (kept 3 months; index = the random ID, so count(DISTINCT index1)
// = active installs). Nothing else is stored: no IP address, no headers, no cookies.
//
// Turning reports down without an extension update:
//   SAMPLE       (variable, 0..1, default 1): the share of installs that should report. Sent back in every answer as
//                `x-parla-sample`; the extension keeps it and only installs whose ID falls under it keep reporting.
//   DAILY_LIMIT  (variable, default 80000): events accepted per UTC day (counted by the BUDGET Durable Object). Over
//                it: 429 + Retry-After until midnight UTC; extensions pause until then.
// Answers 204, 429, 400, 405 or 413.
//
// Deployed as the emitted JavaScript by `python build.py telemetry-deploy` (bindings and limits are there).
const MAX_BYTES = 8192;
const MAX_EVENTS = 20;

/** the Cloudflare bindings this Worker uses (all optional: tests and a bare deploy have none) */
export interface Env {
  SAMPLE?: string;
  DAILY_LIMIT?: string;
  AE?: { writeDataPoint(p: { indexes: string[]; blobs: string[]; doubles: number[] }): void };
  BUDGET?: { idFromName(name: string): unknown; get(id: unknown): { fetch(url: string, init: RequestInit): Promise<Response> } };
}
/** a Durable Object's state (the part Budget uses) */
export interface BudgetState {
  storage: { get(key: string): Promise<unknown>; put(key: string, value: number): Promise<void>; delete(key: string): Promise<unknown> };
}
/** an accepted event: the fields below, cleaned */
export type CleanEvent = { t: string; v: 1; id: string; [field: string]: string | number | boolean | undefined };

type Field = (x: unknown) => string | number | boolean | undefined;
const num = (lo: number, hi: number): Field => (x) => (Number.isFinite(+(x as number)) ? Math.min(hi, Math.max(lo, Math.round(+(x as number)))) : undefined);
const oneOf = (...xs: (string | number)[]): Field => (x) => (xs.includes(x as string | number) ? (x as string | number) : undefined);
const str = (n: number, re = /^[\w .:/@+-]*$/): Field => (x) => (typeof x === 'string' && re.test(x) ? x.slice(0, n) : undefined);
const text = (n: number): Field => (x) => (typeof x === 'string' ? x.replace(/[\u0000-\u001f]/g, ' ').slice(0, n) : undefined);

/** addresses in free text cut to their domain (older builds sent whole URLs in error messages with diagnostics on) */
const domainsOnly = (s: string | number | boolean | undefined) =>
  (typeof s === 'string' ? s.replace(/\b([a-z][a-z0-9+.-]*:\/\/)([^\s/:?#]*)\S*/gi, (_, _scheme, host) => host || '<url>') : s);

const COMMON: Record<string, Field> = {
  v: oneOf(1), ts: num(0, 4102444800000), id: str(32, /^[0-9a-f]{8,32}$/), commit: str(18, /^[0-9a-f]{7,12}(-dirty)?$/),   // the git commit the extension was built from
  ext: str(16, /^[\d.]+$/),
  br: oneOf('chrome', 'edge', 'firefox', 'other'), brv: num(0, 1000), os: oneOf('windows', 'macos', 'linux', 'chromeos', 'android', 'other'),
  cores: num(0, 1024), lang: str(5, /^[A-Za-z-]*$/),
  app: oneOf('meet', 'teams', 'zoom', 'discord', 'slack', 'webex', 'whereby', 'jitsi', 'local', 'other'),
  domain: str(253, /^[a-z0-9.-]+$/i),   // only present when the user turned detailed diagnostics on (the site's domain, nothing more)
};
const MODELS = oneOf('deepfilternet3', 'rnnoise', 'gtcrn', 'custom');
const MODES = oneOf('page', 'isolated');
const TYPES: Record<string, Record<string, Field>> = {
  call: { mode: MODES, model: MODELS, strength: num(0, 100), dur: oneOf('<1m', '<5m', '<15m', '<60m', '>=60m'),
    lat: num(0, 2000), cost: num(0, 100000), p95: num(0, 100000), load: num(0, 1000000), under: num(0, 1e6), over: num(0, 1e6),
    pumpUnder: num(0, 1e6), offShare: num(0, 100), switches: num(0, 1000) },
  error: { code: oneOf('start', 'engine', 'wasm', 'worklet', 'frame', 'model', 'reload', 'other'), msg: (x) => domainsOnly(text(200)(x)), mode: MODES, model: MODELS },
  active: {},   // once a day per install while the browser runs it: counts active installs, calls or not
  settings: { key: oneOf('model', 'strength', 'enabled', 'mode', 'lang', 'diagnostics'), value: (x) => (typeof x === 'number' || typeof x === 'boolean' ? x : str(16)(x)) },
};

/** the event as it may be stored, or null */
export function clean(ev: unknown): CleanEvent | null {
  if (!ev || typeof ev !== 'object') return null;
  const e = ev as Record<string, unknown>, t = e.t;
  if (typeof t !== 'string' || !Object.hasOwn(TYPES, t)) return null;
  const out: Record<string, string | number | boolean> = { t };
  // builds before 2026-11-04 sent `url` (host + path): only its domain is kept
  if (e.domain === undefined && typeof e.url === 'string') e.domain = e.url.split(/[/:?#]/)[0];
  for (const [k, f] of Object.entries({ ...COMMON, ...TYPES[t] })) { const v = f(e[k]); if (v !== undefined) out[k] = v; }
  return out.v === 1 && out.id ? (out as CleanEvent) : null;
}

const today = () => new Date().toISOString().slice(0, 10);
const untilMidnight = () => { const n = new Date(); return Math.ceil((Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate() + 1) - n.getTime()) / 1000); };

/** the daily event counter (one Durable Object instance, SQLite-backed) */
export class Budget {
  state: BudgetState;
  constructor(state: BudgetState) { this.state = state; }
  async fetch(request: Request): Promise<Response> {
    const { day, n, limit } = (await request.json()) as { day: string; n: number; limit: number };
    const key = `d:${day}`;
    const used = ((await this.state.storage.get(key)) as number) || 0;
    if (used >= limit) return Response.json({ ok: false, used });
    await this.state.storage.put(key, used + n);
    // the day before yesterday's counter is not needed any more
    await this.state.storage.delete(`d:${new Date(Date.now() - 2 * 864e5).toISOString().slice(0, 10)}`);
    return Response.json({ ok: true, used: used + n });
  }
}

export default {
  async fetch(request: Request, env: Env = {}, _ctx?: unknown, log: (line: unknown) => void = console.log): Promise<Response> {
    const raw = env.SAMPLE === undefined || env.SAMPLE === '' ? 1 : +env.SAMPLE;
    const sample = Number.isFinite(raw) ? Math.min(1, Math.max(0, raw)) : 1;
    const headers = { 'x-parla-sample': String(sample) };
    const url = new URL(request.url);
    if (url.pathname !== '/v1/e') return new Response(null, { status: 404 });
    if (request.method !== 'POST') return new Response(null, { status: 405, headers: { allow: 'POST' } });
    const len = Number(request.headers.get('content-length')) || 0;
    if (len > MAX_BYTES) return new Response(null, { status: 413 });
    const body = await request.text();
    if (body.length > MAX_BYTES) return new Response(null, { status: 413 });
    let events: unknown;
    try { events = JSON.parse(body); } catch { return new Response(null, { status: 400 }); }
    if (!Array.isArray(events) || events.length > MAX_EVENTS) return new Response(null, { status: 400 });
    const ok = events.map(clean).filter((e) => e !== null);
    if (sample === 0) return new Response(null, { status: 204, headers });   // reports turned off: accept, keep nothing
    // the daily budget (fails open if the counter is unavailable)
    if (env.BUDGET && ok.length) {
      try {
        const stub = env.BUDGET.get(env.BUDGET.idFromName('daily'));
        const r: { ok: boolean } = await (await stub.fetch('https://budget/', { method: 'POST', body: JSON.stringify({ day: today(), n: ok.length, limit: Number(env.DAILY_LIMIT) || 80000 }) })).json();
        if (!r.ok) return new Response(null, { status: 429, headers: { ...headers, 'retry-after': String(untilMidnight()) } });
      } catch { /* counter down: accept */ }
    }
    for (const e of ok) {
      log({ parla: e });   // one structured log line per event (Workers Logs indexes its fields)
      // Analytics Engine: index = the random ID (exact distinct counts), blobs = the words, doubles = the numbers
      env.AE?.writeDataPoint({
        indexes: [e.id],
        blobs: [e.t, e.br, e.os, e.app ?? '', e.model ?? '', e.mode ?? '', e.ext ?? '', e.lang ?? '', e.code ?? '', e.dur ?? '', e.commit ?? ''].map(String),   // blob11 = commit (appended: older columns keep their place)
        doubles: [e.lat ?? 0, e.cost ?? 0, e.under ?? 0, e.brv ?? 0, e.cores ?? 0].map(Number),
      });
    }
    return new Response(null, { status: 204, headers });
  },
};

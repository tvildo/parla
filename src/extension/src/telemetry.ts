// telemetry.ts — the extension's reports (service worker module). docs/telemetry.md has the full field list.
//
// Two switches (settings; companies can set them by policy):
//   telemetry   (default OFF) anonymous technical reports: errors, performance, usage, browser/device, the KIND of
//                             calling app (meet / teams / zoom / ...), a random ID that changes every month
//   diagnostics (default OFF) detailed diagnostics, only with reports on and after the user turned it on in the settings (Firefox also asks
//                             through its own consent): adds the page address (host + path, never the query or
//                             fragment) and a permanent random installation ID
// Never sent: audio, page content, query strings, anything typed. Events are small JSON objects, batched, capped per
// day and sent to the receiver (src/telemetry/worker.ts) with fetch.
// The receiver steers the volume without an extension update: every answer carries `x-parla-sample` (the share of
// installs that should report; an install reports when its ID hashes below it), and 429 + Retry-After pauses all
// sending until then (its daily limit). Plus one `active` report a day, so active installs can be counted.
import { buildInfo, getLocal } from './messages.ts';
import type { Settings, TelemetryEvent } from './messages.ts';

export const ENDPOINT = 'https://t.parla.cc/v1/e';
const DAILY_CAP = 40;         // events per install per day
const QUEUE_MAX = 100;
const MSG_MAX = 200;

const APPS: [string, RegExp][] = [
  ['meet', /(^|\.)meet\.google\.com$/], ['teams', /(^|\.)teams\.(microsoft|live)\.com$/], ['zoom', /(^|\.)zoom\.(us|com)$/],
  ['discord', /(^|\.)discord\.com$/], ['slack', /(^|\.)(slack|app\.slack)\.com$/], ['webex', /(^|\.)webex\.com$/],
  ['whereby', /(^|\.)whereby\.com$/], ['jitsi', /(^|\.)meet\.jit\.si$/],
];
/** the kind of calling app for a page, never its address */
export function appKind(url: string): string {
  try {
    const h = new URL(url).hostname;
    return APPS.find(([, re]) => re.test(h))?.[0] ?? (/^(localhost|127\.0\.0\.1)$/.test(h) ? 'local' : 'other');
  } catch { return 'other'; }
}

const month = () => new Date().toISOString().slice(0, 7);
const day = () => new Date().toISOString().slice(0, 10);
const rid = () => Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, '0')).join('');

/** the effective switches (settings < policy), and Firefox's own consent where it exists */
export async function switches(settings: Settings): Promise<{ telemetry: boolean; diagnostics: boolean }> {
  let telemetry = settings.telemetry === true;   // off unless turned on (or set on by policy)
  let diagnostics = settings.diagnostics === true;
  // Firefox: the user's choices in about:addons (data collection permissions) win as well
  const perms = globalThis.browser?.permissions;
  if (perms?.contains && navigator.userAgent.includes('Firefox')) {
    try {
      if (telemetry) telemetry = await perms.contains({ data_collection: ['technicalAndInteraction'] });
      if (diagnostics) diagnostics = await perms.contains({ data_collection: ['browsingActivity'] });
    } catch { /* older Firefox without data collection permissions: our switches only */ }
  }
  return { telemetry, diagnostics: telemetry && diagnostics };
}

/** the random ID: a new one every month; a permanent one only while diagnostics is on */
async function installId(diagnostics: boolean): Promise<string> {
  const { tid = null, tidMonth = '', pid = null } = await getLocal(['tid', 'tidMonth', 'pid']);
  if (diagnostics) {
    if (pid) return pid;
    const p = rid();
    await chrome.storage.local.set({ pid: p });
    return p;
  }
  if (pid) await chrome.storage.local.remove('pid');
  if (tid && tidMonth === month()) return tid;
  const t = rid();
  await chrome.storage.local.set({ tid: t, tidMonth: month() });
  return t;
}

function browserFacts() {
  const ua = navigator.userAgent;
  const m = ua.match(/(Edg|Firefox|Chrome)\/(\d+)/);
  const br = !m ? 'other' : m[1] === 'Edg' ? 'edge' : m[1].toLowerCase();
  const os = /Windows/.test(ua) ? 'windows' : /Mac OS X/.test(ua) ? 'macos' : /CrOS/.test(ua) ? 'chromeos' : /Android/.test(ua) ? 'android' : /Linux/.test(ua) ? 'linux' : 'other';
  return { br, brv: m ? +m[2] : 0, os, cores: navigator.hardwareConcurrency || 0, ext: chrome.runtime.getManifest().version, lang: (navigator.language || '').slice(0, 5) };
}

// What each event type may carry. The page script runs in the page's world, so a page could forge messages: only
// these fields pass, numbers are clamped, words come from fixed lists (no free text except an error's message).
type Field = (x: unknown) => unknown;
const num = (lo: number, hi: number): Field => (x) => (Number.isFinite(+(x as number)) ? Math.min(hi, Math.max(lo, Math.round(+(x as number)))) : undefined);
const oneOf = (...xs: string[]): Field => (x) => (xs.includes(x as string) ? x : undefined);
const MODELS = oneOf('deepfilternet3', 'rnnoise', 'gtcrn', 'custom');
const MODES = oneOf('page', 'isolated');
const SCHEMA: Record<string, Record<string, Field>> = {
  call: { mode: MODES, model: MODELS, strength: num(0, 100), dur: oneOf('<1m', '<5m', '<15m', '<60m', '>=60m'),
    lat: num(0, 2000), cost: num(0, 100000), p95: num(0, 100000), load: num(0, 1000), under: num(0, 1e6), over: num(0, 1e6),
    pumpUnder: num(0, 1e6), offShare: num(0, 100), switches: num(0, 1000) },
  error: { code: oneOf('start', 'engine', 'wasm', 'worklet', 'frame', 'model', 'reload', 'other'), msg: (x) => (x === undefined ? undefined : String(x)),
    mode: (x) => (x === null ? undefined : MODES(x)), model: MODELS },
  active: {},
  settings: { key: oneOf('model', 'strength', 'enabled', 'mode', 'lang', 'diagnostics'), value: (x) => (typeof x === 'string' ? x.slice(0, 16) : typeof x === 'number' || typeof x === 'boolean' ? x : undefined) },
};
function sanitize(ev: TelemetryEvent): Record<string, unknown> | null {
  const spec = Object.hasOwn(SCHEMA, ev?.t) ? SCHEMA[ev.t] : undefined;
  if (!spec) return null;
  const out: Record<string, unknown> = { t: ev.t };
  for (const [k, f] of Object.entries(spec)) { const v = f(ev[k]); if (v !== undefined) out[k] = v; }
  return out;
}

const clamp = (s: unknown) => String(s ?? '').replace(/\s+/g, ' ').slice(0, MSG_MAX);
/** URLs in free text (error messages): reduced to their domain with diagnostics, to "<url>" without; never a path or query */
const scrub = (s: unknown, diagnostics: boolean) => clamp(s).replace(/\b[a-z][a-z0-9+.-]*:\/\/\S+/gi, (u) => {
  if (diagnostics) { try { const h = new URL(u).hostname; if (h) return h; } catch { /* not a URL */ } }
  return '<url>';
});

/**
 * Queue one event from a page or from the worker itself.
 * ev: { t: 'call' | 'error' | 'settings' | 'active', ...fields }; pageUrl: the tab's address (only its kind is used; its
 * domain only with diagnostics, never the path or query)
 */
export async function record(ev: TelemetryEvent, pageUrl: string | null | undefined, settings: Settings): Promise<boolean> {
  const sw = await switches(settings);
  if (!sw.telemetry) return false;
  const { tq = [], tday = '', tcount = 0, terr = [] } = await getLocal(['tq', 'tday', 'tcount', 'terr']);
  const today = day();
  let count = tday === today ? tcount : 0;
  let errs = tday === today ? terr : [];
  if (count >= DAILY_CAP) return false;
  if (ev.t === 'error') {
    if (errs.includes(ev.code as string)) return false;            // one report per error code per day
    errs = [...errs, ev.code as string].slice(-50);
  }
  const fields = sanitize(ev);
  if (!fields) return false;
  const id = await installId(sw.diagnostics);
  if (!(await sampledIn(id))) return false;
  const out: Record<string, unknown> = { v: 1, ts: Date.now(), id, ...browserFacts(), commit: (await buildInfo()).commit, ...fields };
  if (pageUrl) {
    out.app = appKind(pageUrl);
    if (sw.diagnostics) { try { out.domain = new URL(pageUrl).hostname.slice(0, 253); } catch { /* no address */ } }
  }
  if (out.msg !== undefined) out.msg = scrub(out.msg, sw.diagnostics);
  count++;
  const q = [...tq, out].slice(-QUEUE_MAX);
  await chrome.storage.local.set({ tq: q, tday: today, tcount: count, terr: errs });
  return true;
}

/** the receiver's share of installs that report (x-parla-sample, default all): this ID's place in [0, 1) */
async function sampledIn(id: string): Promise<boolean> {
  const { tsample = 1 } = await getLocal('tsample');
  if (tsample >= 1) return true;
  if (tsample <= 0) return false;
  return parseInt(id.slice(0, 8), 16) / 0x100000000 < tsample;
}

/** once a day while the browser runs the extension: an `active` report (counts installs, calls or not) */
export async function active(settings: Settings): Promise<boolean> {
  const { tactive = '' } = await getLocal('tactive');
  if (tactive === day()) return false;
  await chrome.storage.local.set({ tactive: day() });
  if (await record({ t: 'active' }, null, settings)) await flush(settings);
  return true;
}

let sending: Promise<number> | null = null;
/** send what is queued (batches of 20); keeps the queue on failure, tries again later (next event or wake-up) */
export async function flush(settings: Settings): Promise<number> {
  if (sending) return sending;
  sending = (async () => {
   try {
    // tests point the endpoint elsewhere (storage.local telemetryEndpoint); users can't set it
    const { telemetryEndpoint: endpoint = ENDPOINT } = await getLocal('telemetryEndpoint');
    const sw = await switches(settings);
    if (!sw.telemetry) { await chrome.storage.local.remove('tq'); return 0; }
    const { tq = [], tnext = 0 } = await getLocal(['tq', 'tnext']);
    if (!tq.length || Date.now() < tnext) return 0;
    const batch = tq.slice(0, 20);
    try {
      const r = await fetch(endpoint, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(batch), keepalive: true, credentials: 'omit', referrerPolicy: 'no-referrer' });
      const share = parseFloat(r.headers.get('x-parla-sample') ?? '');
      if (Number.isFinite(share)) await chrome.storage.local.set({ tsample: Math.min(1, Math.max(0, share)) });
      if (r.status === 429) {
        // the receiver's daily limit: pause until it says (at most a day), keep the queue
        const after = Math.min(86400, Math.max(60, Number(r.headers.get('retry-after')) || 3600));
        await chrome.storage.local.set({ tnext: Date.now() + after * 1000 });
        return 0;
      }
      if (!r.ok && r.status !== 400 && r.status !== 413) throw new Error(`HTTP ${r.status}`);
      if (Number.isFinite(share) && share <= 0) { await chrome.storage.local.set({ tq: [], tnext: 0, tfail: 0 }); return 0; }
      const { tq: now = [] } = await getLocal('tq');
      await chrome.storage.local.set({ tq: now.slice(batch.length), tnext: 0, tfail: 0 });
      return batch.length;
    } catch {
      const { tfail = 0 } = await getLocal('tfail');
      // back off: 1 min, 5 min, 30 min, 2 h
      const wait = [60, 300, 1800, 7200][Math.min(tfail, 3)] * 1000;
      await chrome.storage.local.set({ tfail: tfail + 1, tnext: Date.now() + wait });
      return 0;
    }
   } finally { sending = null; }   // every way out, the early returns too
  })();
  return sending;
}

/** a switch changed: forget what it covered */
export async function onSwitch(settings: Settings): Promise<void> {
  const sw = await switches(settings);
  if (!sw.telemetry) await chrome.storage.local.remove(['tq', 'tid', 'tidMonth', 'pid']);
  else if (!sw.diagnostics) await chrome.storage.local.remove('pid');
}

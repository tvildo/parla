// The Parla extension (src/extension, built into build/extension by `python build.py extension`) in
// Playwright's bundled Chromium (branded Chrome ignores --load-extension since v137), with a fake microphone
// that holds only noise (lib.ts fakeMic). Test pages are served by request routing (no server): /plain.html
// (no CSP) and /strict.html with a Teams-like CSP (scripts only from self / nonce, no eval, no WebAssembly).
//   - a site switched off gets the untouched microphone (and gives the input level for the other tests)
//   - plain page: the in-page engine removes the noise, the track looks like the microphone, video passes
//   - strict page: the page can't compile WebAssembly -> isolated mode (extension-origin worker), same result
//   - live control: switching the site off mid-stream passes the input through (no restart)
//   - stopping the returned track ends the session; one microphone opened twice is processed once
//   - an update waits until the call ends; a reload mid-call never leaves the site in silence
import { test, expect, chromium, type BrowserContext, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT, fakeMic, rms } from './lib';

const tr = (code: string) => JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', `${code}.json`), 'utf8'));

const EXT = process.env.PARLA_EXTENSION_DIR || path.join(ROOT, 'build', 'extension');
const ORIGIN = 'http://localhost:9911';

const REC_JS = `
window.capture = async (seconds, constraints = { audio: true }) => {
  const s = await navigator.mediaDevices.getUserMedia(constraints);
  const t = s.getAudioTracks()[0];
  const ctx = new AudioContext({ sampleRate: 48000 });
  const src = ctx.createMediaStreamSource(new MediaStream([t]));
  const sp = ctx.createScriptProcessor(4096, 1, 1);
  const out = new Float32Array(Math.ceil(seconds * 48000));
  let n = 0;
  await new Promise((res) => {
    sp.onaudioprocess = (e) => { const x = e.inputBuffer.getChannelData(0); const k = Math.min(x.length, out.length - n); out.set(x.subarray(0, k), n); n += k; if (n >= out.length) res(); };
    src.connect(sp); sp.connect(ctx.destination);
  });
  sp.disconnect(); src.disconnect();
  window.__track = t;
  const b = new Uint8Array(out.buffer); let bin = '';
  for (let i = 0; i < b.length; i += 0x8000) bin += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return { audio: btoa(bin), label: t.label, settings: t.getSettings(), video: s.getVideoTracks().map((v) => v.label), wrapped: !!window.__viExtension };
};
window.canCompileWasm = async () => { try { await WebAssembly.compile(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])); return true; } catch { return false; } };
`;
const page = (title: string) => `<!doctype html><html><head><title>${title}</title><script src="/rec.js"></script></head><body>${title}</body></html>`;
const STRICT_CSP = "default-src 'self'; script-src 'self' 'nonce-abc'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'";
/** the browser's own echo cancellation / gain control would change levels: off, so levels compare */
const RAW = { echoCancellation: false, autoGainControl: false };

let ctx: BrowserContext;
let inputLevel = 0;

test.beforeAll(async () => {
  test.skip(!fs.existsSync(path.join(EXT, 'manifest.json')), `no built extension in ${EXT} (python build.py extension)`);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'parla-ext-'));
  ctx = await chromium.launchPersistentContext(dir, {
    channel: 'chromium',
    headless: true,
    locale: 'es-ES',                       // the popup's automatic language follows the browser's (tests below)
    permissions: ['microphone', 'camera'],
    args: [
      `--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`,
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${fakeMic()}%noloop`, '--autoplay-policy=no-user-gesture-required',
      '--enable-unsafe-extension-debugging',   // CDP Extensions.loadUnpacked: reload = an update landing mid-call
    ],
  });
  await ctx.route(`${ORIGIN}/**`, (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/rec.js') return route.fulfill({ body: REC_JS, contentType: 'text/javascript' });
    if (u.pathname === '/strict.html') return route.fulfill({ body: page('strict'), contentType: 'text/html', headers: { 'content-security-policy': STRICT_CSP } });
    return route.fulfill({ body: page('plain'), contentType: 'text/html' });
  });
});

test.afterAll(async () => { await ctx?.close(); });

async function worker() {
  let sw = ctx.serviceWorkers().at(-1);
  if (!sw) sw = await ctx.waitForEvent('serviceworker');
  return sw;
}
const settings = async (s: Record<string, unknown>) => (await worker()).evaluate(async (v) => { await chrome.storage.sync.clear(); await chrome.storage.sync.set(v); }, s);
const status = async () => (await worker()).evaluate(() => [...(globalThis as any).viStatus.values()]);

async function capture(p: Page, seconds: number, constraints: unknown = { audio: true }) {
  const errors: string[] = [];
  p.on('pageerror', (e) => errors.push(e.message));
  const r = await p.evaluate(([s, c]) => (window as any).capture(s, c), [seconds, constraints] as const);
  const audio = new Float32Array(Uint8Array.from(Buffer.from(r.audio, 'base64')).buffer);
  return { ...r, audio, errors };
}

test.describe.configure({ mode: 'serial' });

test('a site switched off gets the untouched microphone', async () => {
  test.setTimeout(60_000);
  await settings({ sites: { [ORIGIN]: 'off' } });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  const r = await capture(p, 4, { audio: RAW });
  inputLevel = rms(r.audio, 1, 4);
  expect(r.errors).toEqual([]);
  expect((await status()).some((s: any) => s.origin === ORIGIN && s.active > 0)).toBe(false);
  expect(inputLevel, 'the fake microphone is audible').toBeGreaterThan(0.02);
  await p.close();
});

test('plain page: in-page engine removes the noise, looks like the microphone, video passes through', async () => {
  await settings({ model: 'deepfilternet3', strength: 100 });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  expect(await p.evaluate(() => (window as any).canCompileWasm())).toBe(true);
  const r = await capture(p, 6, { audio: RAW, video: true });
  const st = (await status()).find((s: any) => s.origin === ORIGIN);
  const out = rms(r.audio, 2, 6);
  console.log(JSON.stringify({ mode: st?.mode, model: st?.model, latencyMs: st?.latencyMs, input: +inputLevel.toFixed(4), output: +out.toFixed(4), label: r.label, video: r.video }));
  expect(r.wrapped).toBe(true);
  expect(st?.mode).toBe('page');
  expect(st?.error ?? null).toBeNull();
  expect(out, 'noise removed').toBeLessThan(inputLevel / 5);
  expect(r.label).not.toMatch(/MediaStreamAudioDestination/i);
  expect(r.settings.deviceId).toBeTruthy();
  expect(r.video.length).toBe(1);
  expect(r.errors).toEqual([]);
  await p.close();
});

test('strict CSP page (no WebAssembly): isolated mode, same result', async () => {
  await settings({ model: 'deepfilternet3', strength: 100 });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/strict.html`);
  expect(await p.evaluate(() => (window as any).canCompileWasm()), 'the test page really forbids WebAssembly').toBe(false);
  const r = await capture(p, 6, { audio: RAW });
  const st = (await status()).find((s: any) => s.origin === ORIGIN && s.mode);
  const out = rms(r.audio, 2, 6);
  console.log(JSON.stringify({ mode: st?.mode, latencyMs: st?.latencyMs, pumpUnderruns: st?.pumpUnderruns ?? 0, output: +out.toFixed(4) }));
  expect(st?.mode).toBe('isolated');
  expect(out, 'noise removed').toBeLessThan(inputLevel / 5);
  expect(st?.pumpUnderruns ?? 0, 'audio gaps between page and worker').toBeLessThan(10);
  await p.close();
});

test('popup meter: shows how much quieter the noise is, in both engine modes; strength 0 shows no reduction', async () => {
  test.setTimeout(90_000);
  const id = new URL((await worker()).url()).host;
  const es = tr('es');
  const quieter = new RegExp(`^${es['ext.meter.quieter'].replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace('{n}', '([\\d.]+)')}$`);
  const meterN = async (pop: Page) => {
    const m = quieter.exec((await pop.locator('#meterV').textContent()) ?? '');
    return m ? +m[1] : 0;
  };
  for (const [file, mode] of [['plain.html', 'page'], ['strict.html', 'isolated']] as const) {
    await settings({ model: 'deepfilternet3', strength: 100 });
    const p = await ctx.newPage();
    await p.goto(`${ORIGIN}/${file}`);
    const rec = p.evaluate(() => (window as any).capture(9, { audio: { echoCancellation: false, autoGainControl: false } }));
    const tabId = await (await worker()).evaluate(async (u) => (await chrome.tabs.query({ url: `${u}/*` }))[0]?.id, `${ORIGIN}/${file}`);
    const pop = await ctx.newPage();
    await pop.goto(`chrome-extension://${id}/popup.html?tab=${tabId}`);
    await expect(pop.locator('#meter')).toBeVisible({ timeout: 15_000 });
    // the fake microphone is noise only: everything removed is noise; DFN3 takes it down by far more than 10 dB
    await expect.poll(() => meterN(pop), { timeout: 15_000, message: `${mode}: the meter's number` }).toBeGreaterThan(3);
    const bars = await pop.evaluate(() => ({ mic: parseFloat(document.getElementById('barIn')!.style.width), call: parseFloat(document.getElementById('barOut')!.style.width) }));
    console.log(JSON.stringify({ mode, meter: await pop.locator('#meterV').textContent(), title: await pop.locator('#meterV').getAttribute('title'), bars }));
    expect(bars.mic, 'the microphone bar shows the noise').toBeGreaterThan(30);
    expect(bars.call, 'what the call hears is far below it').toBeLessThan(bars.mic / 2);
    if (mode === 'page') {
      const { default: AxeBuilder } = await import('@axe-core/playwright');
      const r = await new AxeBuilder({ page: pop }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`), 'popup with the meter').toEqual([]);
      // strength 0 = the plain microphone: nothing removed, so no number
      await settings({ model: 'deepfilternet3', strength: 0 });
      await expect.poll(async () => (await pop.locator('#meterV').textContent()) ?? '', { timeout: 15_000 }).toBe('—');
    }
    await pop.close();
    await rec;
    await p.close();
  }
  await settings({});
});

test('live off mid-stream passes the input through', async () => {
  await settings({ model: 'gtcrn', strength: 100 });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  const rec = capture(p, 9, { audio: RAW });
  await expect.poll(async () => (await status()).some((s: any) => s.origin === ORIGIN && s.active > 0), { timeout: 15_000 }).toBe(true);
  await p.waitForTimeout(3000);
  await (await worker()).evaluate((o) => chrome.storage.sync.set({ sites: { [o]: 'off' } }), ORIGIN);
  const r = await rec;
  const on = rms(r.audio, 1.5, 3), off = rms(r.audio, 6.5, 9);
  console.log(JSON.stringify({ on: +on.toFixed(4), off: +off.toFixed(4), input: +inputLevel.toFixed(4) }));
  expect(on).toBeLessThan(inputLevel / 4);
  expect(off).toBeGreaterThan(inputLevel * 0.6);
  await p.close();
});

test('stopping the returned track ends the session', async () => {
  test.setTimeout(60_000);
  await settings({ model: 'gtcrn' });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  await capture(p, 1);
  await expect.poll(async () => (await status()).find((s: any) => s.origin === ORIGIN && s.active > 0)?.active ?? 0).toBe(1);
  await p.evaluate(() => (window as any).__track.stop());
  await expect.poll(async () => Math.max(0, ...(await status()).filter((s: any) => s.origin === ORIGIN).map((s: any) => s.active))).toBe(0);
  await p.close();
});

test('a clone of the cleaned track keeps the session after the original is stopped (call libraries do that)', async () => {
  test.setTimeout(60_000);
  await settings({ model: 'gtcrn', strength: 0 });   // strength 0: the input passes through, so a live clone is audible
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  await capture(p, 1, { audio: RAW });
  const level = await p.evaluate(async () => {
    const w = window as any;
    const clone: MediaStreamTrack = w.__track.clone();
    w.__track.stop();                          // the original goes; the clone must keep the engine
    w.__clone = clone;
    const ctx = new AudioContext();
    const an = ctx.createAnalyser();
    ctx.createMediaStreamSource(new MediaStream([clone])).connect(an);
    const b = new Float32Array(2048); let e = 0, n = 0;
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 50));
      if (i < 10) continue;
      an.getFloatTimeDomainData(b); for (const x of b) { e += x * x; n++; }
    }
    ctx.close();
    return Math.sqrt(e / n);
  });
  expect(level, 'the clone still carries audio').toBeGreaterThan(inputLevel / 4);
  expect((await status()).find((s: any) => s.origin === ORIGIN)?.active).toBe(1);
  await p.evaluate(() => (window as any).__clone.stop());   // the last track: the session ends
  await expect.poll(async () => Math.max(0, ...(await status()).filter((s: any) => s.origin === ORIGIN).map((s: any) => s.active))).toBe(0);
  await p.close();
  await settings({});
});

/** longest run of exact zeros (silence the site would hear), in ms */
function longestSilenceMs(x: Float32Array, from: number) {
  let best = 0, run = 0;
  for (let i = from; i < x.length; i++) { if (x[i] === 0) { run++; if (run > best) best = run; } else run = 0; }
  return (best * 1000) / 48000;
}
/** what an update does to open tabs: the extension is replaced (same id); waits for the new service worker */
async function reloadExtension(old: unknown) {
  const cdp = await ctx.browser()!.newBrowserCDPSession();
  await cdp.send('Extensions.loadUnpacked' as any, { path: EXT });
  await expect.poll(() => ctx.serviceWorkers().some((w) => w !== old), { timeout: 15_000 }).toBe(true);
}

test('the same microphone opened twice in a page (Meet: preview + call) is processed once', async () => {
  test.setTimeout(60_000);
  await settings({ model: 'gtcrn' });
  const p = await ctx.newPage();
  const logs: string[] = [];
  p.on('console', (m) => { if (m.text().startsWith('[vi]')) logs.push(m.text()); });
  await p.goto(`${ORIGIN}/plain.html`);
  const r = await p.evaluate(async () => {
    const a = await navigator.mediaDevices.getUserMedia({ audio: true });
    const b = await navigator.mediaDevices.getUserMedia({ audio: true });
    (window as any).__a = a.getAudioTracks()[0]; (window as any).__b = b.getAudioTracks()[0];
    return { a: a.getAudioTracks()[0].readyState, b: b.getAudioTracks()[0].readyState, same: a.getAudioTracks()[0] === b.getAudioTracks()[0] };
  });
  expect(r).toEqual({ a: 'live', b: 'live', same: false });
  expect(logs.some((l) => /sharing the running session/.test(l))).toBe(true);
  await expect.poll(async () => (await status()).find((s: any) => s.origin === ORIGIN && s.active > 0)?.active ?? 0).toBe(1);
  // the preview stops: the call's track keeps the session; both stopped: the session ends
  await p.evaluate(() => (window as any).__a.stop());
  await p.waitForTimeout(800);
  expect(await p.evaluate(() => (window as any).__b.readyState)).toBe('live');
  expect((await status()).find((s: any) => s.origin === ORIGIN)?.active).toBe(1);
  await p.evaluate(() => (window as any).__b.stop());
  await expect.poll(async () => Math.max(0, ...(await status()).filter((s: any) => s.origin === ORIGIN).map((s: any) => s.active))).toBe(0);
  await p.close();
});

test('an update waits until the call ends, then applies', async () => {
  test.setTimeout(90_000);
  await settings({ model: 'gtcrn' });
  const p = await ctx.newPage();
  await p.goto(`${ORIGIN}/plain.html`);
  await capture(p, 1);
  const sw = await worker();
  await expect.poll(() => sw.evaluate(() => (globalThis as any).viUpdate.sessions()), { timeout: 10_000 }).toBe(1);
  await sw.evaluate(() => { const u = (globalThis as any).viUpdate; u.reload = () => { (globalThis as any).reloaded = true; }; return u.onUpdateAvailable({ version: '9.9.9' }); });
  await p.waitForTimeout(4000);
  // still the same worker, the update is parked
  expect(await sw.evaluate(async () => (await chrome.storage.session.get('pendingUpdate')).pendingUpdate)).toBe('9.9.9');
  expect(await sw.evaluate(() => !!(globalThis as any).reloaded)).toBe(false);
  await p.evaluate(() => (window as any).__track.stop());   // call over -> the update applies (reload)
  await expect.poll(() => sw.evaluate(() => !!(globalThis as any).reloaded), { timeout: 10_000 }).toBe(true);
  await p.close();
});

test('extension reload mid-call (isolated mode): no silence, the engine comes back', async () => {
  await settings({ model: 'gtcrn', strength: 100 });
  const p = await ctx.newPage();
  const logs: string[] = [];
  p.on('console', (m) => { if (m.text().startsWith('[vi]')) logs.push(m.text()); });
  await p.goto(`${ORIGIN}/strict.html`);
  // the fake microphone clip lasts 12 s from the device opening (then digital silence): stay inside it
  const rec = capture(p, 9.5, { audio: RAW });
  await expect.poll(async () => (await status()).some((s: any) => s.origin === ORIGIN && s.mode === 'isolated' && s.active > 0), { timeout: 20_000 }).toBe(true);
  await p.waitForTimeout(1500);
  const sw = await worker();
  await reloadExtension(sw);
  await expect.poll(async () => (await status()).some((s: any) => s.origin === ORIGIN && s.mode === 'isolated' && s.active > 0), { timeout: 30_000 }).toBe(true);
  const r = await rec;
  const silence = longestSilenceMs(r.audio, 48000);
  console.log(JSON.stringify({ silenceMs: +silence.toFixed(1), logs }));
  expect(logs.some((l) => /extension reloaded/.test(l))).toBe(true);
  expect(logs.some((l) => /on again/.test(l))).toBe(true);
  expect(silence, 'the site never hears a gap longer than the pump fallback').toBeLessThan(60);
  await p.close();
});

test('popup and options follow the browser language; the Language setting overrides it (RTL too)', async () => {
  const sw = await worker();
  const id = new URL(sw.url()).host;
  await settings({});
  const p = await ctx.newPage();
  await p.goto(`chrome-extension://${id}/popup.html`);
  await expect(p.locator('html')).toHaveAttribute('lang', 'es');
  await expect(p.getByText(tr('es')['ext.siteOn'])).toBeVisible();
  await settings({ lang: 'ka' });
  await p.reload();
  await expect(p.locator('html')).toHaveAttribute('lang', 'ka');
  await expect(p.getByText(tr('ka')['ext.siteOn'])).toBeVisible();
  await p.goto(`chrome-extension://${id}/options.html`);
  await p.getByRole('tab', { name: tr('ka')['ext.language'] }).click();
  await expect(p.locator('#lang')).toHaveValue('ka');
  await p.locator('#lang').selectOption('ar');
  await expect(p.locator('html')).toHaveAttribute('dir', 'rtl');
  await expect(p.getByRole('tab', { selected: true })).toHaveText(tr('ar')['ext.language']);
  await expect(p.getByRole('heading', { level: 2 }).first()).toHaveText(tr('ar')['ext.language']);
  // tabs: arrow keys move between them (reversed in right-to-left), the hash keeps the tab across a reload
  await p.getByRole('tab', { selected: true }).press('Home');
  await expect(p.getByRole('tab', { selected: true })).toHaveText(tr('ar')['ext.settings']);
  await p.getByRole('tab', { selected: true }).press('ArrowLeft');
  await expect(p.getByRole('tab', { selected: true })).toHaveText(tr('ar')['ext.language']);
  await p.reload();
  await expect(p.getByRole('tabpanel')).toHaveAttribute('id', 'panel-language');
  await p.close();
  await settings({});
});

test('options: "on for all sites by default" (moved from the popup) shows and changes the setting', async () => {
  const id = new URL((await worker()).url()).host;
  await settings({ lang: 'en' });
  const p = await ctx.newPage();
  await p.goto(`chrome-extension://${id}/popup.html`);
  await expect(p.locator('#siteOn')).toBeAttached();
  await expect(p.locator('#allOn')).toHaveCount(0);
  await p.goto(`chrome-extension://${id}/options.html`);
  // what this is: the version and the commit it was built from, also in the manifest (chrome://extensions)
  await expect(p.locator('#version')).toHaveText(/^Parla v\d+\.\d+\.\d+ · ([0-9a-f]{12}(-dirty)?|unknown)$/);
  const man = await p.evaluate(() => chrome.runtime.getManifest() as { version: string; version_name?: string });
  expect(man.version_name).toBe(`${man.version} (${(await p.locator('#version').textContent())!.split(' · ')[1]})`);
  const sw = p.getByRole('switch', { name: tr('en')['ext.allOn'] });
  await expect(sw).toBeChecked();
  await sw.uncheck({ force: true });
  await expect.poll(() => (worker().then((w) => w.evaluate(async () => (await chrome.storage.sync.get('enabled')).enabled)))).toBe(false);
  await p.reload();
  await expect(sw).not.toBeChecked();
  await p.close();
  await settings({});
});

test('privacy: plain switches; diagnostics needs reports on, turns on and off directly, and goes off with reports', async () => {
  const id = new URL((await worker()).url()).host;
  const sync = () => worker().then((w) => w.evaluate(async () => {
    const s = await chrome.storage.sync.get(['telemetry', 'diagnostics']);
    return { telemetry: s.telemetry === true, diagnostics: s.diagnostics === true };
  }));
  await settings({ lang: 'en' });
  const p = await ctx.newPage();
  await p.goto(`chrome-extension://${id}/options.html#privacy`);
  await expect(p.locator('html')).toHaveAttribute('data-ready', '1');
  const reports = p.getByRole('switch', { name: tr('en')['ext.tel.label'] });
  const diag = p.getByRole('switch', { name: tr('en')['ext.diag.label'] });
  await expect(p.getByRole('button')).toHaveCount(0);      // no confirmation box (the tabs are role=tab)
  await expect(diag).toBeDisabled();                        // diagnostics only on top of reports
  await reports.check({ force: true });
  await expect(diag).toBeEnabled();
  await diag.check({ force: true });
  await expect.poll(sync).toEqual({ telemetry: true, diagnostics: true });
  await diag.uncheck({ force: true });
  await expect.poll(sync).toEqual({ telemetry: true, diagnostics: false });
  await diag.check({ force: true });
  await reports.uncheck({ force: true });                   // reports off takes diagnostics with it
  await expect.poll(sync).toEqual({ telemetry: false, diagnostics: false });
  await expect(diag).not.toBeChecked();
  // the popup without a call: no empty meter
  await p.goto(`chrome-extension://${id}/popup.html`);
  await expect(p.locator('#meter')).toBeHidden();
  await p.close();
  await settings({});
});

test('popup and options pass axe (WCAG 2.2 AA), in a left-to-right and a right-to-left language', async () => {
  const { default: AxeBuilder } = await import('@axe-core/playwright');
  const id = new URL((await worker()).url()).host;
  const p = await ctx.newPage();
  for (const lang of ['en', 'ar']) {
    await settings({ lang });
    for (const page of ['popup.html', 'options.html']) {
      await p.goto(`chrome-extension://${id}/${page}`);
      await expect(p.locator('html')).toHaveAttribute('lang', lang);
      const r = await new AxeBuilder({ page: p }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.help} -> ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`), `${lang} ${page}`).toEqual([]);
    }
  }
  await p.close();
  await settings({});
});

test('reports: nothing by default, one anonymous call summary when turned on, only the site domain with diagnostics', async () => {
  const http = await import('node:http');
  const got: any[] = [];
  let share = '1';   // what the receiver answers as x-parla-sample
  const server = http.createServer((req, res) => {
    let b = ''; req.on('data', (c) => (b += c)); req.on('end', () => { try { got.push(...JSON.parse(b)); } catch { /* not ours */ } res.writeHead(204, { 'x-parla-sample': share }); res.end(); });
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
  const port = (server.address() as any).port;
  const sw = await worker();
  // earlier tests queued reports to the real receiver (unreachable here) and backed off: start clean
  await sw.evaluate(async (u) => {
    await chrome.storage.local.remove(['tq', 'tday', 'tcount', 'terr', 'tnext', 'tfail', 'tsample']);
    await chrome.storage.local.set({ telemetryEndpoint: u });
  }, `http://127.0.0.1:${port}/v1/e`);
  const call = async () => {
    const p = await ctx.newPage();
    await p.goto(`${ORIGIN}/plain.html?secret=1#x`);
    await capture(p, 2, { audio: RAW });
    await p.evaluate(() => (window as any).__track.stop());
    await p.waitForTimeout(1200);
    await p.close();
  };
  try {
    // defaults: reports off, nothing is sent
    await settings({ model: 'gtcrn' });
    await call();
    await new Promise((r) => setTimeout(r, 1500));
    expect(got).toEqual([]);
    // turned on: one anonymous call summary
    await settings({ model: 'gtcrn', telemetry: true });
    await call();
    await expect.poll(() => got.filter((e) => e.t === 'call').length, { timeout: 10_000 }).toBe(1);
    const ev = got.find((e) => e.t === 'call');
    console.log(JSON.stringify(ev));
    expect(ev).toMatchObject({ v: 1, t: 'call', app: 'local', mode: 'page', model: 'gtcrn', dur: '<1m' });
    expect(ev.ext).toMatch(/^\d+\.\d+\.\d+$/);
    expect(ev.commit, 'reports carry the commit the extension was built from').toMatch(/^([0-9a-f]{12}(-dirty)?|unknown)$/);
    expect(ev.id).toMatch(/^[0-9a-f]{24}$/);
    expect(ev.domain).toBeUndefined();
    expect(ev.lat).toBeGreaterThan(0);
    expect(ev.cost).toBeGreaterThan(0);
    expect(JSON.stringify(ev)).not.toContain('secret');
    // diagnostics: only the domain (no port, path, query or fragment), a permanent ID
    got.length = 0;
    await settings({ model: 'gtcrn', telemetry: true, diagnostics: true });
    await call();
    await expect.poll(() => got.filter((e) => e.t === 'call').length, { timeout: 10_000 }).toBe(1);
    const d = got.find((e) => e.t === 'call');
    expect(d.domain).toBe('localhost');
    expect(d.url).toBeUndefined();
    expect(JSON.stringify(d)).not.toContain('plain.html');
    expect(d.id).not.toBe(ev.id);
    // the receiver turns reports down to 0 % (no extension update): the next answer carries it, then nothing is sent
    share = '0';
    await settings({ model: 'gtcrn', telemetry: true });
    await call();                                       // this report still goes out and brings back the 0
    await expect.poll(() => sw.evaluate(async () => (await chrome.storage.local.get('tsample')).tsample), { timeout: 10_000 }).toBe(0);
    got.length = 0;
    await call();
    await new Promise((r) => setTimeout(r, 1500));
    expect(got).toEqual([]);
  } finally {
    await sw.evaluate(() => chrome.storage.local.remove(['telemetryEndpoint', 'tq', 'tday', 'tcount', 'terr', 'tnext', 'tfail', 'tsample']));
    await settings({});
    server.close();
  }
});

test('a hostile page that takes over the bridge: only engine files, no fake settings reports, no junk in the status', async () => {
  const sw = await worker();
  await sw.evaluate(async () => { await chrome.storage.local.remove(['tq', 'tday', 'tcount', 'terr', 'tnext', 'tfail', 'tsample']); });
  await settings({ model: 'gtcrn', telemetry: true });
  const p = await ctx.newPage();
  // the page script runs in the page's world: the page can grab its port to the extension
  await p.addInitScript(() => {
    const orig = MessagePort.prototype.postMessage;
    MessagePort.prototype.postMessage = function (this: MessagePort, m: any, t?: any) {
      if (m && m.type === 'config' && !(window as any).grabbed) (window as any).grabbed = this;
      return orig.call(this, m, t);
    } as any;
    (window as any).ask = (msg: any) => new Promise((res) => {
      const port = (window as any).grabbed as MessagePort, inner = port.onmessage;
      port.onmessage = (e) => { if (e.data && e.data.id === msg.id) { port.onmessage = inner; res(e.data); } else inner && inner.call(port, e); };
      orig.call(port, msg);
    });
  });
  await p.goto(`${ORIGIN}/plain.html`);
  await capture(p, 2, { audio: RAW });
  const r = await p.evaluate(async () => {
    const ask = (window as any).ask;
    const out: any = {
      other: await ask({ id: 9001, type: 'fetch', path: 'http://127.0.0.1:1/x', as: 'text' }),
      up: await ask({ id: 9002, type: 'fetch', path: 'engine/../background.js', as: 'text' }),
      own: await ask({ id: 9003, type: 'fetch', path: 'background.js', as: 'text' }),
      engine: await ask({ id: 9004, type: 'fetch', path: 'models/gtcrn/model.json', as: 'text' }),
    };
    // the call ends first (its own status updates would replace the junk below)
    (window as any).__track.stop();
    await new Promise((r) => setTimeout(r, 1500));
    const port = (window as any).grabbed as MessagePort;
    MessagePort.prototype.postMessage.call(port, { type: 'telemetry', ev: { t: 'settings', key: 'model', value: 'gtcrn' } });
    MessagePort.prototype.postMessage.call(port, { type: 'status', active: 1, error: 'x'.repeat(5000), evil: { deep: 1 }, origin: 'https://other.example' });
    return { ...out, engine: !!out.engine.result };
  });
  expect(r.other.error).toBe('not an engine file');
  expect(r.up.error).toBe('not an engine file');
  expect(r.own.error).toBe('not an engine file');
  expect(r.engine).toBe(true);
  await p.waitForTimeout(800);
  const after = await sw.evaluate(async () => ({ q: ((await chrome.storage.local.get('tq')).tq || []) as any[], st: [...(globalThis as any).viStatus.values()] }));
  expect(after.q.filter((e: any) => e.t === 'settings')).toEqual([]);
  const st = after.st.find((s: any) => s.error && s.error.startsWith('xxx'));
  expect(st.error.length).toBe(200);
  expect(st.evil).toBeUndefined();
  expect(st.origin).toBe('http://localhost:9911');   // the bridge's own view of the page, not the page's claim
  await p.close();
  await sw.evaluate(() => chrome.storage.local.remove(['tq', 'tday', 'tcount', 'terr', 'tnext', 'tfail', 'tsample']));
  await settings({});
});

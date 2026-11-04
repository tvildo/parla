// test_browser.ts — the SDK in real Chrome (system Chrome, fake microphone): load, process(), on/off,
// strength, a device change, stop; the cleaned track carries sound. Builds a throwaway page next to an SDK build.
//   node src/sdk/build.ts --out build/sdk-demo
//   node src/sdk/test_browser.ts [--dir build/sdk-demo] [--port 8797] [--headed]
//                                 [--modules <dir with node_modules/playwright-core>]   (default: tests/)
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const arg = (n: string, d: string) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
// --modules: a folder with playwright-core in its node_modules (elsewhere than this repo, e.g. a test pod)
const require = createRequire(path.join(path.resolve(arg('modules', path.join(REPO, 'tests'))), 'package.json'));
const { chromium } = require('playwright-core') as typeof import('playwright-core');
const DIR = path.resolve(REPO, arg('dir', 'build/sdk-demo'));
const PORT = +arg('port', '8797');
const ORIGIN = `http://127.0.0.1:${PORT}`;

fs.writeFileSync(path.join(DIR, 'index.html'), `<!doctype html><meta charset="utf-8"><title>parla sdk test</title>
<script type="module">
import { Parla, ParlaError } from './parla.js';
window.run = async () => {
  const log = [];
  const t0 = performance.now(), ms = () => Math.round(performance.now() - t0);
  try {
    const parla = await Parla.load({ base: './' });
    log.push(['loaded', ms(), parla.model]);
    const mic = (await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } })).getAudioTracks()[0];
    const ns = await parla.process(mic);
    log.push(['processing', ms(), ns.latencyMs]);
    let voices = 0; ns.onVoice = () => voices++;
    // the cleaned track carries sound: its level through an analyser
    const ctx = new AudioContext();
    const an = ctx.createAnalyser(); an.fftSize = 2048;
    ctx.createMediaStreamSource(new MediaStream([ns.track])).connect(an);
    // the loudest 43 ms over 1.5 s (the fake microphone's beep comes and goes), after 300 ms for the last change
    // to arrive (audio from just before it is still on its way between the two contexts)
    const level = async () => { await new Promise((r) => setTimeout(r, 300)); let max = 0; const b = new Float32Array(2048); for (let i = 0; i < 30; i++) { await new Promise((r) => setTimeout(r, 50)); an.getFloatTimeDomainData(b); max = Math.max(max, Math.sqrt(b.reduce((s, x) => s + x * x, 0) / b.length)); } return max; };
    log.push(['level on', ms(), await level()]);
    ns.setStrength(0.5);
    log.push(['level at strength 0.5', ms(), await level()]);
    ns.setStrength(1);
    ns.setEnabled(false);
    log.push(['level off', ms(), await level()]);
    const mic2 = (await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } })).getAudioTracks()[0];
    ns.setInput(mic2);
    log.push(['level after setInput (off)', ms(), await level()]);
    ns.setGain(0.1);
    log.push(['level at gain 0.1 (off)', ms(), await level()]);
    ns.setGain(1);
    ns.setEnabled(true);
    log.push(['voice events', ms(), voices]);
    ns.stop();
    log.push(['stopped', ms(), ns.track.readyState]);
    let refused = null;
    try { await Parla.load({ base: './nowhere/' }); } catch (e) { refused = e instanceof ParlaError ? e.code : String(e); }
    log.push(['missing engine', ms(), refused]);
  } catch (e) { log.push(['error', ms(), e && e.code, String(e && e.stack || e)]); }
  return log;
};
</script>`);

const TYPES: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const f = path.join(DIR, decodeURIComponent(new URL(req.url ?? '/', ORIGIN).pathname));
  if (!f.startsWith(DIR) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream',
    // the CSP an app needs: own scripts, WebAssembly, nothing else
    'content-security-policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; worker-src 'self' blob:" });
  fs.createReadStream(f).pipe(res);
}).listen(PORT, '127.0.0.1');

const browser = await chromium.launch({ channel: process.env.PARLA_CHROME_CHANNEL || 'chrome', headless: !args.includes('--headed'),
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'] });
let failed = 1;
try {
  const page = await (await browser.newContext({ permissions: ['microphone'] })).newPage();
  page.on('console', (m) => console.log(`  [page ${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`  [page error] ${e.message}`));
  await page.goto(`${ORIGIN}/index.html`);
  type Log = [string, number, unknown, ...unknown[]][];
  await page.waitForFunction(() => typeof (window as unknown as { run?: unknown }).run === 'function');
  const log: Log = await page.evaluate(() => (window as unknown as { run: () => Promise<[string, number, unknown, ...unknown[]][]> }).run());
  for (const l of log) console.log(JSON.stringify(l));
  const get = (k: string) => log.find((l) => l[0] === k);
  const n = (k: string) => get(k)?.[2] as number;
  // the fake microphone is a beep, not speech: on, the engine removes most of it; off (bypass), it is there
  const on = n('level on'), off = n('level off'), again = n('level after setInput (off)');
  const ok = get('loaded') && get('processing') && off > 1e-3 && on < off / 4 && n('level at strength 0.5') > on * 4 && n('level at strength 0.5') < off && again > 1e-3 && n('level at gain 0.1 (off)') < again / 4 && n('voice events') > 20 && get('stopped')?.[2] === 'ended' && get('missing engine')?.[2] === 'load';
  console.log(ok ? 'PASS' : 'FAIL');
  failed = ok ? 0 : 1;
} finally { await browser.close(); server.close(); }
process.exit(failed);

// The Firefox build of the extension in the system Firefox (Selenium + geckodriver, a temporary add-on), with
// Firefox's fake microphone (a steady tone: noise to the engine). Not in Docker (needs a real Firefox):
//   python build.py extension --browser firefox && cd tests && node firefox.ts [--headless] [--rate 44100]
// Checks: the add-on wraps getUserMedia, the engine starts ("[vi] on:"), and the tone is removed (output far below
// the same page without the add-on).
import http from 'node:http';
import { Builder } from 'selenium-webdriver';
import firefox from 'selenium-webdriver/firefox.js';

import path from 'node:path';
import { fileURLToPath } from 'node:url';
const EXT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'build', 'extension-firefox');
const PAGE = `<!doctype html><title>ff</title><script>
window.logs = [];
for (const k of ['log', 'info', 'warn', 'error']) { const o = console[k]; console[k] = (...a) => { logs.push(a.join(' ')); o.apply(console, a); }; }
window.rms = async (constraints, seconds) => {
  const s = await navigator.mediaDevices.getUserMedia(constraints);
  const ctx = new AudioContext();
  const an = ctx.createAnalyser(); an.fftSize = 2048;
  ctx.createMediaStreamSource(s).connect(an);
  const b = new Float32Array(2048); let e = 0, n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) {
    await new Promise((r) => setTimeout(r, 50));
    if (performance.now() - t0 < 1500) continue;            // engine start-up
    an.getFloatTimeDomainData(b); for (const x of b) { e += x * x; n++; }
  }
  s.getTracks().forEach((t) => t.stop()); ctx.close();
  return Math.sqrt(e / Math.max(1, n));
};
</script><body>ff</body>`;
const server = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html' }); r.end(PAGE); }).listen(9917, '127.0.0.1');

const opts = new firefox.Options();
// a given Firefox (the test container's release build), else the system's
if (process.env.PARLA_FIREFOX) opts.setBinary(process.env.PARLA_FIREFOX);
if (process.argv.includes('--headless')) opts.addArguments('-headless');
opts.setPreference('media.navigator.streams.fake', true);
opts.setPreference('media.navigator.permission.disabled', true);
opts.setPreference('media.autoplay.default', 0);
// --rate 44100: Firefox's audio runs at that rate (as on a Mac whose default output is a 44.1 kHz device, e.g.
// BlackHole): the extension must not force its own rate, or Firefox can't connect the microphone / the cleaned track
const rate = process.argv.includes('--rate') ? +process.argv[process.argv.indexOf('--rate') + 1] : 0;
if (rate) opts.setPreference('media.cubeb.force_sample_rate', rate);
async function baseline() {
  const d = await new Builder().forBrowser('firefox').setFirefoxOptions(opts).build();
  try {
    await d.get('http://localhost:9917/');
    return await d.executeAsyncScript('const done = arguments[arguments.length - 1]; rms(arguments[0], 4).then(done, (e) => done("ERR " + e));',
      { audio: { echoCancellation: false, autoGainControl: false, noiseSuppression: false } });
  } finally { await d.quit(); }
}
const input = await baseline();
const driver = (await new Builder().forBrowser('firefox').setFirefoxOptions(opts).build()) as firefox.Driver;
let ok = false;
try {
  await driver.installAddon(EXT, true);
  await driver.sleep(1500);
  await driver.get('http://localhost:9917/');
  const raw = { audio: { echoCancellation: false, autoGainControl: false, noiseSuppression: false } };
  const out = await driver.executeAsyncScript('const done = arguments[arguments.length - 1]; rms(arguments[0], 6).then(done, (e) => done("ERR " + e));', raw);
  const info = await driver.executeScript<{ wrapped: boolean; logs: string[]; ua: string; pageRate: number }>('return { wrapped: !!window.__viExtension, logs: window.logs, ua: navigator.userAgent, pageRate: new AudioContext().sampleRate };');
  console.log(JSON.stringify({ inputRms: input, outputRms: out, ...info }, null, 1));
  ok = info.wrapped && typeof out === 'number' && typeof input === 'number' && input > 0.01 && out < input / 10
    && info.logs.some((l) => /\[vi\] on:/.test(l));
} finally {
  await driver.quit();
  server.close();
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);

// The npm package (@tvildo/parla, build/npm) as apps use it: examples/react-vite (built with Vite: the engine files
// become assets; useParla in React StrictMode) and examples/vanilla (no bundler: the browser resolves the package's
// files from node_modules). Fake microphone = noise only (lib.ts): the cleaned track must be far quieter.
import { test, expect, type Page } from '@playwright/test';

const RMS = `async (seconds) => {
  const ctx = new AudioContext();
  const an = ctx.createAnalyser(); an.fftSize = 2048;
  ctx.createMediaStreamSource(new MediaStream([window.parla.track])).connect(an);
  const b = new Float32Array(2048); let e = 0, n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) { await new Promise((r) => setTimeout(r, 50)); an.getFloatTimeDomainData(b); for (const x of b) { e += x * x; n++; } }
  ctx.close();
  return Math.sqrt(e / n);
}`;
const rms = (p: Page, s = 2) => p.evaluate(`(${RMS})(${s})`) as Promise<number>;
const micRms = (p: Page) => p.evaluate(`(async () => {
  const t = (await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false, echoCancellation: false, autoGainControl: false } })).getAudioTracks()[0];
  const saved = window.parla; window.parla = { track: t };
  const r = await (${RMS})(2); window.parla = saved; t.stop(); return r;
})()`) as Promise<number>;

for (const [name, url] of [['react-vite', 'http://127.0.0.1:8796/'], ['vanilla', 'http://127.0.0.1:8795/']] as const) {
  test(`${name}: the package cleans the microphone`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(url);
    await page.getByRole('button', { name: 'Start microphone' }).click();
    await expect.poll(() => page.evaluate(() => !!(window as any).parla?.track && !(window as any).parla?.error && ((window as any).parla.ready ?? true)), { timeout: 30_000 }).toBe(true);
    await page.waitForTimeout(1500);                              // engine start-up
    const input = await micRms(page);
    const out = await rms(page);
    console.log(JSON.stringify({ example: name, input: +input.toFixed(4), output: +out.toFixed(4) }));
    expect(input).toBeGreaterThan(0.02);
    expect(out, 'noise removed').toBeLessThan(input / 5);
    if (name === 'react-vite') {
      await page.getByLabel('Noise suppression').uncheck();      // live off: the input passes
      await page.waitForTimeout(800);
      expect(await rms(page)).toBeGreaterThan(input * 0.5);
    }
    expect(errors).toEqual([]);
  });
}

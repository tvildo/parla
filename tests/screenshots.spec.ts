// Screenshots for the README, the docs and the website, taken from the real builds (python build.py screenshots ->
// SCREENSHOTS=1 npx playwright test --project screenshots). Writes docs/images/*.png and src/site/public/screenshots/.
//   extension-popup.png     the popup while a (generic, made-up) meeting page uses the microphone: the noise meter
//                           live (the fake microphone is background noise: its bar full, the call's nearly empty)
//   extension-settings.png  the settings page (Settings tab)
//   demo.png                the website's live demo after a recording, its noise meter live
// The version line shows only the version (the commit would make every screenshot stale at the next commit).
import { test, expect, chromium } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT, fakeMic } from './lib';

const EXT = path.join(ROOT, 'build', 'extension');
const OUT = [path.join(ROOT, 'docs', 'images'), path.join(ROOT, 'src', 'site', 'public', 'screenshots')];
const SITE = 'http://127.0.0.1:8799';
const save = (name: string, png: Buffer) => { for (const d of OUT) { fs.mkdirSync(d, { recursive: true }); fs.writeFileSync(path.join(d, name), png); } };

// a neutral meeting page: the extension wraps its microphone like any calling app's
const MEETING = `<!doctype html><html><head><title>Weekly sync</title><style>
body { margin: 0; font: 15px system-ui, sans-serif; background: #111318; color: #e8e8ee; display: grid; place-items: center; height: 100vh; }
</style></head><body><p>Weekly sync</p><script>
navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, autoGainControl: false } }).then((s) => { window.mic = s; });
</script></body></html>`;

test.describe.configure({ mode: 'serial' });

test('extension popup and settings', async () => {
  const ctx = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(os.tmpdir(), 'parla-shot-')), {
    channel: 'chromium', headless: true, colorScheme: 'dark', deviceScaleFactor: 2, locale: 'en-US',
    permissions: ['microphone'],
    args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${fakeMic()}`, '--autoplay-policy=no-user-gesture-required'],
  });
  try {
    await ctx.route('https://meet.example.org/**', (r) => r.fulfill({ body: MEETING, contentType: 'text/html' }));
    const sw = ctx.serviceWorkers()[0] || await ctx.waitForEvent('serviceworker');
    const id = new URL(sw.url()).host;
    const call = await ctx.newPage();
    await call.goto('https://meet.example.org/weekly-sync');
    await expect.poll(() => sw.evaluate(() => [...(globalThis as any).viStatus.values()].some((s: any) => s.active > 0)), { timeout: 30_000 }).toBe(true);
    const tabId = await sw.evaluate(async () => (await chrome.tabs.query({ url: 'https://meet.example.org/*' }))[0].id);

    const popup = await ctx.newPage();
    await popup.setViewportSize({ width: 364, height: 600 });
    await popup.goto(`chrome-extension://${id}/popup.html?tab=${tabId}`);
    await expect(popup.locator('#state')).toContainText('Active', { timeout: 15_000 });
    // the meter: wait for its number (measured in the noise), then let the bars settle
    await expect(popup.locator('#meterV')).toHaveText(/quieter/, { timeout: 20_000 });
    await popup.waitForTimeout(1200);
    save('extension-popup.png', await popup.locator('body').screenshot());

    const opts = await ctx.newPage();
    await opts.setViewportSize({ width: 720, height: 400 });
    await opts.goto(`chrome-extension://${id}/options.html`);
    await expect(opts.locator('html')).toHaveAttribute('data-ready', '1');
    await opts.locator('#version').evaluate((el) => { el.textContent = el.textContent!.split(' · ')[0]; });
    await opts.waitForTimeout(300);
    save('extension-settings.png', await opts.locator('body').screenshot());
  } finally {
    await ctx.close();
  }
});

test('website: the live demo', async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 820 }, deviceScaleFactor: 2, locale: 'en-US', permissions: ['microphone'] });
  const page = await ctx.newPage();
  await page.goto(`${SITE}/demo/`);
  await page.getByRole('button', { name: 'Start with my microphone' }).click();
  await expect(page.getByText(/Delay ~\d+ ms/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole('button', { name: /Record 8 seconds/ }).click();
  await expect(page.getByRole('button', { name: /Record again/ })).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('.nmeter-value')).toHaveText(/quieter/, { timeout: 20_000 });   // the meter, live on the noise
  await page.waitForTimeout(800);
  save('demo.png', await page.locator('main.demo').screenshot());
  await ctx.close();
});

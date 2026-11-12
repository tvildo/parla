// Screenshots for the README, the docs and the website, taken from the real builds (python build.py screenshots ->
// SCREENSHOTS=1 npx playwright test --project screenshots). Writes docs/images/*.png. (The website draws its own
// previews from the real popup and demo card: src/site/src/components/previews.ts.)
//   extension-popup.png     the popup while a (generic, made-up) meeting page uses the microphone: the noise meter
//                           live (the fake microphone is background noise: its bar full, the call's nearly empty)
//   extension-settings.png  the settings page (Settings tab)
//   demo.png                the website's live demo after a recording, its noise meter live
// The settings page's version line is hidden (a version would make every screenshot stale at the next release).
import { test, expect, chromium } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT, fakeMic } from './lib';

const EXT = path.join(ROOT, 'build', 'extension');
const DOCS = path.join(ROOT, 'docs', 'images');
const SITE = 'http://127.0.0.1:8799';
const save = (name: string, png: Buffer) => {
  fs.mkdirSync(DOCS, { recursive: true }); fs.writeFileSync(path.join(DOCS, name), png);
};

// a neutral meeting page: the extension wraps its microphone like any calling app's
const MEETING = `<!doctype html><html><head><title>Weekly sync</title><style>
body { margin: 0; font: 15px system-ui, sans-serif; background: #111318; color: #e8e8ee; display: grid; place-items: center; height: 100vh; }
</style></head><body><p>Weekly sync</p><script>
navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, autoGainControl: false } }).then((s) => { window.mic = s; });
</script></body></html>`;

test.describe.configure({ mode: 'serial' });

// the captures, reused for the store images at the end
const shots: Record<'popup' | 'settings' | 'demo', Buffer | null> = { popup: null, settings: null, demo: null };

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
    shots.popup = await popup.locator('body').screenshot();
    save('extension-popup.png', shots.popup);

    const opts = await ctx.newPage();
    await opts.setViewportSize({ width: 720, height: 400 });
    await opts.goto(`chrome-extension://${id}/options.html`);
    await expect(opts.locator('html')).toHaveAttribute('data-ready', '1');
    await opts.locator('#version').evaluate((el) => { (el as HTMLElement).hidden = true; });   // a version goes stale in a picture
    await opts.waitForTimeout(300);
    shots.settings = await opts.locator('body').screenshot();
    save('extension-settings.png', shots.settings);
  } finally {
    await ctx.close();
  }
});

test('website: the live demo', async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 820 }, deviceScaleFactor: 2, locale: 'en-US', permissions: ['microphone'] });
  const page = await ctx.newPage();
  await page.goto(`${SITE}/demo/`);
  // mid-recording: the meter runs only while the microphone is open (it closes after the take)
  await page.getByRole('button', { name: /Record 8 seconds/ }).click();
  await expect(page.getByText(/Delay ~\d+ ms/)).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('.nmeter-value')).toHaveText(/quieter/, { timeout: 7_000 });   // the meter, live on the noise
  await page.waitForTimeout(500);
  const demo = await page.locator('main.demo').screenshot();
  save('demo.png', demo);
  shots.demo = await page.locator('.demo-card').screenshot();
  await ctx.close();
});

// Store listing images (Chrome Web Store, Firefox Add-ons): 1280x800 JPEG (no alpha: the Chrome store refuses it),
// each real capture above in a frame with one line of text, in the site's palette. docs/images/store/.
test('store listing images', async ({ browser }) => {
  test.skip(!shots.popup || !shots.settings || !shots.demo, 'needs the captures above (run the whole file)');
  const en = JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', 'en.json'), 'utf8'));
  const frames: { name: string; title: [string, string]; sub: string; img: Buffer; width: number }[] = [
    { name: 'store-1-popup.jpg', title: [en['site.hero1'], en['site.hero2']], sub: 'AI noise suppression for Meet, Teams, Zoom and any web call. On your device: your audio never leaves it.', img: shots.popup!, width: 400 },
    { name: 'store-2-demo.jpg', title: [en['demo.h1a'], en['demo.h1b']], sub: 'Try it at parla.cc: record a few seconds, then listen with and without Parla.', img: shots.demo!, width: 640 },
    { name: 'store-3-settings.jpg', title: ['Simple settings,', 'per site'], sub: 'Pick the quality, turn it on or off for each site. Free and open source.', img: shots.settings!, width: 640 },
  ];
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const dir = path.join(DOCS, 'store');
  fs.mkdirSync(dir, { recursive: true });
  for (const f of frames) {
    await page.setContent(`<!doctype html><html><head><style>
      html, body { margin: 0; width: 1280px; height: 800px; overflow: hidden; }
      body { background: radial-gradient(1200px 700px at 80% 10%, #2a1f5c 0%, #0c0c14 60%); color: #f4f4f8;
        font: 18px system-ui, "Segoe UI", Roboto, sans-serif; display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 56px; padding: 0 80px; box-sizing: border-box; }
      h1 { font-size: 46px; line-height: 1.1; margin: 0 0 18px; }
      h1 span { background: linear-gradient(90deg, #7c7cff, #b36bff); -webkit-background-clip: text; background-clip: text; color: transparent; }
      p { color: #b4b4c8; font-size: 21px; line-height: 1.45; margin: 0; max-width: 460px; }
      .brand { display: flex; gap: 12px; align-items: center; font-weight: 700; font-size: 22px; margin-bottom: 40px; }
      .logo { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #7c7cff, #b36bff); display: grid; place-items: center; font-weight: 800; }
      img { display: block; max-height: 680px; border-radius: 16px; border: 1px solid #2a2a44; box-shadow: 0 30px 80px -30px #000; }
    </style></head><body><div><div class="brand"><span class="logo">P</span>Parla</div>
      <h1>${f.title[0]} <span>${f.title[1]}</span></h1><p>${f.sub}</p></div>
      <img style="width:${f.width}px" src="data:image/png;base64,${f.img.toString('base64')}"></body></html>`);
    await page.locator('img').evaluate((i: HTMLImageElement) => i.decode());
    fs.writeFileSync(path.join(dir, f.name), await page.screenshot({ type: 'jpeg', quality: 92 }));
  }
  await ctx.close();
});

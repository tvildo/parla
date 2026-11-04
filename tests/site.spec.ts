// The website (build/site, `python build.py site`): every language's pages load, the language menu switches,
// right-to-left languages are marked, and the live demo runs the engine on the (fake) microphone.
import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './lib';

const LANGS: { code: string; dir: string }[] = JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', 'languages.json'), 'utf8'));
const tr = (code: string) => JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', `${code}.json`), 'utf8'));
const prefix = (code: string) => (code === 'en' ? '/' : `/${code}/`);

test('every language: landing, privacy, demo pages, translated, with the right direction', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const { code, dir } of LANGS) {
    const t = tr(code);
    await page.goto(prefix(code));
    await expect(page.locator('html')).toHaveAttribute('lang', code);
    await expect(page.locator('html')).toHaveAttribute('dir', dir);
    await expect(page.getByRole('heading', { level: 1 })).toContainText(t['site.hero2']);
    await page.goto(`${prefix(code)}privacy/`);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(t['privacy.h1']);
    await page.goto(`${prefix(code)}demo/`);
    await expect(page.getByRole('button', { name: t['demo.start'] })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('language menu switches to the same page in another language', async ({ page }) => {
  await page.goto('/privacy/');
  await page.locator('.langs summary').click();
  await page.locator('.langs a[hreflang="ka"]').click();
  await expect(page).toHaveURL(/\/ka\/privacy\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(tr('ka')['privacy.h1']);
});

test('the browser language is suggested once, never forced', async ({ browser }) => {
  const ctx = await browser.newContext({ locale: 'es-ES' });
  const page = await ctx.newPage();
  await page.goto('/');
  await expect(page).toHaveURL(/\/$/);                       // no redirect
  await expect(page.locator('#suggest')).toBeVisible();
  await page.locator('#suggestNo').click();
  await page.reload();
  await expect(page.locator('#suggest')).toBeHidden();       // dismissed: not again
  await ctx.close();
});

test('live demo (Georgian): engine on the microphone, strength, quality switch, recording with progress, two players', async ({ page }) => {
  const t = tr('ka');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('/ka/demo/');
  await page.getByRole('button', { name: t['demo.start'] }).click();
  const delay = new RegExp(t['demo.delay'].replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace('\\{ms\\}', '\\d+').replace('{ms}', '\\d+'));
  await expect(page.getByText(delay)).toBeVisible({ timeout: 30_000 });
  // one noise meter: the original and "with Parla" layers in one bar, and how much quieter the noise is
  await expect(page.locator('.nmeter-track [role=meter]')).toHaveCount(2);
  const quieter = new RegExp(`^${t['demo.quieter'].replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace('{n}', '([\\d.]+)')}$`);
  await expect.poll(async () => +(quieter.exec((await page.locator('.nmeter-value').textContent()) ?? '')?.[1] ?? 0), { timeout: 15_000 }).toBeGreaterThan(3);
  console.log(JSON.stringify({ meter: await page.locator('.nmeter-value').textContent(), dB: await page.locator('.nmeter-value').getAttribute('title') }));
  const slider = page.getByRole('slider', { name: new RegExp(t['demo.ns']) });
  await expect(slider).toHaveValue('100');
  await slider.fill('60');
  await expect(slider).toHaveAttribute('aria-valuetext', '60%');
  await page.getByLabel(t['demo.quality']).selectOption('gtcrn');
  await expect(page.getByText(delay)).toBeVisible({ timeout: 30_000 });
  await page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') }).click();
  await expect(page.getByRole('progressbar')).toBeVisible();
  await expect(page.getByText(t['demo.left'].replace('{s}', '8'))).toBeVisible();
  await expect(page.getByRole('button', { name: t['demo.recordAgain'] })).toBeVisible({ timeout: 20_000 });
  // two players: with Parla first, the original second
  await expect(page.locator('.players figcaption')).toHaveText([t['demo.withParla'], t['demo.original']]);
  const durations = await page.locator('.players audio').evaluateAll((els) => Promise.all(els.map((a) => new Promise<number>((res) => {
    const el = a as HTMLAudioElement;
    const done = () => res(el.duration);
    if (el.readyState >= 1) done(); else el.addEventListener('loadedmetadata', done, { once: true });
  }))));
  expect(durations.length).toBe(2);
  await page.getByRole('button', { name: t['demo.stop'] }).click();
  await expect(page.getByRole('button', { name: t['demo.start'] })).toBeVisible();
  expect(errors.filter((e) => !/favicon/.test(e))).toEqual([]);
});

test('demo without microphone permission: only the reason and a retry', async ({ browser }) => {
  const ctx = await browser.newContext({ permissions: [] });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('denied', 'NotAllowedError'); };
  });
  const t = tr('en');
  await page.goto('/demo/');
  await page.getByRole('button', { name: t['demo.start'] }).click();
  await expect(page.getByRole('alert')).toContainText(t['demo.err.blocked']);
  await expect(page.getByRole('button', { name: t['demo.retry'] })).toBeVisible();
  await expect(page.locator('.demo-grid')).toHaveCount(0);           // nothing else
  await expect(page.getByRole('slider')).toHaveCount(0);
  await ctx.close();
});

// The website (build/site, `python build.py site`): every language's pages load, the language menu switches,
// right-to-left languages are marked, and the live demo runs the engine on the (fake) microphone.
import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './lib';

const LANGS: { code: string; dir: string }[] = JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', 'languages.json'), 'utf8'));
const tr = (code: string) => JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', `${code}.json`), 'utf8'));
const prefix = (code: string) => (code === 'en' ? '/' : `/${code}/`);

test('every language: landing, privacy, security, demo pages, translated, with the right direction', async ({ page }) => {
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
    await page.goto(`${prefix(code)}security/`);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(t['security.h1']);
    await expect(page.locator('main a[href$="/security/advisories/new"]')).toHaveText(t['security.report.link']);
    await page.goto(`${prefix(code)}demo/`);
    await expect(page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('"See it" draws the real popup and demo card, translated, no pictures', async ({ page }) => {
  for (const code of ['en', 'ka', 'ar']) {
    const t = tr(code);
    await page.goto(prefix(code));
    const see = page.locator('#see');
    await expect(see.locator('img, picture')).toHaveCount(0);
    const popup = see.locator('.preview-popup');
    await expect(popup).toHaveAttribute('aria-label', t['site.shots.popup']);
    await expect(popup.locator('.title')).toHaveText(t['ext.title']);       // inside its shadow root
    await expect(popup.locator('#siteOn')).toBeChecked();
    await expect(popup.locator('.body')).toHaveAttribute('dir', LANGS.find((l) => l.code === code)!.dir);
    const demo = see.locator('.preview-demo');
    await expect(demo.locator('h2')).toHaveText(t['demo.yourMic']);
    await expect(demo.getByRole('progressbar')).toHaveCount(1);
  }
});

test('main menu: Trust and Developers open as submenus, one at a time, Escape closes; the current page is marked', async ({ page }) => {
  const t = tr('en');
  await page.goto('/');
  const trust = page.locator('details.menu', { has: page.locator('summary', { hasText: t['site.nav.trust'] }) });
  const dev = page.locator('details.menu', { has: page.locator('summary', { hasText: t['site.nav.dev'] }) });
  await expect(trust.getByRole('link', { name: t['site.nav.security'] })).toBeHidden();
  await trust.locator('summary').click();
  await expect(trust.getByRole('link', { name: t['site.nav.privacy'] })).toBeVisible();
  await dev.locator('summary').click();                                        // opening one closes the other
  await expect(dev.getByRole('link', { name: t['site.nav.npm'] })).toBeVisible();
  await expect(trust).not.toHaveAttribute('open', '');
  await page.keyboard.press('Escape');
  await expect(dev).not.toHaveAttribute('open', '');
  await expect(dev.locator('summary')).toBeFocused();
  await trust.locator('summary').click();
  await trust.getByRole('link', { name: t['site.nav.security'] }).click();
  await expect(page).toHaveURL(/\/security\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(t['security.h1']);
  await expect(trust.locator('summary')).toHaveClass(/here/);
  await expect(trust.locator('a[aria-current="page"]')).toHaveText(t['site.nav.security']);   // in the closed submenu
});

test('phone: the main menu is a panel behind the menu button; a group opens in place, a link or Escape closes it', async ({ page }) => {
  const t = tr('en');
  await page.setViewportSize({ width: 390, height: 800 });
  await page.goto('/');
  const toggle = page.getByRole('button', { name: t['site.nav.menu'] });
  const panel = page.locator('#mainnav');
  await expect(panel).toBeHidden();
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(page.locator('details.theme summary')).toBeVisible();          // theme and language stay in the top row
  await expect(page.locator('.langs summary')).toBeVisible();
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(panel.getByRole('link', { name: t['site.nav.try'] })).toBeVisible();
  await expect(panel.locator('a.cta')).toHaveText(t['site.nav.install']);
  await panel.locator('summary', { hasText: t['site.nav.trust'] }).click();
  await expect(panel.getByRole('link', { name: t['site.nav.security'] })).toBeVisible();
  await page.keyboard.press('Escape');                                         // closes the group and the panel
  await expect(panel).toBeHidden();
  await toggle.click();
  await panel.getByRole('link', { name: t['site.nav.how'] }).click();          // same-page link: the panel closes
  await expect(panel).toBeHidden();
  await toggle.click();
  await panel.locator('a.cta').click();
  await expect(page).toHaveURL(/\/install\/$/);
});

test('theme: follows the system, a choice is remembered, the popup preview follows it', async ({ page }) => {
  const t = tr('en');
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const popupBg = () => page.locator('.preview-popup .body').evaluate((el) => getComputedStyle(el).backgroundColor);
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', /./);           // System: no choice stored
  expect(await bg()).toBe('rgb(12, 12, 20)');
  expect(await popupBg()).toBe('rgb(12, 12, 20)');
  const summary = page.locator('details.theme summary');
  await expect(summary).toHaveAttribute('aria-label', `${t['site.theme']}: ${t['site.theme.system']}`);
  await summary.click();
  await page.getByRole('button', { name: t['site.theme.light'] }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(summary).toHaveAttribute('aria-label', `${t['site.theme']}: ${t['site.theme.light']}`);
  await expect(summary).toBeFocused();
  expect(await bg()).toBe('rgb(246, 246, 251)');
  expect(await popupBg()).toBe('rgb(246, 246, 251)');                                    // the popup's own light colours
  await page.reload();                                                                     // remembered, applied before paint
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.goto('/security/');
  expect(await bg()).toBe('rgb(246, 246, 251)');
  await summary.click();
  await expect(page.getByRole('button', { name: t['site.theme.light'] })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: t['site.theme.system'] }).click();              // back to the system's (dark)
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', /./);
  expect(await bg()).toBe('rgb(12, 12, 20)');
  await page.emulateMedia({ colorScheme: 'light' });
  expect(await bg()).toBe('rgb(246, 246, 251)');
});

test('"Install Parla" (menu and home page) leads to the install page: the extension, then the npm package, in every language', async ({ page }) => {
  for (const { code } of LANGS) {
    const t = tr(code);
    await page.goto(prefix(code));
    await expect(page.locator('.nav-links a.cta')).toHaveText(t['site.nav.install']);
    await expect(page.locator('#install')).toHaveCount(0);                     // no install section on the home page
    await page.locator('.hero .btn.primary').click();
    await expect(page).toHaveURL(new RegExp(`${prefix(code)}install/$`));
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(t['install.title']);
    await expect(page.locator('#npm h2')).toHaveText(t['install.npm.h']);
    await expect(page.locator('#npm .cmd')).toHaveText('npm install @tvildo/parla');
  }
});

test("install page: icons on every card, the visitor's browser marked and first (Firefox, iPhone, Edge, Android Chrome)", async ({ browser }) => {
  const t = tr('en');
  const UA: [string, string, string | null][] = [
    ['firefox', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0', 'firefox'],
    ['iphone', 'Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.0 Mobile/15E148 Safari/604.1', 'safari'],
    ['edge', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0', 'chrome'],
    ['android', 'Mozilla/5.0 (Linux; Android 16; Pixel 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Mobile Safari/537.36', null],
  ];
  for (const [name, userAgent, want] of UA) {
    const ctx = await browser.newContext({ userAgent });
    const page = await ctx.newPage();
    await page.goto('/install/');
    await expect(page.locator('.browser h2 svg.icon')).toHaveCount(3 + 1 + 1 + 2);    // chrome/brave/opera, firefox, safari, node/npm
    await expect(page.locator('.browser .btn.primary svg.icon')).toHaveCount(3);       // Add to Chrome, Add to Firefox, docs
    if (want) {
      await expect(page.locator('.browser.mine'), name).toHaveAttribute('id', want);
      await expect(page.locator('.browsers > .browser').first(), name).toHaveAttribute('id', want);
      await expect(page.locator('.browser.mine .yours')).toHaveText(t['install.yours']);
    } else {
      await expect(page.locator('.browser.mine'), name).toHaveCount(0);
      await expect(page.locator('.browsers > .browser').first()).toHaveAttribute('id', 'chrome');
    }
    await ctx.close();
  }
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

test('live demo (Georgian): one button opens the microphone, records and closes it, two players, strength, quality switch, record again', async ({ page }) => {
  const t = tr('ka');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('/ka/demo/');
  await page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') }).click();   // one button: microphone + recording
  const delay = new RegExp(t['demo.delay'].replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace('\\{ms\\}', '\\d+').replace('{ms}', '\\d+'));
  await expect(page.getByText(delay)).toBeVisible({ timeout: 30_000 });
  // one noise meter: the original and "with Parla" layers in one bar, and how much quieter the noise is
  await expect(page.locator('.nmeter-track [role=meter]')).toHaveCount(2);
  const quieter = new RegExp(`^${t['demo.quieter'].replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace('{n}', '([\\d.]+)')}$`);
  await expect.poll(async () => +(quieter.exec((await page.locator('.nmeter-value').textContent()) ?? '')?.[1] ?? 0), { timeout: 15_000 }).toBeGreaterThan(3);
  console.log(JSON.stringify({ meter: await page.locator('.nmeter-value').textContent(), dB: await page.locator('.nmeter-value').getAttribute('title') }));
  await expect(page.getByRole('progressbar')).toBeVisible();
  await expect(page.getByRole('button', { name: t['demo.recordAgain'] })).toBeVisible({ timeout: 20_000 });
  // the take is done: the microphone is closed (no delay shown, no stop button), the meter is empty
  await expect(page.getByText(delay)).toHaveCount(0);
  await expect(page.locator('.demo-one button')).toHaveCount(1);
  // two players: with Parla first, the original second
  await expect(page.locator('.players figcaption')).toHaveText([t['demo.withParla'], t['demo.original']]);
  const durations = await page.locator('.players audio').evaluateAll((els) => Promise.all(els.map((a) => new Promise<number>((res) => {
    const el = a as HTMLAudioElement;
    const done = () => res(el.duration);
    if (el.readyState >= 1) done(); else el.addEventListener('loadedmetadata', done, { once: true });
  }))));
  expect(durations.length).toBe(2);
  const slider = page.getByRole('slider', { name: new RegExp(t['demo.ns']) });
  await expect(slider).toHaveValue('100');
  await slider.fill('60');
  await expect(slider).toHaveAttribute('aria-valuetext', '60%');
  await page.getByLabel(t['demo.quality']).selectOption('gtcrn');
  await page.getByRole('button', { name: t['demo.recordAgain'] }).click();
  await expect(page.getByText(delay)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('progressbar')).toBeVisible();
  await expect(page.getByText(t['demo.left'].replace('{s}', '8'))).toBeVisible();
  await expect(page.getByRole('button', { name: t['demo.recordAgain'] })).toBeEnabled({ timeout: 20_000 });
  await expect(page.getByText(delay)).toHaveCount(0);
  expect(errors.filter((e) => !/favicon/.test(e))).toEqual([]);
});

test('demo without microphone permission: the card becomes the reason and a retry', async ({ browser }) => {
  const ctx = await browser.newContext({ permissions: [] });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('denied', 'NotAllowedError'); };
  });
  const t = tr('en');
  await page.goto('/demo/');
  await page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') }).click();
  await expect(page.getByRole('alert')).toContainText(t['demo.err.blocked']);
  await expect(page.getByRole('button', { name: t['demo.retry'] })).toBeVisible();
  await expect(page.locator('.demo-card')).toHaveCount(1);            // the card is replaced, nothing else
  await expect(page.getByRole('slider')).toHaveCount(0);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.locator('.demo .lead')).toBeVisible();
  await ctx.close();
});

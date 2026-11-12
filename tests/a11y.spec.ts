// Accessibility of the website: axe-core (WCAG 2.0/2.1/2.2 A + AA rules) on every page type, in the dark and the light
// theme, in a left-to-right, a right-to-left and a non-Latin language, the demo idle and running; and the demo by keyboard only.
import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './lib';

const tr = (code: string) => JSON.parse(fs.readFileSync(path.join(ROOT, 'src', 'i18n', `${code}.json`), 'utf8'));
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

async function axe(page: Page, what: string) {
  const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  const v = r.violations.map((x) => `${x.id} (${x.impact}): ${x.help} -> ${x.nodes.slice(0, 3).map((n) => n.target.join(' ')).join(' | ')}`);
  expect(v, `${what}: accessibility violations`).toEqual([]);
}

for (const lang of ['en', 'ka', 'ar']) {
  const p = lang === 'en' ? '/' : `/${lang}/`;
  test(`axe in the dark theme: landing, privacy, security, demo (${lang})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' });   // Playwright's default is light: that one is the test below
    await page.goto(p);
    await axe(page, `${lang} landing`);
    await page.locator('.langs summary').click();               // the open language menu too
    await axe(page, `${lang} landing, language menu open`);
    await page.locator('details.menu:not(.langs) summary').first().click();   // a submenu of the main menu
    await axe(page, `${lang} landing, main submenu open`);
    await page.goto(`${p}privacy/`);
    await axe(page, `${lang} privacy`);
    await page.goto(`${p}security/`);
    await axe(page, `${lang} security`);
    await page.goto(`${p}demo/`);
    await axe(page, `${lang} demo idle`);
    await page.getByRole('button', { name: tr(lang)['demo.record'].replace('{s}', '8') }).click();
    await expect(page.getByRole('progressbar')).toBeVisible({ timeout: 30_000 });
    await axe(page, `${lang} demo recording`);
    await expect(page.getByRole('button', { name: tr(lang)['demo.recordAgain'] })).toBeEnabled({ timeout: 20_000 });
    await axe(page, `${lang} demo recorded`);
  });
}

for (const lang of ['en', 'ar']) {
  const p = lang === 'en' ? '/' : `/${lang}/`;
  test(`axe in the light theme: landing, menus, privacy, security, demo (${lang})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto(p);
    await expect(page.locator('html')).not.toHaveAttribute('data-theme', /./);
    await axe(page, `${lang} landing, light`);
    await page.locator('details.theme summary').click();
    await axe(page, `${lang} landing, light, theme menu open`);
    await page.goto(`${p}privacy/`);
    await axe(page, `${lang} privacy, light`);
    await page.goto(`${p}security/`);
    await axe(page, `${lang} security, light`);
    await page.goto(`${p}demo/`);
    await axe(page, `${lang} demo idle, light`);
    await page.getByRole('button', { name: tr(lang)['demo.record'].replace('{s}', '8') }).click();
    await expect(page.getByRole('progressbar')).toBeVisible({ timeout: 30_000 });
    await axe(page, `${lang} demo recording, light`);
  });
}

test('axe on a phone: the menu panel open (en, ar)', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  for (const [p, lang] of [['/', 'en'], ['/ar/', 'ar']]) {
    await page.goto(p);
    await axe(page, `${lang} phone, menu closed`);
    await page.getByRole('button', { name: tr(lang)['site.nav.menu'] }).click();
    await page.locator('#mainnav details.menu summary').first().click();
    await axe(page, `${lang} phone, menu open`);
  }
});

test('demo by keyboard only: record, strength, quality, record again', async ({ page }) => {
  const t = tr('en');
  await page.goto('/demo/');
  // the skip link is the first stop and lands on the main content
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: t['site.skip'] })).toBeFocused();
  const start = page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') });
  for (let i = 0; i < 30 && !(await start.evaluate((el) => el === document.activeElement)); i++) await page.keyboard.press('Tab');
  await expect(start).toBeFocused();
  await page.keyboard.press('Enter');   // microphone + recording
  await expect(page.getByRole('status')).toContainText(t['demo.started'], { timeout: 30_000 });
  const again = page.getByRole('button', { name: t['demo.recordAgain'] });
  await expect(again).toBeEnabled({ timeout: 30_000 });
  // strength: a slider, arrow keys move it
  const slider = page.getByRole('slider', { name: new RegExp(t['demo.ns']) });
  await slider.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(slider).toHaveValue('95');
  // quality: a dropdown, arrow keys change it
  const quality = page.getByLabel(t['demo.quality']);
  await quality.focus();
  await page.keyboard.press('ArrowDown');
  await expect(quality).toHaveValue('rnnoise');
  await expect(again).toBeEnabled({ timeout: 30_000 });   // the new quality has loaded
  await again.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('progressbar')).toBeVisible();
  await expect(again).toBeEnabled({ timeout: 20_000 });
});

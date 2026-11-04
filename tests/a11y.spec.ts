// Accessibility of the website: axe-core (WCAG 2.0/2.1/2.2 A + AA rules) on every page type in a left-to-right, a
// right-to-left and a non-Latin language, the demo idle and running; and the demo by keyboard only.
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
  test(`axe: landing, privacy, demo (${lang})`, async ({ page }) => {
    await page.goto(p);
    await axe(page, `${lang} landing`);
    await page.locator('.langs summary').click();               // the open language menu too
    await axe(page, `${lang} landing, language menu open`);
    await page.goto(`${p}privacy/`);
    await axe(page, `${lang} privacy`);
    await page.goto(`${p}demo/`);
    await axe(page, `${lang} demo idle`);
    await page.getByRole('button', { name: tr(lang)['demo.start'] }).click();
    await expect(page.getByRole('button', { name: tr(lang)['demo.stop'] })).toBeVisible({ timeout: 30_000 });
    await axe(page, `${lang} demo running`);
  });
}

test('demo by keyboard only: start, strength, quality, record', async ({ page }) => {
  const t = tr('en');
  await page.goto('/demo/');
  // the skip link is the first stop and lands on the main content
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: t['site.skip'] })).toBeFocused();
  const start = page.getByRole('button', { name: t['demo.start'] });
  for (let i = 0; i < 30 && !(await start.evaluate((el) => el === document.activeElement)); i++) await page.keyboard.press('Tab');
  await expect(start).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: t['demo.stop'] })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('status')).toContainText(t['demo.started']);
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
  await expect(page.getByRole('button', { name: t['demo.stop'] })).toBeEnabled({ timeout: 30_000 });
  const rec = page.getByRole('button', { name: t['demo.record'].replace('{s}', '8') });
  await expect(rec).toBeEnabled({ timeout: 30_000 });   // the new quality has loaded
  await rec.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: t['demo.recordAgain'] })).toBeVisible({ timeout: 20_000 });
});

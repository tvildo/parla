// make_icons.ts — the extension's icons (16, 32, 48, 128 px PNG) from the website's logo, src/site/public/favicon.svg,
// rendered by Chrome (the tests' Playwright). The PNGs are committed; run this again only when the logo changes.
//   node tools/make_icons.ts
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(REPO, 'tests', 'package.json'));
const { chromium } = require('playwright-core') as typeof import('playwright-core');

const svg = fs.readFileSync(path.join(REPO, 'src', 'site', 'public', 'favicon.svg'), 'utf8');
const OUT = path.join(REPO, 'src', 'extension', 'src', 'icons');
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ channel: process.env.PARLA_CHROME_CHANNEL || 'chrome' });
try {
  for (const size of [16, 32, 48, 128]) {
    const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
    await page.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block;width:${size}px;height:${size}px}</style>${svg}`);
    await page.screenshot({ path: path.join(OUT, `icon-${size}.png`), omitBackground: true });
    await page.close();
  }
} finally { await browser.close(); }
console.log(`icons -> ${path.relative(REPO, OUT)}`);

// i18n_check.ts — every language in src/i18n/languages.json has every key of en.json, the same {placeholders},
// no empty values and no extra keys. Exit 1 on any problem. (Node built-ins only.)
//   node tools/i18n_check.ts
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src', 'i18n');
type Strings = Record<string, string>;
const read = (f: string) => JSON.parse(fs.readFileSync(path.join(DIR, f), 'utf8'));
const en: Strings = read('en.json');
const langs: { code: string }[] = read('languages.json');
const holes = (s: unknown) => [...String(s).matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');

let bad = 0;
const status: Strings = {};
for (const { code } of langs) {
  if (code === 'en') continue;
  const f = `${code}.json`;
  if (!fs.existsSync(path.join(DIR, f))) { console.log(`${code}: missing ${f}`); bad++; continue; }
  let t: Strings;
  try { t = read(f); } catch (e) { console.log(`${code}: not JSON (${(e as Error).message})`); bad++; continue; }
  status[code] = t._status || '?';
  for (const k of Object.keys(en)) {
    if (k === '_status') continue;
    if (!(k in t)) { console.log(`${code}: missing ${k}`); bad++; continue; }
    if (typeof t[k] !== 'string' || !t[k].trim()) { console.log(`${code}: empty ${k}`); bad++; continue; }
    if (holes(t[k]) !== holes(en[k])) { console.log(`${code}: ${k} placeholders {${holes(t[k])}} != {${holes(en[k])}}`); bad++; }
  }
  for (const k of Object.keys(t)) if (!(k in en)) { console.log(`${code}: extra key ${k}`); bad++; }
  if (t['ext.description'] && t['ext.description'].length > 132) { console.log(`${code}: ext.description > 132 chars`); bad++; }
  if (t['ext.name'] && t['ext.name'].length > 45) { console.log(`${code}: ext.name > 45 chars`); bad++; }
}
console.log(`i18n: ${langs.length} languages, ${Object.keys(en).length - 1} keys; status ${Object.entries(status).map(([c, s]) => `${c}:${s}`).join(' ')}`);
if (bad) { console.log(`${bad} problem(s)`); process.exit(1); }

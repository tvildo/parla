// release_age.ts — supply chain: no package version younger than 3 days enters the repository (a new malicious
// release is usually found and pulled within days; docs/releasing.md#supply-chain).
//
//   node tools/release_age.ts [--base <git ref>] [--days 3]
//
// Compares every lockfile (each npm project's package-lock.json, uv.lock) in the working tree with the same file at --base
// (default origin/master) and checks each package version that is new there:
//   - npm: its publish time from the registry (https://registry.npmjs.org/<name>, `time[version]`); local packages
//     (file:, links) are skipped
//   - Python: the upload times uv.lock records for its files (the newest one: files can be added to an old release)
// Exits 1 when one is younger than --days. A base that git can't resolve (a new branch, a shallow clone without it)
// means every entry is checked. The same rule at install time: `min-release-age` in every .npmrc (npm 11.10+),
// `exclude-newer` in pyproject.toml (uv); Dependabot waits a week (`cooldown`).
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const git = (...a: string[]) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8', maxBuffer: 256 << 20 });
const args = process.argv.slice(2);
const opt = (name: string, def: string) => (args.includes(name) ? args[args.indexOf(name) + 1] : def);
const BASE = opt('--base', 'origin/master');
const DAYS = Number(opt('--days', '3'));
if (!(DAYS > 0)) throw new Error(`--days must be a positive number, not ${opt('--days', '')}`);
const NOW = Date.now();

type Entry = { eco: 'npm' | 'pypi'; name: string; version: string; file: string; uploaded?: string };

function show(ref: string, file: string): string | null {
  try { return execFileSync('git', ['show', `${ref}:${file}`], { cwd: REPO, encoding: 'utf8', maxBuffer: 256 << 20, stdio: ['ignore', 'pipe', 'ignore'] }); } catch { return null; }
}

/** registry packages of a package-lock.json (v2/v3 `packages` map) */
function npmEntries(text: string, file: string): Entry[] {
  const lock = JSON.parse(text) as { packages?: Record<string, { version?: string; resolved?: string; link?: boolean; name?: string }> };
  const out: Entry[] = [];
  for (const [key, p] of Object.entries(lock.packages ?? {})) {
    if (!key || p.link || !p.version || !p.resolved?.startsWith('https://registry.npmjs.org/')) continue;
    const name = p.name ?? key.slice(key.lastIndexOf('node_modules/') + 'node_modules/'.length);
    out.push({ eco: 'npm', name, version: p.version, file });
  }
  return out;
}

/** registry packages of a uv.lock, with the newest upload time of their files */
function uvEntries(text: string, file: string): Entry[] {
  const out: Entry[] = [];
  for (const block of text.split(/^\[\[package\]\]$/m).slice(1)) {
    const name = /^name = "([^"]+)"/m.exec(block)?.[1];
    const version = /^version = "([^"]+)"/m.exec(block)?.[1];
    const times = [...block.matchAll(/upload-time = "([^"]+)"/g)].map((m) => m[1]).sort();
    if (!name || !version || !/^source = \{ registry = /m.test(block)) continue;   // the project itself, local paths
    out.push({ eco: 'pypi', name, version, file, uploaded: times.at(-1) });
  }
  return out;
}

const entries = (text: string, file: string) => (file.endsWith('uv.lock') ? uvEntries(text, file) : npmEntries(text, file));
const id = (e: Entry) => `${e.eco}:${e.name}@${e.version}`;

let base: string | null = null;
try { base = git('rev-parse', '--verify', '--quiet', `${BASE}^{commit}`).trim(); } catch { /* not available */ }
if (!base) console.warn(`release_age: base ${BASE} not found, checking every locked version`);

const files = git('ls-files', '--cached', '--others', '--exclude-standard', '--', '*package-lock.json', 'uv.lock')
  .split('\n').filter(Boolean);
const fresh = new Map<string, Entry>();
for (const file of files) {
  const old = base ? show(base, file) : null;
  const before = new Set(old ? entries(old, file).map(id) : []);
  for (const e of entries(fs.readFileSync(path.join(REPO, file), 'utf8'), file)) if (!before.has(id(e)) && !fresh.has(id(e))) fresh.set(id(e), e);
}

async function npmTime(name: string, version: string): Promise<string | undefined> {
  const r = await fetch(`https://registry.npmjs.org/${name.replace('/', '%2f')}`, { headers: { accept: 'application/json' } });
  if (!r.ok) throw new Error(`registry ${name}: HTTP ${r.status}`);
  return ((await r.json()) as { time?: Record<string, string> }).time?.[version];
}

const todo = [...fresh.values()];
const young: string[] = [];
const workers = Array.from({ length: 8 }, async () => {
  for (let e = todo.shift(); e; e = todo.shift()) {
    const t = e.eco === 'npm' ? await npmTime(e.name, e.version) : e.uploaded;
    if (!t) { young.push(`${id(e)} (${e.file}): no publish time found`); continue; }
    const days = (NOW - Date.parse(t)) / 86_400_000;
    if (!(days >= DAYS)) young.push(`${id(e)} (${e.file}): published ${t}, ${days.toFixed(1)} days ago`);
  }
});
await Promise.all(workers);

console.log(`release_age: ${fresh.size} new locked versions since ${base ? base.slice(0, 12) : 'the beginning'}, minimum age ${DAYS} days`);
if (young.length) {
  console.error(`release_age: younger than ${DAYS} days (wait, or pick an older version):\n  ${young.sort().join('\n  ')}`);
  process.exit(1);
}

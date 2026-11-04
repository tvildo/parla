// next_version.ts — the next release version from the git history (semantic versioning, conventional commits).
//
//   node tools/next_version.ts [--bump auto|patch|minor|major]
//
// Prints `version=X.Y.Z`, `tag=vX.Y.Z`, `bump=...` and `previous=...` lines (the release workflow appends them to
// $GITHUB_OUTPUT), or `version=` when nothing was committed since the last tag.
//   - the last release = the highest vX.Y.Z tag reachable from HEAD; none yet = the first release, the version in
//     src/extension/src/manifest.json as it is
//   - the bump, from the commit messages since that tag: `type!:` or a `BREAKING CHANGE` line -> major, `feat:` ->
//     minor, anything else -> patch. `--bump` overrides it (manual runs).
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const git = (...a: string[]) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8' }).trim();
const args = process.argv.slice(2);
const forced = args.includes('--bump') ? args[args.indexOf('--bump') + 1] : 'auto';
if (!['auto', 'patch', 'minor', 'major'].includes(forced)) throw new Error(`--bump auto|patch|minor|major, not ${forced}`);

type V = [number, number, number];
const parse = (s: string): V | null => { const m = /^v?(\d+)\.(\d+)\.(\d+)$/.exec(s); return m ? [+m[1], +m[2], +m[3]] : null; };
const cmp = (a: V, b: V) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
const show = (v: V) => v.join('.');

const tags = git('tag', '--merged', 'HEAD', '--list', 'v*').split('\n').map((t) => ({ t, v: parse(t) }))
  .filter((x): x is { t: string; v: V } => !!x.v).sort((a, b) => cmp(b.v, a.v));
const last = tags[0];

let version: V, bump: string;
if (!last) {
  const m = JSON.parse(fs.readFileSync(path.join(REPO, 'src', 'extension', 'src', 'manifest.json'), 'utf8'));
  const v = parse(m.version);
  if (!v) throw new Error(`manifest version ${m.version} is not X.Y.Z`);
  version = v;
  bump = 'first';
} else {
  const log = git('log', `${last.t}..HEAD`, '--format=%B%x00');
  const messages = log.split('\0').map((s) => s.trim()).filter(Boolean);
  if (!messages.length) { console.log('version=\ntag=\nbump=none\nprevious=' + last.t); process.exit(0); }
  bump = forced !== 'auto' ? forced
    : messages.some((m) => /^\w+(\([^)]*\))?!:/.test(m) || /^BREAKING[ -]CHANGE:/m.test(m)) ? 'major'
    : messages.some((m) => /^feat(\([^)]*\))?:/m.test(m)) ? 'minor'
    : 'patch';
  const [a, b, c] = last.v;
  version = bump === 'major' ? [a + 1, 0, 0] : bump === 'minor' ? [a, b + 1, 0] : [a, b, c + 1];
}
console.log(`version=${show(version)}\ntag=v${show(version)}\nbump=${bump}\nprevious=${last?.t ?? ''}`);

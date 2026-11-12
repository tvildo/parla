// next_version.ts — the next release version of the extension, the SDK and the npm package, from the git history
// (semantic versioning, conventional commits). The website has no version: it is deployed on every merge.
//
//   node tools/next_version.ts [--bump auto|patch|minor|major]
//
// Prints `version=X.Y.Z`, `tag=vX.Y.Z`, `bump=...`, `previous=...` and `current=X.Y.Z` lines (the release workflow
// appends them to $GITHUB_OUTPUT); `version=` when nothing that ships in the extension / SDK / npm package changed
// since the last tag. `current` = the release the website offers for download: the new version, else the last one.
// The commits that count are listed on stderr.
//   - the last release = the highest vX.Y.Z tag reachable from HEAD; none yet = the first release, the version in
//     src/extension/src/manifest.json as it is
//   - a commit counts when it changes a file outside the website-only paths (SITE_ONLY), or an `ext.*` text in
//     src/i18n (the extension's texts; the other keys are the website's). Merge commits don't count: their changes
//     are in the commits they merge; nor do `docs:`, `ci:`, `test:` and `style:` commits (nothing that runs changes).
//   - the bump, from the messages of the commits that count: `type!:` or a `BREAKING CHANGE` line -> major,
//     `feat:` -> minor, anything else -> patch. `--bump` overrides it and forces a release (manual runs).
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const git = (...a: string[]) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8', maxBuffer: 64 << 20 }).trim();
const args = process.argv.slice(2);
const forced = args.includes('--bump') ? args[args.indexOf('--bump') + 1] : 'auto';
if (!['auto', 'patch', 'minor', 'major'].includes(forced)) throw new Error(`--bump auto|patch|minor|major, not ${forced}`);

// Paths whose changes never reach the extension, the SDK or the npm package. Anything not listed counts (a new
// file nobody thought about releases rather than being left out).
const SITE_ONLY = [
  /^src\/site\//,          // the website
  /^src\/telemetry\//,     // the report receiver (deployed on every merge, like the website)
  /^Dockerfile$/, /^\.dockerignore$/,   // the website image
  /^docs\//, /\.md$/,      // documentation (the npm page's README follows with the next release)
  /^tests\//, /^examples\//,
  /^\.github\//, /^\.gitignore$/,
  /^tools\/next_version\.ts$/, /^tools\/release\//, /^tools\/release_age\.ts$/,
  // the Python model tools and their locked environment: no build runs them (a regenerated model changes models/)
  /^tools\/[^/]+\.py$/, /^pyproject\.toml$/, /^uv\.lock$/, /^\.python-version$/,
];
const siteOnly = (f: string) => SITE_ONLY.some((r) => r.test(f));

// src/i18n/<lang>.json holds the website's texts and the extension's (`ext.*`): only the latter count
const extTexts = (rev: string, file: string): string => {
  let all: Record<string, string> = {};
  try { all = JSON.parse(git('show', `${rev}:${file}`)); } catch { /* not there in that commit */ }
  return JSON.stringify(Object.entries(all).filter(([k]) => k.startsWith('ext.')).sort());
};
const NO_RELEASE = /^(docs|ci|test|style)(\([^)]*\))?:/;
const counts = (sha: string): boolean => !NO_RELEASE.test(git('log', '-1', '--format=%s', sha)) &&
  git('diff-tree', '--no-commit-id', '--name-only', '-r', sha).split('\n').filter(Boolean).some((f) =>
    /^src\/i18n\/[^/]+\.json$/.test(f) ? extTexts(`${sha}^`, f) !== extTexts(sha, f) : !siteOnly(f));

type V = [number, number, number];
const parse = (s: string): V | null => { const m = /^v?(\d+)\.(\d+)\.(\d+)$/.exec(s); return m ? [+m[1], +m[2], +m[3]] : null; };
const cmp = (a: V, b: V) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
const show = (v: V) => v.join('.');
const out = (version: string, bump: string, previous: string, current: string) =>
  console.log(`version=${version}\ntag=${version && `v${version}`}\nbump=${bump}\nprevious=${previous}\ncurrent=${current}`);

const tags = git('tag', '--merged', 'HEAD', '--list', 'v*').split('\n').map((t) => ({ t, v: parse(t) }))
  .filter((x): x is { t: string; v: V } => !!x.v).sort((a, b) => cmp(b.v, a.v));
const last = tags[0];

if (!last) {
  const m = JSON.parse(fs.readFileSync(path.join(REPO, 'src', 'extension', 'src', 'manifest.json'), 'utf8'));
  const v = parse(m.version);
  if (!v) throw new Error(`manifest version ${m.version} is not X.Y.Z`);
  out(show(v), 'first', '', show(v));
  process.exit(0);
}

const shas = git('log', `${last.t}..HEAD`, '--no-merges', '--format=%H').split('\n').filter(Boolean);
const shipped = shas.filter(counts);
for (const sha of shas) console.error(`${shipped.includes(sha) ? 'release' : 'site   '} ${git('log', '-1', '--format=%h %s', sha)}`);
if (!shipped.length && (forced === 'auto' || !shas.length)) {
  out('', 'none', last.t, show(last.v));   // nothing for the extension / SDK / npm: the website alone
  process.exit(0);
}
const messages = shipped.map((sha) => git('log', '-1', '--format=%B', sha));
const bump = forced !== 'auto' ? forced
  : messages.some((m) => /^\w+(\([^)]*\))?!:/.test(m) || /^BREAKING[ -]CHANGE:/m.test(m)) ? 'major'
  : messages.some((m) => /^feat(\([^)]*\))?:/m.test(m)) ? 'minor'
  : 'patch';
const [a, b, c] = last.v;
const next = show(bump === 'major' ? [a + 1, 0, 0] : bump === 'minor' ? [a, b + 1, 0] : [a, b, c + 1]);
out(next, bump, last.t, next);

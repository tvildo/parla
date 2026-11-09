// build.ts — assemble the unpacked browser extension (Manifest V3) into build/extension/ (Node built-ins only).
//
//   node src/extension/build.ts [--out build/extension] [--wasm build/wasm] [--models deepfilternet3,gtcrn,rnnoise]
//                            [--zip]   (also writes build/extension.zip for the stores)
//                            [--browser firefox]   (-> build/extension-firefox: Firefox 140+ manifest)
//                            [--browser safari]    (-> build/extension-safari: the web extension inside the Safari app,
//                                                   src/extension/safari; see its README)
//                            [--version X.Y.Z]   (the release version in the built manifest; CI passes the next tag)
//
// Contents: src/extension/src/* (pages, styles, manifest; the scripts as emitted JavaScript from build/js/, so run
// `python build.py ts` first: build.py does) + engine/ (worklet bundle, engine wasm builds) + models/<name>/
// (packages, shipped inside the extension: works without any server). The worklet and its imports (vi_engine,
// resampler, otel_export) are bundled into ONE classic script, engine/worklet.bundle.js, because it is loaded three
// ways: as an AudioWorklet module in the page (mode "page"), from a blob when the page refuses the extension
// URL, and with importScripts() in the extension-origin worker (mode "isolated", engine_worker.ts).
// Paths are relative to the repository root.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';
import { bundle as bundleWorklet, classic, emitted, notices } from '../../tools/worklet_bundle.ts';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const arg = (n: string, d: string) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const BROWSER = arg('browser', 'chrome');
if (!['chrome', 'firefox', 'safari'].includes(BROWSER)) throw new Error(`--browser chrome|firefox|safari, not ${BROWSER}`);
const OUT = path.resolve(REPO, arg('out', BROWSER === 'chrome' ? 'build/extension' : `build/extension-${BROWSER}`));
const WASM = path.resolve(REPO, arg('wasm', 'build/wasm'));
const MODELS = arg('models', 'deepfilternet3,gtcrn,rnnoise').split(',').filter(Boolean);

const bundle = (entry: string) => bundleWorklet(entry, { repo: REPO, by: 'src/extension/build.ts' });
// loaded as plain scripts (content scripts, a worker, a worklet): no import/export may survive the compile
const CLASSIC = new Set(['inject.js', 'bridge.js', 'engine_frame.js', 'engine_worker.js', 'pump_worklet.js']);

function copyDir(src: string, dst: string, filter: (name: string) => boolean = () => true) {
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, d, filter);
    else if (filter(e.name)) fs.copyFileSync(s, d);
  }
}

fs.rmSync(OUT, { recursive: true, force: true });
// pages, styles, manifest, policy schema from the source; the code as emitted (messages.ts is types + one helper)
copyDir(path.join(REPO, 'src', 'extension', 'src'), OUT, (n) => !n.endsWith('.ts'));
const JS = path.dirname(emitted(REPO, 'src', 'extension', 'src', 'background.ts'));
// inject.js imports the popup's echo gate: bundled into it (echo_gate.js isn't shipped on its own)
const BUNDLED = new Set(['echo_gate.js']);
for (const f of fs.readdirSync(JS).filter((n) => n.endsWith('.js') && !BUNDLED.has(n))) {
  fs.writeFileSync(path.join(OUT, f), f === 'inject.js' ? bundle(path.join(JS, f)) : CLASSIC.has(f) ? classic(path.join(JS, f)) : fs.readFileSync(path.join(JS, f)));
}
// translations (src/i18n): locales/<code>.json = the ext.* keys for the popup / options (i18n.ts), and
// _locales/<store code>/messages.json for the manifest's name and description (store listings; languages the
// stores don't know fall back to English there)
{
  const I18N = path.join(REPO, 'src', 'i18n');
  const langs: { code: string; name: string; dir?: string; store?: string }[] = JSON.parse(fs.readFileSync(path.join(I18N, 'languages.json'), 'utf8'));
  fs.mkdirSync(path.join(OUT, 'locales'), { recursive: true });
  fs.writeFileSync(path.join(OUT, 'locales', 'languages.json'), JSON.stringify(langs.map(({ code, name, dir }) => ({ code, name, dir }))));
  const en: Record<string, string> = JSON.parse(fs.readFileSync(path.join(I18N, 'en.json'), 'utf8'));
  for (const { code, store } of langs) {
    const f = path.join(I18N, `${code}.json`);
    const all: Record<string, string> = { ...en, ...(fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : {}) };
    const ext = Object.fromEntries(Object.entries(all).filter(([k]) => k.startsWith('ext.')));
    fs.writeFileSync(path.join(OUT, 'locales', `${code}.json`), JSON.stringify(ext));
    if (store) {
      fs.mkdirSync(path.join(OUT, '_locales', store), { recursive: true });
      fs.writeFileSync(path.join(OUT, '_locales', store, 'messages.json'), JSON.stringify({
        name: { message: all['ext.name'] }, description: { message: all['ext.description'] },
      }, null, 1));
    }
  }
}
if (BROWSER === 'firefox') {
  // Firefox MV3: event-page background scripts (no service_worker), an add-on id (storage.managed / signing),
  // MAIN-world content scripts and scripting.executeScript({world}) since 128; data collection: anonymous technical
  // reports and detailed diagnostics, both optional; nothing is sent unless the user also turns them on in our settings
  const mf = path.join(OUT, 'manifest.json');
  const m = JSON.parse(fs.readFileSync(mf, 'utf8'));
  m.background = { scripts: [m.background.service_worker], type: 'module' };
  delete m.minimum_chrome_version;
  delete m.storage;   // managed_schema is Chrome's; Firefox reads policies from its own policies.json / managed storage manifest
  for (const w of m.web_accessible_resources) delete w.use_dynamic_url;
  // 140 (ESR) / Android 142: the first versions that know data_collection_permissions (MAIN-world scripts need 128)
  m.browser_specific_settings = {
    gecko: { id: 'parla@parla.cc', strict_min_version: '140.0', data_collection_permissions: { required: ['none'], optional: ['technicalAndInteraction', 'browsingActivity'] } },
    gecko_android: { strict_min_version: '142.0' },
  };
  m.developer = { name: 'tvildo', url: 'https://github.com/tvildo' };
  fs.writeFileSync(mf, JSON.stringify(m, null, 2) + '\n');
}
if (BROWSER === 'safari') {
  // Safari web extension (MV3, inside a small app: src/extension/safari): no Chrome-only keys; the minimum version is
  // the first Safari with MAIN-world content scripts at document_start (the getUserMedia wrapper needs them)
  const mf = path.join(OUT, 'manifest.json');
  const m = JSON.parse(fs.readFileSync(mf, 'utf8'));
  delete m.minimum_chrome_version;
  delete m.storage;   // managed_schema is Chrome's
  for (const w of m.web_accessible_resources) delete w.use_dynamic_url;
  m.browser_specific_settings = { safari: { strict_min_version: '18.0' } };
  fs.writeFileSync(mf, JSON.stringify(m, null, 2) + '\n');
}
// Version and commit: --version X.Y.Z (CI: the next semantic tag, docs/releasing.md; default: the source manifest's) and
// the git commit it was built from ($PARLA_COMMIT, else $GITHUB_SHA, else git; `-dirty` = uncommitted changes). They go
// into build.json (the settings page shows them, the reports carry them) and, for Chrome, the manifest's version_name
// (shown on chrome://extensions). A rebuild from a source archive without .git: PARLA_COMMIT=<sha> for the same bytes.
const VERSION = arg('version', '');
if (VERSION && !/^\d+\.\d+\.\d+$/.test(VERSION)) throw new Error(`--version X.Y.Z, not ${VERSION}`);
function gitCommit(): string {
  const env = process.env.PARLA_COMMIT || process.env.GITHUB_SHA;
  if (env) return env.slice(0, 12);
  try {
    const run = (...a: string[]) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    return run('rev-parse', '--short=12', 'HEAD') + (run('status', '--porcelain', '--untracked-files=no') ? '-dirty' : '');
  } catch { return 'unknown'; }
}
{
  const mf = path.join(OUT, 'manifest.json');
  const m = JSON.parse(fs.readFileSync(mf, 'utf8'));
  if (VERSION) m.version = VERSION;
  const commit = gitCommit();
  if (BROWSER === 'chrome') m.version_name = `${m.version} (${commit})`;   // Firefox has no version_name
  fs.writeFileSync(mf, JSON.stringify(m, null, 2) + '\n');
  fs.writeFileSync(path.join(OUT, 'build.json'), JSON.stringify({ version: m.version, commit }) + '\n');
}
fs.mkdirSync(path.join(OUT, 'engine'), { recursive: true });
fs.writeFileSync(path.join(OUT, 'engine', 'worklet.bundle.js'), bundle(emitted(REPO, 'src', 'engine', 'js', 'worklet.ts')));
for (const b of ['relaxed', 'simd', 'nosimd']) {
  const f = path.join(WASM, `vi_rs_${b}.wasm`);
  if (!fs.existsSync(f)) throw new Error(`missing ${f} (run: python build.py wasm)`);
  fs.copyFileSync(f, path.join(OUT, 'engine', `vi_rs_${b}.wasm`));
}
const shipped: { name: string; display_name: string; sample_rate: number | null; engine: string }[] = [];
for (const m of MODELS) {
  const src = path.join(REPO, 'models', m);
  const man = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
  // the package's manifest.json ships as model.json: the Chrome Web Store refuses a zip with more than one
  // manifest.json (the extension's own is the only one allowed); inject.ts reads models/<name>/model.json
  copyDir(src, path.join(OUT, 'models', m), (n) => !n.endsWith('.md') && n !== 'manifest.json');
  fs.copyFileSync(path.join(src, 'manifest.json'), path.join(OUT, 'models', m, 'model.json'));
  shipped.push({ name: m, display_name: man.display_name || m, sample_rate: man.sample_rate || man.pipeline?.sample_rate || null, engine: man.engine });
}
fs.writeFileSync(path.join(OUT, 'models', 'index.json'), JSON.stringify({ models: shipped }, null, 1));
// the licences: Parla's, and the models' (each package's LICENSE is in models/<name>/ too)
fs.copyFileSync(path.join(REPO, 'LICENSE'), path.join(OUT, 'LICENSE'));
fs.writeFileSync(path.join(OUT, 'NOTICE'), notices(REPO, MODELS));

let bytes = 0;
const manifests: string[] = [];
(function walk(d: string) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else { bytes += fs.statSync(p).size; if (e.name === 'manifest.json') manifests.push(path.relative(OUT, p)); } } })(OUT);
// the Chrome Web Store rejects a package with more than one manifest.json anywhere in it
if (manifests.length !== 1) throw new Error(`exactly one manifest.json allowed in the extension, found: ${manifests.join(', ')}`);
console.log(`extension -> ${OUT} (${(bytes / 1e6).toFixed(1)} MB; models: ${MODELS.join(', ')})`);

if (args.includes('--zip')) {
  // store upload: a plain zip (stored + deflate), no external tools
  const files: string[] = [];
  (function walk(d: string) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else files.push(p); } })(OUT);
  const parts: Buffer[] = [], central: Buffer[] = [];
  let off = 0;
  const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc32 = (b: Buffer) => { let c = 0xffffffff; for (const x of b) c = crcT[(c ^ x) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  for (const f of files) {
    const name = Buffer.from(path.relative(OUT, f).replace(/\\/g, '/'));
    const raw = fs.readFileSync(f), def = zlib.deflateRawSync(raw), crc = crc32(raw);
    const h = Buffer.alloc(30); h.writeUInt32LE(0x04034b50, 0); h.writeUInt16LE(20, 4); h.writeUInt16LE(8, 8); h.writeUInt32LE(crc, 14);
    h.writeUInt32LE(def.length, 18); h.writeUInt32LE(raw.length, 22); h.writeUInt16LE(name.length, 26);
    parts.push(h, name, def);
    const c = Buffer.alloc(46); c.writeUInt32LE(0x02014b50, 0); c.writeUInt16LE(20, 4); c.writeUInt16LE(20, 6); c.writeUInt16LE(8, 10); c.writeUInt32LE(crc, 16);
    c.writeUInt32LE(def.length, 20); c.writeUInt32LE(raw.length, 24); c.writeUInt16LE(name.length, 28); c.writeUInt32LE(off, 42);
    central.push(c, name);
    off += 30 + name.length + def.length;
  }
  const cd = Buffer.concat(central);
  const end = Buffer.alloc(22); end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(files.length, 8); end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(off, 16);
  const zip = `${OUT}.zip`;
  fs.writeFileSync(zip, Buffer.concat([...parts, cd, end]));
  console.log(`zip -> ${zip} (${(fs.statSync(zip).size / 1e6).toFixed(1)} MB)`);
}

// build.ts — the Parla browser SDK for an app to ship with its own files (Node built-ins only).
//
//   node src/sdk/build.ts [--wasm build/wasm] [--models deepfilternet3] [--out build/sdk]
//                      [--js-out <dir>] [--assets-out <dir>] [--version X.Y.Z (a release: the next tag)]
//
// Output (--out, one folder):   parla.js, assets.js, react.js, vi_engine.js (+ .d.ts each)   the module the app imports
// (the npm package, with the files resolved by bundlers, is npm.ts; both copy the code `python build.py ts` emitted)
//                               engine/worklet.bundle.js, engine/vi_rs_{relaxed,simd,nosimd}.wasm
//                               models/<name>/ (manifest, graph, weights), NOTICE.txt
// --js-out / --assets-out put the two halves where an app wants them instead (its source tree, which its
// bundler compiles, and its static files, served at the `base` the app passes to Parla.load).
//
// --wasm: the engine builds from `python build.py wasm`. Paths are relative to the repository root.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { bundle, emitted, notices, sdkCode, stampVersion } from '../../tools/worklet_bundle.ts';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const arg = (n: string, d: string) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const OUT = path.resolve(REPO, arg('out', 'build/sdk'));
const JS_OUT = path.resolve(REPO, arg('js-out', OUT));
const ASSETS_OUT = path.resolve(REPO, arg('assets-out', OUT));
const WASM = path.resolve(REPO, arg('wasm', 'build/wasm'));
const MODELS = arg('models', 'deepfilternet3').split(',').filter(Boolean);

function copyDir(src: string, dst: string, filter: (name: string) => boolean = () => true) {
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, d, filter);
    else if (filter(e.name)) fs.copyFileSync(s, d);
  }
}

for (const d of new Set([OUT, JS_OUT, ASSETS_OUT])) fs.rmSync(d, { recursive: true, force: true });
// assets.js: the source version (no bundled files: the app passes `base`); react.js: the React hook
sdkCode(REPO, JS_OUT);
if (arg('version', '')) stampVersion(JS_OUT, arg('version', ''));

fs.mkdirSync(path.join(ASSETS_OUT, 'engine'), { recursive: true });
fs.writeFileSync(path.join(ASSETS_OUT, 'engine', 'worklet.bundle.js'), bundle(emitted(REPO, 'src', 'engine', 'js', 'worklet.ts'), { repo: REPO, by: 'src/sdk/build.ts' }));
for (const b of ['relaxed', 'simd', 'nosimd']) {
  const f = path.join(WASM, `vi_rs_${b}.wasm`);
  if (!fs.existsSync(f)) throw new Error(`missing ${f} (run: python build.py wasm)`);
  fs.copyFileSync(f, path.join(ASSETS_OUT, 'engine', `vi_rs_${b}.wasm`));
}
const notice = ['Parla noise suppression SDK (Apache-2.0). The model packages keep their own licences:', ''];
for (const m of MODELS) {
  const src = path.join(REPO, 'models', m);
  const man = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
  copyDir(src, path.join(ASSETS_OUT, 'models', m), (n) => !n.endsWith('.md'));
  notice.push(`${m}: ${man.license || 'see the package'}${man.source ? ` (${man.source})` : ''}`);
}
fs.writeFileSync(path.join(ASSETS_OUT, 'NOTICE.txt'), notice.join('\n') + '\n\n' + notices(REPO, MODELS));

// a hash of everything shipped: apps can name the assets' folder by it (cacheable forever)
const h = crypto.createHash('sha256');
let bytes = 0;
const files: string[] = [];
(function walk(d: string) { for (const e of fs.readdirSync(d, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else files.push(p); } })(ASSETS_OUT);
if (JS_OUT !== ASSETS_OUT) for (const f of ['parla.js', 'vi_engine.js']) files.push(path.join(JS_OUT, f));
for (const f of files) { const b = fs.readFileSync(f); h.update(path.basename(f)).update(b); bytes += b.length; }
const version = arg('version', '') || JSON.parse(fs.readFileSync(path.join(REPO, 'src', 'sdk', 'package.json'), 'utf8')).version;
console.log(`parla sdk ${version} (${h.digest('hex').slice(0, 12)}) -> ${JS_OUT === ASSETS_OUT ? OUT : `${JS_OUT} + ${ASSETS_OUT}`} (${(bytes / 1e6).toFixed(1)} MB; models: ${MODELS.join(', ')})`);

// ts_emit.ts — the browser code as JavaScript: every .ts file tsconfig.json compiles, with its types erased by Node's own
// type stripping (the types become spaces: every line and column stays where it is in the source, comments and
// formatting too, so stack traces and the shipped files read like the source). Relative `./x.ts` imports become
// `./x.js`. Writes build/js/<same path>.js. Type checking and the SDK's .d.ts files are tsc's (python build.py ts).
//   node tools/ts_emit.ts [root ...]   (default: all of tsconfig.json's; Node built-ins only, no compiler needed)
import fs from 'node:fs';
import path from 'node:path';
import { stripTypeScriptTypes } from 'node:module';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// the same roots as tsconfig.json's "include" (src/types holds only declarations)
const ROOTS = process.argv.length > 2 ? process.argv.slice(2) : ['src/engine/js', 'src/extension/src', 'src/sdk/src', 'src/telemetry/worker.ts'];

const files: string[] = [];
for (const r of ROOTS) {
  const p = path.join(REPO, r);
  if (p.endsWith('.ts')) files.push(p);
  else for (const f of fs.readdirSync(p)) if (f.endsWith('.ts') && !f.endsWith('.d.ts')) files.push(path.join(p, f));
}
for (const f of files) {
  const js = stripTypeScriptTypes(fs.readFileSync(f, 'utf8'), { mode: 'strip' })
    .replace(/(\bfrom\s*'\.\.?\/[^']*)\.ts'/g, "$1.js'")
    .replace(/(\bimport\(\s*'\.\.?\/[^']*)\.ts'/g, "$1.js'");
  const out = path.join(REPO, 'build', 'js', path.relative(REPO, f)).replace(/\.ts$/, '.js');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, js);
}
console.log(`ts -> build/js (${files.length} files)`);

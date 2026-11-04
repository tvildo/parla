# src/sdk/: Parla in a web app's own code (browser) — the npm package `@tvildo/parla`

User docs: `docs/` (getting started, features, React, npm and assets, API). This file: how it is built.

- TypeScript: `src/parla.ts`, `src/react.ts` (`useParla`, `ParlaProvider`, `getParla`), `src/assets.ts` (source:
  `null` = no bundled files, pass `base`). Shipped as JavaScript + generated `.d.ts` (`python build.py ts` -> build/js;
  `tools/worklet_bundle.ts` `sdkCode()` copies them, rewriting the declarations' `./x.ts` imports to `./x.js`); members
  marked `private` / `@internal` stay out of the types.
- `npm.ts` (`python build.py npm`) -> `build/npm/` + `build/tvildo-parla-<version>.tgz`: `dist/` with the code, a
  generated `dist/assets.js` (one literal `new URL('./engine/…' | './models/…', import.meta.url)` per file: bundlers
  emit each as an asset, browsers without a bundler resolve it natively), `dist/engine/` (worklet bundle + 3 wasm),
  `dist/models/{deepfilternet3,rnnoise,gtcrn}/`, README (`README.npm.md`), LICENSE, NOTICE. 8.5 MB packed.
- `Parla.load()` without `base` uses `ASSETS`; with `base` everything comes from there (self-hosting, CDN).
- Used by: `examples/react-vite` (Vite build; dev server needs `optimizeDeps.exclude`), `examples/vanilla` (no
  bundler), the website's demo (`src/site` depends on `file:../../build/npm`). Tests: `tests/package.spec.ts`.
- `build.ts` (`python build.py sdk`) is the older copy-the-files build (code + engine/ + models/ for an app that
  serves them itself, then `base`).


The second way to ship the engine (the first is the extension, `src/extension/`). Here an app cleans its own microphone track before it sends it. Nothing is forked: the SDK reuses `src/engine/js/worklet.ts` (bundled), `src/engine/js/vi_engine.ts` and the model packages. Everything runs on the user's device; there is no account, key or server.

## API (`src/parla.ts`; apps get `parla.js` + `parla.d.ts`)

```js
const parla = await Parla.load({ base: '/static/parla/', model: 'deepfilternet3' });
const ns = await parla.process(micTrack);     // throws ParlaError: the app keeps micTrack
pc.addTrack(ns.track, stream);                // or sender.replaceTrack(ns.track) mid-call
ns.setEnabled(false);                         // live on/off, ramped (no click)
ns.setStrength(0.6);                          // how much of the cleaned signal (the rest: the input, delayed to match)
ns.setGain(1.5);                              // input level before the engine (a mic sensitivity setting; also in bypass)
ns.onVoice = ({ speaking, prob, levelDb }) => {};   // speech on the cleaned output, ~19/s (a speaking mic button)
ns.setInput(otherMic);                        // device change, same output track
ns.stop();
parla.preload();                              // fetch the model ahead of the first call
```

- **The worklet gets the wasm's bytes, not a compiled Module.** Chrome 154 drops a message carrying a `WebAssembly.Module` posted to an AudioWorklet: a silent `messageerror` in the worklet, so 'init' never answered (Chrome 156 delivers it). `ViRuntime` takes either; every host here posts bytes.
- **One AudioContext (48 kHz) per Parla, one engine node per processed track.** `process()` refuses to start (`ParlaError 'engine'`) when the browser keeps the context suspended (no user gesture yet): otherwise the cleaned track would be silent.
- **Failures after start**: the worklet catches an engine exception and passes the input through
  (`{type:'error', fatal}`); `ParlaStream.onError` / `.error` (also for an ended input track, and the browser's
  `processorerror`); `useParla` then returns `track: null` + `error`. `stop()` posts `close`: the worklet frees its
  engines and ends. `close()` makes the Parla refuse `process()`, also mid-start.
- **Errors** are `ParlaError` with a code: `unsupported` (no WebAssembly / AudioWorklet, or a CSP without `'wasm-unsafe-eval'`), `load` (a file), `engine`. The app keeps using its plain microphone. Audio never stops because of the SDK.

## Build

```sh
python build.py wasm                                  # engine builds -> build/wasm/
node src/sdk/build.ts [--models deepfilternet3] [--out build/sdk]
  [--js-out <app>/src/lib/parla --assets-out <app>/public/parla]      # the two halves where the app wants them
node src/sdk/test_browser.ts [--dir build/sdk-demo]  # real Chrome, fake mic: load, process, on/off, setInput, stop
```

`test_browser.ts` serves an SDK build with the app CSP above (`--modules <dir>` points at a `node_modules` with playwright-core; default `tests/`). Its checks:
- processing starts;
- the fake microphone's beep is mostly removed when on, about half at strength 0.5, and present when off and after `setInput`;
- `setGain(0.1)` gives a tenth of the level (each change settles 300 ms before measuring: audio from before it is still on its way between the two audio contexts);
- speech events arrive (~19/s);
- `stop` ends the track;
- a missing engine is a `load` error.

- **Output:** `parla.js`, `parla.d.ts` and `vi_engine.js` (the app's bundler compiles them); `engine/` (`worklet.bundle.js`, 3 wasm builds); `models/<name>/`; `NOTICE.txt` (the models' licences).
- **Sizes:** 9.5 MB with DeepFilterNet3, most of it its 8.5 MB of weights. The app serves `engine/` and `models/` itself at `base`; they need cache revalidation, or a versioned path.
- **The app's CSP** needs `script-src 'self' 'wasm-unsafe-eval'`.
- **`src/vi_engine.ts`** only re-exports `src/engine/js/vi_engine.ts` for use inside the repo; the builds ship the real file (as `vi_engine.js` + `.d.ts`).

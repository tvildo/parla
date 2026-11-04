# The npm package and its files

`@tvildo/parla` contains everything it needs: the code, the engine (an AudioWorklet script and three WebAssembly
builds, ~260 KB each) and the three models (~9.5 MB in total, mostly the best model's weights). There is no CDN and no
server: the browser loads these files from wherever your app serves them.

```
dist/parla.js, parla.d.ts, react.js, react.d.ts, vi_engine.js, assets.js
dist/engine/worklet.bundle.js, vi_rs_relaxed.wasm, vi_rs_simd.wasm, vi_rs_nosimd.wasm
dist/models/deepfilternet3/ rnnoise/ gtcrn/   (manifest.json, graph.json, weights.bin)
```

## With a bundler (nothing to configure)

`dist/assets.js` refers to every file with `new URL('./engine/…', import.meta.url)`. Vite, webpack 5, Rollup,
esbuild and Parcel recognise this pattern and copy each file into your build as a hashed asset, then rewrite the URL.
So `Parla.load()` needs no options.

The browser downloads only what it uses: one wasm build (the fastest its CPU supports), the worklet script, and the
files of the model in use. The other models stay unused files in your build output.

Vite's dev server needs `optimizeDeps: { exclude: ['@tvildo/parla'] }` (see [React](react.md#vite)).

## Without a bundler

Serve the package's `dist/` folder as static files and import `parla.js` from it. Browsers resolve
`import.meta.url` themselves, so the engine files are found next to it. See [`examples/vanilla`](../examples/vanilla).

## Serving the files from somewhere else

Pass `base`: the URL of a folder that contains `engine/` and `models/` (a copy of `dist/engine` and `dist/models`):

```js
const parla = await Parla.load({ base: 'https://static.example.com/parla/0.1.0/' });
// or from a public CDN that mirrors npm (the user's browser then contacts that CDN):
const parla = await Parla.load({ base: 'https://cdn.jsdelivr.net/npm/@tvildo/parla@0.1.0/dist/' });
```

The files must be served with CORS headers when they come from another origin (the AudioWorklet script and the wasm
are fetched by the page).

## Content Security Policy

Your page needs:

```
script-src 'self' 'wasm-unsafe-eval';     # 'wasm-unsafe-eval' lets the page compile WebAssembly
```

plus the origin of `base` in `script-src` and `connect-src` if you serve the files from elsewhere.

If your page restricts `connect-src`: bundlers inline small files as `data:` URLs (Vite does below 4 KB, which covers the
models' `manifest.json`), and Parla fetches them, so allow `data:` there (`connect-src 'self' data:`) or turn inlining
off for the package's files (Vite: `build.assetsInlineLimit: (file) => (file.includes('@tvildo/parla') ? false : undefined)`).

## Caching

The bundled file names contain a content hash: cache them forever (`Cache-Control: public, max-age=31536000,
immutable`). With `base`, use a versioned path (as above) or revalidation.

## Package size

| | Download (once per user) |
|---|---|
| Engine (one wasm build + worklet) | ~300 KB (~110 KB compressed) |
| `deepfilternet3` model | ~8.6 MB |
| `rnnoise` model | ~0.35 MB |
| `gtcrn` model | ~0.4 MB |

Use `parla.preload()` to fetch the model before the first call, e.g. when the user opens the call screen.

## Publishing (maintainers)

```sh
python build.py wasm && python build.py npm      # build/npm/ + build/tvildo-parla-<version>.tgz
python build.py examples && python build.py test # the examples and every browser test on the packed package
cd build/npm && npm publish --access public
```

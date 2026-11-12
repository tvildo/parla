# Development

How the project is laid out, built and tested.

## Layout

| Path | What |
|---|---|
| `src/engine/` | the engine: a Rust workspace with no third-party crates (`vi_kernels` SIMD math, `vi_audio` DSP, `vi_engine` contracts, `vi_graph` generic model runtime, `vi_dfn` DeepFilterNet3, `vi_rnnoise`, `vi_rs` registry + C ABI for wasm + `vi_cli`, `vi_otel` optional telemetry) and `js/` (the AudioWorklet host, the wasm glue, resampler, model store) |
| `src/extension/` | the browser extension (Manifest V3; Chrome, Edge, Firefox) |
| `src/sdk/` | the npm package `@tvildo/parla` (`npm.ts` builds it), React hook included |
| `src/site/` | the website (Astro, static): landing, privacy, live demo in every language |
| `src/i18n/` | every user-visible string, one JSON per language (`languages.json` lists them) |
| `models/` | model packages (manifest + graph + weights) and the format spec, [`models/FORMAT.md`](../models/FORMAT.md) |
| `tools/` | converters (ONNX -> package), reference executors used for validation, build helpers, `i18n_check.ts` |
| `tests/` | browser tests (Playwright, run in Docker), the Firefox check, the screenshot script |
| `examples/` | apps on the packed npm package: `react-vite`, `vanilla` |
| `docs/` | these docs |
| `AGENTS.md` (here and in `src/engine`, `src/engine/vi_otel`, `src/extension`, `src/sdk`) | notes for AI coding agents and contributors: how each part works, decisions, verified numbers |
| `.github/workflows/ci.yml` | build and test on every push; releases from `master` ([releasing](releasing.md)) |
| `build.py` | every build step (Python standard library only) |
| `pyproject.toml`, `uv.lock` | the Python tools' dependencies (numpy, soundfile; mypy), locked with [uv](https://docs.astral.sh/uv/) |
| `Dockerfile` | the website as a container (nginx on :8080) |

## Build

Needs Rust (the version in `src/engine/rust-toolchain.toml`; rustup installs it), [binaryen](https://github.com/WebAssembly/binaryen)'s
`wasm-opt` (on `PATH` or `WASM_OPT`), Node 24+ with npm 11.10+ (it honours the 3-day minimum release age in the
`.npmrc` files), Python 3.11+, [uv](https://docs.astral.sh/uv/) for `check` and the
model tools, and Docker for the tests. Install npm packages with
`npm ci` (the lockfiles); every project's `.npmrc` turns install scripts off. How the build is pinned:
[releasing → supply chain](releasing.md#supply-chain).

```sh
python build.py wasm                        # the engine -> build/wasm/
python build.py check                       # type-check all the TypeScript and Python
python build.py extension                   # Chrome / Edge -> build/extension/ (--zip for the store)
python build.py extension --browser firefox # Firefox -> build/extension-firefox/
python build.py npm                         # the npm package -> build/npm/ + build/tvildo-parla-<version>.tgz
python build.py sdk                         # the SDK as plain files -> build/sdk/
python build.py examples                    # examples/ on the packed package
python build.py site                        # the website -> build/site/
python build.py serve                       # http://127.0.0.1:8765/ (the website)
python build.py screenshots                 # docs/images (+ store/: the store listing images) from the current builds
```

`--version X.Y.Z` on `extension`, `npm` and `sdk` stamps a release version into the build (CI does it with the next
tag, `node tools/next_version.ts`); without it the builds keep the version in the source. Every extension build
records the git commit it was built from (`build.json`).

The website as a container: `docker build -t parla-site . && docker run -p 8080:8080 parla-site` (CI builds and checks
this image on every push). The source is all here; hosting parla.cc is not part of the project.

## Test

```sh
python build.py test                        # every browser test in Docker: Chromium (Playwright), the SDK, then Firefox
python build.py test -- --project site      # one project: extension | site | package
cd tests && node firefox.ts --headless      # the Firefox build in this machine's Firefox (also part of `test`)
python build.py regress                     # every model, output bit-exact against tests/golden.json
cd src/engine && cargo test --release && cargo clippy --release -- -D clippy::correctness
node tools/i18n_check.ts                    # every language has every string and placeholder
```

What the browser tests cover: the extension (page and isolated mode, live settings, updates during a call,
languages, accessibility), the website (every language, the language menu, the demo, accessibility with axe-core and
the keyboard), and the npm package in the two example apps. The test microphone is a generated noise signal: no
recordings of people are used anywhere.

## TypeScript

All the code outside Rust is TypeScript (strict). Node 24 runs the build scripts, tools and tests as they are (type
stripping); browsers get JavaScript:

- `python build.py ts` (the extension and SDK builds run it): `tsc` checks the browser code and writes the SDK's `.d.ts`
  files, `tools/ts_emit.ts` writes the JavaScript to `build/js/` with the types erased by Node: every line and column
  stays where it is in the source, comments too, so the shipped files and stack traces read like the source.
- `python build.py check`: type-check everything: the browser code (`tsconfig.json`), the build scripts and tools
  (`tsconfig.node.json`), the browser tests (`tests/tsconfig.json`), then the Python (below). `npm run check` does the
  TypeScript part only.
- Only syntax that can be erased (`erasableSyntaxOnly`: no `enum`, `namespace` or constructor parameter properties).
  Imports name the `.ts` file (`import { Parla } from './parla.ts'`); the emitted JavaScript imports `./parla.js`.
- The first `build.py ts` / `check` runs `npm ci` here (TypeScript and the type packages: `package.json`).

## Python

`build.py` and the model tools in `tools/*.py` are typed Python, checked with `mypy --strict` (settings in
`pyproject.toml`). Dependencies are locked with [uv](https://docs.astral.sh/uv/): `pyproject.toml` lists them, `uv.lock`
pins every version with its checksums, and `.python-version` picks the interpreter uv uses.

- `build.py` needs only the standard library: any `python3` (3.11+) runs it, with nothing to install (the Docker
  images and CI call it directly).
- The tools need numpy (and soundfile to read WAV / FLAC): run them in the locked environment, e.g.
  `uv run python tools/graph_ref.py models/gtcrn in.wav out.f32`. `uv run` creates `.venv/` from `uv.lock` on first use.
- `python build.py check` runs `uv run --locked mypy`: it fails when `uv.lock` doesn't match `pyproject.toml`. After
  changing dependencies, run `uv lock` and commit both files. Dependabot updates `uv.lock` weekly.
- Shapes of the data: `tools/onnx_pb.py` (`Model`, `Graph`, `Node`) for ONNX files, `tools/onnx2graph.py`
  (`GraphNode`) and `tools/graph_ref.py` (`Json`, `Attrs`: values parsed from JSON stay `Any`) for packages.

## Translations

Strings live in `src/i18n/<code>.json` (flat keys: `site.*`, `privacy.*`, `demo.*`, `ext.*`; `{name}` placeholders;
`_one` / `_other` plurals). English is the source and the fallback. To fix a translation, edit the file and set
`"_status": "reviewed"` once a native speaker has checked it. To add a language, add it to `languages.json` and
create its file; it then appears on the website and in the extension. `node tools/i18n_check.ts` must pass.

## Models

Only permissively licensed weights, used as data: our own inference and signal processing, never the upstream
project's code. Each package records its licence, source, version and checksums in `manifest.json`; see `NOTICE`.
Converters and reference implementations are in `tools/`.

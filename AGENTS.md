# Parla (open source) — agent notes

Parla is open source (Apache-2.0): a **browser extension** that removes noise from the microphone in web calls, and an
**SDK** (`@tvildo/parla`, React hook included) that lets developers build their own noise-cancellation apps. Everything
runs on the user's device; the website (parla.cc) presents both. See `README.md` for the overview.

> Keep this file and the nested `AGENTS.md` files current: whenever something changes, update the relevant section
> without being asked. User-visible changes also go in `CHANGELOG.md` (fixes phrased as the symptom).

## Where things are

| Path | What | Docs |
|---|---|---|
| `src/engine/` | Rust workspace + `js/` browser runtime (TypeScript) | [src/engine/AGENTS.md](src/engine/AGENTS.md), [vi_otel](src/engine/vi_otel/AGENTS.md) |
| `src/extension/` | MV3 extension for Chrome / Edge / Firefox (getUserMedia wrapper, page / isolated engine modes, popup, options, policy) | [src/extension/AGENTS.md](src/extension/AGENTS.md) |
| `src/sdk/` | the npm package `@tvildo/parla` (`Parla.load` / `process(track)`, React hook) | [src/sdk/AGENTS.md](src/sdk/AGENTS.md) |
| `.github/workflows/ci.yml`, `tools/next_version.ts` | CI: branches `development` (build + browser tests, files as the run's artifact, nothing published) and `master` (two streams: the extension / SDK / npm share `vX.Y.Z`, released only when a commit since the last tag changes something they ship (anything outside `next_version.ts`'s `SITE_ONLY` paths, or an `ext.*` text in src/i18n): the next semantic version from those commits, stamped into every build with `--version`; tag + GitHub release with the Chrome / Firefox zips, npm .tgz, SDK zip; Chrome Web Store, Firefox Add-ons, npm, each skipped while its secrets are missing. The website (GitHub Pages) and the report receiver deploy on every merge, after the release job; the site offers the latest release). Every build records version + git commit (`build.json`, Chrome `version_name`, settings page, reports' `ext` / `commit`). Listings exist (Chrome Web Store, Firefox Add-ons; links in `src/site/src/stores.ts`); store secrets set 2026-11-09 (Chrome through the Google Cloud project `parla-release`, OAuth client `parla-ci`, app in production so the refresh token does not expire; AMO JWT key checked against the API); the automated uploads run with the next release. **Supply chain**: actions by commit SHA, exact Node, Rust via `src/engine/rust-toolchain.toml` + `cargo --locked`, binaryen by SHA-256, `npm ci` everywhere with `ignore-scripts=true` (`.npmrc` in every npm project) + `npm audit signatures`, store tools from `tools/release/package-lock.json`, Docker bases by digest, `persist-credentials: false`, Dependabot (`.github/dependabot.yml`) into `development`, told to skip Rust majors/minors and Node majors (deliberate upgrades: 2026-11-04 Rust 1.91.1 -> 1.99.0, all 7 models byte-identical in `regress`, wasm ~4 % smaller; Node 24 -> 26.13.0 in CI + image, minimum stays 24). CI also builds the website image and checks `/` and `/demo/` | [docs/releasing.md](docs/releasing.md) |
| `docs/` | user / developer docs (Markdown): getting started, features, React, npm and assets, API, extension, accessibility | [docs/README.md](docs/README.md) |
| `examples/` | `react-vite`, `vanilla` on the packed package | — |
| `src/site/` | the website, Astro (`src/views/` Home, Privacy, DemoPage; `src/layouts/Base.astro`; `src/components/Demo.tsx` React island on the SDK; `src/styles/`; `nginx.conf` for the image) | below |
| `src/i18n/` | **every user-visible string**: `en.json` (source) + 20 translations, `languages.json` (code, native name, dir, store code) | below |
| `models/` | packages + `FORMAT.md` + `index.json` | [models/FORMAT.md](models/FORMAT.md) |
| `tools/` | `ts_emit.ts` (the browser code as JavaScript, types erased by Node, lines unchanged), `i18n_check.ts` (every language has every key and placeholder), `onnx2graph.py` (+ `onnx_pb.py`), `make_*.py` packagers, `graph_ref.py` / `dfn_ref.py` / `rnnoise_ref.py` (numpy reference executors), `check_*.py`, `worklet_bundle.ts`, `wasm_index.py` | — |
| `tests/` | Playwright: `extension.spec.ts`, `site.spec.ts`; `Dockerfile` (Playwright image + Python + a pinned Firefox release, geckodriver, PulseAudio null sink) and `run_all.sh` (Chromium tests, SDK, then `firefox.ts`: the Firefox build in Firefox, Selenium; also runs on the host's Firefox) | below |
| `build.py` | `wasm`, `native`, `regress`, `ts`, `check`, `extension`, `sdk`, `npm`, `site`, `serve`, `test` | — |
| `package.json`, `tsconfig*.json`, `src/types/` | TypeScript for the whole tree (below) | [docs/development.md](docs/development.md#typescript) |
| `Dockerfile` | the website image (wasm + site build, nginx :8080) | — |

## Rules

- **Everything runs on the user's device.** No account. Audio, page content and typed text never leave the browser.
  The extension's only network traffic is its reports (`src/extension/src/telemetry.ts` -> `src/telemetry/worker.ts`
  at https://t.parla.cc/v1/e, docs/telemetry.md): **everything OFF by default** (decided by the user 2026-10-27:
  privacy first): anonymous technical reports only when the user turns them on (or a policy does); detailed diagnostics
  (the site's domain only, never path / query, since 2026-11-04; permanent ID) only on top of that (its own switch, off by default; Firefox also asks its permission). The website and the npm package send nothing. New report fields need: both
  schemas (telemetry.ts `SCHEMA` and worker.ts `TYPES`), docs/telemetry.md, and the privacy texts if they change what
  is collected. The
  receiver's `SAMPLE` (sent back as `x-parla-sample`, the extension obeys it by ID) and `DAILY_LIMIT` (429 until
  midnight UTC) turn reports down without an extension update; deployed by `python build.py telemetry-deploy` (stdlib, Cloudflare API; env CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID; the bindings live there, no wrangler.toml) or the release job `telemetry` (environment `telemetry`, vars TELEMETRY_SAMPLE / TELEMETRY_DAILY_LIMIT); installs are counted in Analytics Engine
  (`parla_reports`, index = random ID; one `active` report per install per day).
- **Only permissively licensed model weights**, used as data: our own inference and DSP code, never the upstream
  project's code (no onnxruntime, no their PyTorch/TF/DSP in inference). Provenance in each `manifest.json` + `NOTICE`.
- **No recordings of people** in the repo (tests use generated signals; demo media must be synthetic).
- **Credit**: "made by tvildo" (github.com/tvildo) in the site footer, the extension's popup footer, options page and
  manifest description; the Firefox manifest's `developer`.
- **End-user UI names quality levels, never models**: "Best quality · full band" (deepfilternet3), "Light · full band"
  (rnnoise), "Lightest · wideband" (gtcrn).
- **Accessibility is tested**: new UI must keep axe at zero violations (WCAG 2.2 AA) and work by keyboard.
- **A call never breaks because of Parla**: every failure path falls back to the plain microphone.
- **TypeScript everywhere outside Rust** (user, 2026-10-27), strict, erasable syntax only. Browser code (engine
  `js/`, extension, SDK, telemetry Worker) ships as `build/js/` (`python build.py ts`: tsc checks + writes the SDK's
  `.d.ts`; `tools/ts_emit.ts` erases the types with Node's stripper, so lines and comments stay put); build scripts,
  tools and tests run as `.ts` under Node 24. `python build.py check` = all three projects at zero errors. Imports name
  `.ts` files. Shared types: `src/extension/src/messages.ts` (settings, storage.local keys, every message between the
  extension's parts), `WorkletIn` / `WorkletOut` in `src/engine/js/worklet.ts` (the engine's port protocol).
  The extension's classic scripts (inject, bridge, engine_frame, engine_worker, pump_worklet) may only `import type`;
  the build refuses an import or export left in them.
- Model packages are self-contained (no file references outside `models/<name>/`) and carry the upstream licence
  text (`models/<name>/LICENSE`); `tools/worklet_bundle.ts` `notices()` puts NOTICE + those texts into every
  distribution (extension `NOTICE`, npm `NOTICE` + `dist/models/*/LICENSE`, the site's `licenses.txt`).
- Wasm builds remap the engine's and the Rust toolchain's paths (`build.py wasm`): no local path in a shipped file.
- Build outputs go under `build/` (gitignored); never commit wasm.

## Site

- **Astro** (static output, pinned 7.3.8 + @astrojs/react 7.0.1): pages `src/pages/index.astro` (English at the root) and
  `src/pages/[lang]/*.astro` (`/ka/`, `/es/`, ...) render the same views per language. Each page has `hreflang`
  alternates, a language menu (`<details>` in the header) and a one-time "also in <your language>" banner from the
  browser language (never a redirect; dismissal in localStorage `parla.lang`).
- `build.py site` runs `tools/i18n_check.ts`, puts the SDK's assets (`engine/`, `models/`) at the root, builds Astro
  and rewrites root-absolute links to page-relative ones (`relative_links`): one build works at `/` and under the
  `/d/` preview.
- Palette and type of the original holding page (dark `#0c0c14` with an indigo radial glow, accent `#7c7cff` -> `#b36bff`,
  system font stack) in `src/styles/site.css`; RTL via `dir` + logical properties.
- The live demo (`src/components/Demo.tsx` + `audio.ts`, `client:only="react"`) gets its strings (`demo.*`) and the
  relative path to the engine's files from the page. One card: Original vs With Parla meters, quality (dropdown) +
  noise suppression strength (slider 0-100 %), then the buttons: "Start" before; "Stop" + "Record 8 s" (red dot) once
  the microphone runs. Recording shows a progress bar with the time left, then two seekable players in the same card
  (with Parla first, the original second) and the button becomes "Record again". No microphone (permission denied, no device, unsupported browser):
  only the reason and "Try again" (no live listening, no on/off switch: decided by the user 2026-10-27).
- `npm run dev` in `src/site` (Astro dev server); the demo then needs `engine/`/`models/`: run `python build.py site`
  once and use `python build.py serve` for the full thing.
- Source links point to `github.com/tvildo/parla` (the public repository, not created yet).

## README, docs, screenshots

- `README.md` is for **users** (what it does, screenshots, install, the demo, a short "for developers" with links).
  Everything technical (layout, build, tests, translations, models) is in `docs/` (`docs/development.md` for
  contributors; getting started, features, React, npm, API, extension, accessibility for SDK users).
- Screenshots are generated, never hand-made: `python build.py screenshots` (`tests/screenshots.spec.ts`, local
  Chromium for its fonts) writes `docs/images/*.png` (README, docs) and `docs/images/store/*.jpg` (1280x800, the Chrome / Firefox listings) from the current extension and site
  builds. The website uses no screenshots: its "See it" section renders the real popup (`src/extension/src/popup.html` +
  `ui.css`, texts from the page language) and the real demo card at build time (`src/site/src/components/previews.ts`);
  the site build fails when the popup changes in a way the preview can't follow. The popup shows a made-up meeting page (`meet.example.org`)
  through `popup.html?tab=<id>`. Retake them after UI changes.

## Languages (i18n)

- One source of truth: `src/i18n/<code>.json`, flat keys `site.*`, `privacy.*`, `demo.*`, `ext.*`; `{name}`
  placeholders; plurals as `<key>_one` / `<key>_other` (chosen by `{n}`). English is the source and the fallback for a
  missing key. 21 languages: en, zh-CN, zh-TW, hi, es, fr, ar, bn, pt-BR, ru, ur, id, de, ja, mr, te, tr, ta, vi, ko,
  ka; RTL: ar, ur.
- **All translations are machine drafts** (`"_status": "draft"`, 2026-10-27) until a native speaker reviews them
  (then `"reviewed"`). Georgian: the user reviews.
- Add a string: add the key to `en.json` and to every language (the check fails otherwise), use it via `t()`.
  Add a language: `languages.json` + `<code>.json`; it appears on the site, in the menus and in the extension.
- `ext.name` <= 45 and `ext.description` <= 132 characters (store limits; checked).

## Tests

`python build.py test` builds `tests/Dockerfile` (mcr.microsoft.com/playwright, Chromium + Node + Python) and runs it
with this tree mounted at `/oss` (node_modules in a Docker volume): `extension.spec.ts` (8 tests: off = untouched mic,
page mode, isolated mode under a strict CSP, live off, stop, shared mic, update waits for the call, reload mid-call
with no silence), `site.spec.ts` (pages, live demo), then the SDK's `src/sdk/test_browser.ts`. The fake microphone is a
generated WAV (noise + hum + whistle, `tests/lib.ts`): suppression on removes ~39 dB, off passes it.
Without Docker: `cd tests && npm ci && npx playwright install chromium && npx playwright test`.

## Verified (2026-10-26)

- Engine: `cargo test` green; the 7 packages bit-identical to the outputs from before the licensing code was removed
  (same input, same hashes); `tests/golden.json` recorded on AMD64-win32 (Linux: record on first run).
- Browser (Docker, 2026-10-27): extension 10/10 (incl. languages, axe), site 4/4 (every language, menu, suggestion,
  Georgian demo), a11y 4/4 (axe on every page type in en/ka/ar, keyboard-only demo), package 2/2 (React + Vite and
  no-bundler examples, -50 dB), SDK test PASS; Firefox 158 build PASS (`tests/firefox.ts`), `web-ext lint` clean.
- Accessibility target WCAG 2.2 AA (docs/accessibility.md); any axe violation fails the suite.
- Website image builds and serves (`docker build .`).

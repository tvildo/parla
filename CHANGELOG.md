# Changelog

User-visible changes. Fixes are written as the symptom someone saw.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- The report receiver can be deployed from this repository (`python build.py telemetry-deploy`, or with every
  release), so the whole service is open source, not just its code.
- Releases: `development` is built and tested on every push; every merge into `master` becomes the next semantic
  version (from the commit messages), a GitHub release with the Chrome and Firefox extensions, the npm package and the
  SDK files, and goes to the Chrome Web Store, Firefox Add-ons and npm (docs/releasing.md).
- The extension's settings page shows its version and the commit it was built from; reports carry both.
- Extension popup: a live **Background noise** meter during calls: your microphone against what the call hears, and
  how much quieter the noise sounds (measured in your pauses).
- First open-source release layout: the engine (`src/engine`), the browser extension (`src/extension`), the SDK
  (`src/sdk`), the website with a live demo (`src/site`), model packages (`models`), Apache-2.0.
- `build.py`: one script for every build step (wasm, extension, SDK, site, regression, local server) and `build.py
  test`, which runs the browser tests in Docker.
- Website: landing page, privacy page, and a live demo that cleans your own microphone in the page (quality levels,
  on/off, live listening with headphones, an 8-second A/B recording).
- `Dockerfile`: the website as a container (nginx on :8080).
- Firefox version of the extension (Firefox 128+): `python build.py extension --browser firefox`.
- 21 languages for the website and the extension: English, 简体中文, 繁體中文, हिन्दी, Español, Français, العربية (RTL), বাংলা,
  Português (Brasil), Русский, اردو (RTL), Bahasa Indonesia, Deutsch, 日本語, मराठी, తెలుగు, Türkçe, தமிழ், Tiếng Việt, 한국어,
  ქართული. The site has a page per language and offers your browser's language; the extension follows the browser and
  has a Language setting. Translations are drafts until reviewed by native speakers.
- The website is built with Astro (static).
- Extension reports: anonymous technical reports (errors, performance, settings, browser, the kind of calling app;
  a monthly random ID), off until the user turns them on in the settings; detailed diagnostics (page address, permanent ID)
  only when turned on. Receiver: a Cloudflare Worker (`src/telemetry/worker.ts`, Workers Logs). Privacy texts updated
  in all 21 languages; `docs/telemetry.md` lists every field. One `active` report a day counts active installs
  (Analytics Engine, 3 months); the receiver can turn reports down or off for every install (`--sample`) and has a
  daily limit after which extensions pause until midnight UTC.
- The extension uses the Parla colours (the website's palette).
- A user-friendly README with screenshots; the website has a "See it" section with the extension and the demo.
  Developer notes (layout, build, tests, translations) are in `docs/development.md`. Screenshots come from
  `python build.py screenshots`.
- Live demo in one card: quality as a dropdown, a noise suppression strength slider, "Stop" and "Record" once the
  microphone runs; recording shows a progress bar and ends with two players you can seek (with Parla, then the
  original) and "Record again"; without microphone access only the reason and "Try again" are
  shown. Live listening and the on/off switch were removed.
- npm package `@tvildo/parla` with the engine and models inside (bundlers pick the files up as assets; no CDN, no
  copy step), a React hook (`@tvildo/parla/react`: `useParla`, `ParlaProvider`), and examples (React + Vite, plain
  HTML). The website's live demo uses the package.
- Documentation in `docs/`: getting started, features (speech detection, strength, gain, device changes, quality
  levels), React, npm and assets, API, extension (incl. policy keys), accessibility.
- Accessibility (WCAG 2.2 AA): skip link, visible focus, stronger contrast for small text, a spoken status and
  readable level meters in the demo, labelled controls and quieter live regions in the extension popup, right-to-
  left layouts; checked with axe-core and a keyboard-only test.
- The website and the extension say what Parla is: AI noise suppression that runs on your device, open source, made
  by tvildo.

### Changed
- Built with Rust 1.99 (engine files ~4 % smaller, model outputs byte-identical) and Node 26; website image on
  nginx 1.31.8, tests on Playwright 1.64.1. CI now also builds and checks the website image.
- Build and release pipeline hardened against supply-chain attacks: everything pinned (actions by commit, Node,
  Rust, binaryen by checksum, Docker images by digest), npm only from lockfiles with install scripts off and
  registry signatures checked, the store upload tools locked, Dependabot for updates (docs/releasing.md).
- Website demo: the two level bars are one noise meter, as in the extension: the original microphone and the voice
  with Parla in one bar, and how much quieter the background noise is. The number stops at 16× (40 dB): beyond that it
  is silence to the ear.
- Detailed diagnostics send only the site's domain (e.g. `meet.google.com`), no longer the page path; addresses in
  error messages are cut to their domain too. The receiver drops paths sent by older builds.
- Extension settings page: Settings, Language and Privacy are tabs. "On for all sites by default" moved there from
  the popup (the popup keeps the switch for the current site).
- All the JavaScript is now TypeScript (strict): the engine's browser runtime, the extension, the SDK, the telemetry
  receiver, the build scripts, tools and tests. The npm package's types are generated from the code
  (`dist/*.d.ts`), the React example is TypeScript, and the shipped JavaScript keeps the source's lines and comments.
  `python build.py check` type-checks everything; Node 24+ is needed to build.
- No account and no licence checks: the extension and the SDK work fully, unlimited, without signing in.
- The extension no longer asks for the `identity` permission.

### Fixed
- The Chrome Web Store refused the extension zip ("More than one manifest file was found in the package"): the
  model packages' manifest.json ship as model.json inside the extension, and the build fails if a second
  manifest.json ever appears.
- Extension settings: a second quick click on a switch could be undone by the refresh after the first one.
- `build.py examples` failed after the SDK was rebuilt with the same version (npm checksum of the packed file).
- Extension settings, Privacy: a "Turn on detailed diagnostics?" box with Turn on / Cancel was always shown and its
  buttons seemed to do nothing. The box is gone: detailed diagnostics is a plain switch (still off by default, what
  it sends is written under it). The popup no longer shows an empty noise meter when there is no call.
- Each stopped stream kept its whole engine (the wasm instance and the model's weights, 8.5 MB for the best quality) in
  memory until the page closed: SDK streams, the website demo's quality switches, and the extension's in-page mode on
  every call. The engine is now freed when a stream stops, and the extension closes its audio after the last call.
- A call app that cloned the cleaned microphone and stopped the original (common in call libraries) sent silence.
- An engine failure in the middle of a call silently muted the cleaned track; the input now passes through and the
  SDK reports it (`ParlaStream.onError`, `useParla` -> `error`). An unplugged microphone is reported the same way.
- Two quick quality changes in the extension could leave the wrong model running and an unused one in memory.
- The "users cannot turn it off" policy (`lockEnabled`) could be bypassed with the popup's "on for all sites" switch.
- A wrong `base` (a server answering with an HTML page) was reported as "this browser can't run Parla"; it is now a
  `load` error, and every documented error arrives as a `ParlaError`. `close()` during `process()` no longer leaves
  a stream on a closed audio context.
- Isolated mode stopped working for the rest of a tab after one slow start, or when the page removed its frame.
- When the engine could not start, the extension turned the browser's noise suppression on even where the site had
  turned it off.
- The popup's strength slider jumped back while being dragged; a refusal ("set by your organization") was not shown.
- An update waiting for calls to end could be held back forever by an open page; it now applies after 12 hours.
- The engine's WebAssembly files contained the build machine's toolchain path.
- The model weights' licence texts (MIT, BSD-3-Clause, CC BY 4.0) now ship with every copy: `models/<name>/LICENSE`,
  the extension's `NOTICE`, the npm package, and the website's `licenses.txt` (footer link).
- The extension had no icons (required by the stores).
- TypeScript projects with `moduleResolution: "node"` could not find the types of `@tvildo/parla/react`.
- The popup's strength slider started in the middle (50 %) until the settings loaded; it starts at the default, 100 %.
  An install updated from an earlier build drops that build's account data and goes back to 100 % strength once.
- On Chrome 154 the in-page engine never started (the browser silently dropped the compiled engine sent to the audio
  worklet); the engine is now sent as bytes.
- Recordings and calls could be silent with the extension on, when a page opened the microphone without a click first
  (the browser's autoplay rules kept the extension's audio paused). It now starts at the page's request and, if still
  paused, at the next click or key press; the console shows `[vi] audio running` / `suspended`.
- A hostile web page could talk to the extension's page channel and have it fetch any file of the extension, send
  made-up "settings" reports or fill the popup with long text. Pages now get only the engine's own files, reports from
  pages are limited to call summaries and errors, status keeps known fields (text cut to 200 characters), settings and
  other tabs' status answer extension pages only, and the isolated-mode frame starts at most 8 engine workers.
- Firefox warned "Reading manifest: Warning processing storage" when loading the add-on (a Chrome-only manifest key).

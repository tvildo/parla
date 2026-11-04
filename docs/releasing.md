# Releasing

Two branches:

| Branch | What happens on a push (the workflow [.github/workflows/ci.yml](../.github/workflows/ci.yml)) |
|---|---|
| `development` | Everyday work. Everything is built and tested (type-check, engine, extension for Chrome and Firefox, npm package, SDK files, browser tests, and the website image, started and checked). The files are attached to the run as the `parla` artifact. Nothing is tagged or published. Pull requests get the same. |
| `master` | Releases. A merge into `master` builds and tests the same way, then creates the next version tag, a GitHub release with the files to download, publishes the extension to the Chrome Web Store and Firefox Add-ons and the package to npm, deploys the report receiver to Cloudflare, and the website to GitHub Pages (parla.cc). |

## The version

Semantic versioning, from the commit messages since the last tag ([conventional commits](https://www.conventionalcommits.org/)):

| Commits since the last release | Next version (from 0.4.2) |
|---|---|
| any `feat!:` / `fix!:` (a `!`), or a line `BREAKING CHANGE: …` | **1.0.0** (major) |
| any `feat: …` / `feat(popup): …` | **0.5.0** (minor) |
| anything else (`fix:`, `docs:`, `chore:`, merges, …) | **0.4.3** (patch) |

The first release takes the version in `src/extension/src/manifest.json` (0.1.0). A manual run of the workflow on
`master` (*Actions → Build and release → Run workflow*) can force `patch`, `minor` or `major`. Locally:
`node tools/next_version.ts` prints what the next release would be.

The version is stamped into the builds, not committed: the extension's manifest, the npm package's `package.json` and
the SDK's `VERSION` all carry it, so the files in the repository keep the last hand-set number.

## What a release contains

The tag `vX.Y.Z` on the merged commit, and a GitHub release *Parla vX.Y.Z* with notes from the merged pull requests
and these files:

| File | What |
|---|---|
| `parla-chrome-vX.Y.Z.zip` | the extension for Chrome and Edge (load it unpacked, or the store's package) |
| `parla-firefox-vX.Y.Z.zip` | the extension for Firefox 128+ |
| `tvildo-parla-X.Y.Z.tgz` | the npm package (`npm install ./tvildo-parla-X.Y.Z.tgz`) |
| `parla-sdk-vX.Y.Z.zip` | the SDK as plain files for apps without npm (all three quality levels) |

Every build also knows what it was built from: `build.json` in the extension (`{"version", "commit"}`), shown at the
bottom of the extension's settings page ("Parla v0.2.0 · 3f2a9c1b7e04"), in Chrome's `version_name`
(chrome://extensions) and sent with the extension's reports when they are on (`ext`, `commit`;
[telemetry](telemetry.md)). Local builds from uncommitted changes say `-dirty`.

## One-time setup

The first version goes to each store by hand: the listing (description, screenshots, category), the privacy answers
and, for Chrome, the developer registration can't be done by the workflow. Until the secrets below exist, the store
and npm steps are skipped (a notice in the run), and the GitHub release still happens.

**Chrome Web Store**

1. Register as a Chrome Web Store developer and upload `parla-chrome-*.zip` once in the
   [developer dashboard](https://chrome.google.com/webstore/devconsole). Its ID is `CWS_EXTENSION_ID`; the
   publisher ID (dashboard → Account) is `CWS_PUBLISHER_ID`.
2. In Google Cloud: enable the *Chrome Web Store API*, create an OAuth client (type *Desktop app*) and a refresh token
   for it (steps: [chrome-webstore-upload-keys](https://github.com/fregante/chrome-webstore-upload-keys)).

**Firefox Add-ons**

1. Submit `parla-firefox-*.zip` once on [addons.mozilla.org](https://addons.mozilla.org/developers/) (listed), with
   the source code and how to build it: `npm ci`, `python build.py wasm --only 0` (1, 2), then
   `PARLA_COMMIT=<commit> python build.py extension --browser firefox --version X.Y.Z` (Rust 1.99.0, binaryen 133, Node
   26 on Linux; the result is byte-identical).
2. Create API credentials under *Tools → Manage API Keys*: the *JWT issuer* and *JWT secret*.

**npm**: trusted publishing, no token. On npmjs.com, the package's *Settings → Trusted Publisher → GitHub Actions*:
user `tvildo`, repository `parla`, workflow `ci.yml`, environment `stores`; and under *Publishing access* "Require
two-factor authentication and disallow tokens". Then set the repository variable `NPM_TRUSTED_PUBLISHING` to `true`.
Each release publishes with a signed provenance record (the commit and build it came from). The very first version of
a package has to be published by hand (`npm publish <file>.tgz --access public` in a terminal, with 2FA).

**GitHub**

- *Settings → Environments → New environment* `stores` (optionally with yourself as a required reviewer: then every
  store / npm release waits for a click), with these secrets:

  | Secret | From |
  |---|---|
  | `CWS_EXTENSION_ID`, `CWS_PUBLISHER_ID`, `CWS_CLIENT_ID`, `CWS_CLIENT_SECRET`, `CWS_REFRESH_TOKEN` | Chrome Web Store, Google Cloud |
  | `AMO_JWT_ISSUER`, `AMO_JWT_SECRET` | Firefox Add-ons API key |
  
- *New environment* `telemetry` (optional: the report receiver is deployed with every release): secrets
  `CLOUDFLARE_API_TOKEN` (Workers Scripts: Edit, Workers Custom Domains: Edit, Zone: Read) and
  `CLOUDFLARE_ACCOUNT_ID`; variables `TELEMETRY_SAMPLE` (default 1) and `TELEMETRY_DAILY_LIMIT` (default 80000).
  Without the secrets the step is skipped. The receiver can also be deployed by hand: [telemetry](telemetry.md#running-the-receiver).
- Default branch `development`; protect `master` so it only changes through pull requests (*Settings → Branches*;
  for a private repository this needs a paid plan, public repositories have it free).

## Supply chain

What the pipeline trusts, and how each piece is pinned:

| Piece | Pinned by |
|---|---|
| GitHub actions | full commit SHA (the version in a comment) |
| Node | exact version (`NODE_VERSION` in the workflow); Docker images by digest |
| Rust | `src/engine/rust-toolchain.toml` (1.99.0); the engine uses no third-party crates, `cargo build --locked` |
| binaryen (`wasm-opt`) | version + SHA-256 of the download (workflow and `Dockerfile`) |
| npm packages | `package-lock.json` in every project, installed with `npm ci`; install scripts off (`.npmrc`: `ignore-scripts=true`); `npm audit signatures` checks the registry signatures |
| Store upload tools | `tools/release/package-lock.json` (no `npx` of the latest version) |
| Docker base images | `@sha256:` digests (`Dockerfile`, `tests/Dockerfile`); downloads inside them (Firefox, geckodriver, binaryen) by SHA-256 |

Checkouts don't keep the GitHub token (`persist-credentials: false`); only the release job can write (to create the
tag and release), and the store and npm credentials live in the `stores` environment. Dependabot
(`.github/dependabot.yml`) opens weekly pull requests into `development` to move the pins forward (GitHub actions,
npm packages, the Docker images); CI tests them there, including the website image. When the Playwright image is
updated, update `@playwright/test` in `tests/package.json` to the same version.

**Upgrading Rust or Node** is done by hand, everywhere at once, not by Dependabot (it is told to skip them):

- **Rust**: `channel` in `src/engine/rust-toolchain.toml`, the `rustup` lines in the workflow, the `rust:` image in
  `Dockerfile` (by digest). Then `python build.py regress`: every model's output must stay byte-identical (if a new
  compiler changes the numbers, check them and record new results with `--record`).
- **Node**: `NODE_VERSION` in the workflow and the `node:` image in `Dockerfile` (by digest). The minimum for
  contributors (Node 24+, for running TypeScript directly) is separate.

To look up a digest: `docker buildx imagetools inspect <image>:<tag>`.

## Security checks

[.github/workflows/security.yml](../.github/workflows/security.yml) runs on every push and pull request, and weekly:

| Check | What |
|---|---|
| CodeQL (SAST) | the TypeScript / JavaScript, the Rust engine and the workflows (`security-extended` queries) |
| zizmor (SAST) | the GitHub workflows: injection, over-broad permissions, unpinned actions, credential leaks |
| Dependency review | pull requests that add a package with a known vulnerability (moderate or worse) fail |
| OWASP ZAP baseline (DAST) | the website image, started in CI: headers, cookies, information leaks; the report is the run's `zap-report` artifact |

Findings appear in the repository's *Security → Code scanning*. GitHub's secret scanning with push protection refuses
pushes that contain a credential, and Dependabot alerts cover every lockfile (including `src/site` and `examples/`,
which Dependabot doesn't update automatically: they use the locally built package).

## Building a release locally

```sh
npm ci
for i in 0 1 2; do python build.py wasm --only $i; done
python build.py extension --zip --version 0.2.0                     # build/extension.zip
python build.py extension --browser firefox --zip --version 0.2.0   # build/extension-firefox.zip
python build.py npm --version 0.2.0                                 # build/tvildo-parla-0.2.0.tgz
python build.py sdk --models deepfilternet3,gtcrn,rnnoise --version 0.2.0
```

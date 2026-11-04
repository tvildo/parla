# The browser extension

Parla for Chrome, Edge and Firefox (140+) cleans the microphone in every web calling app — Google Meet, Microsoft
Teams, Zoom (web), Discord, Slack and any other site that uses the microphone — without changes to the app.

## Install

Until it is in the Chrome Web Store and Firefox Add-ons, download `parla-chrome-vX.Y.Z.zip` or
`parla-firefox-vX.Y.Z.zip` from the [latest release](https://github.com/tvildo/parla/releases/latest) and unzip it
(or build it from source, [how to build](development.md#build): `build/extension`, `build/extension-firefox`). Then:

- **Chrome / Edge**: open `chrome://extensions`, turn on *Developer mode*, click *Load unpacked* and choose the
  unzipped folder.
- **Firefox**: open `about:debugging`, click *This Firefox* → *Load Temporary Add-on* and choose `manifest.json` in
  the unzipped folder (it stays until Firefox restarts, until the add-on is signed).

## Using it

Click the toolbar icon:

- **On for this site** — per site; the default for all sites is in Settings.
- **Quality** — Best quality · full band / Light · full band / Lightest · wideband.
- **Strength** — how much of the cleaned voice (100 % = all).
- The status line shows when it is active, the delay, and whether you are speaking (the dot).
- **Background noise** (during a call): a live bar, your microphone (striped) against what the call hears (solid);
  the gap is the noise Parla removes. The number says how much quieter the background noise sounds, measured in your
  pauses ("8× quieter"; hover for the dB). Computed on your device while the popup is open; nothing is sent.

Settings (the link in the popup), in tabs:

- **Settings**: where the engine runs (automatic, in the page, or isolated for sites that forbid WebAssembly, such
  as Teams), **On for all sites by default**, and the per-site choices.
- **Language**: automatic (the browser's) or any of the 21.
- **Privacy**: anonymous technical reports and detailed diagnostics, both off by default ([telemetry](telemetry.md)).

At the bottom: the version and the git commit it was built from ("Parla v0.2.0 · 3f2a9c1b7e04"; also on
`chrome://extensions` as the version name). Please include it when you report a problem.

If anything goes wrong, the site gets your plain microphone and the popup says why. Updates wait until your calls end.

## Languages

English, 简体中文, 繁體中文, हिन्दी, Español, Français, العربية, বাংলা, Português (Brasil), Русский, اردو, Bahasa Indonesia,
Deutsch, 日本語, मराठी, తెలుగు, Türkçe, தமிழ், Tiếng Việt, 한국어, ქართული. Corrections welcome: `src/i18n/<code>.json`.

## Enterprise policy

IT can install the extension with Chrome/Edge policy and preset its settings (managed storage, schema in
`src/extension/src/managed_schema.json`). Values set by policy are shown locked in the popup.

| Key | Type | |
|---|---|---|
| `enabled` | boolean | on for every site by default |
| `lockEnabled` | boolean | users cannot turn it off |
| `model` | string | `deepfilternet3`, `rnnoise`, `gtcrn` |
| `strength` | integer | 0–100 |
| `sites` | object | per origin `"on"` / `"off"`, e.g. `{"https://teams.microsoft.com": "on"}` |
| `onlyListedSites` | boolean | process audio only on sites set to `"on"` |
| `lang` | string | `"auto"` or a language code (`en`, `ka`, `es`, `zh-CN`, …) |
| `telemetry` | boolean | anonymous technical reports (default `false`); `true` is the organization's consent for its users |
| `diagnostics` | boolean | detailed diagnostics (default `false`); `true` is the organization's consent for its users |

## Privacy

Audio is processed on your computer and never leaves it. There is no account. The extension runs on all sites only so
that it can wrap the microphone wherever a calling app asks for it; it reads no page content.

Reports (settings page → Privacy; details and the exact fields in [telemetry](telemetry.md)):

- **Anonymous technical reports — off** unless you turn them on: errors, performance (delay, processing
  time, audio gaps), the settings in use, browser and OS, and the *kind* of calling app (`meet`, `teams`, `zoom`, …,
  `other`) — never its address. A random ID that changes every month. About one small report per call.
- **Detailed diagnostics — off** unless you turn them on (Firefox also asks its own permission): adds the
  site's domain (e.g. `meet.google.com`, never the rest of the address) and a permanent random ID.
- Never sent: audio, page content, anything you type. Firefox shows both in about:addons as optional data collection.

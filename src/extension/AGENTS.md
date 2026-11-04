# src/extension/ — Parla as a browser extension (Chrome / Edge / Firefox, Manifest V3)

The extension wraps `navigator.mediaDevices.getUserMedia`, so any web calling app (Meet, Teams, Zoom web, Discord, ...)
gets noise-suppressed microphone audio without changing its code. Same engine, worklet and models as the SDK; nothing
is forked: the build bundles `src/engine/js/worklet.ts`. No account, no server: the engine and the models ship inside
the extension and everything runs on the user's device.

## How it works

| File (`src/`) | World / context | Does |
|---|---|---|
| `inject.ts` | page, MAIN world, `document_start`, all frames | wraps getUserMedia: real mic (browser noise suppression off; echo cancellation / AGC as the site asked) -> engine -> a track that looks like the mic (label, settings, constraints, stop, 'ended' follow the real one) + original video. Never breaks a call: any failure -> the plain microphone + a `[vi]` console line + popup notice |
| `bridge.ts` | content script, ISOLATED world | MessageChannel to `inject.ts` (hello captured at document_start); settings from the service worker; fetches engine + model files from the package (page CSP never applies); creates the engine frame; pushes config changes live |
| `background.ts` | service worker (module) | effective settings = defaults < user (storage.sync) < **enterprise policy** (storage.managed, `managed_schema.json`); per-tab status + badge; **updates**: held while a call runs (`onUpdateAvailable`), applied when the last session port closes; after install / update the content scripts are injected into open tabs |
| `pump_worklet.ts` + `engine_frame.html/js` + `engine_worker.ts` | isolated mode | pages whose CSP forbids WebAssembly (Teams: nonce-only scripts, no eval): the page runs only a JS pump worklet; quanta go over a MessagePort to a worker in an extension-origin frame that runs the same worklet code with shims (AudioWorkletProcessor, registerProcessor, sampleRate, currentFrame), +3 quanta (8 ms) jitter buffer |
| `popup.*`, `options.*`, `ui.css` | extension pages | this-site on/off, quality, strength, live status (mode, delay, speaking), live **noise meter** (popup -> `chrome.tabs.connect(tab, {name:'meter'})` -> bridge tells the page `meter on` -> page script forwards the worklet's `voice` events, `inDb` / `outDb` = input vs output RMS over ~53 ms, ~19/s, only while a popup is open; number = input − output in pauses (≥ 6 samples without speech), EMA, shown as loudness `2^(dB/10)`× quieter); options = tabs (WAI-ARIA, arrow keys, `#settings` / `#language` / `#privacy` in the URL): Settings (engine mode, per-site list), Language, Privacy; policy-set values shown locked |
| `telemetry.ts` | service worker module | the reports (docs/telemetry.md): switches `telemetry` (default off) / `diagnostics` (default off; Firefox's own data-collection consent is checked too), monthly / permanent random ID, app kind from the host (never the address unless diagnostics), a per-type field whitelist (`SCHEMA`: the page script runs in the page's world, so messages can be forged), 40 events/day, queue + back-off, `fetch` to https://t.parla.cc/v1/e (`storage.local.telemetryEndpoint` overrides it for tests). Fed by the page script: one `call` summary at hang-up (it asks the engine for `stats` first, 300 ms max) and `error` events; by the worker: `settings` changes |
| `messages.ts` | types (+ `getLocal`) | the settings, the storage.local keys (`LocalStore`), and every message: page script <-> bridge (`PageToBridge` / `BridgeToPage`), bridge / popup / options -> service worker (`ToBackground`). What arrives from a page is still checked at run time (`background.ts` `pageStatus`, the telemetry schema): the types describe our own code, not a hostile page |
| `i18n.ts` | extension pages | translations: `locales/<code>.json` (the `ext.*` keys of `src/i18n`, written by `build.ts`), language = setting `lang` (Options; policy key `lang`) or `auto` = the browser's languages matched to ours; `data-i18n` / `data-i18n-title` attributes, `t(key, vars)`, `<html lang dir>`. Not `chrome.i18n`: Chrome's UI-language list lacks e.g. Georgian. The manifest's name / description use `__MSG_*__` from `_locales/<store code>/messages.json` (store listings; languages the stores don't know fall back to English) |

Tracks: the session's output track (`MediaStreamAudioDestinationNode`) stays private; the page gets disguised clones,
each counted as a user (`clone()` too: call libraries clone and stop the original); the last one stopped ends the
session. Settings reach the engine one change at a time per session (`apply` is serialized). The page's AudioContext is
closed when no call is left (`idle()`; `starting` keeps it open while a getUserMedia is in flight). Updates wait for
calls, at most 12 h (`UPDATE_MAX_WAIT`). `lockEnabled` (policy) ignores the user's off switches (background `siteEnabled`,
`set` refuses). Icons: `src/icons/` from the site's logo (`node tools/make_icons.ts`). The build ships `LICENSE` and a
`NOTICE` with every model's licence text.

Modes: `auto` (default) = page, then isolated; `page` / `isolated` forced in Options. Live changes: strength / on-off
ramp inside the engine (no clicks), a model change crossfades; a session that was starting while settings changed
applies the newest settings when it is up. In page mode the worklet gets the wasm's **bytes** (Chrome 154 drops a
message carrying a `WebAssembly.Module` to an AudioWorklet); the page compiles once only to learn whether its CSP allows
WebAssembly.

## Build and test

```
python build.py wasm && python build.py extension [--zip]   # -> build/extension/ (10.2 MB with 3 models)
python build.py extension --browser firefox [--zip]          # -> build/extension-firefox/ (Firefox 140+)
python build.py test                                         # browser tests in Docker (tests/extension.spec.ts)
cd tests && node firefox.ts [--headless]                    # the Firefox build in this machine's Firefox (Selenium); `python build.py test` runs it in Docker too (Firefox 158, PulseAudio null sink)
```

**Firefox**: one source tree; `build.ts --browser firefox` rewrites the manifest: `background.scripts` (Firefox MV3
has no background service worker; the same module runs as an event page), `browser_specific_settings.gecko` (id
`parla@parla.cc`, min 140 / `gecko_android` min 142: the first versions with `data_collection_permissions` (AMO warns
on anything lower; MAIN-world content scripts and `scripting.executeScript({world})` alone would need 128);
`data_collection_permissions: none`), `developer`, no `minimum_chrome_version` / `use_dynamic_url`. `web-ext lint`: 0 errors / warnings. Firefox 158
(2026-10-27, `tests/firefox.ts`): page mode, fake-mic tone 0.0707 -> 0.000045 RMS (-64 dB). Load it by hand:
about:debugging -> This Firefox -> Load Temporary Add-on -> `build/extension-firefox/manifest.json` (gone after a
restart until it is signed on addons.mozilla.org). Not yet checked in Firefox: isolated mode (strict-CSP sites).
Load unpacked in Chrome/Edge: chrome://extensions -> Developer mode -> Load unpacked -> `build/extension`. Tests use
Playwright's bundled Chromium (branded Chrome ignores `--load-extension` since v137) with a generated fake microphone
(noise + hum, no voice): suppression on removes ~39 dB of it in page and isolated mode, off passes it through.

## Updates and reloads (never break a call)

- The page script keeps the site's track for the whole call: real mic -> [engine] -> wet gain + dry gain ->
  destination. When the extension goes away (update / reload) the bridge says `lost` (its session port to
  the service worker disconnects); page mode keeps running, isolated mode switches to the plain mic (dry)
  and the pump itself plays the mic after ~21 ms without replies. The new version re-injects a bridge;
  its hello replaces the port (messages from a replaced port are ignored) and dead engines are rebuilt,
  then crossfaded in once their output has started (no gap the length of their latency).
- Tests reload through CDP `Extensions.loadUnpacked` (`--enable-unsafe-extension-debugging`): an
  unpacked extension calling `chrome.runtime.reload()` itself under automation is unloaded ~3 s after
  start; the update-hold test swaps `viUpdate.reload`.
- Permissions are frozen (`storage`, `scripting` + host access): a new warning-bearing permission in an
  update disables the extension until the user accepts.

## Rules / gotchas

- The same microphone opened twice in a page (Meet: preview + call; same deviceId / echo cancellation / AGC) shares
  one session (refcounted track clones): processed once.
- End-user UI names quality levels, never models: "Best quality · full band" (deepfilternet3), "Lightest · wideband"
  (gtcrn), "Light · full band" (rnnoise); mapping in `popup.ts` `QUALITY`. Package ids only in the policy schema and
  console logs.
- `chrome.storage.managed.get` takes ~2 s on its first call: the worker caches the policy (memory + storage.local) and
  the bridge prefetches the config at document_start; the page waits up to 6 s once.
- Level tests request the mic with echo cancellation / AGC off: Chrome's adaptive gain would change the levels. Real
  calls keep the site's settings.
- Web-accessible resources (`engine/*`, `models/*`, the pump and frame) let any page detect the extension; accepted
  (no secrets in them).

## Not done yet

Real-site verification by hand (Meet, Teams, Zoom web; in Firefox too), a virtual "Parla" entry in the device list,
Firefox isolated mode check, store listings (Chrome Web Store, addons.mozilla.org: AMO reviews the engine source; it is
public) + privacy disclosure (nothing collected).

**Model manifests in the package**: `models/<name>/manifest.json` ships as `models/<name>/model.json` (the Chrome Web
Store rejects a zip with more than one `manifest.json`, 2026-11-05); `inject.ts` fetches `model.json`; `build.ts` fails
the build if the package holds anything but exactly one `manifest.json`.

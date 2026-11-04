# Parla documentation

Parla removes background noise and other voices from a microphone, live, on the user's device. Use it as a
**browser extension** (any web calling app, no code) or as a **library** in your own web app.

| | |
|---|---|
| [Getting started](getting-started.md) | install `@tvildo/parla`, clean a microphone in ten lines |
| [Features](features.md) | quality levels, speech detection, strength, gain, device changes, latency, what happens on errors |
| [React](react.md) | the `useParla` hook, `ParlaProvider`, a call example |
| [npm package and assets](npm-and-assets.md) | how the engine files reach the browser (bundlers, no bundler, CDN), CSP, caching |
| [API reference](api.md) | `Parla`, `ParlaStream`, `ParlaError`, React types |
| [Browser extension](extension.md) | install, settings, languages, enterprise policy |
| [Telemetry](telemetry.md) | what the extension's reports contain, the two switches, the receiver, retention |
| [Accessibility](accessibility.md) | what the website and the extension support, how it is tested |
| [Development](development.md) | **for contributors**: project layout, building every part, tests, translations, models |
| [Releasing](releasing.md) | **for maintainers**: the GitHub workflow, version tags, publishing to the Chrome and Firefox stores |

Examples you can run: [`examples/react-vite`](../examples/react-vite) (Vite + React) and
[`examples/vanilla`](../examples/vanilla) (one HTML file, no build step).

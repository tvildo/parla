# API reference

Types ship with the package (`dist/parla.d.ts`, `dist/react.d.ts`).

## `@tvildo/parla`

### `Parla.load(options?) → Promise<Parla>`

Fetches and compiles the engine. Model files are fetched on the first `process()` (or `preload()`).

| Option | Default | |
|---|---|---|
| `model` | `'deepfilternet3'` | `'rnnoise'`, `'gtcrn'` ([features](features.md)) |
| `base` | the package's own files | URL of a folder with `engine/` and `models/` ([assets](npm-and-assets.md)) |

Throws `ParlaError` (`unsupported`, `load`).

### `parla.process(track, options?) → Promise<ParlaStream>`

A cleaned copy of a microphone track. Call it after a user gesture.

| Option | Default | |
|---|---|---|
| `enabled` | `true` | start with noise suppression on or off |
| `strength` | `1` | 0..1 |
| `gain` | `1` | input level |

Throws `ParlaError` (`engine`, `load`). The input track is never stopped by Parla.

### `parla.preload() → Promise<void>`

Downloads the model now, so the first `process()` doesn't wait for it.

### `parla.close()`

Stops every stream and closes the AudioContext. A closed Parla refuses `process()` (also one still starting); load a
new one.

### `ParlaStream`

| Member | |
|---|---|
| `track` | the cleaned `MediaStreamTrack` |
| `input` | the current input track |
| `latencyMs` | the engine's delay |
| `enabled`, `strength` | current settings |
| `setEnabled(on)` | on/off, ramped, same timing |
| `setStrength(0..1)` | cleaned share of the output, ramped |
| `setGain(x)` | input level, ramped |
| `setInput(track)` | another microphone, same output track |
| `onVoice` | `({ speaking, prob, levelDb }) => void`, ~19 times a second ([speech detection](features.md#speech-detection)) |
| `onError` | `(error: ParlaError) => void`: the engine stopped mid-call (the output then carries the plain input) or the input track ended. Switch back to your own track |
| `error` | the `ParlaError` once `onError` fired, else `null` |
| `stop()` | releases the engine and the output track |

### `ParlaError`

`error.code`: `'unsupported'` (no WebAssembly / AudioWorklet, or blocked by the page's CSP), `'load'` (a file could not
be fetched, or the server answered something else: a wrong `base`), `'engine'` (no user gesture yet, or the engine failed). `error.message` has the details.

### `BUILD`

The WebAssembly build this browser runs: `'relaxed'`, `'simd'`, `'nosimd'`, or `null` without WebAssembly.

### `VERSION`

The package version.

## `@tvildo/parla/react`

### `useParla(track, options?) → { track, ready, error, speaking, levelDb, latencyMs }`

See [React](react.md). Options: `enabled`, `strength`, `gain`, `model`, `base`. If the engine stops mid-call, `track`
turns `null` again and `error` says why: send the input track then.

### `<ParlaProvider model? base?>`

Default `model` / `base` for the hooks below it.

### `getParla({ model?, base? }) → Promise<Parla>`

The shared `Parla` the hooks use (one per `model` + `base`).

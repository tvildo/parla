# @tvildo/parla

AI noise suppression for your web app's microphone. Parla removes background noise **and other voices** from a
microphone track, live, entirely on the user's device (WebAssembly in an AudioWorklet). No account, no API key, no
server: the engine and its models ship inside this package.

```sh
npm install @tvildo/parla
```

```js
import { Parla } from '@tvildo/parla';

const mic = (await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } })).getAudioTracks()[0];
const parla = await Parla.load();             // with a bundler the engine files come from this package
const ns = await parla.process(mic);          // call it after a click: browsers start audio only after a user gesture
peerConnection.addTrack(ns.track);            // the cleaned microphone
ns.onVoice = ({ speaking }) => showTalking(speaking);
```

React:

```jsx
import { useParla } from '@tvildo/parla/react';

function Call({ mic }) {
  const { track, ready, speaking } = useParla(mic);   // track: the cleaned microphone once ready
  useEffect(() => { if (ready) sender.replaceTrack(track); }, [ready, track]);
  return <MicIcon active={speaking} />;
}
```

## Features

- **Noise and voice removal**: three quality levels: `deepfilternet3` (best, full band 48 kHz, default),
  `rnnoise` (light, full band), `gtcrn` (lightest, wideband).
- **Speech detection**: `onVoice({ speaking, prob, levelDb })` about 19 times a second, on the cleaned voice: a
  "you are talking" indicator, or a hint when someone talks while muted.
- **Live controls without clicks**: `setEnabled(on)`, `setStrength(0..1)` (keep some of the room), `setGain(x)`.
- **Device changes mid-call**: `setInput(otherMic)` keeps the same output track.
- **Low delay**: ~33 ms with the best model.
- **Fails safe**: if the engine can't start, `process()` throws and your app keeps the plain microphone.
- **Private**: audio never leaves the browser; the package makes no requests except for its own files.

## Engine files

The package contains the engine (~260 KB of wasm per build) and the three models (~9.5 MB, mostly the best model's
weights). Your bundler (Vite, webpack 5, Rollup, esbuild, Parcel) emits them as assets automatically; the browser
downloads only the wasm build it can run and the model in use. Without a bundler, serve `dist/engine` and
`dist/models` yourself and pass `Parla.load({ base: '/path/to/them/' })`. Your page's Content Security Policy needs
`script-src 'wasm-unsafe-eval'`.

## Docs

Getting started, React, features, assets and CSP, API: <https://github.com/tvildo/parla/tree/master/docs>.

Apache-2.0. Model weights keep their own permissive licences (see `NOTICE`). Made by [tvildo](https://github.com/tvildo).

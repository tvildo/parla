# Features

## Noise and voice removal

Parla runs neural networks trained to keep the speaker's voice and remove everything else: keyboard, fans, traffic,
music, a TV, and **other people talking in the background**. Three quality levels, chosen with `model`:

| `model` | Quality | Audio | Cost | Delay |
|---|---|---|---|---|
| `deepfilternet3` (default) | best | full band (48 kHz) | ~6 % of one CPU core | ~33 ms |
| `rnnoise` | light | full band (48 kHz) | ~3 % | ~10 ms (+ buffering) |
| `gtcrn` | lightest | wideband (16 kHz) | ~2 % | ~16 ms (+ buffering) |

```js
const parla = await Parla.load({ model: 'rnnoise' });
```

`ns.latencyMs` reports the actual delay of a running stream. Show users quality names, not model names.

## Speech detection

Every stream reports voice activity on the **cleaned** signal, about 19 times a second:

```js
ns.onVoice = ({ speaking, prob, levelDb }) => {
  micButton.classList.toggle('talking', speaking);   // speaking: true while someone talks
  // prob: speech probability 0..1; levelDb: level of the cleaned voice in dBFS
};
```

Because it runs after noise removal, background noise does not count as speech. It combines a level detector
(adaptive noise floor, 30 ms attack, 250 ms hangover) with the model's own speech estimate where the model has one.
Typical uses: a "you are talking" ring, "you are muted" hints, active-speaker highlighting, push-to-talk helpers.

## Live controls (no clicks, no gaps)

```js
ns.setEnabled(false);   // noise suppression off: the original voice, delayed by the same latency (ramped)
ns.setEnabled(true);
ns.setStrength(0.7);    // 70 % cleaned + 30 % original: keeps a little of the room
ns.setGain(1.5);        // input level before the engine (a microphone sensitivity setting)
```

All changes ramp inside the engine over a few milliseconds and keep the timing identical, so toggling during a call
never clicks or shifts the audio.

## Device changes mid-call

```js
const other = (await navigator.mediaDevices.getUserMedia({ audio: { deviceId, noiseSuppression: false } })).getAudioTracks()[0];
ns.setInput(other);     // the output track (ns.track) stays the same: nothing to renegotiate
```

## Fails safe

- `Parla.load()` and `process()` throw a `ParlaError` with a `code`: `unsupported` (no WebAssembly or AudioWorklet,
  or a Content Security Policy without `'wasm-unsafe-eval'`), `load` (a file could not be fetched), `engine` (no user
  gesture yet, or the engine failed to start). Keep the plain microphone in that case.
- A running stream never goes silent because of Parla.

## Private by design

Audio is processed in the page, in an AudioWorklet, with WebAssembly. Parla sends nothing anywhere and has no
analytics. The only network requests are for its own files (engine and model), which you can serve yourself.

## Performance

- One `AudioContext` (48 kHz) per `Parla`, one engine per processed track. `parla.preload()` downloads the model
  ahead of the first call.
- The engine picks the fastest WebAssembly build the browser supports (relaxed SIMD, SIMD or plain).
- Long frames are spread over several audio callbacks, so the main cost per callback stays low (no audio glitches on
  slower machines).

## Browser support

Tested on desktop: current Chrome / Edge and Firefox 158 (the extension needs Chromium 116+ or Firefox 128+). Needed: WebAssembly and AudioWorklet;
the engine falls back to a build without SIMD where SIMD is missing. Safari and mobile browsers should work (they
have both) but are not tested yet; the best model needs a recent phone.

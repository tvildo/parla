# Getting started

```sh
npm install @tvildo/parla
```

```js
import { Parla } from '@tvildo/parla';

button.onclick = async () => {
  // 1. the microphone, with the browser's own noise suppression off (Parla replaces it)
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } });
  const mic = stream.getAudioTracks()[0];

  // 2. the engine (fetched and compiled once; with a bundler its files come from the package)
  const parla = await Parla.load();

  // 3. a cleaned copy of the microphone
  try {
    const ns = await parla.process(mic);
    peerConnection.addTrack(ns.track, stream);   // send the cleaned track instead of `mic`
    ns.onVoice = ({ speaking }) => talkingIndicator.hidden = !speaking;
  } catch (e) {
    peerConnection.addTrack(mic, stream);        // Parla could not start: the call goes on with the plain microphone
  }
};
```

Three things to know:

- **Start after a click.** Browsers only start audio after a user gesture. `process()` throws a `ParlaError`
  with code `engine` when the page has had no click yet.
- **Turn the browser's noise suppression off** for the track you give Parla (`noiseSuppression: false`). Echo
  cancellation and automatic gain control can stay on.
- **Keep the plain microphone as a fallback.** Every error is a `ParlaError` (codes `unsupported`, `load`,
  `engine`); catch it and use `mic` itself. Parla never makes audio stop.

Next: [features](features.md), [React](react.md), [how the engine files are loaded](npm-and-assets.md).

## Without npm

Copy `node_modules/@tvildo/parla/dist/` (or the unpacked tarball) to your static files and import it directly; the
browser finds the engine files next to it:

```html
<script type="module">
  import { Parla } from '/static/parla/parla.js';
  const parla = await Parla.load();
</script>
```

See [`examples/vanilla`](../examples/vanilla).

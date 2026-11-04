# React

`@tvildo/parla/react` has a hook and an optional provider. React 18 and 19 are supported (StrictMode included).

```sh
npm install @tvildo/parla
```

## `useParla(track, options)`

```jsx
import { useParla } from '@tvildo/parla/react';

function MicControls({ mic }) {               // mic: a MediaStreamTrack from getUserMedia (or null)
  const [enabled, setEnabled] = useState(true);
  const { track, ready, error, speaking, levelDb, latencyMs } = useParla(mic, { enabled, strength: 1 });

  if (error) return <p role="alert">Noise suppression is unavailable; using your plain microphone.</p>;
  if (!ready) return <p role="status">Starting noise suppression…</p>;
  return (
    <>
      <span className={speaking ? 'mic talking' : 'mic'} aria-label={speaking ? 'Speaking' : 'Quiet'} />
      <label><input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> Noise suppression</label>
    </>
  );
}
```

| Returned | |
|---|---|
| `track` | the cleaned microphone (`MediaStreamTrack`), `null` until ready |
| `ready` | the engine runs on this track |
| `error` | a `ParlaError` when it could not start (keep using the input track) |
| `speaking`, `levelDb` | speech detection on the cleaned voice (`levelDb` updates ~5 times a second) |
| `latencyMs` | the engine's delay |

| Option | Default | |
|---|---|---|
| `enabled` | `true` | live on/off, ramped |
| `strength` | `1` | 0..1, how much of the cleaned signal |
| `gain` | `1` | input level before the engine |
| `model` | `deepfilternet3` | `rnnoise`, `gtcrn` (see [features](features.md)) |
| `base` | — | where the engine files are served; not needed with a bundler |

Changing `enabled`, `strength` or `gain` applies live. Changing the input track, `model` or `base` starts a new stream
(the old one is stopped). The stream stops when the component unmounts; the input track stays yours.

## `ParlaProvider`

Options for every hook below it, e.g. a model chosen in your settings:

```jsx
import { ParlaProvider } from '@tvildo/parla/react';

<ParlaProvider model={settings.quality === 'light' ? 'rnnoise' : 'deepfilternet3'}>
  <App />
</ParlaProvider>
```

All hooks with the same `model`/`base` share one engine and one AudioContext.

## A call with WebRTC

```jsx
function Call({ pc }) {
  const [mic, setMic] = useState(null);
  const { track, ready } = useParla(mic);
  const sender = useRef(null);

  async function join() {                       // after a click: audio may start
    const s = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } });
    const t = s.getAudioTracks()[0];
    sender.current = pc.addTrack(t, s);         // start with the plain microphone…
    setMic(t);
  }

  // …and switch to the cleaned one as soon as it runs (no renegotiation)
  useEffect(() => { if (ready && sender.current) sender.current.replaceTrack(track); }, [ready, track]);

  return <button onClick={join}>Join</button>;
}
```

## Vite

Production builds need nothing. For the dev server, keep the package out of the dependency pre-bundle (it finds its
engine files next to itself):

```js
// vite.config.js
export default { optimizeDeps: { exclude: ['@tvildo/parla'] } };
```

## Next.js / server rendering

Parla runs in the browser only. Use the hook in a client component (`'use client'`) and pass it a track you got in
the browser; nothing runs during server rendering.

A complete app: [`examples/react-vite`](../examples/react-vite).

// A minimal call screen: the microphone through Parla, a "speaking" light, on/off and strength, and the cleaned
// voice played back (use headphones). In a real call you would send `track` over your RTCPeerConnection.
import { useEffect, useRef, useState } from 'react';
import { useParla } from '@tvildo/parla/react';

declare global {
  interface Window { parla?: { track: MediaStreamTrack | null; ready: boolean; speaking: boolean } }
}

export function App() {
  const [mic, setMic] = useState<MediaStreamTrack | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [strength, setStrength] = useState(1);
  const { track, ready, error, speaking, levelDb, latencyMs } = useParla(mic, { enabled, strength });
  const audio = useRef<HTMLAudioElement>(null);

  // the cleaned track goes wherever the microphone would: here an <audio> element (and window.parla for tests)
  useEffect(() => {
    window.parla = { track, ready, speaking };
    if (audio.current) audio.current.srcObject = track ? new MediaStream([track]) : null;
  }, [track, ready, speaking]);

  async function start() {
    const s = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false } });
    setMic(s.getAudioTracks()[0]);
  }

  return (
    <main>
      <h1>Parla + React</h1>
      {!mic && <button onClick={start}>Start microphone</button>}
      {mic && !ready && !error && <p role="status">Loading the engine…</p>}
      {error && <p role="alert">Parla could not start ({error.message}); the plain microphone is used.</p>}
      {ready && (
        <>
          <p role="status">Clean · delay {Math.round(latencyMs)} ms · {speaking ? 'speaking' : 'quiet'} · {Math.round(levelDb)} dBFS</p>
          <label><input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> Noise suppression</label>
          <label> Strength <input type="range" min="0" max="1" step="0.05" value={strength} onChange={(e) => setStrength(+e.target.value)} /></label>
        </>
      )}
      <audio ref={audio} autoPlay />
    </main>
  );
}

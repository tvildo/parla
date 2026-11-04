// react.ts — Parla for React (`@tvildo/parla/react`).
//
//   const [mic, setMic] = useState(null);                       // a microphone track from getUserMedia
//   const { track, ready, error, speaking, levelDb, latencyMs } = useParla(mic, { enabled, strength, gain });
//   // send `track` (the cleaned microphone) instead of `mic`; until `ready`, `track` is null: keep sending `mic`
//
//   <ParlaProvider model="deepfilternet3" base="/static/parla/">   // optional: options for every hook below it
//
// One Parla (engine + one AudioContext) per base/model is shared by all hooks; each hook owns one processed stream
// and stops it when the input track changes or the component unmounts (React 18/19 StrictMode-safe: the second
// mount gets its own stream, the first one is stopped).
import { createContext, createElement, useContext, useEffect, useRef, useState } from 'react';
import type { ReactElement, ReactNode } from 'react';
import { Parla } from './parla.ts';
import type { ParlaError, ParlaStream } from './parla.ts';

export interface ParlaOptions {
  /** noise suppression on (default true); changes apply live, ramped (no click) */
  enabled?: boolean;
  /** how much of the cleaned signal, 0..1 (default 1) */
  strength?: number;
  /** input level before the engine (default 1) */
  gain?: number;
  /** "deepfilternet3" (best, full band; default) | "rnnoise" (light, full band) | "gtcrn" (lightest, wideband) */
  model?: string;
  /** where engine/ and models/ are served; omit with a bundler (the package's own files are used) */
  base?: string;
}
export interface ParlaState {
  /** the cleaned microphone; null until ready, and again after an error (keep sending the input track then) */
  track: MediaStreamTrack | null;
  ready: boolean;
  error: ParlaError | Error | null;
  /** speech detected on the cleaned output */
  speaking: boolean;
  /** level of the cleaned output, dBFS (updated ~5 times a second) */
  levelDb: number;
  /** the engine's delay */
  latencyMs: number;
}

const Options = createContext<{ model?: string; base?: string }>({});
const shared = new Map<string, Promise<Parla>>();   // `${base}|${model}` -> Promise<Parla>

/** the shared Parla for these options (one engine + AudioContext per base/model), created once */
export function getParla({ base, model = 'deepfilternet3' }: { base?: string; model?: string } = {}): Promise<Parla> {
  const key = `${base || ''}|${model}`;
  if (!shared.has(key)) {
    const p = Parla.load({ base, model });
    p.catch(() => shared.delete(key));
    shared.set(key, p);
  }
  return shared.get(key)!;
}

/** default options (model, base) for every useParla below it */
export function ParlaProvider({ model, base, children }: { model?: string; base?: string; children?: ReactNode }): ReactElement {
  return createElement(Options.Provider, { value: { model, base } }, children);
}

const IDLE: ParlaState = { track: null, ready: false, error: null, speaking: false, levelDb: -120, latencyMs: 0 };

/** The cleaned copy of a microphone track. */
export function useParla(input: MediaStreamTrack | null, options: ParlaOptions = {}): ParlaState {
  const defaults = useContext(Options);
  const { enabled = true, strength = 1, gain = 1 } = options;
  const model = options.model ?? defaults.model;
  const base = options.base ?? defaults.base;
  const [state, setState] = useState<ParlaState>(IDLE);
  const stream = useRef<ParlaStream | null>(null);

  useEffect(() => {
    if (!input) { setState(IDLE); return undefined; }
    let cancelled = false;
    setState({ ...IDLE });
    getParla({ base, model })
      .then((p) => p.process(input, { enabled, strength, gain }))
      .then((s) => {
        if (cancelled) { s.stop(); return; }
        stream.current = s;
        // the engine stopped mid-call (or the microphone went away): back to the input track, with the reason
        s.onError = (error) => { if (!cancelled) setState({ ...IDLE, error }); };
        let last = 0;
        s.onVoice = ({ speaking, levelDb }) => {
          // speaking changes at once; the level about 5 times a second (no re-render per audio event)
          const now = performance.now();
          setState((st) => (st.speaking !== speaking || now - last > 200 ? ((last = now), { ...st, speaking, levelDb }) : st));
        };
        setState({ ...IDLE, track: s.track, ready: true, latencyMs: s.latencyMs });
      })
      .catch((error) => { if (!cancelled) setState({ ...IDLE, error }); });
    return () => {
      cancelled = true;
      stream.current?.stop();
      stream.current = null;
    };
    // enabled / strength / gain change live (effects below), without a new stream
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, model, base]);

  useEffect(() => { stream.current?.setEnabled(enabled); }, [enabled, state.ready]);
  useEffect(() => { stream.current?.setStrength(strength); }, [strength, state.ready]);
  useEffect(() => { stream.current?.setGain(gain); }, [gain, state.ready]);

  return state;
}

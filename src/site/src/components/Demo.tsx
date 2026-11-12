// The live demo (a React island on /demo/ in every language), one card: one button opens the microphone (the SDK's
// engine on it, live noise meter), records 8 seconds and closes it again; the two takes play back with and without
// Parla. Quality (dropdown) and noise suppression strength apply to the next take (strength also live). Strings come
// from the page (demo.* keys of src/i18n/<lang>.json). Without microphone access the card is replaced by the error and
// a retry (heading, lead and note stay). The same card, frozen mid-recording, is the home page's picture (DemoPreview).
import { useEffect, useRef, useState } from "react";
import { QUALITIES, Session, levelDb, meterScale, type Model } from "./audio";

type Phase = { kind: "idle" } | { kind: "starting" } | { kind: "running"; latencyMs: number } | { kind: "error"; message: string };
type Strings = Record<string, string>;
type T = (key: string, vars?: Record<string, string | number>) => string;

const RECORD_S = 8;

const translator = (s: Strings): T => (key, vars = {}) =>
  String(s[`demo.${key}`] ?? key).replace(/\{(\w+)\}/g, (m, v) => (v in vars ? String(vars[v]) : m));

export default function Demo({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [model, setModel] = useState<Model>("deepfilternet3");
  const [strength, setStrength] = useState(100);
  const [speaking, setSpeaking] = useState(false);
  const session = useRef<Session | null>(null);
  // recording: elapsed seconds while it runs (null otherwise), the two takes when done
  const [rec, setRec] = useState<Rec>({ elapsed: null, urls: null });

  const mounted = useRef(true);
  useEffect(() => () => { mounted.current = false; session.current?.stop(); session.current = null; }, []);
  useEffect(() => () => { if (rec.urls) { URL.revokeObjectURL(rec.urls.original); URL.revokeObjectURL(rec.urls.cleaned); } }, [rec.urls]);

  /** the one button: opens the microphone, records RECORD_S seconds, closes the microphone, shows the two takes */
  async function record() {
    const s = await start();
    if (!s) return;
    setRec({ elapsed: 0, urls: null });
    try {
      const urls = await s.record(RECORD_S, (elapsed) => { if (session.current === s) setRec((r) => ({ ...r, elapsed })); });
      if (mounted.current && session.current === s) setRec({ elapsed: null, urls });
      else { URL.revokeObjectURL(urls.original); URL.revokeObjectURL(urls.cleaned); }
    } catch (e) {
      if (mounted.current && session.current === s) { setRec({ elapsed: null, urls: null }); setPhase({ kind: "error", message: explain(e, t) }); }
    } finally {
      if (session.current === s) { s.stop(); session.current = null; if (mounted.current) setPhase((p) => (p.kind === "running" ? { kind: "idle" } : p)); }
    }
  }

  async function start(): Promise<Session | null> {
    setPhase({ kind: "starting" });
    let s: Session | null = null;
    try {
      s = await Session.start();
      if (!mounted.current) { s.stop(); return null; }   // left the page while the microphone was opening
      session.current = s;
      s.onVoice = setSpeaking;
      s.setStrength(strength / 100);
      const latencyMs = await s.use(model);
      if (!mounted.current) return null;
      setPhase({ kind: "running", latencyMs });
      return s;
    } catch (e) {
      s?.stop();
      session.current = null;
      if (mounted.current) setPhase({ kind: "error", message: explain(e, t) });
      return null;
    }
  }

  function changeStrength(v: number) {
    setStrength(v);
    session.current?.setStrength(v / 100);
  }

  const running = phase.kind === "running";
  const sess = session.current;

  return (
    <main className="demo">
      <h1>{t("h1a")} <span className="grad">{t("h1b")}</span></h1>
      <p className="lead">{t("lead")}</p>

      <Card t={t} phase={phase} model={model} strength={strength} speaking={speaking} rec={rec}
        inNode={running ? sess?.inMeter ?? null : null} outNode={running ? sess?.outMeter ?? null : null}
        onRecord={record} onModel={setModel} onStrength={changeStrength} />

      <p className="note">{t("note")}</p>
    </main>
  );
}

type Rec = { elapsed: number | null; urls: { original: string; cleaned: string } | null };
interface CardProps {
  t: T; phase: Phase; model: Model; strength: number; speaking: boolean; rec: Rec;
  inNode: AnalyserNode | null; outNode: AnalyserNode | null;
  onRecord: () => void; onModel: (m: Model) => void; onStrength: (v: number) => void;
  /** a fixed meter (the home page's preview, rendered without JavaScript) */
  still?: Still;
}

/** the demo card: what the live demo shows, and (with fixed state) the home page's preview of it */
function Card({ t, phase, model, strength, speaking, rec, inNode, outNode, onRecord, onModel, onStrength, still }: CardProps) {
  const running = phase.kind === "running";
  // no microphone (no permission, no device, an unsupported browser): the card is only the reason and a retry
  if (phase.kind === "error") {
    return (
      <section className="card demo-card demo-one demo-error" role="alert">
        <p>{phase.message}</p>
        <div className="actions">
          <button className="btn primary" onClick={onRecord}>{t("retry")}</button>
        </div>
      </section>
    );
  }
  return (
    <section className="card demo-card demo-one" aria-labelledby="demo-mic">
      <h2 id="demo-mic">{t("yourMic")}</h2>
      <p className="sr-only" role="status">{running ? t("started") : phase.kind === "starting" ? t("starting") : ""}</p>
      <NoiseMeter inNode={inNode} outNode={outNode} speaking={running && speaking} still={still} t={t} />

      <div className="controls">
        <div className="field">
          <label htmlFor="demo-quality">{t("quality")}</label>
          <select id="demo-quality" value={model} disabled={phase.kind === "starting" || rec.elapsed !== null} onChange={(e) => onModel(e.target.value as Model)}>
            {QUALITIES.map((q) => <option key={q.model} value={q.model}>{t(`q.${q.label}`)} · {t(`q.${q.detail}`)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="demo-strength">{t("ns")} <span className="muted">{strength}%</span></label>
          <input id="demo-strength" type="range" min={0} max={100} step={5} value={strength}
            aria-valuetext={`${strength}%`} onChange={(e) => onStrength(+e.target.value)} />
        </div>
      </div>

      <div className="row">
        <div className="actions left">
          <button className="btn primary" onClick={onRecord}
            disabled={phase.kind === "starting" || rec.elapsed !== null}>
            <span className="rec-dot" aria-hidden="true" />
            {phase.kind === "starting" ? t("starting") : rec.urls ? t("recordAgain") : t("record", { s: RECORD_S })}
          </button>
        </div>
        {running && <span className="muted small">{t("delay", { ms: Math.round(phase.latencyMs) })}</span>}
      </div>

      {rec.elapsed !== null && <Progress elapsed={rec.elapsed} t={t} />}
      {rec.elapsed === null && rec.urls && <Players urls={rec.urls} t={t} />}
    </section>
  );
}

/**
 * The home page's picture of the demo: the real card, mid-recording, rendered to static HTML at build time (so it
 * follows every change to the card, in every language). Not interactive: the page marks it inert.
 */
export function DemoPreview({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const none = () => {};
  return (
    <Card t={t} phase={{ kind: "running", latencyMs: 40 }} model="deepfilternet3" strength={100} speaking
      rec={{ elapsed: 5, urls: null }} inNode={null} outNode={null} onRecord={none} onModel={none} onStrength={none}
      still={{ in: 0.8, out: 0.22, text: t("quieter", { n: 6 }) }} />
  );
}

/**
 * One bar, two layers: the original microphone (striped) and what the call hears with Parla (solid); the gap is the
 * noise removed. The number is measured in pauses only (no speech for 0.3 s: everything removed there is noise) and
 * shown as loudness: 10 dB less sounds half as loud, so "N× quieter" = 2^(dB/10), capped at 40 dB (16×: beyond that it
 * is silence to the ear, and noise-only input would read thousands). Same meter as the extension's popup.
 */
const MAX_DB = 40;
type Still = { in: number; out: number; text: string };
function NoiseMeter({ inNode, outNode, speaking, still, t }: { inNode: AnalyserNode | null; outNode: AnalyserNode | null; speaking: boolean; still?: Still; t: T }) {
  const inBar = useRef<HTMLDivElement>(null);
  const outBar = useRef<HTMLDivElement>(null);
  const [text, setText] = useState(still?.text ?? "—");
  const [db, setDb] = useState<number | null>(null);
  const talk = useRef({ speaking, since: 0 });
  talk.current.speaking = speaking;
  if (speaking) talk.current.since = performance.now();
  useEffect(() => {
    const bars = [inBar.current, outBar.current];
    if (!inNode || !outNode) {
      for (const b of bars) if (b) { b.style.transform = "scaleX(0)"; b.setAttribute("aria-valuenow", "0"); }
      setText("—"); setDb(null);
      return;
    }
    const bi = new Float32Array(inNode.fftSize), bo = new Float32Array(outNode.fftSize);
    let raf = 0, n = 0, cut: number | null = null, floor: number | null = null, shown = "";
    const tick = () => {
      const i = levelDb(inNode, bi), o = levelDb(outNode, bo);
      const vi = meterScale(i), vo = Math.min(vi, meterScale(o));
      inBar.current!.style.transform = `scaleX(${vi.toFixed(3)})`;
      outBar.current!.style.transform = `scaleX(${vo.toFixed(3)})`;
      if (!talk.current.speaking && performance.now() - talk.current.since > 300) {
        floor = floor === null ? i : floor + 0.03 * (i - floor);
        const d = Math.max(0, i - o);
        cut = cut === null ? d : cut + 0.03 * (d - cut);
      }
      // text and values for assistive technology a few times a second, not every frame
      if (++n % 15 === 0) {
        inBar.current!.setAttribute("aria-valuenow", String(Math.round(vi * 100)));
        outBar.current!.setAttribute("aria-valuenow", String(Math.round(vo * 100)));
        const x = cut === null ? 0 : 2 ** (Math.min(cut, MAX_DB) / 10);
        const next = floor !== null && floor < -55 ? t("quiet") : x >= 1.2 ? t("quieter", { n: x < 10 ? x.toFixed(1).replace(/\.0$/, "") : Math.round(x) }) : "—";
        if (next !== shown) { shown = next; setText(next); }
        setDb(cut === null ? null : Math.round(cut));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inNode, outNode]);
  return (
    <div className="nmeter">
      <div className="nmeter-head">
        <span className="meter-label">{t("noise")}{speaking ? <span className="dot" title={t("speaking")} /> : null}</span>
        <span className="nmeter-value" title={db !== null ? `−${db} dB` : undefined}>{text}</span>
      </div>
      <div className="nmeter-track">
        <div className="nmeter-bar in" ref={inBar} style={still && { transform: `scaleX(${still.in})` }} role="meter" aria-label={t("original")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={0} />
        <div className="nmeter-bar out" ref={outBar} style={still && { transform: `scaleX(${still.out})` }} role="meter" aria-label={t("withParla")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={0} />
      </div>
      <div className="nmeter-legend" aria-hidden="true">
        <span><i className="key in" />{t("original")}</span>
        <span><i className="key out" />{t("withParla")}</span>
      </div>
    </div>
  );
}

function Progress({ elapsed, t }: { elapsed: number; t: T }) {
  const left = Math.max(0, Math.ceil(RECORD_S - elapsed));
  return (
    <div className="recorder">
      <div className="progress" role="progressbar" aria-label={t("recording", { s: RECORD_S })}
        aria-valuemin={0} aria-valuemax={RECORD_S} aria-valuenow={Math.round(elapsed)} aria-valuetext={t("left", { s: left })}>
        <div className="progress-bar" style={{ transform: `scaleX(${(elapsed / RECORD_S).toFixed(3)})` }} />
      </div>
      <p className="muted small" aria-hidden="true">{t("left", { s: left })}</p>
    </div>
  );
}

/** the two takes: with Parla first, the original second (seekable players) */
function Players({ urls, t }: { urls: { original: string; cleaned: string }; t: T }) {
  return (
    <div className="players">
      <figure>
        <figcaption>{t("withParla")}</figcaption>
        <audio controls src={urls.cleaned} preload="auto" />
      </figure>
      <figure>
        <figcaption>{t("original")}</figcaption>
        <audio controls src={urls.original} preload="auto" />
      </figure>
    </div>
  );
}

function explain(e: unknown, t: T): string {
  const err = e as { name?: string; code?: string; message?: string };
  if (err?.name === "NotAllowedError" || err?.name === "SecurityError") return t("err.blocked");
  if (err?.name === "NotFoundError" || err?.name === "OverconstrainedError") return t("err.noMic");
  if (err?.code === "unsupported") return t("err.unsupported");
  return t("err.other", { message: err?.message ?? String(e) });
}

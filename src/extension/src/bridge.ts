// bridge.ts — content script in the extension's ISOLATED world: the link between the page script
// (inject.ts, MAIN world, which wraps getUserMedia) and the extension.
//
// Why it exists: the page script runs under the PAGE's rules (its CSP, no chrome.* APIs). This script runs
// under the extension's: it reads settings from the service worker, fetches the engine and
// model files from the extension package (page CSP never applies), creates the engine frame for the
// isolated mode, and passes status back for the toolbar badge and popup.
//
// Channel: a MessageChannel; port2 goes to the page script with a 'vi-bridge-hello' window message that
// inject.ts captures first (it listens from document_start and stops the event).
//   page -> {id, type:'config'}                      -> {id, result: config + base URL}
//   page -> {id, type:'fetch', path, as:'text'|'buffer'} -> {id, result} (buffer transferred)
//   page -> {id, type:'frame', sampleRate} + [ctlPort, audioPort] -> engine frame + worker, ports handed over
//   page -> {type:'status', ...} | {type:'telemetry', ev}   (a report: the service worker checks every field)
//   page -> {type:'meter', inDb, outDb, speaking}    live levels, forwarded to open popups ('meter' ports)
//   bridge -> {type:'config', config}                pushed when settings change
//   bridge -> {type:'meter', on}                     a popup's meter opened (on) / the last one closed
//   bridge -> {type:'lost'}                          the extension was updated / reloaded: this bridge is
//                                                    dead; the new version injects a new bridge (new hello)
import type { MeterSample, PageConfig, PageToBridge, TabStatus } from './messages.ts';

(() => {
  const w = window as Window & { __viBridge?: boolean };
  if (w.__viBridge) return;
  w.__viBridge = true;
  const ch = new MessageChannel();
  const port = ch.port1;
  const origin = location.origin !== 'null' ? location.origin : (document.referrer ? new URL(document.referrer).origin : '');
  const base = chrome.runtime.getURL('');
  let frame: HTMLIFrameElement | null = null, frameReady: Promise<HTMLIFrameElement> | null = null;
  // everything here answers the page script, which the page itself can replace: treat each message as the page's
  const FETCHABLE = /^(engine|models)\/[\w./-]+$|^pump_worklet\.js$/;

  const config = async (): Promise<PageConfig> => ({ ...(await chrome.runtime.sendMessage({ type: 'config', origin })), base });
  // asked for now, before the site asks for the microphone (wakes the service worker early)
  let first: Promise<PageConfig> | null = config();

  function engineFrame(): Promise<HTMLIFrameElement> {
    // a frame the page removed (an app re-rendering its body) is made again; a failed one is retried next time
    if (frameReady && frame?.isConnected) return frameReady;
    frame?.remove();
    const ready: Promise<HTMLIFrameElement> = new Promise((res, rej) => {
      const f = (frame = document.createElement('iframe'));
      f.src = chrome.runtime.getURL('engine_frame.html');
      f.allow = 'autoplay';
      f.setAttribute('aria-hidden', 'true');
      f.tabIndex = -1;
      f.style.cssText = 'position:fixed;width:0;height:0;border:0;opacity:0;pointer-events:none;';
      const t = setTimeout(() => rej(new Error('the site blocked the engine frame')), 5000);
      f.addEventListener('load', () => { clearTimeout(t); res(f); }, { once: true });
      (document.body || document.documentElement).appendChild(f);
    });
    frameReady = ready;
    ready.catch(() => { if (frameReady === ready) frameReady = null; });
    return ready;
  }

  // anything may arrive here (the page can replace the page script): every field is checked before use
  port.onmessage = async (e: MessageEvent<PageToBridge>) => {
    const m = (e.data || {}) as PageToBridge & { id?: number };
    const reply = (result: unknown, transfer: Transferable[] = []) => port.postMessage({ id: m.id, result }, transfer);
    const fail = (err: unknown) => port.postMessage({ id: m.id, error: String((err as Error | null)?.message || err) });
    try {
      if (m.type === 'config') { const c = first ? await first.catch(() => config()) : await config(); first = null; reply(c); }
      else if (m.type === 'fetch') {
        // the page asks for it, so only the engine's own files (getURL passes a full address through unchanged)
        if (typeof m.path !== 'string' || !FETCHABLE.test(m.path) || m.path.includes('..')) throw new Error('not an engine file');
        const r = await fetch(chrome.runtime.getURL(m.path));
        if (!r.ok) throw new Error(`${m.path}: ${r.status}`);
        if (m.as === 'text') reply(await r.text());
        else { const b = await r.arrayBuffer(); reply(b, [b]); }
      } else if (m.type === 'frame') {
        const f = await engineFrame();
        f.contentWindow!.postMessage({ type: 'vi-engine', sampleRate: m.sampleRate, build: m.build }, new URL(base).origin, [...e.ports]);
        reply(true);
      } else if (m.type === 'status') { sessionCount((m as TabStatus).active || 0); chrome.runtime.sendMessage({ ...m, origin }).catch(() => {}); }
      else if (m.type === 'telemetry') chrome.runtime.sendMessage({ type: 'telemetry', ev: m.ev }).catch(() => {});
      else if (m.type === 'meter' && meters.size) {
        const db = (x: unknown) => (Number.isFinite(+(x as number)) ? Math.max(-120, Math.min(0, +(x as number))) : -120);
        const s: MeterSample = { inDb: db(m.inDb), outDb: db(m.outDb), speaking: m.speaking === true };
        for (const p of meters) p.postMessage(s);
      }
    } catch (err) {
      fail(err);
    }
  };

  // the popup's live meter (popup.ts): a 'meter' port from the extension while the popup is open
  const meters = new Set<chrome.runtime.Port>();
  chrome.runtime.onConnect.addListener((p) => {
    if (p.name !== 'meter' || p.sender?.id !== chrome.runtime.id) return;
    meters.add(p);
    port.postMessage({ type: 'meter', on: true });
    p.onDisconnect.addListener(() => { meters.delete(p); if (!meters.size) port.postMessage({ type: 'meter', on: false }); });
  });

  // settings changed (popup, options, policy): push the new config to the page
  chrome.storage.onChanged.addListener(async (changes, area) => {
    if (area === 'local') return;
    try { port.postMessage({ type: 'config', config: await config() }); } catch { /* extension reloaded */ }
  });

  // While this page has a live microphone session, an open port to the service worker tells us when the
  // extension goes away (update / reload / disable): it disconnects and chrome.runtime.id is gone. A plain
  // worker restart (idle timeout) also disconnects it: reconnect. No port without a session (no wake-ups).
  let sw: chrome.runtime.Port | null = null, live = false;
  function watch() {
    if (!live || sw) return;
    try { sw = chrome.runtime.connect({ name: 'session' }); } catch { return port.postMessage({ type: 'lost' }); }
    sw.onDisconnect.addListener(() => {
      sw = null;
      if (!chrome.runtime || !chrome.runtime.id) port.postMessage({ type: 'lost' });
      else setTimeout(watch, 1000);
    });
  }
  function sessionCount(n: number) {
    live = n > 0;
    if (live) watch();
    else if (sw) { try { sw.disconnect(); } catch { /* gone */ } sw = null; }
  }

  window.postMessage({ type: 'vi-bridge-hello' }, '*', [ch.port2]);
})();

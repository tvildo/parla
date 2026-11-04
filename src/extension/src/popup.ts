// popup.ts — the toolbar popup: this site on/off, model, strength, live status and noise meter of the current tab
// (the default for all sites is in the options page).
// Settings go to the service worker (which refuses what the organization's policy sets); running streams
// pick changes up live (bridge.ts pushes the new config to the page).
import { loadI18n, applyI18n } from './i18n.ts';
import type { Translate } from './i18n.ts';
import type { ToBackground, TabStatusReply, SetReply, MeterSample } from './messages.ts';

// the page's controls (inputs and selects: value / checked / disabled)
const $ = (id: string) => document.getElementById(id) as HTMLInputElement;
function send(m: { type: 'tabStatus'; tabId: number }): Promise<TabStatusReply>;
function send(m: Extract<ToBackground, { type: 'set' }>): Promise<SetReply>;
function send(m: ToBackground): Promise<unknown> { return chrome.runtime.sendMessage(m); }
// live regions re-announce on every write: only write when the text changes (the popup refreshes every 0.7 s)
const say = (id: string, text: string) => { if ($(id).textContent !== text) $(id).textContent = text; };
// the UI names quality levels, never the models behind them
const QUALITY: Record<string, string> = { deepfilternet3: 'ext.qs.best', gtcrn: 'ext.qs.lightest', rnnoise: 'ext.qs.light' };

const first = (await send({ type: 'tabStatus', tabId: -1 })).settings.lang || 'auto';
let t: Translate & { setting?: string } = await loadI18n(first);
t.setting = first;
applyI18n(document, t);

// the active tab; popup.html?tab=<id> shows another tab (opened as a page: previews, screenshots, tests)
const forced = Number(new URLSearchParams(location.search).get('tab') ?? 0);
const [tab] = forced ? [await chrome.tabs.get(forced)] : await chrome.tabs.query({ active: true, currentWindow: true });
let origin = '';
try { origin = new URL(tab.url!).origin; } catch { /* chrome:// etc. */ }
const usable = /^https:|^http:\/\/(localhost|127\.0\.0\.1)(:|$)/.test(origin);
$('site').textContent = usable ? new URL(origin).host : t('ext.notAvailable');

// a refusal from the service worker (a setting the organization's policy sets): shown until the next change
let notice = '';

// ---- the live meter: microphone vs what the call hears (~19 samples a second over a 'meter' port to the tab) ----
// The number is measured in pauses only (no speech for ~0.3 s): there everything removed is noise. It is shown as
// loudness, "N× quieter" (rule of thumb: 10 dB less sounds half as loud), capped at 40 dB = 16× (beyond that it is
// silence to the ear; the title keeps the measured dB).
const meter = (() => {
  const LO = -70, HI = -10;                                  // the bar's dBFS range
  const pct = (db: number) => `${(100 * Math.min(1, Math.max(0, (db - LO) / (HI - LO)))).toFixed(1)}%`;
  let p: chrome.runtime.Port | null = null, target = { inDb: -120, outDb: -120 }, shown = { inDb: -120, outDb: -120 };
  let pause = 0, cut: number | null = null, floor: number | null = null, raf = 0;
  const ease = (from: number, to: number) => from + (to - from) * (to > from ? 0.45 : 0.12);   // fast up, slow down
  function frame() {
    shown = { inDb: ease(shown.inDb, target.inDb), outDb: ease(shown.outDb, target.outDb) };
    $('barIn').style.width = pct(shown.inDb);
    $('barOut').style.width = pct(Math.min(shown.outDb, shown.inDb));
    raf = requestAnimationFrame(frame);
  }
  function sample(s: MeterSample) {
    target = s;
    pause = s.speaking ? 0 : pause + 1;
    if (pause >= 6) {                                         // ~0.3 s without speech: a pure-noise window
      floor = floor === null ? s.inDb : floor + 0.1 * (s.inDb - floor);
      const d = Math.max(0, s.inDb - s.outDb);
      cut = cut === null ? d : cut + 0.1 * (d - cut);
    }
    let text = '—';
    if (floor !== null && floor < -65) text = t('ext.meter.quiet');
    else if (cut !== null && 2 ** (cut / 10) >= 1.2) { const n = 2 ** (Math.min(cut, 40) / 10); text = t('ext.meter.quieter', { n: n < 10 ? n.toFixed(1).replace(/\.0$/, '') : String(Math.round(n)) }); }
    say('meterV', text);
    $('meterV').title = cut !== null ? `−${Math.round(cut)} dB` : '';
  }
  return {
    connect() {
      if (p || !tab.id) return;
      try { p = chrome.tabs.connect(tab.id, { name: 'meter' }); } catch { p = null; return; }
      p.onMessage.addListener(sample);
      p.onDisconnect.addListener(() => { p = null; });
      if (!raf) raf = requestAnimationFrame(frame);
    },
    reset() {
      if (p) { try { p.disconnect(); } catch { /* gone */ } p = null; }
      cancelAnimationFrame(raf); raf = 0; pause = 0; cut = floor = null;
      target = shown = { inDb: -120, outDb: -120 };
    },
  };
})();

async function refresh() {
  const r = await send({ type: 'tabStatus', tabId: tab.id! });
  const s = r.settings, st = r.status, policy = new Set(r.policy);
  if ((s.lang || 'auto') !== t.setting) { t = await loadI18n(s.lang); t.setting = s.lang || 'auto'; applyI18n(document, t); }
  const siteSetting = s.sites[origin];
  const siteOn = s.lockEnabled && policy.has('lockEnabled') ? true : siteSetting ? siteSetting === 'on' : s.enabled && !s.onlyListedSites;
  $('siteOn').checked = siteOn;
  $('siteOn').disabled = !usable || policy.has('lockEnabled') && !!s.lockEnabled;
  $('model').value = s.model;
  $('model').disabled = policy.has('model');
  // not while the user is dragging it (the refresh would pull the thumb back)
  if (document.activeElement !== $('strength')) {
    $('strength').value = String(s.strength);
    $('strengthV').textContent = `${s.strength}%`;
    $('strength').setAttribute('aria-valuetext', `${s.strength}%`);
  }
  $('strength').disabled = policy.has('strength');
  $('managed').hidden = r.policy.length === 0;

  const on = !!st && (st.active ?? 0) > 0;
  $('meter').hidden = !on;
  if (on) meter.connect(); else meter.reset();
  $('mic').classList.toggle('live', !!(on && st.speaking));
  $('mic').classList.toggle('on', !!on);
  if (!usable) say('state', '');
  else if (on) {
    const mode = t(st!.mode === 'isolated' ? 'ext.modeIsolated' : 'ext.modePage');
    say('state', t('ext.active', { n: st.active!, quality: t(QUALITY[st!.model ?? ''] ?? 'ext.qs.custom'), mode }) +
      (st!.latencyMs ? t('ext.delay', { ms: Math.round(st!.latencyMs) }) : '') + (st!.enabled === false ? t('ext.passingThrough') : ''));
  } else say('state', t(siteOn ? 'ext.startsReload' : 'ext.offForSite'));
  const err = notice || (st && st.error) || '';
  $('error').hidden = !err;
  say('error', err);

}

$('siteOn').addEventListener('change', async () => {
  const r = await send({ type: 'set', origin, value: $('siteOn').checked ? 'on' : 'off' });
  notice = r && 'error' in r ? r.error : '';
  refresh();
});
$('model').addEventListener('change', () => send({ type: 'set', key: 'model', value: $('model').value }).then(refresh));
$('strength').addEventListener('input', () => { const v = $('strength').value; $('strengthV').textContent = `${v}%`; $('strength').setAttribute('aria-valuetext', `${v}%`); });
$('strength').addEventListener('change', () => send({ type: 'set', key: 'strength', value: +$('strength').value }).then(refresh));

await refresh();
setInterval(refresh, 700);

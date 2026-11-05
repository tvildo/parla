// background.ts — the extension's service worker: effective settings and per-tab status.
//
// Settings: defaults < the user's choices (chrome.storage.sync, popup / options) < enterprise policy
// (chrome.storage.managed, managed_schema.json; IT pushes the extension + its config with Chrome/Edge
// policy). Policy values win and the popup shows them locked.
//
// No account, no server: the engine and the models ship inside the extension and run on this device. Reports
// (telemetry.ts): off by default; anonymous technical reports and detailed diagnostics only when the user turns them on.
import * as telemetry from './telemetry.ts';
import { getLocal } from './messages.ts';
import type { Settings, PageConfig, TabStatus, ToBackground } from './messages.ts';
//
// Messages (runtime.sendMessage):
//   bridge -> {type:'config', origin}            -> effective config for that page
//   bridge -> {type:'status', ...}               per-tab status (badge, popup)
//   popup  -> {type:'tabStatus', tabId}          -> {status, settings, policy}
//   popup  -> {type:'set', key, value, origin?}  user setting (refused when the policy sets it)
//   bridge -> {type:'telemetry', ev}             a report from the page (call summary, error); telemetry.ts decides
//   bridge port 'session'                        open while the tab has a live microphone session
//
// Updates never break a call: while any tab has a live session, a downloaded update waits
// (onUpdateAvailable; Chrome holds it while we listen) and is applied when the last session ends. After an
// install or update the content scripts are injected into the tabs that are already open, so a page that
// was in a call when the update landed gets its engine back (inject.ts revives it) without a reload.

const DEFAULTS: Settings = { enabled: true, model: 'deepfilternet3', strength: 100, sites: {}, onlyListedSites: false, mode: 'auto', lang: 'auto', telemetry: false, diagnostics: false };   // reports: off until the user turns them on
const status = new Map<number, TabStatus>();   // tabId -> latest status from the page
// read by the test suite (tests/extension.spec.ts)
const hooks = globalThis as unknown as { viStatus: typeof status; viUpdate: unknown };
hooks.viStatus = status;

// Policy: chrome.storage.managed.get takes ~2 s on its first call (measured), too slow for a page waiting
// on getUserMedia. Read it once at start, keep it in memory, refresh on change; the last known policy is
// also kept in storage.local so a restarted worker answers at once.
let managedNow: Partial<Settings> | null = null;
const managedFirst = refreshManaged();
async function refreshManaged(): Promise<Partial<Settings>> {
  // Safari has no managed storage (no enterprise policy there): no policy
  const m = (chrome.storage.managed ? await chrome.storage.managed.get(null).catch(() => ({})) : {}) as Partial<Settings>;
  managedNow = m;
  await chrome.storage.local.set({ managedCache: m });
  return m;
}
async function managedPolicy(): Promise<Partial<Settings>> {
  if (managedNow) return managedNow;
  const { managedCache } = await getLocal('managedCache');
  return managedCache || managedFirst;
}
chrome.storage.onChanged.addListener((_, area) => { if (area === 'managed') refreshManaged(); });

async function settings(): Promise<{ merged: Settings; managed: Partial<Settings> }> {
  const [user, managed] = await Promise.all([chrome.storage.sync.get(null) as Promise<Partial<Settings>>, managedPolicy()]);
  return { merged: { ...DEFAULTS, ...user, ...managed, sites: { ...(user.sites || {}), ...(managed.sites || {}) } }, managed };
}

function siteEnabled(s: Settings, origin: string, managed: Partial<Settings> = {}): boolean {
  if (managed.lockEnabled) return managed.sites?.[origin] !== 'off' && managed.enabled !== false;
  const v = s.sites[origin];
  if (v === 'on') return true;
  if (v === 'off') return false;
  return s.enabled && !s.onlyListedSites;
}

async function config(origin: string): Promise<Omit<PageConfig, 'base'>> {
  const { merged: s, managed } = await settings();
  telemetry.active(s).catch(() => {});   // first page with a microphone today (a worker can live past midnight)
  return {
    enabled: siteEnabled(s, origin, managed),
    model: s.model,
    strength: Math.max(0, Math.min(100, +s.strength)) / 100,
    mode: s.mode,
    telemetry: s.telemetry === true,
    version: chrome.runtime.getManifest().version,
  };
}

function badge(tabId: number, st: TabStatus | null) {
  const on = !!st && (st.active ?? 0) > 0;
  chrome.action.setBadgeText?.({ tabId, text: st && st.error ? '!' : on ? 'ON' : '' })?.catch(() => {});
  chrome.action.setBadgeBackgroundColor?.({ tabId, color: st && st.error ? '#f59e0b' : '#22c55e' })?.catch(() => {});
}

// What pages can send. bridge.ts relays the page script's messages, and the page script runs in the page's own
// world: a hostile page can replace it and send anything the bridge relays. So from content scripts: only
// config / status / telemetry, status keeps known fields (short strings), telemetry only call summaries and errors.
// Settings and other tabs' status: extension pages only (popup, options).
const STATUS: Record<string, 'number' | 'string' | 'boolean'> = { active: 'number', build: 'string', error: 'string', mode: 'string', model: 'string', enabled: 'boolean',
  strength: 'number', latencyMs: 'number', pumpUnderruns: 'number', speaking: 'boolean', levelDb: 'number' };
function pageStatus(m: Record<string, unknown>): TabStatus {
  const st: Record<string, unknown> = {};
  for (const [k, type] of Object.entries(STATUS)) {
    const v = m[k];
    if (v === null) st[k] = null;
    else if (typeof v === type && (type !== 'number' || Number.isFinite(v))) st[k] = typeof v === 'string' ? v.slice(0, 200) : v;
  }
  return st as TabStatus;
}
const fromExtensionPage = (sender: chrome.runtime.MessageSender) => sender.id === chrome.runtime.id && typeof sender.url === 'string' && sender.url.startsWith(chrome.runtime.getURL(''));

chrome.runtime.onMessage.addListener((msg: unknown, sender, reply) => {
  (async () => {
    if (!msg || typeof msg !== 'object') return reply(null);
    // checked at run time below: a content script relays what its page sent
    const m = msg as ToBackground;
    if (m.type === 'config') return reply(await config(String(m.origin)));
    if (m.type === 'status' && sender.tab?.id !== undefined) {
      const tabId = sender.tab.id;
      const prev = status.get(tabId) || {};
      const st: TabStatus = { ...prev, ...pageStatus(m as unknown as Record<string, unknown>), origin: typeof m.origin === 'string' ? m.origin : prev.origin, at: Date.now() };
      status.set(tabId, st);
      badge(tabId, st);
      return reply(true);
    }
    if (m.type === 'telemetry' && sender.tab && m.ev && typeof m.ev === 'object' && (m.ev.t === 'call' || m.ev.t === 'error')) {
      const { merged } = await settings();
      if (await telemetry.record(m.ev, sender.tab.url, merged)) telemetry.flush(merged);
      return reply(true);
    }
    if (!fromExtensionPage(sender)) return reply(null);
    if (m.type === 'tabStatus') {
      const { merged, managed } = await settings();
      return reply({ status: status.get(m.tabId) || null, settings: merged, policy: Object.keys(managed) });
    }
    if (m.type === 'set') {
      const { managed } = await settings();
      const key = m.origin ? 'sites' : m.key;
      if (!key) return reply({ error: 'no setting named' });
      if (managed.lockEnabled && ((key === 'enabled' && m.value === false) || (key === 'sites' && m.value === 'off'))) {
        return reply({ error: 'set by your organization' });
      }
      if (key in managed && !(key === 'sites' && !(m.origin! in (managed.sites || {})))) return reply({ error: 'set by your organization' });
      if (m.origin) {
        const { sites = {} } = (await chrome.storage.sync.get('sites')) as Partial<Settings>;
        if (m.value === null) delete sites[m.origin]; else sites[m.origin] = m.value as Settings['sites'][string];
        await chrome.storage.sync.set({ sites });
      } else await chrome.storage.sync.set({ [key]: m.value });
      const { merged: now } = await settings();
      if (m.key === 'telemetry' || m.key === 'diagnostics') await telemetry.onSwitch(now);
      // which setting changed and to what (never the per-site list: that is browsing data)
      if (!m.origin && ['model', 'strength', 'enabled', 'mode', 'lang', 'diagnostics'].includes(key)) {
        const value = m.key === 'strength' ? Math.round(+(m.value as number) / 10) * 10 : m.value;
        if (await telemetry.record({ t: 'settings', key, value }, null, now)) telemetry.flush(now);
      }
      return reply({ ok: true });
    }
    reply(null);
  })().catch((e) => reply({ error: String(e && e.message || e) }));
  return true;   // async reply
});

chrome.tabs.onRemoved.addListener((id) => status.delete(id));

// ---- updates: never in the middle of a call ----
const sessionPorts = new Set<chrome.runtime.Port>();
const UPDATE_MAX_WAIT = 12 * 3600e3;   // 12 h: then the update is applied even mid-call (the call keeps its engine)
const startedAt = Date.now();
chrome.runtime.onConnect.addListener((p) => {
  if (p.name !== 'session') return;
  sessionPorts.add(p);
  p.onDisconnect.addListener(async () => {
    sessionPorts.delete(p);
    maybeUpdate();
  });
});
async function maybeUpdate() {
  const { pendingUpdate, pendingSince = Date.now() } = (await chrome.storage.session.get(['pendingUpdate', 'pendingSince'])) as { pendingUpdate?: unknown; pendingSince?: number };
  if (!pendingUpdate) return;
  const overdue = Date.now() - pendingSince > UPDATE_MAX_WAIT;
  if (!overdue) setTimeout(maybeUpdate, Math.min(UPDATE_MAX_WAIT, 3600e3));   // checked again, call or not
  // a freshly started worker: bridges reconnect their session ports within ~1 s, wait for them
  const wait = 3000 - (Date.now() - startedAt);
  if (wait > 0) return void setTimeout(maybeUpdate, wait);
  if (sessionPorts.size > 0 && !overdue) return;
  await chrome.storage.session.remove(['pendingUpdate', 'pendingSince']);
  viUpdate.reload();
}
async function onUpdateAvailable(details: { version?: string }) {
  const { pendingSince } = await chrome.storage.session.get('pendingSince');
  await chrome.storage.session.set({ pendingUpdate: details.version || true, pendingSince: pendingSince ?? Date.now() });
  maybeUpdate();
}
// Safari has no onUpdateAvailable: the App Store updates the app, and with it the extension
chrome.runtime.onUpdateAvailable?.addListener(onUpdateAvailable);
maybeUpdate();   // a worker restarted while an update was waiting
// reports queued while offline; the daily `active` report
settings().then(async ({ merged }) => { if (!(await telemetry.active(merged))) await telemetry.flush(merged); }).catch(() => {});
// e2e hooks (tests swap `reload`: an unpacked extension that reloads itself under automation is unloaded)
const viUpdate = { onUpdateAvailable, sessions: () => sessionPorts.size, reload: () => chrome.runtime.reload() };
hooks.viUpdate = viUpdate;

// open tabs get the content scripts (new install: wrap future getUserMedia calls; update: a new bridge,
// the page script from the old version stays and reconnects to it)
// Settings format 2 (the open-source build): an install coming from an earlier build loses that build's account /
// licence data, and strength goes back to the default (100 %) once.
const SETTINGS_VERSION = 2;
async function migrate() {
  const { settingsVersion = 0 } = await getLocal('settingsVersion');
  if (settingsVersion >= SETTINGS_VERSION) return;
  await chrome.storage.local.remove(['devToken', 'account', 'lease', 'licNext', 'licOnline', 'licError', 'licAttempts', 'licState', 'usage', 'lastReport', 'device']);
  await chrome.storage.sync.remove(['strength', 'licenseUrl', 'tags']);
  await chrome.storage.local.set({ settingsVersion: SETTINGS_VERSION });
}

chrome.runtime.onInstalled.addListener(async ({ reason }) => {
  if (reason !== 'install' && reason !== 'update') return;
  await migrate().catch(() => {});
  const tabs = await chrome.tabs.query({ url: ['https://*/*', 'http://localhost/*', 'http://127.0.0.1/*'] }).catch(() => []);
  for (const t of tabs) {
    if (t.id === undefined) continue;
    const target = { tabId: t.id, allFrames: true };
    try {
      await chrome.scripting.executeScript({ target, files: ['inject.js'], world: 'MAIN' });
      await chrome.scripting.executeScript({ target, files: ['bridge.js'] });
    } catch { /* tab closed, discarded, or not scriptable */ }
  }
});
chrome.tabs.onUpdated.addListener((id, info) => { if (info.status === 'loading') { status.delete(id); badge(id, null); } });

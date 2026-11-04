// options.ts — engine mode, the default for all sites + per-site list, language, privacy (reports). Policy-managed values are shown read-only.
import { loadI18n, applyI18n, languages } from './i18n.ts';
import { buildInfo } from './messages.ts';
import type { ToBackground, TabStatusReply } from './messages.ts';

// the page's controls (inputs and selects: value / checked / disabled)
const $ = (id: string) => document.getElementById(id) as HTMLInputElement;
function send(m: { type: 'tabStatus'; tabId: number }): Promise<TabStatusReply>;
function send(m: ToBackground): Promise<unknown>;
function send(m: ToBackground): Promise<unknown> { return chrome.runtime.sendMessage(m); }

for (const l of await languages()) $('lang').append(new Option(l.name, l.code));

// tabs (WAI-ARIA tabs pattern): one panel at a time, arrows / Home / End move between tabs, the hash remembers the tab
const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role=tab]')];
function selectTab(tab: HTMLButtonElement, focus = false) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute('aria-controls')!)!.hidden = !on;
  }
  history.replaceState(null, '', `#${tab.id.slice(4)}`);
  if (focus) tab.focus();
}
for (const t of tabs) {
  t.onclick = () => selectTab(t);
  t.onkeydown = (e) => {
    const i = tabs.indexOf(t), rtl = document.documentElement.dir === 'rtl';
    const step = { ArrowRight: rtl ? -1 : 1, ArrowLeft: rtl ? 1 : -1 }[e.key];
    const next = step ? tabs[(i + step + tabs.length) % tabs.length] : e.key === 'Home' ? tabs[0] : e.key === 'End' ? tabs[tabs.length - 1] : null;
    if (next) { e.preventDefault(); selectTab(next, true); }
  };
}
selectTab(tabs.find((t) => t.id === `tab-${location.hash.slice(1)}`) ?? tabs[0]);

// every change bumps `gen`: a load() that started before a newer change drops its (older) values, so a quick second
// click is never undone by the first click's refresh
let gen = 0;
const changed = () => ++gen;

async function load() {
  const my = ++gen;
  const r = await send({ type: 'tabStatus', tabId: -1 });
  const s = r.settings, policy = new Set(r.policy);
  const t = await loadI18n(s.lang);
  if (my !== gen) return;
  applyI18n(document, t);
  // the credit line: translated text around the author link
  const [before, after] = t('ext.credit').split('{author}');
  const a = Object.assign(document.createElement('a'), { href: 'https://github.com/tvildo', target: '_blank', rel: 'noopener', textContent: 'tvildo' });
  const site = Object.assign(document.createElement('a'), { href: 'https://parla.cc', target: '_blank', rel: 'noopener', textContent: 'parla.cc' });
  $('credit').replaceChildren(before ?? '', a, after ?? '', ' · ', site);
  $('lang').value = s.lang || 'auto';
  $('lang').disabled = policy.has('lang');
  $('telemetry').checked = s.telemetry === true;
  $('telemetry').disabled = policy.has('telemetry');
  $('diagnostics').checked = s.telemetry === true && s.diagnostics === true;
  $('diagnostics').disabled = policy.has('diagnostics') || s.telemetry !== true;
  $('allOn').checked = s.enabled;
  $('allOn').disabled = policy.has('enabled') || policy.has('lockEnabled') && !!s.lockEnabled;
  $('mode').value = s.mode || 'auto';
  $('mode').disabled = policy.has('mode');
  if (policy.has('mode')) $('mode').title = t('ext.managedBy');
  $('managed').hidden = r.policy.length === 0;
  const sites = Object.entries(s.sites);
  $('sites').replaceChildren(...(sites.length ? sites.map(([o, v]) => {
    const row = document.createElement('div');
    row.className = 'row between';
    const name = document.createElement('span'); name.textContent = `${o} — ${t(v === 'on' ? 'ext.siteState.on' : 'ext.siteState.off')}`;
    const b = document.createElement('button'); b.className = 'secondary'; b.textContent = t('ext.reset');
    b.onclick = () => send({ type: 'set', origin: o, value: null }).then(load);
    row.append(name, b);
    return row;
  }) : [Object.assign(document.createElement('span'), { textContent: t('ext.noSites') })]));
}


$('allOn').onchange = async () => { changed(); await send({ type: 'set', key: 'enabled', value: $('allOn').checked }); await load(); };
$('mode').onchange = async () => { changed(); await send({ type: 'set', key: 'mode', value: $('mode').value }); await load(); };
$('lang').onchange = async () => { changed(); await send({ type: 'set', key: 'lang', value: $('lang').value }); await load(); };

// Firefox has its own consent for data collection (about:addons): ask it too, from this click
const firefoxConsent = (kind: string, on: boolean): Promise<boolean> => {
  const p = globalThis.browser?.permissions as { request(p: FirefoxDataCollection): Promise<boolean>; remove(p: FirefoxDataCollection): Promise<boolean> } | undefined;
  if (!p || !navigator.userAgent.includes('Firefox')) return Promise.resolve(true);
  return (on ? p.request({ data_collection: [kind] }) : p.remove({ data_collection: [kind] }).then(() => true)).catch(() => !on);
};
$('telemetry').onchange = async () => {
  changed();
  const on = $('telemetry').checked;
  if (!(await firefoxConsent('technicalAndInteraction', on))) $('telemetry').checked = false;
  await send({ type: 'set', key: 'telemetry', value: $('telemetry').checked });
  if (!on) await send({ type: 'set', key: 'diagnostics', value: false });
  await load();
};
// detailed diagnostics: a plain switch (off by default; what it sends is written right under it). Firefox asks its
// own data-collection permission from this click; refused there = stays off.
$('diagnostics').onchange = async () => {
  changed();
  const on = $('diagnostics').checked;
  const ok = await firefoxConsent('browsingActivity', on);
  await send({ type: 'set', key: 'diagnostics', value: on && ok });
  await load();
};
// what this is: version and the git commit it was built from (build.json)
const build = await buildInfo();
$('version').textContent = `Parla v${build.version} · ${build.commit}`;
await load();
document.documentElement.dataset.ready = '1';   // first load done (tests wait for it before clicking)

// i18n.ts — the popup's and options page's translations (ES module). Not chrome.i18n: Chrome's UI-language list lacks
// languages we ship (e.g. Georgian), and users can pick a language in Options. Files: locales/<code>.json (the ext.*
// keys of src/i18n/<code>.json, written by build.mjs) + locales/languages.json.
//   const t = await loadI18n(setting)   // setting: 'auto' (browser language) or a code
//   t('ext.active', { n: 2, quality, mode })   // plural keys: <key>_one / _few / ... / _other by {n} (Intl.PluralRules)
//   applyI18n(document, t)              // [data-i18n] -> textContent, [data-i18n-title] -> title; <html lang dir>
export interface Language { code: string; name: string; dir?: 'ltr' | 'rtl' }
/** a translation function, with the language it speaks */
export interface Translate {
  (key: string, vars?: Record<string, string | number>): string;
  code: string;
  dir: 'ltr' | 'rtl';
}

const url = (p: string) => chrome.runtime.getURL(p);
const json = async (p: string) => (await fetch(url(p))).json();

let languagesP: Promise<Language[]> | null = null;
export function languages(): Promise<Language[]> {
  languagesP = languagesP || json('locales/languages.json');
  return languagesP;
}

/** the shipped language for a BCP 47 tag: exact, then script/region rules for Chinese, then the base language */
export function match(tag: string | null | undefined, codes: string[]): string | null {
  const t = String(tag || '').toLowerCase();
  const exact = codes.find((c) => c.toLowerCase() === t);
  if (exact) return exact;
  if (t.startsWith('zh')) return /hant|tw|hk|mo/.test(t) ? 'zh-TW' : 'zh-CN';
  const base = t.split('-')[0];
  return codes.find((c) => c.toLowerCase() === base) || codes.find((c) => c.toLowerCase().split('-')[0] === base) || null;
}

export async function resolve(setting?: string): Promise<string> {
  const codes = (await languages()).map((l) => l.code);
  if (setting && setting !== 'auto' && codes.includes(setting)) return setting;
  for (const tag of navigator.languages || [navigator.language]) {
    const c = match(tag, codes);
    if (c) return c;
  }
  return 'en';
}

export async function loadI18n(setting = 'auto'): Promise<Translate> {
  const code = await resolve(setting);
  const en = await json('locales/en.json');
  const tr = code === 'en' ? {} : await json(`locales/${code}.json`).catch(() => ({}));
  const all: Record<string, string> = { ...en, ...tr };
  const dir = (await languages()).find((l) => l.code === code)?.dir || 'ltr';
  const plural = new Intl.PluralRules(code);
  const t = ((key: string, vars: Record<string, string | number> = {}) => {
    let k = key;
    // plural keys: <key>_<CLDR category> (one, few, many, ...) for this language, else <key>_other
    if ('n' in vars && `${key}_other` in all) { const c = `${key}_${plural.select(+vars.n)}`; k = c in all ? c : `${key}_other`; }
    return String(all[k] ?? key).replace(/\{(\w+)\}/g, (m, v) => (v in vars ? String(vars[v]) : m));
  }) as Translate;
  t.code = code;
  t.dir = dir;
  return t;
}

export function applyI18n(doc: Document, t: Translate): void {
  doc.documentElement.lang = t.code;
  doc.documentElement.dir = t.dir;
  for (const el of doc.querySelectorAll<HTMLElement>('[data-i18n]')) el.textContent = t(el.dataset.i18n!);
  for (const el of doc.querySelectorAll<HTMLElement>('[data-i18n-title]')) el.title = t(el.dataset.i18nTitle!);
}

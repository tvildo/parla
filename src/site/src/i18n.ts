// Translations for the site: one JSON per language in src/i18n (shared with the extension). English is the source;
// a missing key in another language falls back to English.
import languages from "../../i18n/languages.json";

type Dict = Record<string, string>;
const files = import.meta.glob<Dict>("../../i18n/*.json", { eager: true, import: "default" });
const dict = (code: string): Dict => files[`../../i18n/${code}.json`] ?? {};

export const LANGS = (languages as { code: string; name: string; dir: "ltr" | "rtl" }[]);
export const DEFAULT = "en";
/** URL path segment of a language: "" for English (the root), else the code */
export const seg = (code: string) => (code === DEFAULT ? "" : `${code}/`);

export type T = ((key: string, vars?: Record<string, string | number>) => string) & { code: string; dir: "ltr" | "rtl" };

export function translator(code: string): T {
  const all = { ...dict(DEFAULT), ...dict(code) };
  const t = ((key: string, vars: Record<string, string | number> = {}) => {
    let k = key;
    if ("n" in vars && `${key}_one` in all) k = `${key}_${+vars.n === 1 ? "one" : "other"}`;
    return String(all[k] ?? key).replace(/\{(\w+)\}/g, (m, v) => (v in vars ? String(vars[v]) : m));
  }) as T;
  t.code = code;
  t.dir = LANGS.find((l) => l.code === code)?.dir ?? "ltr";
  return t;
}

/** every key with a prefix (e.g. "demo."), for passing a component its strings */
export function pick(code: string, prefix: string): Dict {
  const all = { ...dict(DEFAULT), ...dict(code) };
  return Object.fromEntries(Object.entries(all).filter(([k]) => k.startsWith(prefix)));
}

/** getStaticPaths for the [lang] routes: every language except the default (served at the root) */
export const langPaths = () => LANGS.filter((l) => l.code !== DEFAULT).map((l) => ({ params: { lang: l.code } }));

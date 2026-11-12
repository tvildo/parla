// The home page's "See it" pictures, made from the real things at build time instead of screenshots, so they follow
// every change to the originals and are translated like the rest of the page:
//   popupPreview(t)  the extension's popup: its own popup.html + ui.css (src/extension/src), texts from the page
//                    language's ext.* keys, filled with a sample state (on, cleaning a call, the meter mid-noise)
//   demoPreview(t)   the live demo's card (Demo.tsx DemoPreview) rendered to static HTML with the site's demo styles
// Each comes as a declarative shadow root, so the original styles and the site's (both have .card, .logo, .foot, h1 …)
// don't touch each other. The build fails when the popup changes in a way the preview can't follow (an element it fills
// is gone, a text has no translation, a style hook moved): fix the preview here, never let it drift silently.
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import popupHtml from "../../../extension/src/popup.html?raw";
import uiCss from "../../../extension/src/ui.css?raw";
import siteCss from "../styles/site.css?raw";
import demoCss from "../styles/demo.css?raw";
import { DemoPreview } from "./Demo";
import { pick, type T } from "../i18n";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fail = (what: string): never => { throw new Error(`site preview of the extension popup: ${what} (src/site/src/components/previews.ts)`); };

/** one replacement that must happen exactly once */
function once(s: string, re: RegExp, to: (...m: string[]) => string, what: string): string {
  const n = [...s.matchAll(new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g"))].length;
  if (n !== 1) fail(`${what}: expected once, found ${n} times`);
  return s.replace(re, to as (...a: string[]) => string);
}

/** the start tag of the element with this id, edited by `edit` */
const tag = (s: string, id: string, edit: (t: string) => string) =>
  once(s, new RegExp(`<\\w+\\b[^>]*\\bid="${id}"[^>]*>`), (m) => edit(m), `element #${id}`);
/** the text of the element with this id (it must hold plain text) */
const text = (s: string, id: string, value: string) =>
  once(s, new RegExp(`(<(\\w+)\\b[^>]*\\bid="${id}"[^>]*>)[^<]*(</\\2>)`), (_m, a, _t, b) => a + esc(value) + b, `text of #${id}`);
const unhide = (t: string) => (/\shidden(?=[\s>])/.test(t) ? t.replace(/\shidden(?=[\s>])/, "") : fail(`${t} is no longer hidden by default`));
const addAttr = (attr: string) => (t: string) => t.replace(/>$/, ` ${attr}>`);

export function popupPreview(t: T): string {
  const tr = (key: string, vars?: Record<string, string | number>) => {
    const v = t(key, vars);
    return v === key ? fail(`no translation for ${key}`) : v;
  };
  const body = /<body class="popup">([\s\S]*?)<script\b/.exec(popupHtml)?.[1] ?? fail("no <body class=\"popup\"> … <script>");
  // every data-i18n text and title, from the page's language
  let h = body.replace(/(<(\w+)\b[^>]*\bdata-i18n="([^"]+)"[^>]*>)[^<]*(<\/\2>)/g, (_m, a, _t, key, b) => a + esc(tr(key)) + b);
  h = h.replace(/(\bdata-i18n-title="([^"]+)"[^>]*?\btitle=")[^"]*"/g, (_m, a, key) => a + esc(tr(key)) + '"');
  const leftover = [...h.matchAll(/\bdata-i18n="([^"]+)"[^>]*>([^<]*)</g)].filter((m) => m[2] !== esc(tr(m[1])));
  if (leftover.length) fail(`texts not translated: ${leftover.map((m) => m[1]).join(", ")} (an element with markup inside?)`);
  // a sample state: on for a call site, cleaning one microphone stream, the meter in the middle of some noise
  h = text(h, "site", "call.example.com");
  h = tag(h, "siteOn", addAttr("checked"));
  h = tag(h, "mic", (x) => x.replace('class="mic"', 'class="mic on live"'));
  h = text(h, "state", tr("ext.active", { n: 1, quality: tr("ext.qs.best"), mode: tr("ext.modePage") }) + tr("ext.delay", { ms: 40 }));
  h = text(h, "strengthV", "100%");
  h = tag(h, "meter", unhide);
  h = text(h, "meterV", tr("ext.meter.quieter", { n: 6 }));
  h = tag(h, "barIn", addAttr('style="width: 80%"'));
  h = tag(h, "barOut", addAttr('style="width: 22%"'));
  // headings inside a picture would join the page's outline
  h = h.replace(/<(\/?)h[1-6]\b/g, "<$1div");
  // the page's <body> is a box here: its rules move to .body / .popup
  if ((uiCss.match(/:root \{/g) ?? []).length !== 2) fail("ui.css: expected two :root { rules (dark and light)");
  // always the dark look (the site is dark): the light scheme's colours go
  let css = once(uiCss, /@media \(prefers-color-scheme: light\) \{\s*:root \{[^}]*\}\s*\}/, () => "", "ui.css light-scheme :root block");
  css = css.replace(/:root \{/g, ":host {");
  css = once(css, /^body \{/m, () => ".body {", "ui.css body {");
  css = once(css, /^body\.popup \{/m, () => ".popup {", "ui.css body.popup {");
  css = once(css, /^body\.popup main \{/m, () => ".popup main {", "ui.css body.popup main {");
  if (/\bbody\.popup\b/.test(css)) fail("ui.css has more body.popup rules: map them here");
  return `<template shadowrootmode="open"><style>${css}</style><div class="body popup" dir="${t.dir}" lang="${t.code}">${h}</div></template>`;
}

export function demoPreview(t: T): string {
  const card = renderToStaticMarkup(createElement(DemoPreview, { strings: pick(t.code, "demo.") }));
  return `<template shadowrootmode="open"><style>${siteCss}${demoCss}:host{display:block}.demo-one{margin:0}</style>` +
    `<div dir="${t.dir}" lang="${t.code}">${card}</div></template>`;
}

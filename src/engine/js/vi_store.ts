// vi_store.ts — in-browser model store ("mini Ollama"): pull model packages from the static
// registry (models/index.json + manifests) or from local files, cache them in the browser
// (Cache API, works offline once pulled), verify sha256, list / delete installed models.
// A package = manifest.json + its files (graph.json, weights.bin, ...); pure data, no code.
//
//   const store = new ViStore('../models/');
//   await store.registry()          -> [{name, manifest, installed}]
//   await store.pull('gtcrn')       -> manifest (fetch + verify + cache)
//   await store.addFiles(fileList)  -> manifest (local manifest.json + referenced files)
//   await store.list()              -> [{name, display_name, bytes, license, sample_rate, latency, local_only, engine}]
//   await store.get('gtcrn')        -> {manifest, graphJson, weights: ArrayBuffer|null}
//   await store.remove('gtcrn')
// Storage: Cache API ('vi-models-v1'), keys `vi-store/<name>/<file>`; falls back to an in-memory
// map when caches are unavailable (non-secure context / private mode).
// Availability: `baseUrl` may be a list of mirrors (e.g. CDN, then the customer's own copy), tried in
// order, so one origin being down doesn't block first downloads; cached packages work offline.
// Engine wasm is cached by content (sha256 from plugins/index.json): updates reach devices as soon as
// the index can be read; offline, the last known module is used.

import type { Manifest } from './vi_engine.ts';

const CACHE = 'vi-models-v1';

/** a manifest as stored here: the package's own fields + what the store adds */
export interface StoredManifest extends Manifest {
  display_name?: string;
  license?: string;
  local_only?: boolean;
  sample_rate?: number;
  latency?: number;
  frame_size?: number;
  sha256?: Record<string, string>;
  _installed: number;
  _bytes: number;
  _files: string[];
}
export interface StoredModel {
  name: string; display_name: string; engine: string; bytes: number; license?: string; local_only: boolean;
  sample_rate?: number; latency?: number; frame_size?: number; manifest: StoredManifest;
}
export interface RegistryEntry { name: string; manifest?: Manifest; url?: string; error?: string; installed: boolean }

async function sha256hex(buf: ArrayBuffer): Promise<string | null> {
  if (!globalThis.crypto || !crypto.subtle) return null;
  const h = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function packageFiles(m: Manifest): string[] {
  const f: string[] = [];
  if (m.graph) f.push(m.graph);
  if (m.weights) f.push(m.weights);
  return f;
}

export class ViStore {
  readonly bases: URL[];
  readonly base: URL;
  private mem = new Map<string, ArrayBuffer>();

  constructor(baseUrl: string | string[] = '../models/') {
    const here = globalThis.location ? location.href : 'http://localhost/';
    this.bases = (Array.isArray(baseUrl) ? baseUrl : [baseUrl]).map((b) => new URL(b, here));
    this.base = this.bases[0];   // cache keys: always the first base, whichever mirror served the bytes
  }

  /** fetch `rel` from the first mirror that answers OK; -> {r, url} */
  private async _fetchAny(rel: string): Promise<{ r: Response; url: URL }> {
    let last: unknown = null;
    for (const b of this.bases) {
      try {
        const url = new URL(rel, b);
        const r = await fetch(url);
        if (r.ok) return { r, url };
        last = new Error(`${url}: HTTP ${r.status}`);
      } catch (e) { last = e; }
    }
    throw last || new Error(`no mirror for ${rel}`);
  }

  private async _cache(): Promise<Cache | null> {
    try { return globalThis.caches ? await caches.open(CACHE) : null; } catch { return null; }
  }
  private _key(name: string, file: string): string { return new URL(`vi-store/${encodeURIComponent(name)}/${encodeURIComponent(file)}`, this.base).href; }
  private async _put(name: string, file: string, buf: ArrayBuffer | Uint8Array, type = 'application/octet-stream'): Promise<void> {
    const c = await this._cache();
    if (c) await c.put(this._key(name, file), new Response(buf as BodyInit, { headers: { 'Content-Type': type, 'Content-Length': String(buf.byteLength) } }));
    else this.mem.set(this._key(name, file), buf instanceof Uint8Array ? buf.slice().buffer : buf);
  }
  private async _getBuf(name: string, file: string): Promise<ArrayBuffer | null> {
    const c = await this._cache();
    if (c) { const r = await c.match(this._key(name, file)); return r ? await r.arrayBuffer() : null; }
    return this.mem.get(this._key(name, file)) || null;
  }

  /** Models offered by the static registry (index.json). */
  async registry(): Promise<RegistryEntry[]> {
    const idx: { models: string[] } = await (await this._fetchAny('index.json')).r.json();
    const installed = new Set((await this.list()).map((m) => m.name));
    const out: RegistryEntry[] = [];
    for (const name of idx.models) {
      try {
        const { r, url } = await this._fetchAny(`${name}/manifest.json`);
        const m = await r.json();
        out.push({ name, manifest: m, url: url.href, installed: installed.has(name) });
      } catch (e) { out.push({ name, error: String(e), installed: installed.has(name) }); }
    }
    return out;
  }

  /** Fetch a package from the static path, verify sha256 (if listed), cache it. */
  async pull(name: string, onProgress: (file: string, bytes: number) => void = () => {}): Promise<StoredManifest> {
    const { r: mr, url: murl } = await this._fetchAny(`${name}/manifest.json`);
    const mtext = await mr.text();
    const m: Manifest = JSON.parse(mtext);
    const files: Record<string, ArrayBuffer> = {};
    for (const f of packageFiles(m)) {
      const r = await fetch(new URL(f, murl));
      if (!r.ok) throw new Error(`${name}: ${f}: HTTP ${r.status} (local-only weights? load them from files instead)`);
      files[f] = await r.arrayBuffer();
      onProgress(f, files[f].byteLength);
    }
    return this._install(name, m, files);
  }

  /** Install from local files (FileList / File[]): manifest.json + the files it references
   *  (matched by base name). Lets users run models that are not on any server. */
  async addFiles(fileList: FileList | File[]): Promise<StoredManifest> {
    const byName = new Map([...fileList].map((f) => [f.name, f]));
    const mf = byName.get('manifest.json');
    if (!mf) throw new Error('select manifest.json together with its graph/weights files');
    const m: Manifest = JSON.parse(await mf.text());
    const files: Record<string, ArrayBuffer> = {};
    for (const f of packageFiles(m)) {
      const base = f.split('/').pop()!;
      const file = byName.get(base);
      if (!file) throw new Error(`missing file ${base} for ${m.name}`);
      files[f] = await file.arrayBuffer();
    }
    return this._install(m.name, m, files);
  }

  private async _install(name: string, m: Manifest, files: Record<string, ArrayBuffer>): Promise<StoredManifest> {
    const sums = (m.sha256 || {}) as Record<string, string>;
    for (const [f, buf] of Object.entries(files)) {
      const want = sums[f];
      if (want) {
        const got = await sha256hex(buf);
        if (got && got !== want) throw new Error(`${name}: sha256 mismatch for ${f}`);
      }
    }
    for (const [f, buf] of Object.entries(files)) await this._put(name, f, buf);
    const meta: StoredManifest = { ...m, _installed: Date.now(), _bytes: Object.values(files).reduce((a, b) => a + b.byteLength, 0), _files: Object.keys(files) };
    await this._put(name, 'manifest.json', new TextEncoder().encode(JSON.stringify(meta)), 'application/json');
    return meta;
  }

  private async _manifest(name: string): Promise<StoredManifest | null> {
    const b = await this._getBuf(name, 'manifest.json');
    return b ? JSON.parse(new TextDecoder().decode(b)) : null;
  }

  /** Installed packages (from the cache). */
  async list(): Promise<StoredModel[]> {
    const c = await this._cache();
    const keys = c ? (await c.keys()).map((r) => r.url) : [...this.mem.keys()];
    const names = new Set<string>();
    for (const k of keys) {
      const m = k.match(/vi-store\/([^/]+)\/manifest\.json$/);
      if (m) names.add(decodeURIComponent(m[1]));
    }
    const out: StoredModel[] = [];
    for (const n of names) {
      const m = await this._manifest(n);
      if (!m) continue;
      out.push({ name: n, display_name: m.display_name || n, engine: m.engine, bytes: m._bytes, license: m.license, local_only: !!m.local_only,
        sample_rate: m.sample_rate || (m.pipeline as { sample_rate?: number } | undefined)?.sample_rate, latency: m.latency, frame_size: m.frame_size, manifest: m });
    }
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }

  /** Package contents for engine creation (offline). */
  async get(name: string): Promise<{ manifest: StoredManifest; graphJson: string | null; weights: ArrayBuffer | null } | null> {
    const m = await this._manifest(name);
    if (!m) return null;
    const g = m.graph ? await this._getBuf(name, m.graph) : null;
    const graphJson = g ? new TextDecoder().decode(g) : null;
    const weights = m.weights ? await this._getBuf(name, m.weights) : null;
    return { manifest: m, graphJson, weights };
  }

  /** Engine wasm module bytes, cache-first (sha256 from web/plugins/index.json). `file` is
   *  relative to web/, e.g. 'plugins/vi_graph_simd.wasm' or 'vi_rs_simd.wasm'. */
  async wasm(file: string, webBase: string | URL): Promise<ArrayBuffer> {
    const latest = new URL(`vi-store/_wasm/${file}`, this.base).href;           // last known module
    const byHash = (h: string) => new URL(`vi-store/_wasm/sha256/${h}`, this.base).href;  // content-addressed
    const c = await this._cache();
    let want: string | null = null;
    try {
      const idx: { files: Record<string, { sha256?: string }> } = await (await fetch(new URL('plugins/index.json', webBase), { cache: 'no-cache' })).json();
      want = idx.files[file]?.sha256 ?? null;
    } catch { /* offline / index unreachable: fall back to the last known module */ }
    if (c && want) { const r = await c.match(byHash(want)); if (r) return await r.arrayBuffer(); }
    if (c && !want) { const r = await c.match(latest); if (r) return await r.arrayBuffer(); }
    const r = await fetch(new URL(file, webBase));
    if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);   // never cache an error page as the module
    const buf = await r.arrayBuffer();
    const got = await sha256hex(buf);
    if (want && got && want !== got) throw new Error(`${file}: sha256 mismatch`);
    if (c) {
      const prev = await c.match(latest);
      const prevSha = prev && prev.headers.get('x-vi-sha256');
      const resp = () => new Response(buf, { headers: { 'Content-Type': 'application/wasm', 'x-vi-sha256': got || '' } });
      if (got) await c.put(byHash(got), resp());
      await c.put(latest, resp());
      if (prevSha && prevSha !== got) await c.delete(byHash(prevSha));   // drop the replaced version
    }
    return buf;
  }

  async remove(name: string): Promise<void> {
    const m = await this._manifest(name);
    const files = ['manifest.json', ...((m && m._files) || [])];
    const c = await this._cache();
    for (const f of files) { if (c) await c.delete(this._key(name, f)); else this.mem.delete(this._key(name, f)); }
  }
}

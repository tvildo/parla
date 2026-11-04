// messages.ts — the extension's settings and the messages between its parts (types, and one typed storage helper).
//
//   page script (inject.ts) <-> bridge (bridge.ts): a MessageChannel; PageToBridge / BridgeToPage
//   bridge, popup, options -> service worker (background.ts): runtime.sendMessage; ToBackground
// The page script runs in the page's own world, so whatever arrives from it (through the bridge) is checked at run
// time too: these types describe what our own code sends, not what a hostile page could.

export type SiteSetting = 'on' | 'off';
export type Mode = 'auto' | 'page' | 'isolated';

/** effective settings: defaults < the user's choices (storage.sync) < policy (storage.managed) */
export interface Settings {
  enabled: boolean;
  model: string;
  /** 0..100 */
  strength: number;
  sites: Record<string, SiteSetting>;
  onlyListedSites: boolean;
  mode: Mode;
  lang: string;
  telemetry: boolean;
  diagnostics: boolean;
  lockEnabled?: boolean;
}

/** what a page gets (background.ts config()) */
export interface PageConfig {
  enabled: boolean;
  model: string;
  /** 0..1 */
  strength: number;
  mode: Mode;
  telemetry: boolean;
  version: string;
  /** the extension's base URL (added by the bridge) */
  base: string;
}

/** a tab's status for the badge and popup (the page script's report(), filtered by background.ts) */
export interface TabStatus {
  active?: number;
  build?: string;
  error?: string | null;
  mode?: Mode | null;
  model?: string;
  enabled?: boolean;
  strength?: number;
  latencyMs?: number;
  pumpUnderruns?: number;
  speaking?: boolean;
  levelDb?: number;
  origin?: string;
  at?: number;
}

/** live levels for the popup's meter (~19 a second, only while a popup is open): RMS dBFS of the microphone and of
 *  what the call hears over the same ~53 ms; the gap is what was removed */
export interface MeterSample { inDb: number; outDb: number; speaking: boolean }

/** what the extension was built from (build.json, written by build.ts): shown on the settings page, sent in reports */
export interface BuildInfo { version: string; commit: string }
let buildInfoP: Promise<BuildInfo> | null = null;
export const buildInfo = (): Promise<BuildInfo> => (buildInfoP ??= fetch(chrome.runtime.getURL('build.json'))
  .then((r) => r.json() as Promise<BuildInfo>)
  .catch(() => ({ version: chrome.runtime.getManifest().version, commit: 'unknown' })));

/** a report from the page (telemetry.ts checks every field again) */
export type TelemetryEvent = { t: string; [field: string]: unknown };

export type ToBackground =
  | { type: 'config'; origin: string }
  | ({ type: 'status' } & TabStatus)
  | { type: 'telemetry'; ev: TelemetryEvent }
  | { type: 'tabStatus'; tabId: number }
  | { type: 'set'; key?: keyof Settings; origin?: string; value: unknown };

export interface TabStatusReply { status: TabStatus | null; settings: Settings; policy: string[] }
export type SetReply = { ok: true } | { error: string };

/** what the extension keeps in storage.local (never synced) */
export interface LocalStore {
  // policy
  managedCache: Partial<Settings>;
  settingsVersion: number;
  // reports (telemetry.ts)
  tq: Record<string, unknown>[];
  tday: string;
  tcount: number;
  terr: string[];
  tnext: number;
  tfail: number;
  tsample: number;
  tactive: string;
  tid: string | null;
  tidMonth: string;
  pid: string | null;
  /** tests only: another receiver */
  telemetryEndpoint: string;
}
/** storage.local.get, typed: the keys asked for, each possibly missing */
export const getLocal = <K extends keyof LocalStore>(keys: K | K[]): Promise<Partial<Pick<LocalStore, K>>> =>
  chrome.storage.local.get(keys) as Promise<Partial<Pick<LocalStore, K>>>;

/** page script -> bridge */
export type PageToBridge =
  | { id: number; type: 'config' }
  | { id: number; type: 'fetch'; path: string; as: 'text' | 'buffer' }
  | { id: number; type: 'frame'; sampleRate: number; build: string }
  | ({ type: 'status' } & TabStatus)
  | { type: 'telemetry'; ev: TelemetryEvent }
  | ({ type: 'meter' } & MeterSample);

/** bridge -> page script */
export type BridgeToPage =
  | { id: number; result?: unknown; error?: string }
  | { type: 'config'; config: PageConfig }
  | { type: 'lost' }
  | { type: 'meter'; on: boolean };

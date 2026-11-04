// Browser globals TypeScript's DOM library doesn't describe.

// ---- the AudioWorklet global scope (src/engine/js/worklet.ts, src/extension/src/pump_worklet.ts) ----
declare const sampleRate: number;
declare const currentFrame: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: unknown);
}
declare function registerProcessor(name: string, processor: new (options?: any) => AudioWorkletProcessor): void;

// ---- a classic worker (src/extension/src/engine_worker.ts) ----
declare function importScripts(...urls: string[]): void;

// ---- Firefox's promise-based WebExtension API: only what Parla uses (data collection consent) ----
interface FirefoxDataCollection { data_collection: string[] }
declare var browser: undefined | {
  permissions?: {
    contains(p: FirefoxDataCollection): Promise<boolean>;
    request(p: FirefoxDataCollection): Promise<boolean>;
  };
};

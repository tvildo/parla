// The engine's files inside the npm package, as URLs the bundler resolves (`new URL(..., import.meta.url)`: Vite,
// webpack 5, Rollup, esbuild and Parcel emit each file as an asset). In the source tree there are none: pass `base`.
// The npm build (build.mjs --npm) writes the real map here.
export const ASSETS: Record<string, URL> | null = null;

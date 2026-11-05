// The Parla website: static pages in every language (src/i18n), the live demo as a React island.
// The demo uses the npm package (build/npm): its engine files become hashed assets in _astro/. build.py site makes
// the output path-relative, so the same build works at the root and under a preview prefix (/d/).
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  site: 'https://parla.cc',
  trailingSlash: 'always',
  // the CSS is small: inlined, so no request blocks the first paint
  build: { format: 'directory', inlineStylesheets: 'always' },
  integrations: [react()],
  vite: {
    server: { fs: { allow: [fileURLToPath(new URL('../..', import.meta.url))] } },
    // the package finds its engine files with new URL(..., import.meta.url): keep it out of the dev pre-bundle
    optimizeDeps: { exclude: ['@tvildo/parla'] },
  },
});

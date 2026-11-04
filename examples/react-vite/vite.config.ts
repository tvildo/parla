import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // the dev server pre-bundles dependencies, which would move parla's `new URL(..., import.meta.url)` away from its
  // files: keep it out of the pre-bundle (production builds need nothing)
  optimizeDeps: { exclude: ['@tvildo/parla'] },
});

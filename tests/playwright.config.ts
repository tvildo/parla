// Browser tests. Build first: python build.py wasm && python build.py extension && python build.py site && python build.py npm
//   cd tests && npm ci && npx playwright install chromium && npx playwright test
import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fakeMic } from './lib';

const SITE_PORT = 8799;

export default defineConfig({
  testDir: '.',
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  projects: [
    // the extension (build/extension) in Playwright's bundled Chromium (branded Chrome ignores --load-extension)
    { name: 'extension', testMatch: /extension\.spec\.ts/ },
    // README / docs / website screenshots, on demand only (python build.py screenshots)
    ...(process.env.SCREENSHOTS ? [{
      name: 'screenshots', testMatch: /screenshots\.spec\.ts/,
      use: { launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${fakeMic()}`, '--autoplay-policy=no-user-gesture-required'] } },
    }] : []),
    // the npm package as apps use it (examples/react-vite built with Vite, examples/vanilla without a bundler)
    {
      name: 'package', testMatch: /package\.spec\.ts/,
      use: {
        permissions: ['microphone'],
        launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${fakeMic()}`, '--autoplay-policy=no-user-gesture-required'] },
      },
    },
    // the website's live demo (build/site) in Chromium with a fake microphone
    {
      name: 'site', testMatch: /(site|a11y)\.spec\.ts/,
      use: {
        baseURL: `http://127.0.0.1:${SITE_PORT}`,
        permissions: ['microphone'],
        launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${fakeMic()}`, '--autoplay-policy=no-user-gesture-required'] },
      },
    },
  ],
  webServer: [
    { command: `python ${path.join('..', 'build.py')} serve ${SITE_PORT}`, url: `http://127.0.0.1:${SITE_PORT}/index.html`, reuseExistingServer: true },
    { command: `python ${path.join('..', 'build.py')} serve 8796 --root ${path.join('..', 'examples', 'react-vite', 'dist')}`, url: 'http://127.0.0.1:8796/', reuseExistingServer: true },
    { command: `python ${path.join('..', 'build.py')} serve 8795 --root ${path.join('..', 'examples', 'vanilla')}`, url: 'http://127.0.0.1:8795/', reuseExistingServer: true },
  ],
});

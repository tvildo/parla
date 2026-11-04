#!/bin/sh
# Every browser test, in the container (tests/Dockerfile): Chromium (extension, site, package) with Playwright, the
# SDK in Chromium, then the Firefox build of the extension in Firefox. Arguments go to `playwright test`.
set -e
npm ci --no-audit --no-fund
npx playwright test "$@"
# the SDK build copies the code `python build.py ts` emitted on the host (build/js, mounted with the tree)
node ../src/sdk/build.ts --out build/sdk-demo
node ../src/sdk/test_browser.ts
# Firefox needs an audio device for its AudioContexts: PulseAudio with a null sink
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
pulseaudio -D --exit-idle-time=-1 --log-target=stderr 2>/dev/null || true
pactl load-module module-null-sink sink_name=null >/dev/null && pactl set-default-sink null
node firefox.ts --headless

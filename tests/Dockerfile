# Browser tests in a container: Playwright's image (Chromium + Node) + Python for `build.py serve`, and a real Firefox
# (release build, geckodriver, Selenium) with PulseAudio on a null sink for the Firefox extension check.
# Built and run by `python build.py test` (the oss/ tree is mounted at /oss; build artifacts come from the host).
# pinned by digest (the tag can move); the version matches @playwright/test in package-lock.json
FROM mcr.microsoft.com/playwright:v1.65.0-noble@sha256:6e3fb709361521eb27161606e6da6d2674e5e89fa16516cb37d6fdd91372273d
RUN apt-get update && apt-get install -y --no-install-recommends python3 pulseaudio xz-utils curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/python
# Firefox release + geckodriver, pinned by checksum (Mozilla's SHA256SUMS; the geckodriver release asset)
ARG FIREFOX=158.0
ARG FIREFOX_SHA256=9549b054ec4cf85f50f6a19d8c6fb79580312e71ebc4ec2cb217ec9af09dc12a
ARG GECKODRIVER=0.37.1
ARG GECKODRIVER_SHA256=e815130ea95983e162ae91843b48d3a3ce991735635fce83a647afde21e09f7e
RUN curl -fsSL -o /tmp/ff.tar.xz "https://ftp.mozilla.org/pub/firefox/releases/${FIREFOX}/linux-x86_64/en-US/firefox-${FIREFOX}.tar.xz" \
 && echo "${FIREFOX_SHA256}  /tmp/ff.tar.xz" | sha256sum -c - && tar -xJf /tmp/ff.tar.xz -C /opt && rm /tmp/ff.tar.xz \
 && curl -fsSL -o /tmp/gd.tgz "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER}/geckodriver-v${GECKODRIVER}-linux64.tar.gz" \
 && echo "${GECKODRIVER_SHA256}  /tmp/gd.tgz" | sha256sum -c - && tar -xzf /tmp/gd.tgz -C /usr/local/bin && rm /tmp/gd.tgz
# PARLA_FIREFOX: the browser firefox.ts drives; SE_OFFLINE: Selenium uses this geckodriver, downloads nothing
ENV PARLA_CHROME_CHANNEL=chromium CI=1 PARLA_FIREFOX=/opt/firefox/firefox SE_OFFLINE=true XDG_RUNTIME_DIR=/tmp/runtime
WORKDIR /oss/tests
CMD ["sh", "run_all.sh"]

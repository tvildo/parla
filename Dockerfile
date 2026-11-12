# The Parla website (landing, privacy, live demo + the engine it runs), built from source and served by
# nginx on :8080. `docker build -t parla-site . && docker run -p 8080:8080 parla-site`
FROM rust:1.99.1-slim-bookworm@sha256:f2c004e3d2d8b547e8cf7ffbc29205385f4e71d23e53e4e383065db5e9f822cd AS wasm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl python3 \
 && rm -rf /var/lib/apt/lists/* && rustup target add wasm32-unknown-unknown
ARG BINARYEN=version_133
ARG BINARYEN_SHA256=2dc9c7813f5375db93d96ead4b78222fcc3e2677bbb832297af4797782a37489
RUN curl -fsSL -o /tmp/b.tgz "https://github.com/WebAssembly/binaryen/releases/download/${BINARYEN}/binaryen-${BINARYEN}-x86_64-linux.tar.gz" \
 && echo "${BINARYEN_SHA256}  /tmp/b.tgz" | sha256sum -c - \
 && tar -xzf /tmp/b.tgz -C /opt && ln -s /opt/binaryen-${BINARYEN}/bin/wasm-opt /usr/local/bin/wasm-opt && rm /tmp/b.tgz
WORKDIR /src
COPY build.py build.py
COPY src/engine/ src/engine/
# only the bundle (vi_rs_*): the site's engine; per-plugin modules are for other hosts
RUN for i in 0 1 2; do python3 build.py wasm --only $i --out /out/wasm; done

FROM node:26.13.0-bookworm-slim@sha256:bc0a2afd3b74ce12222fc8bab99271e66257f021078f5f4a5d687a77d280c1d9 AS site
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# the TypeScript compiler first (its own layer)
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund --ignore-scripts
COPY build.py LICENSE NOTICE tsconfig.json ./
COPY tools/worklet_bundle.ts tools/i18n_check.ts tools/ts_emit.ts tools/
COPY models/ models/
# everything tsconfig.json compiles (the site needs the engine and the SDK; the rest is checked with them)
COPY src/types/ src/types/
COPY src/engine/js/ src/engine/js/
COPY src/extension/src/ src/extension/src/
COPY src/telemetry/ src/telemetry/
COPY src/sdk/ src/sdk/
COPY src/i18n/ src/i18n/
COPY src/site/ src/site/
COPY --from=wasm /out/wasm /out/wasm
RUN python3 build.py site --wasm /out/wasm --out /out/site

FROM nginx:1.31.8-alpine@sha256:785835ea65db4bba6f021acc0f3be91b0fef76a53fe41f8588c6698d2aed0606
RUN rm -f /etc/nginx/conf.d/default.conf
COPY src/site/nginx.conf /etc/nginx/conf.d/site.conf
COPY --from=site /out/site /usr/share/nginx/html
EXPOSE 8080

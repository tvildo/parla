# Security

## Reporting a vulnerability

Please report it privately: [open a security advisory](https://github.com/tvildo/parla/security/advisories/new)
(Security → Report a vulnerability). Don't open a public issue for it, so it can be fixed before it is made public.
Say what is affected (the extension and its version, the npm package, the website), how to reproduce it, and what
someone could do with it.

Supported: the latest release of the extension (Chrome Web Store, Firefox Add-ons), the latest `@tvildo/parla` on npm,
and the website at parla.cc.

## How Parla is kept safe

The website's [security page](https://parla.cc/security/) explains it for everyone. The details:

- At run time: audio is processed on the device (WebAssembly, no server receives it); the extension's permissions
  (`storage`, `scripting` and host access) and its content security policy are in
  [src/extension/src/manifest.json](src/extension/src/manifest.json), how it checks what web pages send in
  [src/extension/AGENTS.md](src/extension/AGENTS.md); reports, off by default: [docs/telemetry.md](docs/telemetry.md).
- In development: pinned and checksummed dependencies, no release younger than 3 days, CodeQL, zizmor, dependency
  review, OWASP ZAP, secret scanning: [docs/releasing.md](docs/releasing.md#supply-chain).

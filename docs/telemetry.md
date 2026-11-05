# Telemetry (the extension's reports)

The browser extension can send small reports so problems on real machines can be found and fixed, **only if the user
turns them on**: out of the box it sends nothing. The website and the npm package send nothing.

## The two switches

In the extension's settings, section **Privacy** (companies can set both by policy: `telemetry`, `diagnostics`):

| | Default | What it adds |
|---|---|---|
| **Anonymous technical reports** | **off** | errors, performance, the settings in use, browser and OS, the *kind* of calling app, a random ID that changes every month |
| **Detailed diagnostics** | **off** | the domain of the site (e.g. `meet.google.com`: no path, port, search part or fragment) and a permanent random ID — only after you turn it on |

Turning reports off deletes the queue and the IDs at once. Firefox also lists both as optional data collection in
`about:addons` (technical and interaction data / browsing activity); its choice wins as well.

**Never sent**: audio (not even levels of it), page content, anything typed, page titles, the search part (`?…`) or
fragment (`#…`) of an address, cookies, the IP address (the receiver doesn't store it).

## What a report looks like

Every report is one JSON object of ~300 bytes:

```json
{ "v": 1, "t": "call", "ts": 1793053526164, "id": "cb708dcdd3706c33ed927f2c",
  "ext": "0.1.0", "br": "chrome", "brv": 156, "os": "windows", "cores": 16, "lang": "en-US",
  "app": "meet", "mode": "page", "model": "deepfilternet3", "strength": 100, "dur": "<5m",
  "lat": 33, "cost": 996, "p95": 1010, "load": 79, "under": 0, "over": 1, "offShare": 0, "switches": 0 }
```

| Field | |
|---|---|
| `v`, `t`, `ts` | format version, type (`active`, `call`, `error`, `settings`), time |
| `id` | random; a new one every month (permanent only with detailed diagnostics) |
| `ext`, `commit` | extension version and the git commit it was built from (as on the settings page) |
| `br`, `brv`, `os`, `cores`, `lang` | browser + major version, OS, CPU cores, browser language |
| `app` | kind of calling app: `meet`, `teams`, `zoom`, `discord`, `slack`, `webex`, `whereby`, `jitsi`, `local`, `other` |
| `domain` | **only with detailed diagnostics**: the site's domain, e.g. `meet.google.com` (never the rest of the address; the receiver also cuts an older build's `url` field down to this) |

Per type:

- **`active`** (once a day while the browser runs the extension, calls or not): only the common fields above. This
  is what counts active installs.
- **`call`** (one per call, when it ends): `mode` (in-page / isolated), `model` (quality level), `strength`
  (rounded to 10), `dur` (duration bucket), `lat` (delay, ms), `cost` (engine time per 10 ms of audio, µs), `p95`
  (slowest audio callbacks, µs), `load` (model load time, ms), `under` / `over` / `pumpUnder` (audio gaps),
  `offShare` (% of the call with suppression off), `switches` (quality changes).
- **`error`**: `code` (`start`, `engine`, …), `msg` (the error text, 200 characters; addresses in it are replaced by
  `<url>`, or by their domain when detailed diagnostics are on), `mode`, `model`.
- **`settings`**: `key` (`model`, `strength`, `enabled`, `mode`, `lang`, `diagnostics`) and its new value. The per-site
  list is never sent.

Limits in the extension: at most 40 reports per install per day, one per error code per day, a queue of at most 100
kept while offline, sent in batches of up to 20 with back-off on failure.

## Where reports go

`https://t.parla.cc/v1/e`: a small receiver on Cloudflare, [`src/telemetry/worker.ts`](../src/telemetry/worker.ts) (the
whole receiver; `node src/telemetry/worker.test.ts` tests it). It accepts only the fields above (unknown fields are
dropped, numbers clamped, words from fixed lists, strings cut) and keeps no IP address or headers.

**Kept**: each report for at most 7 days; a summary without page addresses (type, browser, OS, app kind, quality,
mode, versions, language, error code, call length, delay, engine cost) for at most 3 months. Nothing else.

**Turning reports down**: every answer from the receiver says which share of installs should report
(`x-parla-sample`); the extension keeps it and reports only if its random ID falls under it, so `0` turns reports off
for every install without an extension update. When the receiver has had enough reports for the day it answers `429`
with `Retry-After`, and extensions pause until then (keeping at most 100 queued).

Installs that turned reports off are not counted anywhere; the stores' own numbers (Chrome Web Store weekly users,
Firefox Add-ons daily users) include them.

## Running the receiver

The receiver is part of this repository and runs as a Cloudflare Worker. Deploy it (or your own copy, for a fork)
with the standard library only, no Cloudflare tooling:

```sh
export CLOUDFLARE_API_TOKEN=...   # Workers Scripts: Edit, Workers Custom Domains: Edit, Zone: Read
export CLOUDFLARE_ACCOUNT_ID=...
python build.py telemetry-deploy [--sample 0..1] [--daily-limit N] [--log-sampling 0..1] [--host H] [--name N]
```

`--sample` is the share of installs that report (`0` = off for everyone), `--daily-limit` the events accepted per UTC
day. A fork that runs its own receiver also points its extension at it (`src/extension/src/telemetry.ts`). The
release workflow deploys it on every merge into `master` ([releasing](releasing.md)).

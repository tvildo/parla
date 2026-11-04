# rust/vi_otel — tiny OpenTelemetry-compatible telemetry (metrics + traces)

Std-only (no third-party crates), native + wasm32. Real-time safe recording on the audio thread,
OTLP/HTTP **JSON** encoding off the audio thread, transport outside the core (JS `fetch` in the
browser, optional `std::net` HTTP natively). Not wired into the engine yet (see "Integration plan").

| Path | What |
|---|---|
| `src/common.rs` | `Key`, `StringValue`, `Value`, `KeyValue` (all `Copy`, `&'static str`), `InstrumentationScope`, attribute-set hash, `Stats` / `DropReason` |
| `src/metrics.rs` | `SdkMeterProvider`, `MeterProvider` trait, `Meter`, `InstrumentBuilder`, `Counter<u64/f64>`, `UpDownCounter<i64>`, `Gauge<u64/i64/f64>`, `Histogram<f64>` + `Bound*` handles |
| `src/trace.rs` | `SdkTracerProvider`, `TracerProvider` trait, `Tracer`, `SpanBuilder`, `Span`, `SpanContext`, `Context`, `Sampler`, `SpanKind`, `Status`, lock-free span ring |
| `src/otlp.rs`, `src/json.rs` | OTLP JSON encoders; fmt-free JSON writer (ints, hex, escaped strings, doubles) |
| `src/time.rs` | pluggable clock (`set_now_ns`; wasm: host-pushed), SplitMix64 id generator |
| `src/global.rs` | lazily created global providers (`global::meter`, `global::tracer`, `update_resource`) |
| `src/c_abi.rs` | feature `c-abi`: wasm exports `otel_*` |
| `src/http.rs` | feature `native-http`: HTTP/1.1 POST over `TcpStream`, `OtlpHttpExporter` |
| `../js/otel_export.ts` | browser transport (6.3 KB raw / 2.6 KB gzip, no deps); `js/test_otel_export.ts` Node test against the real wasm |
| `tests/otel.rs`, `tests/golden/*.json` | goldens + semantics; `tests/no_alloc.rs` counting-allocator check |
| `examples/bench.rs`, `examples/collector_demo.rs` | micro-bench; simulated engine telemetry -> real collector (one-shot validation, or `--forever` multi-session generator for the dashboard) |
| `size_probe/` | standalone wasm cdylib (own `[workspace]`) measuring the wasm cost |

Features: `c-abi`, `native-http`, `single-thread` (plain cells instead of atomics; only sound if a
provider is never shared across threads; on wasm32 without `+atomics` atomics already compile to plain
loads/stores, so it changes nothing there).

## API (subset of opentelemetry-rust, same shapes)

```rust
use vi_otel::{global, KeyValue, Context, Resource};
use vi_otel::metrics::{MeterProvider, SdkMeterProvider};
use vi_otel::trace::{SdkTracerProvider, TracerProvider, Sampler, Status};

global::set_resource(Resource::builder().with_service_name("vi-engine")
    .with_attributes([KeyValue::new("service.version", "0.3.0")]).build());
let m = global::meter("vi.engine");
let dur = m.f64_histogram("vi.engine.process.duration").with_unit("us")
    .with_boundaries(vec![25., 50., 100., 200., 400., 800., 1600., 2667., 5000.]).build();
let dur_gtcrn = dur.bind(&[KeyValue::new("vi.model", "gtcrn")]);      // setup
let overruns = m.u64_counter("vi.engine.overruns").build();
dur_gtcrn.record(123.0);                                                // audio thread: ~14 ns
overruns.add(1, &[KeyValue::new("vi.model", "gtcrn")]);                // audio thread: ~18 ns (hash + probe)

let t = global::tracer("vi.engine");
let mut load = t.start("vi.model.load");
load.set_attribute(KeyValue::new("vi.model", "gtcrn"));
let mut compile = t.start_with_context("vi.wasm.compile", &Context::from_span(&load));
compile.add_event("module.compiled", [KeyValue::new("bytes", 250_072i64)]);
compile.end();
load.set_status(Status::Ok);
load.end();                                                             // or drop

let mut buf = Vec::new();                                               // exporter thread
global::meter_provider().export_json(&mut buf);                         // -> POST /v1/metrics
let mut scratch = Vec::new();
global::tracer_provider().export_json(&mut scratch, 512, &mut buf);     // -> POST /v1/traces
```

Non-global providers: `SdkMeterProvider::builder().with_resource(r).with_start_time(ns).with_stats(s)
.with_cardinality_limit(n).build()`, `SdkTracerProvider::builder().with_sampler(..).with_span_capacity(n)
.with_id_seed(s).with_stats(s).with_enabled(b).build()`. Share one `Arc<Stats>` between the two so span
drops show up in the metrics export.

### Mapping to the official crates

| Official (`opentelemetry` 0.2x / `opentelemetry_sdk`) | vi_otel | Notes |
|---|---|---|
| `KeyValue::new(k, v)`, `Key`, `Value::{Bool,I64,F64,String}`, `StringValue` | same | no `Value::Array`; strings are `&'static str` (`From<String>` leaks: setup-time only) |
| `global::meter(name)`, `global::tracer(name)`, `global::set_meter_provider` / `set_tracer_provider` | same | providers are lazily created (no no-op phase); `set_*` returns false once in use; extra `global::update_resource` |
| `MeterProvider::meter`, `meter_with_scope(InstrumentationScope)` | same | scope = name + version (no schema url / scope attributes) |
| `meter.u64_counter / f64_counter / i64_up_down_counter / f64_histogram / u64_gauge / i64_gauge / f64_gauge` | same | no `f64_up_down_counter`, `u64_histogram`, observable (async) instruments |
| `.with_description().with_unit().build()`, histogram `.with_boundaries(Vec<f64>)` | same | extra `.with_cardinality_limit(n)` (official: views) |
| `counter.add(v, &attrs)`, `histogram.record(v, &attrs)`, `gauge.record(v, &attrs)` | same | extra `bind(&attrs)` -> `Bound*` handle (the official API removed bound instruments) |
| `SdkMeterProvider::builder().with_resource().build()` | same | no readers / exporters / views / delta temporality; pull with `export_json` |
| `Resource::builder().with_service_name().with_attributes().build()`, `builder_empty()` | same | defaults: `service.name=unknown_service`, `telemetry.sdk.{name=vi_otel,language=rust,version}` |
| `tracer.start(name)`, `start_with_context(name, &cx)`, `SpanBuilder::from_name(..).with_kind().with_start_time().with_attributes().start(&t)` | same | inherent methods (no `Tracer`/`Span` traits to import); names `&'static str`; start time = ns since epoch (official: `SystemTime`) |
| `span.set_attribute`, `set_attributes`, `add_event(name, attrs)`, `set_status(Status::Ok / Status::error(..))`, `update_name`, `end()`, `end_with_timestamp`, `span_context()`, `is_recording()` | same | `add_event` takes any `IntoIterator<Item = KeyValue>` (arrays: no alloc); `end_with_timestamp(u64 ns)`; no links, no `record_error` |
| `Context` + `TraceContextExt::with_remote_span_context` | `Context::new().with_remote_span_context(sc)`, `Context::from_span(&span)` | explicit only: no `Context::current()` / attach guards |
| `Sampler::{AlwaysOn, AlwaysOff, TraceIdRatioBased}`, `ParentBased` | same | children always follow the parent (implicit ParentBased); `set_sampler` at runtime |
| `SpanKind`, `Status`, `TraceId`, `SpanId`, `SpanContext::{trace_id, span_id, is_sampled, is_valid}` | same | no `TraceState`, flags reduced to `sampled` |
| propagators, baggage, logs, exporters (OTLP gRPC / protobuf), batch processors | omitted | transport is the host's job (JS fetch / `http.rs`) |

Swapping in the official crate later: keep call sites on the shared subset, put `vi_otel` vs
`opentelemetry` behind a cargo feature in a thin `telemetry` module; the main source changes are trait
imports (`use opentelemetry::trace::{Tracer, Span, TraceContextExt}`), `SystemTime` start times, and
replacing `bind()` handles with plain `add`/`record` (the official SDK resolves attributes per call,
allocating/locking: not audio-thread safe — that is why vi_otel exists).

## Real-time safety rules

Recording after setup never allocates, locks, formats or does syscalls (except the native clock read in
`span.start/end` without explicit timestamps; wasm reads a host-pushed atomic). Verified by
`tests/no_alloc.rs` (counting global allocator, includes slot claims, overflow and ring-full drops).

- **Setup (any thread, may allocate/lock):** building providers, meters, instruments, `bind()`, resources,
  `KeyValue` from `String`, `global::*` first use.
- **Audio thread:** `add` / `record` (bound or with a small `&[KeyValue]` of static strings), span
  `start` / `set_attribute` / `add_event` / `end`. Never build attributes from `format!`/`to_string()` there.
- Each instrument has a fixed table of attribute-set slots (cardinality limit, default 8, power of two)
  + slot 0 = overflow series `otel.metric.overflow=true` (OTel SDK spec behaviour). A new attribute set
  claims a slot with one CAS and copies its (Copy) attributes in; the exporter reads slots after an
  acquire flag. Attribute sets are hashed order-independently (FNV-1a per key/value, SplitMix finalizer,
  63-bit); hash equality = same series (no deep compare; collision chance negligible at 8 sets).
- Values are 64-bit lock-free cells: u64/i64 `fetch_add`, f64 via CAS loop (sum, min, max), gauges
  store bits. Histogram slot = `[count, sum, min, max, buckets...]`, bucket via binary search over the
  explicit bounds (OTel semantics: bucket i = `(b[i-1], b[i]]`). Exported `count` = sum of buckets read.
- Spans are fixed-size `Copy` records (8 attributes, 4 events x 3 attributes; overflow counted in the
  OTLP `droppedAttributesCount` / `droppedEventsCount`). `end()` pushes into a bounded lock-free MPMC
  ring (Vyukov; capacity default 128, `SpanData` = 1032 B native); full ring = span dropped and counted.
- Drops: `vi_otel.dropped{reason}` (scope `vi_otel`, monotonic sum, emitted when non-zero):
  `span_queue_full`, `metric_cardinality`, `metric_invalid_value` (NaN histogram sample, negative/NaN
  counter increment), `metric_attributes_truncated` (> 6 attributes).
- Temporality is cumulative (start = provider start), so a lost metrics POST loses nothing; spans are
  at-most-once.

## Encoding (OTLP/HTTP JSON, `Content-Type: application/json`)

opentelemetry-proto JSON mapping: lowerCamelCase fields; 64-bit integers as strings
(`startTimeUnixNano`, `timeUnixNano`, `asInt`, `count`, `bucketCounts`, `intValue`); doubles as JSON
numbers (`asDouble`, `sum`, `min`, `max`, `explicitBounds`, `doubleValue`; NaN/Infinity as the strings
`"NaN"`/`"Infinity"`); `traceId`/`spanId`/`parentSpanId` lowercase hex; enums as integers
(`aggregationTemporality: 2` = cumulative, `kind: 1` = internal, `status.code` 1 ok / 2 error).
Sum: `aggregationTemporality` + `isMonotonic` (counters true, up-down false); gauge: data points without
start time; histogram: `count, sum, bucketCounts, explicitBounds, min, max`. Unused instruments/scopes
are omitted. Doubles: exact integers below 2^53, otherwise 15 significant digits (round-trip tested to
1e-14 relative), written without `core::fmt` (`json.rs`). Encoders append to a caller-owned `Vec<u8>`
(capacity reused). Examples: `tests/golden/metrics.json`, `tests/golden/spans.json`.

## Transport

**Browser (wasm, feature `c-abi`)** — the host owns the network:

| Export | |
|---|---|
| `otel_buf_alloc(n) -> ptr`, `otel_buf_free(ptr, n)` | scratch memory |
| `otel_set_time_ms(f64)` / `otel_set_time_ns(u64)` | wall clock (wasm has none): call per quantum / before snapshots |
| `otel_seed(lo, hi)` | id PRNG seed (`crypto.getRandomValues`) |
| `otel_set_enabled(i32)` | **opt-in, default off**: snapshots return 0, spans don't record |
| `otel_set_trace_ratio(f64)`, `otel_set_span_capacity(u32)` | sampling; ring size (before first span) |
| `otel_resource_str/num/bool(...)` | resource attributes (leaked, set once) |
| `otel_snapshot_metrics(buf, cap)`, `otel_snapshot_spans(buf, cap)` | `>0` bytes, `0` nothing/disabled, `<0` = -needed (spans stay staged) |
| `otel_dropped(reason) -> f64`, `otel_instrument(kind, name, unit, bounds) -> id`, `otel_record(id, v)` | stats; JS-side instruments without attributes (WebRTC stats etc., lock-free) |

`web/otel_export.js`: `otelBind(exports)` (worklet/wasm side: `tick`, `seed`, `enable`, `ratio`,
`resource`, `instrument`, `snapshot`) and `createOtelExporter({endpoint, snapshot, intervalMs=15000,
enabled=false, sampleRate=1, headers, maxQueue=16, onError, fetch})` (main thread: interval batching,
session sampling, retry on 408/429/5xx with 1 s..5 min jittered exponential backoff, newer metrics
snapshot supersedes queued ones, 4xx dropped, `pagehide`/hidden -> `fetch(keepalive)` within the 64 KiB
budget else `navigator.sendBeacon`). In an AudioWorklet there is no `fetch`: call `snapshot()` in the
worklet, post the two `Uint8Array`s over the port, and give the exporter `snapshot: () => lastReceived`.
`browserResource(extra)` = coarse attributes only (brand/version/platform/mobile, cores, device memory).
Collector side: a host serves the collector same-origin (e.g. an `/otlp/` reverse-proxy path), so no CORS is needed.

**Native (feature `native-http`)**: `http::post(addr, path, body, timeout)` (HTTP/1.1, `Connection:
close`, plain http only) and `OtlpHttpExporter::new("127.0.0.1:4318").export_metrics(&mp) /
export_spans(&tp)`; run it on its own thread every N seconds.

## Budgets (measured 2026-10-23)

Recording (i9-11900K, Windows, `cargo run --release --example bench`, best of 7 x 20M, 0 allocations):

| Operation | ns/op |
|---|---|
| bound u64 counter add | 4.1 |
| u64 counter add with 2 attributes (hash + probe) | 18.3 |
| f64 counter add, no attributes (CAS) | 8.8 |
| bound f64 histogram record (16 bounds: bucket + count + sum + min + max) | 13.2 |
| f64 histogram record with 1 attribute | 21.6 |
| bound f64 gauge record | 0.5 |
| span start + 1 attribute + 1 event + end (2 `SystemTime` reads, 1 KB ring copy) | 123 |
| metrics `export_json` (6 instruments, 1.5 KB) / spans (512 spans, 111 KB) | 0.8-1.5 us / 150-230 us |

Wasm (`size_probe`, wasm32-unknown-unknown, opt-level z, panic=abort, + `wasm-opt -Oz` binaryen 133):

| | raw | gzip |
|---|---|---|
| baseline (std + dlmalloc + panic) | 14,831 | 6,732 |
| + vi_otel (all instruments, spans, encoders, C ABI) | 42,746 | 19,803 |
| delta | **27.9 KB** | **13.1 KB** |
| same with LTO: 8,034 -> 35,494 | 27.5 KB | 12.6 KB |

Attribution of the delta (code-section bytes before wasm-opt, from the `name` section): encoding
(otlp + json) 6.8 KB + recording (slots, cells, spans, ring) 4.5 KB = **11.3 KB (within the ~15 KB
recording+encoding budget)**; setup/builders/resource 4.4 KB; C ABI 2.9 KB; std pulled in (realloc,
`OnceLock`, `Mutex`, `from_utf8`, Vec growth) ~8.7 KB — most of which the engine bundle already
contains, so the marginal cost inside `vi_rs_*.wasm` will be lower. `core::fmt` is not pulled in by
vi_otel. Removed during tuning: `slice::sort` (4 KB, now insertion), `String::from_utf8_lossy` (1.4 KB),
`Box<[T]>` shrink paths, `OnceLock` per JS instrument, initialized span arrays (`MaybeUninit`).

JS helper: 6,327 B raw / 2,650 B gzip.

## Tests

`cargo test --release -p vi_otel` (+ `--all-features`, `--features single-thread`): JSON doubles
(goldens + 20k random round-trips), string escaping, metrics/spans goldens (`UPDATE_GOLDEN=1` to
regenerate, review the diff), histogram bucketing (unsorted/duplicate/NaN bounds, NaN sample),
attribute order independence + bind sharing + instrument dedupe, cardinality overflow, attribute
truncation, ring-buffer drops + reuse + drop metric, sampling ratio/off/parenting/disable, span
attribute/event limits, 4-thread concurrent recording, no allocation while recording.
JS: `node src/engine/vi_otel/js/test_otel_export.ts [otel.wasm] [collector url]` (needs
`size_probe/out/otel.wasm`: snapshot/resource/JS instruments/drain/negative-size retry; exporter
retry/backoff/supersede with a fake fetch; optional real POST).

## Collector validation / local viewer (Windows box)

Validated 2026-10-23 against the official collector, twice with a debug-exporter collector config (former `compose/otelcol-debug.yaml`, in git history):
Docker image `otel/opentelemetry-collector-contrib:0.162.0` (digest `sha256:39923a8e...74be6`) and the
same-version native release binary `otelcol_0.162.0_windows_amd64.tar.gz` (sha256 `480381f4...146b8f82`
= published `.sha256`; used while the Docker daemon was busy). The golden files, `collector_demo`
(histograms/sums/gauges incl. cardinality overflow + `vi_otel.dropped`, parent/child spans) and the JS
helper over the real wasm C ABI all returned `200 {"partialSuccess":{}}` (no rejected points), and the
`debug` exporter printed every value as encoded (sum monotonic/cumulative, gauges, histogram
count/sum/min/max/bounds/bucket counts, bool/int/double/string attributes, trace/parent/span ids,
kind, status code + message, events). Malformed JSON -> HTTP 400 (negative control).

Any OTLP/HTTP collector works (e.g. collector -> Prometheus + Tempo -> Grafana). `collector_demo` (example) stays as a one-shot validator.
Prometheus names: dots -> underscores plus unit/`_total` suffixes (`vi_engine_process_duration_microseconds_bucket`,
`vi_audio_input_level_dBFS_bucket`, `vi_audio_suppression_db_dB_bucket`, `vi_engine_arena_bytes`,
`vi_webrtc_packets_lost_total`, `vi_otel_dropped_total{reason}`); attributes become labels
(`vi.model` -> `vi_model`, `browser.name` -> `browser_name`), `job` = `service.name`, `instance` =
`service.instance.id`. Every panel was checked through Grafana `/api/ds/query`.

## Proposed engine telemetry (not wired yet — engine files belong to the engine work)

Metric names follow OTel semantic-convention style (`vi.` namespace, units UCUM).

| Instrument | Kind / unit | Attributes | Where |
|---|---|---|---|
| `vi.engine.process.duration` | histogram, `us`, bounds 25..5000 (2667 = 128 frames at 48 kHz) | `vi.model`, `vi.backend` | per render quantum / `engine_process` call (bound handle) |
| `vi.engine.frame.duration` | histogram, `us` | `vi.model` | per model frame (hop) |
| `vi.engine.overruns` | counter `{quantum}` | `vi.model` | processing time > real-time budget |
| `vi.engine.underruns` | counter `{quantum}` | `vi.model` | worklet had no output ready (ring underflow) |
| `vi.engine.op.duration` | histogram, `us` | `vi.model`, `vi.op` (bounded: top op kinds) | sampled: every Nth frame via `run_profiled` |
| `vi.engine.arena.bytes` | gauge `By` | `vi.model` | after graph planning |
| `vi.engine.resets` | counter | `vi.model`, `vi.reason` (`nan`, `inf`, `manual`) | non-finite output -> state reset |
| `vi.model.load` | span (+ events `weights.fetched`, `graph.planned`) | `vi.model`, `vi.model.sha256`, bytes | manifest/package load |
| `vi.wasm.compile.duration` | histogram `ms` + span `vi.wasm.compile` | `vi.wasm.variant` (simd/relaxed/nosimd), plugin | JS: `WebAssembly.compile` (JS instrument) |
| `vi.audio.input.level`, `vi.audio.output.level` | histogram `dBFS` (-90..0) | — | per quantum RMS (no audio content) |
| `vi.audio.suppression.db` | histogram `dB` | `vi.model` | input vs output energy per quantum |
| `vi.audio.speech_presence` | histogram `1` (0..1) | `vi.model` | model VAD/mask mean where available |
| `vi.webrtc.jitter` | histogram `ms` | — | JS `getStats()` (JS instrument) |
| `vi.webrtc.packets_lost`, `vi.webrtc.packets_received`, `vi.webrtc.concealed_samples` | counter | — | JS `getStats()` deltas (received: loss %) |
| `vi.model.load.duration` | histogram `ms` | `vi.model` | same interval as the `vi.model.load` span (spans are not in Prometheus) |

Resource attributes (set once by JS through `otel_resource_*`): `service.name=vi-web`,
`service.version`, `vi.engine.build` (git hash), `browser.name`, `browser.version`, `os.type`,
`browser.mobile`, `host.cpu.count` (hardwareConcurrency), `host.memory.gb` (deviceMemory),
`vi.wasm.simd`, `vi.wasm.relaxed_simd`, `audio.sample_rate`, `audio.base_latency`, `vi.model`,
`vi.model.sha256`.
Keep `vi.model` on data points too (dashboard filter); with the collector's resource-to-label
conversion a resource attribute and a data-point attribute of the same key collide on one label.

Privacy: never audio samples, transcripts, user ids, IP-derived data, full UA strings or URLs with
query strings; levels/suppression are aggregated histograms only. Opt-in, **default off**
(`otel_set_enabled`, `createOtelExporter({enabled})`), session sampling (`sampleRate`), trace ratio
sampling; the collector config additionally deletes `user.id` / `client.address`.

## Browser integration (done 2026-10-24)

- Separate module `rust/vi_otel_wasm` -> `plugins/vi_otel.wasm` (39.9 KB raw / 18.5 KB gzip; engine bundles
  unchanged). C ABI extended with `otel_instrument_attrs` (instrument bound to a `key=value` line set;
  cardinality 64 per JS instrument, up to 512 handles).
- `web/worklet.js` (`{type:'otel'}`): per lane, instruments bound to `vi.model` / `vi.build` /
  `vi.engine`; the active lane closes a window every 375 quanta (1 s @ 48 kHz) into `vi.engine.cost`
  (us of engine time per 10 ms of audio), `vi.worklet.load` (% of the quantum budget), `vi.worklet.quanta` /
  `overruns` / `underruns`, `vi.audio.input.level` / `output.level` (dBFS), `vi.audio.suppression` (dB),
  `vi.engine.latency` (ms). Snapshots on `{type:'otelSnapshot'}`; main-thread values via `{type:'otelRecord'}`.
- A host page turns it on (opt-in) and runs `createOtelExporter` against its collector path; resource = coarse
  browser attributes. The extension and the website send no telemetry.
- Not done: engine-internal (Rust-side) spans/op timings; WebRTC stats metrics.

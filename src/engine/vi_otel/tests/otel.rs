//! Encoding goldens + recording semantics. Regenerate goldens: `UPDATE_GOLDEN=1 cargo test -p vi_otel`.

use std::path::PathBuf;
use std::sync::Arc;
use vi_otel::metrics::{MeterProvider, SdkMeterProvider};
use vi_otel::trace::{Sampler, SdkTracerProvider, SpanKind, Status, TracerProvider};
use vi_otel::{Context, DropReason, InstrumentationScope, KeyValue, Resource, Stats};

const T0: u64 = 1_761_200_000_000_000_000; // 2025-10-23T06:13:20Z
const MS: u64 = 1_000_000;

fn golden(name: &str, got: &[u8]) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() || !p.exists() {
        std::fs::write(&p, got).unwrap();
        return;
    }
    let want = std::fs::read(&p).unwrap();
    assert!(
        want == got,
        "{name} differs from golden\n got: {}\nwant: {}",
        String::from_utf8_lossy(got),
        String::from_utf8_lossy(&want)
    );
}

fn resource() -> Resource {
    Resource::builder()
        .with_service_name("vi-test")
        .with_attributes([KeyValue::new("service.version", "0.1.0"), KeyValue::new("vi.engine.build", "abc123")])
        .build()
}

fn count(hay: &[u8], needle: &str) -> usize {
    let h = String::from_utf8_lossy(hay);
    h.matches(needle).count()
}

#[test]
fn golden_metrics() {
    let mp = SdkMeterProvider::builder().with_resource(resource()).with_start_time(T0).build();
    let m = mp.meter_with_scope(InstrumentationScope::builder("vi.engine").with_version("0.1.0").build());
    let frames = m.u64_counter("vi.engine.frames").with_description("Frames processed").with_unit("{frame}").build();
    frames.add(3, &[KeyValue::new("vi.model", "gtcrn")]);
    frames.add(2, &[KeyValue::new("vi.model", "dfn")]);
    frames.add(4, &[KeyValue::new("vi.model", "gtcrn")]);
    let energy = m.f64_counter("vi.audio.energy").build();
    energy.add(0.25, &[]);
    energy.add(1.5, &[]);
    let inflight = m.i64_up_down_counter("vi.engine.instances").build();
    inflight.add(-3, &[]);
    inflight.add(5, &[]);
    let arena = m.u64_gauge("vi.engine.arena.bytes").with_unit("By").build();
    arena.record(123_456, &[KeyValue::new("vi.model", "gtcrn")]);
    let level = m.f64_gauge("vi.audio.input.level").with_unit("dBFS").build();
    level.record(-23.5, &[]);
    let dur = m
        .f64_histogram("vi.engine.process.duration")
        .with_description("Engine time per render quantum")
        .with_unit("us")
        .with_boundaries(vec![100.0, 200.0, 500.0])
        .build();
    let b = dur.bind(&[KeyValue::new("vi.model", "gtcrn"), KeyValue::new("vi.simd", true)]);
    for v in [50.0, 150.0, 150.5, 1000.0] {
        b.record(v);
    }
    // unused instrument: not exported
    let _unused = m.u64_counter("vi.engine.unused").build();
    let mut out = Vec::new();
    mp.export_json_at(T0 + 10_000 * MS, &mut out);
    golden("metrics.json", &out);
}

#[test]
fn golden_spans() {
    let tp = SdkTracerProvider::builder().with_resource(resource()).with_id_seed(42).build();
    let t = tp.tracer_with_scope(InstrumentationScope::builder("vi.engine").with_version("0.1.0").build());
    let mut load = t
        .span_builder("vi.model.load")
        .with_kind(SpanKind::Internal)
        .with_start_time(T0)
        .with_attributes([KeyValue::new("vi.model", "gtcrn"), KeyValue::new("vi.model.bytes", 2_100_000i64)])
        .start(&t);
    load.add_event_with_timestamp("weights.parsed", T0 + 12 * MS, [KeyValue::new("tensors", 87i64)]);
    let cx = Context::from_span(&load);
    let mut compile = t.span_builder("vi.wasm.compile").with_start_time(T0 + 13 * MS).start_with_context(&t, &cx);
    compile.set_attribute(KeyValue::new("vi.wasm.simd", true));
    compile.set_attribute(KeyValue::new("vi.wasm.ratio", 0.5));
    compile.set_status(Status::Ok);
    compile.end_with_timestamp(T0 + 40 * MS);
    load.set_status(Status::error("bad \"manifest\""));
    load.end_with_timestamp(T0 + 55 * MS);
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    assert_eq!(tp.export_json(&mut scratch, 100, &mut out), 2);
    golden("spans.json", &out);
    // ring drained
    assert_eq!(tp.export_json(&mut scratch, 100, &mut out), 0);
    assert!(out.is_empty());
}

#[test]
fn histogram_bucketing() {
    let mp = SdkMeterProvider::builder().with_start_time(T0).build();
    let h = mp.meter("t").f64_histogram("h").with_boundaries(vec![5.0, 1.0, 2.0, 2.0, f64::NAN]).build();
    for v in [0.5, 1.0, 1.5, 2.0, 5.0, 6.0, f64::NAN, -1.0] {
        h.record(v, &[]);
    }
    let mut out = Vec::new();
    mp.export_json_at(T0, &mut out);
    let s = String::from_utf8(out).unwrap();
    // (-inf,1] (1,2] (2,5] (5,inf)
    assert!(s.contains("\"count\":\"7\",\"sum\":15,\"bucketCounts\":[\"3\",\"2\",\"1\",\"1\"],\"explicitBounds\":[1,2,5],\"min\":-1,\"max\":6"), "{s}");
    assert_eq!(mp.stats().get(DropReason::MetricInvalidValue), 1);
    assert!(s.contains("metric_invalid_value"), "{s}");
}

#[test]
fn bind_and_unordered_attributes_share_a_slot() {
    let mp = SdkMeterProvider::builder().with_start_time(T0).build();
    let c = mp.meter("t").u64_counter("c").build();
    let a = KeyValue::new("a", 1i64);
    let b = KeyValue::new("b", "x");
    let bound = c.bind(&[a, b]);
    bound.add(10);
    c.add(5, &[b, a]);
    let c2 = mp.meter("t").u64_counter("c").build(); // same instrument
    c2.add(1, &[a, b]);
    c.add(7, &[a]);
    let mut out = Vec::new();
    mp.export_json_at(T0, &mut out);
    assert_eq!(count(&out, "\"asInt\""), 2, "{}", String::from_utf8_lossy(&out));
    assert_eq!(count(&out, "\"asInt\":\"16\""), 1);
    assert_eq!(count(&out, "\"asInt\":\"7\""), 1);
}

#[test]
fn cardinality_overflow() {
    let mp = SdkMeterProvider::builder().with_start_time(T0).build();
    let c = mp.meter("t").u64_counter("c").with_cardinality_limit(2).build();
    for i in 0..5i64 {
        c.add(1, &[KeyValue::new("i", i)]);
    }
    let mut out = Vec::new();
    mp.export_json_at(T0, &mut out);
    assert_eq!(count(&out, "otel.metric.overflow"), 1);
    assert_eq!(count(&out, "\"asInt\":\"3\""), 2, "{}", String::from_utf8_lossy(&out)); // overflow series + drop counter
    assert_eq!(mp.stats().get(DropReason::MetricCardinality), 3);
}

#[test]
fn attributes_truncated() {
    let mp = SdkMeterProvider::builder().with_start_time(T0).build();
    let g = mp.meter("t").f64_gauge("g").build();
    let kv: Vec<KeyValue> = (0..9i64).map(|i| KeyValue::new(["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"][i as usize], i)).collect();
    g.record(1.0, &kv);
    assert_eq!(mp.stats().get(DropReason::MetricAttributesTruncated), 1);
    let mut out = Vec::new();
    mp.export_json_at(T0, &mut out);
    assert_eq!(count(&out, "\"key\":\"k"), vi_otel::metrics::MAX_ATTRS);
}

#[test]
fn ring_buffer_drops() {
    let stats = Arc::new(Stats::default());
    let tp = SdkTracerProvider::builder().with_span_capacity(4).with_stats(stats.clone()).build();
    let mp = SdkMeterProvider::builder().with_stats(stats.clone()).with_start_time(T0).build();
    let t = tp.tracer("t");
    for _ in 0..10 {
        t.start("s").end();
    }
    assert_eq!(tp.pending(), 4);
    assert_eq!(tp.dropped_spans(), 6);
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    assert_eq!(tp.export_json(&mut scratch, 3, &mut out), 3);
    assert_eq!(count(&out, "\"spanId\""), 3);
    assert_eq!(tp.export_json(&mut scratch, 100, &mut out), 1);
    // ring reusable after draining
    for _ in 0..4 {
        t.start("s2").end();
    }
    assert_eq!(tp.pending(), 4);
    assert_eq!(tp.dropped_spans(), 6);
    // shared stats -> exported as metric by the meter provider
    mp.export_json_at(T0, &mut out);
    assert!(count(&out, "\"stringValue\":\"span_queue_full\"}}],\"startTimeUnixNano\":\"1761200000000000000\",\"timeUnixNano\":\"1761200000000000000\",\"asInt\":\"6\"") == 1, "{}", String::from_utf8_lossy(&out));
}

#[test]
fn sampling_and_parenting() {
    let tp = SdkTracerProvider::builder().with_sampler(Sampler::AlwaysOff).with_id_seed(1).build();
    let t = tp.tracer("t");
    let s = t.start("off");
    assert!(!s.is_recording());
    assert!(s.span_context().is_valid());
    drop(s);
    assert_eq!(tp.pending(), 0);

    tp.set_sampler(Sampler::TraceIdRatioBased(0.25));
    let mut n = 0;
    for _ in 0..4000 {
        if t.start("r").is_recording() {
            n += 1;
        }
    }
    assert!((800..1200).contains(&n), "{n}");

    tp.drain_into(&mut Vec::new(), usize::MAX);
    tp.set_sampler(Sampler::AlwaysOn);
    let parent = t.start("p");
    let child = t.start_with_context("c", &Context::from_span(&parent));
    assert_eq!(child.span_context().trace_id(), parent.span_context().trace_id());
    assert_ne!(child.span_context().span_id(), parent.span_context().span_id());
    drop(child);
    drop(parent);
    let mut v = Vec::new();
    tp.drain_into(&mut v, 10000);
    assert_eq!(v.len(), 2);
    let last = &v[..];
    assert_eq!(last[0].name(), "c");
    assert_eq!(last[0].parent_span_id(), last[1].span_context().span_id());

    // disabled provider: nothing recorded
    tp.set_enabled(false);
    assert!(!t.start("x").is_recording());
}

#[test]
fn span_attribute_and_event_limits() {
    let tp = SdkTracerProvider::builder().build();
    let t = tp.tracer("t");
    let mut s = t.start("s");
    let keys = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"];
    for k in keys {
        s.set_attribute(KeyValue::new(k, 1i64));
    }
    s.set_attribute(KeyValue::new("a0", 2i64)); // overwrite, not a drop
    for _ in 0..6 {
        s.add_event("e", [KeyValue::new("x", 1i64)]);
    }
    s.end();
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    tp.export_json(&mut scratch, 10, &mut out);
    let txt = String::from_utf8_lossy(&out);
    assert!(txt.contains("\"droppedAttributesCount\":2"), "{txt}");
    assert!(txt.contains("\"droppedEventsCount\":2"), "{txt}");
    assert!(txt.contains("{\"key\":\"a0\",\"value\":{\"intValue\":\"2\"}}"), "{txt}");
}

#[cfg(not(feature = "single-thread"))]
#[test]
fn concurrent_recording() {
    let mp = SdkMeterProvider::builder().with_start_time(T0).build();
    let c = mp.meter("t").u64_counter("c").build();
    let h = mp.meter("t").f64_histogram("h").with_boundaries(vec![0.5]).build();
    let tp = SdkTracerProvider::builder().with_span_capacity(1 << 16).build();
    let t = tp.tracer("t");
    std::thread::scope(|sc| {
        for k in 0..4i64 {
            let (c, h, t) = (c.clone(), h.clone(), t.clone());
            sc.spawn(move || {
                for i in 0..100_000 {
                    c.add(1, &[KeyValue::new("k", k % 2)]);
                    h.record((i % 2) as f64, &[]);
                    if i % 10 == 0 {
                        t.start("s").end();
                    }
                }
            });
        }
    });
    let mut out = Vec::new();
    mp.export_json_at(T0, &mut out);
    let s = String::from_utf8_lossy(&out);
    assert_eq!(s.matches("\"asInt\":\"200000\"").count(), 2, "{s}");
    assert!(s.contains("\"count\":\"400000\",\"sum\":200000,\"bucketCounts\":[\"200000\",\"200000\"]"), "{s}");
    assert_eq!(tp.pending(), 40_000);
}

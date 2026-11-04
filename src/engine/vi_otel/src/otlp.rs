//! OTLP/HTTP JSON encoding (opentelemetry-proto JSON mapping, `Content-Type: application/json`):
//! `ExportMetricsServiceRequest` (POST /v1/metrics) and `ExportTraceServiceRequest`
//! (POST /v1/traces). Field names lowerCamelCase, 64-bit integers (timestamps, counts, asInt,
//! bucketCounts, intValue) as decimal strings, doubles (sum/min/max/explicitBounds/asDouble) as
//! JSON numbers, trace/span ids as lowercase hex, enums as integers.

use crate::common::{KeyValue, Value, DROP_NAMES};
use crate::json::W;
use crate::metrics::{lock, Inst, Kind, SdkMeterProvider, H_B0, H_COUNT, H_MAX, H_MIN, H_SUM};
use crate::trace::{SdkTracerProvider, SpanData, Status};

/// AGGREGATION_TEMPORALITY_CUMULATIVE
const CUMULATIVE: &[u8] = b"2";

fn attrs(w: &mut W, kvs: &[KeyValue]) {
    w.raw(b"[");
    for (i, kv) in kvs.iter().enumerate() {
        if i > 0 {
            w.ch(b',');
        }
        w.raw(b"{\"key\":");
        w.str(kv.key.as_str());
        w.raw(b",\"value\":{");
        match kv.value {
            Value::String(s) => {
                w.raw(b"\"stringValue\":");
                w.str(s.as_str())
            }
            Value::I64(v) => {
                w.raw(b"\"intValue\":");
                w.i64_str(v)
            }
            Value::F64(v) => {
                w.raw(b"\"doubleValue\":");
                w.f64(v)
            }
            Value::Bool(v) => {
                w.raw(b"\"boolValue\":");
                w.raw(if v { b"true" } else { b"false" })
            }
        }
        w.raw(b"}}");
    }
    w.raw(b"]");
}

fn scope(w: &mut W, name: &str, version: Option<&str>) {
    w.raw(b"{\"scope\":{\"name\":");
    w.str(name);
    if let Some(v) = version {
        w.raw(b",\"version\":");
        w.str(v);
    }
    w.raw(b"},");
}

fn resource(w: &mut W, kvs: &[KeyValue]) {
    w.raw(b"{\"resource\":{\"attributes\":");
    attrs(w, kvs);
    w.raw(b"},");
}

fn has_data(i: &Inst) -> bool {
    (0..i.n_slots()).any(|s| i.slot_attrs(s).is_some())
}

fn times(w: &mut W, start: Option<u64>, now: u64) {
    if let Some(s) = start {
        w.raw(b",\"startTimeUnixNano\":");
        w.u64_str(s);
    }
    w.raw(b",\"timeUnixNano\":");
    w.u64_str(now);
}

fn metric(w: &mut W, i: &Inst, start: u64, now: u64) {
    w.raw(b"{\"name\":");
    w.str(i.name);
    if !i.description.is_empty() {
        w.raw(b",\"description\":");
        w.str(i.description);
    }
    if !i.unit.is_empty() {
        w.raw(b",\"unit\":");
        w.str(i.unit);
    }
    let (open, start) = match i.kind {
        Kind::CounterU64 | Kind::CounterF64 | Kind::UpDownI64 => (&b",\"sum\":{\"dataPoints\":["[..], Some(start)),
        Kind::GaugeU64 | Kind::GaugeI64 | Kind::GaugeF64 => (&b",\"gauge\":{\"dataPoints\":["[..], None),
        Kind::HistogramF64 => (&b",\"histogram\":{\"dataPoints\":["[..], Some(start)),
    };
    w.raw(open);
    let mut first = true;
    for s in 0..i.n_slots() {
        let a = match i.slot_attrs(s) {
            Some(a) => a,
            None => continue,
        };
        if !first {
            w.ch(b',');
        }
        first = false;
        let c = &i.cells[s * i.stride..(s + 1) * i.stride];
        w.raw(b"{\"attributes\":");
        attrs(w, a);
        times(w, start, now);
        match i.kind {
            Kind::CounterU64 | Kind::GaugeU64 => {
                // asInt is int64: clamp u64 beyond i64::MAX
                w.raw(b",\"asInt\":");
                w.u64_str(c[0].get().min(i64::MAX as u64))
            }
            Kind::UpDownI64 | Kind::GaugeI64 => {
                w.raw(b",\"asInt\":");
                w.i64_str(c[0].get() as i64)
            }
            Kind::CounterF64 | Kind::GaugeF64 => {
                w.raw(b",\"asDouble\":");
                w.f64(f64::from_bits(c[0].get()))
            }
            Kind::HistogramF64 => {
                // count = sum of buckets read now (consistent even while the audio thread records)
                let nb = i.bounds.len() + 1;
                let mut count = 0u64;
                for b in 0..nb {
                    count += c[H_B0 + b].get();
                }
                let _ = c[H_COUNT].get();
                w.raw(b",\"count\":");
                w.u64_str(count);
                w.raw(b",\"sum\":");
                w.f64(f64::from_bits(c[H_SUM].get()));
                w.raw(b",\"bucketCounts\":[");
                for b in 0..nb {
                    if b > 0 {
                        w.ch(b',');
                    }
                    w.u64_str(c[H_B0 + b].get());
                }
                w.raw(b"],\"explicitBounds\":[");
                for (k, x) in i.bounds.iter().enumerate() {
                    if k > 0 {
                        w.ch(b',');
                    }
                    w.f64(*x);
                }
                w.ch(b']');
                if count > 0 {
                    w.raw(b",\"min\":");
                    w.f64(f64::from_bits(c[H_MIN].get()));
                    w.raw(b",\"max\":");
                    w.f64(f64::from_bits(c[H_MAX].get()));
                }
            }
        }
        w.ch(b'}');
    }
    w.ch(b']');
    match i.kind {
        Kind::CounterU64 | Kind::CounterF64 => {
            w.raw(b",\"aggregationTemporality\":");
            w.raw(CUMULATIVE);
            w.raw(b",\"isMonotonic\":true");
        }
        Kind::UpDownI64 => {
            w.raw(b",\"aggregationTemporality\":");
            w.raw(CUMULATIVE);
            w.raw(b",\"isMonotonic\":false");
        }
        Kind::HistogramF64 => {
            w.raw(b",\"aggregationTemporality\":");
            w.raw(CUMULATIVE);
        }
        _ => {}
    }
    w.raw(b"}}");
}

/// `ExportMetricsServiceRequest` JSON (appends to `out`).
pub fn metrics(mp: &SdkMeterProvider, now: u64, out: &mut Vec<u8>) {
    let inner = &*mp.inner;
    let res = lock(&inner.resource);
    let scopes = lock(&inner.scopes);
    let insts = lock(&inner.insts);
    let w = &mut W(out);
    w.raw(b"{\"resourceMetrics\":[");
    resource(w, res.attrs());
    w.raw(b"\"scopeMetrics\":[");
    let mut first_scope = true;
    for (si, sc) in scopes.iter().enumerate() {
        if !insts.iter().any(|i| i.scope == si && has_data(i)) {
            continue;
        }
        if !first_scope {
            w.ch(b',');
        }
        first_scope = false;
        scope(w, sc.name, sc.version);
        w.raw(b"\"metrics\":[");
        let mut first = true;
        for i in insts.iter().filter(|i| i.scope == si && has_data(i)) {
            if !first {
                w.ch(b',');
            }
            first = false;
            metric(w, i, inner.start_ns, now);
        }
        w.raw(b"]}");
    }
    // internal drop counters (non-zero only)
    let st = &inner.stats;
    if st.total() > 0 {
        if !first_scope {
            w.ch(b',');
        }
        scope(w, crate::resource::SDK_NAME, Some(crate::resource::SDK_VERSION));
        w.raw(b"\"metrics\":[{\"name\":\"vi_otel.dropped\",\"description\":\"Telemetry items dropped or degraded by vi_otel\",\"unit\":\"{item}\",\"sum\":{\"dataPoints\":[");
        let mut first = true;
        for (cell, name) in st.0.iter().zip(DROP_NAMES) {
            let v = cell.get();
            if v == 0 {
                continue;
            }
            if !first {
                w.ch(b',');
            }
            first = false;
            w.raw(b"{\"attributes\":[{\"key\":\"reason\",\"value\":{\"stringValue\":");
            w.str(name);
            w.raw(b"}}]");
            times(w, Some(inner.start_ns), now);
            w.raw(b",\"asInt\":");
            w.u64_str(v);
            w.ch(b'}');
        }
        w.raw(b"],\"aggregationTemporality\":2,\"isMonotonic\":true}}]}");
    }
    w.raw(b"]}]}");
}

fn span(w: &mut W, d: &SpanData) {
    w.raw(b"{\"traceId\":");
    w.hex(&d.ctx.trace_id().0);
    w.raw(b",\"spanId\":");
    w.hex(&d.ctx.span_id().0);
    if d.parent.0 != [0; 8] {
        w.raw(b",\"parentSpanId\":");
        w.hex(&d.parent.0);
    }
    w.raw(b",\"name\":");
    w.str(d.name);
    w.raw(b",\"kind\":");
    w.u64(d.kind as u64);
    w.raw(b",\"startTimeUnixNano\":");
    w.u64_str(d.start);
    w.raw(b",\"endTimeUnixNano\":");
    w.u64_str(d.end);
    w.raw(b",\"attributes\":");
    attrs(w, d.attributes());
    if d.dropped_attrs > 0 {
        w.raw(b",\"droppedAttributesCount\":");
        w.u64(d.dropped_attrs as u64);
    }
    w.raw(b",\"events\":[");
    for (k, e) in d.events().iter().enumerate() {
        if k > 0 {
            w.ch(b',');
        }
        w.raw(b"{\"timeUnixNano\":");
        w.u64_str(e.time);
        w.raw(b",\"name\":");
        w.str(e.name);
        w.raw(b",\"attributes\":");
        attrs(w, e.attributes());
        if e.dropped > 0 {
            w.raw(b",\"droppedAttributesCount\":");
            w.u64(e.dropped as u64);
        }
        w.ch(b'}');
    }
    w.ch(b']');
    if d.dropped_events > 0 {
        w.raw(b",\"droppedEventsCount\":");
        w.u64(d.dropped_events as u64);
    }
    w.raw(b",\"status\":{");
    match d.status {
        Status::Unset => {}
        Status::Ok => w.raw(b"\"code\":1"),
        Status::Error { description } => {
            if !description.is_empty() {
                w.raw(b"\"message\":");
                w.str(description);
                w.ch(b',');
            }
            w.raw(b"\"code\":2")
        }
    }
    w.raw(b"}}");
}

/// `ExportTraceServiceRequest` JSON for `spans` (appends to `out`), grouped by scope.
pub fn spans(tp: &SdkTracerProvider, spans: &[SpanData], out: &mut Vec<u8>) {
    let inner = &*tp.inner;
    let res = lock(&inner.resource);
    let scopes = lock(&inner.scopes);
    let w = &mut W(out);
    w.raw(b"{\"resourceSpans\":[");
    resource(w, res.attrs());
    w.raw(b"\"scopeSpans\":[");
    let mut first_scope = true;
    for (si, sc) in scopes.iter().enumerate() {
        let si = si as u32;
        if !spans.iter().any(|d| d.scope == si) {
            continue;
        }
        if !first_scope {
            w.ch(b',');
        }
        first_scope = false;
        scope(w, sc.name, sc.version);
        w.raw(b"\"spans\":[");
        for (k, d) in spans.iter().filter(|d| d.scope == si).enumerate() {
            if k > 0 {
                w.ch(b',');
            }
            span(w, d);
        }
        w.raw(b"]}");
    }
    w.raw(b"]}]}");
}

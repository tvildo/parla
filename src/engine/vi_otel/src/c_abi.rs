//! C ABI over the global providers (feature `c-abi`), for wasm hosts. JS glue: js/otel_export.js.
//! Telemetry is opt-in: until `otel_set_enabled(1)` snapshots return 0 and spans don't record
//! (metric instruments still count locally; nothing leaves the module).
//!
//!   otel_buf_alloc(n) -> ptr / otel_buf_free(ptr, n)       scratch memory for strings / snapshots
//!   otel_set_time_ms(ms: f64) / otel_set_time_ns(ns: u64)  host wall clock (wasm has none)
//!   otel_seed(lo: u32, hi: u32)                            id PRNG seed (crypto randomness)
//!   otel_set_enabled(on: i32)                              opt-in switch (default off)
//!   otel_set_trace_ratio(r: f64)                           root-span sampling ratio 0..1
//!   otel_set_span_capacity(n: u32) -> 1 if applied         before the first span only
//!   otel_resource_str(kp, kl, vp, vl) / otel_resource_num(kp, kl, v: f64, is_int: i32)
//!   otel_resource_bool(kp, kl, v: i32)                     resource attributes (leaked: set once)
//!   otel_snapshot_metrics(buf, cap) -> i32                 OTLP JSON for POST /v1/metrics
//!   otel_snapshot_spans(buf, cap) -> i32                   OTLP JSON for POST /v1/traces
//!       > 0: bytes written; 0: nothing to send / disabled; < 0: buffer too small, -needed
//!       (spans stay staged until a call with a large enough buffer)
//!   otel_dropped(reason: u32) -> f64                       drop counters (4 = total)
//!   otel_instrument(kind, np, nl, up, ul, bounds: *const f64, nb) -> id (0 = error)
//!       kind 0 f64 counter, 1 i64 up-down counter, 2 f64 gauge, 3 f64 histogram
//!   otel_instrument_attrs(kind, np, nl, up, ul, bounds, nb, ap, al) -> id (0 = error)
//!       same, bound to an attribute set given as UTF-8 "key=value" lines (`ap`/`al`); one id per
//!       (instrument, attribute set), e.g. one per model x wasm build. Setup time (allocates).
//!   otel_record(id, v: f64)                                record (lock-free, no alloc)

use crate::common::{DropReason, KeyValue, StringValue, Value};
use crate::global;
use crate::metrics::{BoundCounter, BoundGauge, BoundHistogram, BoundUpDownCounter};
use crate::trace::{Sampler, SpanData};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering::*};
use std::sync::Mutex;

static ENABLED: AtomicBool = AtomicBool::new(false);

struct Staging {
    out: Vec<u8>,
    spans: Vec<u8>,
    scratch: Vec<SpanData>,
}
static STAGING: Mutex<Staging> = Mutex::new(Staging { out: Vec::new(), spans: Vec::new(), scratch: Vec::new() });

unsafe fn s(p: *const u8, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    match std::str::from_utf8(std::slice::from_raw_parts(p, n)) {
        Ok(t) => t.into(),
        Err(_) => "?".into(),
    }
}

#[no_mangle]
pub extern "C" fn otel_buf_alloc(n: usize) -> *mut u8 {
    match Layout::from_size_align(n.max(8), 8) {
        Ok(l) => unsafe { alloc(l) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `p` must come from `otel_buf_alloc(n)` with the same `n`.
#[no_mangle]
pub unsafe extern "C" fn otel_buf_free(p: *mut u8, n: usize) {
    if let Ok(l) = Layout::from_size_align(n.max(8), 8) {
        if !p.is_null() {
            dealloc(p, l)
        }
    }
}

#[no_mangle]
pub extern "C" fn otel_set_time_ms(ms: f64) {
    crate::time::set_now_ns((ms * 1e6) as u64)
}

#[no_mangle]
pub extern "C" fn otel_set_time_ns(ns: u64) {
    crate::time::set_now_ns(ns)
}

#[no_mangle]
pub extern "C" fn otel_seed(lo: u32, hi: u32) {
    let seed = ((hi as u64) << 32) | lo as u64;
    crate::time::seed(seed);
    global::tracer_provider().reseed(seed);
}

#[no_mangle]
pub extern "C" fn otel_set_enabled(on: i32) {
    ENABLED.store(on != 0, Relaxed);
    global::tracer_provider().set_enabled(on != 0);
}

#[no_mangle]
pub extern "C" fn otel_set_trace_ratio(r: f64) {
    global::tracer_provider().set_sampler(Sampler::TraceIdRatioBased(r));
}

#[no_mangle]
pub extern "C" fn otel_set_span_capacity(n: u32) -> i32 {
    global::set_default_span_capacity(n as usize) as i32
}

fn set_res(kv: KeyValue) {
    global::update_resource(|r| r.set(kv));
}

/// # Safety
/// Pointers must reference `kl` / `vl` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn otel_resource_str(kp: *const u8, kl: usize, vp: *const u8, vl: usize) {
    set_res(KeyValue::new(s(kp, kl), s(vp, vl)));
}

/// # Safety
/// `kp` must reference `kl` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn otel_resource_num(kp: *const u8, kl: usize, v: f64, is_int: i32) {
    let v = if is_int != 0 { Value::I64(v as i64) } else { Value::F64(v) };
    set_res(KeyValue::new(s(kp, kl), v));
}

/// # Safety
/// `kp` must reference `kl` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn otel_resource_bool(kp: *const u8, kl: usize, v: i32) {
    set_res(KeyValue::new(s(kp, kl), v != 0));
}

unsafe fn emit(src: &[u8], buf: *mut u8, cap: usize) -> i32 {
    if src.len() > cap {
        return -(src.len() as i32);
    }
    std::ptr::copy_nonoverlapping(src.as_ptr(), buf, src.len());
    src.len() as i32
}

/// # Safety
/// `buf` must reference `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn otel_snapshot_metrics(buf: *mut u8, cap: usize) -> i32 {
    if !ENABLED.load(Relaxed) {
        return 0;
    }
    let mut guard = crate::metrics::lock(&STAGING);
    let st = &mut *guard;
    global::meter_provider().export_json(&mut st.out);
    if st.out.ends_with(b"\"scopeMetrics\":[]}]}") {
        return 0;
    }
    emit(&st.out, buf, cap)
}

/// # Safety
/// `buf` must reference `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn otel_snapshot_spans(buf: *mut u8, cap: usize) -> i32 {
    if !ENABLED.load(Relaxed) {
        return 0;
    }
    let mut guard = crate::metrics::lock(&STAGING);
    let st = &mut *guard;
    if st.spans.is_empty() {
        let Staging { spans, scratch, .. } = &mut *st;
        // cap the batch so one request stays well below the collector's default body limits
        global::tracer_provider().export_json(scratch, 512, spans);
    }
    if st.spans.is_empty() {
        return 0;
    }
    let r = emit(&st.spans, buf, cap);
    if r > 0 {
        st.spans.clear();
    }
    r
}

#[no_mangle]
pub extern "C" fn otel_dropped(reason: u32) -> f64 {
    let st = global::meter_provider().stats();
    let r = match reason {
        0 => DropReason::SpanQueueFull,
        1 => DropReason::MetricCardinality,
        2 => DropReason::MetricInvalidValue,
        3 => DropReason::MetricAttributesTruncated,
        _ => return st.total() as f64,
    };
    st.get(r) as f64
}

// --- instruments recorded from JS -------------------------------------------------------------

enum JsInst {
    C(BoundCounter<f64>),
    U(BoundUpDownCounter<i64>),
    G(BoundGauge<f64>),
    H(BoundHistogram<f64>),
}

const MAX_JS: usize = 512;
/// Attribute sets per JS instrument (models x builds x runs on one page stay well below).
const JS_CARDINALITY: usize = 64;
// leaked boxes, published once (lock-free lookup in otel_record)
static JS: [AtomicPtr<JsInst>; MAX_JS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_JS];
static JS_N: AtomicUsize = AtomicUsize::new(0);

/// # Safety
/// Pointers must reference the given number of readable bytes / f64s.
#[no_mangle]
pub unsafe extern "C" fn otel_instrument(
    kind: u32,
    np: *const u8,
    nl: usize,
    up: *const u8,
    ul: usize,
    bounds: *const f64,
    nb: usize,
) -> u32 {
    otel_instrument_attrs(kind, np, nl, up, ul, bounds, nb, std::ptr::null(), 0)
}

/// # Safety
/// Pointers must reference the given number of readable bytes / f64s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn otel_instrument_attrs(
    kind: u32,
    np: *const u8,
    nl: usize,
    up: *const u8,
    ul: usize,
    bounds: *const f64,
    nb: usize,
    ap: *const u8,
    al: usize,
) -> u32 {
    if kind > 3 {
        return 0;
    }
    let id = JS_N.fetch_add(1, Relaxed);
    if id >= MAX_JS {
        return 0;
    }
    // "k=v" lines -> attribute set (strings leaked: setup time, a bounded number of sets)
    let attrs: Vec<KeyValue> = s(ap, al)
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| KeyValue::new(k.trim().to_string(), v.trim().to_string()))
        .collect();
    let m = global::meter("vi.web");
    let name: StringValue = s(np, nl).into();
    let unit: StringValue = s(up, ul).into();
    let inst = match kind {
        0 => JsInst::C(m.f64_counter(name).with_unit(unit).with_cardinality_limit(JS_CARDINALITY).build().bind(&attrs)),
        1 => JsInst::U(m.i64_up_down_counter(name).with_unit(unit).with_cardinality_limit(JS_CARDINALITY).build().bind(&attrs)),
        2 => JsInst::G(m.f64_gauge(name).with_unit(unit).with_cardinality_limit(JS_CARDINALITY).build().bind(&attrs)),
        _ => {
            let mut b = m.f64_histogram(name).with_unit(unit).with_cardinality_limit(JS_CARDINALITY);
            if nb > 0 {
                b = b.with_boundaries(std::slice::from_raw_parts(bounds, nb).to_vec());
            }
            JsInst::H(b.build().bind(&attrs))
        }
    };
    JS[id].store(Box::into_raw(Box::new(inst)), Release);
    id as u32 + 1
}

#[no_mangle]
pub extern "C" fn otel_record(id: u32, v: f64) {
    let Some(p) = JS.get((id as usize).wrapping_sub(1)) else { return };
    let p = p.load(Acquire);
    if p.is_null() {
        return;
    }
    // SAFETY: published once with Release, never freed
    match unsafe { &*p } {
        JsInst::C(c) => c.add(v),
        JsInst::U(c) => c.add(v as i64),
        JsInst::G(g) => g.record(v),
        JsInst::H(h) => h.record(v),
    }
}

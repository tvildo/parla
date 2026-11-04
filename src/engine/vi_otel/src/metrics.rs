//! Metrics: `SdkMeterProvider` -> `Meter` -> instruments (`Counter`, `UpDownCounter`, `Gauge`,
//! `Histogram`), mirroring `opentelemetry::metrics`.
//!
//! Real-time model: every instrument owns a fixed table of attribute-set slots allocated at
//! `build()` (cardinality limit, default 8, plus slot 0 = overflow series). Recording hashes the
//! attribute set (order-independent, no allocation), probes the table, and updates 64-bit
//! lock-free cells. The first record of a new attribute set claims a free slot with one CAS and
//! copies the (Copy, `&'static str`) attributes in: still no allocation, no lock. `bind(&attrs)`
//! resolves the slot once and returns a handle whose `add`/`record` is a single atomic op
//! (histogram: count + bucket + sum/min/max).
//!
//! Aggregation temporality is cumulative (start time = provider start); the exporter reads the
//! cells off the audio thread.

use crate::cell::Cell64;
use crate::common::{attrs_hash, DropReason, InstrumentationScope, Key, KeyValue, Stats, StringValue, Value};
use crate::resource::Resource;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};

/// Max attributes stored per data point (longer sets are truncated, `metric_attributes_truncated`).
pub const MAX_ATTRS: usize = 6;
/// Default number of attribute sets per instrument (rounded up to a power of two).
pub const DEFAULT_CARDINALITY_LIMIT: usize = 8;
/// OTel SDK default explicit bucket boundaries.
pub const DEFAULT_BOUNDARIES: [f64; 15] =
    [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0];

const OVERFLOW_KV: KeyValue =
    KeyValue { key: Key::from_static_str("otel.metric.overflow"), value: Value::Bool(true) };

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    CounterU64,
    CounterF64,
    UpDownI64,
    GaugeU64,
    GaugeI64,
    GaugeF64,
    HistogramF64,
}

#[doc(hidden)]
pub struct SlotHead {
    hash: Cell64,
    ready: Cell64,
    attrs: UnsafeCell<[KeyValue; MAX_ATTRS]>,
    n: UnsafeCell<usize>,
}
// SAFETY: `attrs`/`n` are written once by the thread that won the CAS on `hash`, before
// `ready` is released; readers only look at them after an acquire load of `ready`.
unsafe impl Sync for SlotHead {}
unsafe impl Send for SlotHead {}

// histogram cell layout per slot: [count, sum, min, max, bucket0..bucketN]
pub(crate) const H_COUNT: usize = 0;
pub(crate) const H_SUM: usize = 1;
pub(crate) const H_MIN: usize = 2;
pub(crate) const H_MAX: usize = 3;
pub(crate) const H_B0: usize = 4;

#[doc(hidden)]
pub struct Inst {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) unit: &'static str,
    pub(crate) kind: Kind,
    pub(crate) bounds: Vec<f64>,
    pub(crate) stride: usize,
    pub(crate) scope: usize,
    mask: usize,
    heads: Vec<SlotHead>,
    pub(crate) cells: Vec<Cell64>,
    stats: Arc<Stats>,
}

impl Inst {
    fn new(b: &Builder, scope: usize, stats: Arc<Stats>) -> Inst {
        // sorted, deduplicated, finite (insertion sort: tiny inputs, avoids pulling in slice::sort)
        let mut bounds: Vec<f64> = Vec::new();
        if b.kind == Kind::HistogramF64 {
            let src: &[f64] = match &b.boundaries {
                Some(v) => v,
                None => &DEFAULT_BOUNDARIES,
            };
            for &x in src {
                if !x.is_finite() {
                    continue;
                }
                if bounds.contains(&x) {
                    continue;
                }
                bounds.push(x);
                let mut i = bounds.len() - 1;
                while i > 0 && bounds[i - 1] > x {
                    bounds.swap(i - 1, i);
                    i -= 1;
                }
            }
        }
        let stride = if b.kind == Kind::HistogramF64 { H_B0 + bounds.len() + 1 } else { 1 };
        let limit = b.limit.max(1).next_power_of_two();
        let n_slots = limit + 1;
        let mut heads = Vec::with_capacity(n_slots);
        let mut cells = Vec::with_capacity(n_slots * stride);
        for _ in 0..n_slots {
            heads.push(SlotHead {
                hash: Cell64::new(0),
                ready: Cell64::new(0),
                attrs: UnsafeCell::new([KeyValue::EMPTY; MAX_ATTRS]),
                n: UnsafeCell::new(0),
            });
            for k in 0..stride {
                let v = match k {
                    H_MIN if stride > 1 => f64::INFINITY.to_bits(),
                    H_MAX if stride > 1 => f64::NEG_INFINITY.to_bits(),
                    _ => 0,
                };
                cells.push(Cell64::new(v));
            }
        }
        Inst {
            name: b.name,
            description: b.description,
            unit: b.unit,
            kind: b.kind,
            bounds,
            stride,
            scope,
            mask: limit - 1,
            heads,
            cells,
            stats,
        }
    }

    pub(crate) fn n_slots(&self) -> usize {
        self.heads.len()
    }

    fn publish(&self, head: &SlotHead, attrs: &[KeyValue]) {
        let n = attrs.len().min(MAX_ATTRS);
        if attrs.len() > MAX_ATTRS {
            self.stats.inc(DropReason::MetricAttributesTruncated);
        }
        // SAFETY: we won the CAS on head.hash; nobody reads attrs before `ready` is set.
        unsafe {
            (&mut *head.attrs.get())[..n].copy_from_slice(&attrs[..n]);
            *head.n.get() = n;
        }
        head.ready.set_release(1);
    }

    /// Cell base index of the slot for this attribute set (claims a slot on first use).
    #[inline]
    pub(crate) fn slot(&self, attrs: &[KeyValue]) -> usize {
        let h = attrs_hash(attrs);
        let mut i = (h as usize) & self.mask;
        for _ in 0..=self.mask {
            let head = &self.heads[1 + i];
            let cur = head.hash.get();
            if cur == h {
                return (1 + i) * self.stride;
            }
            if cur == 0 {
                if head.hash.cas(0, h) {
                    self.publish(head, attrs);
                    return (1 + i) * self.stride;
                }
                if head.hash.get() == h {
                    return (1 + i) * self.stride;
                }
            }
            i = (i + 1) & self.mask;
        }
        self.stats.inc(DropReason::MetricCardinality);
        let head = &self.heads[0];
        if head.ready.get() == 0 && head.hash.cas(0, 1) {
            self.publish(head, &[OVERFLOW_KV]);
        }
        0
    }

    /// Attributes of slot `s` if it holds data.
    pub(crate) fn slot_attrs(&self, s: usize) -> Option<&[KeyValue]> {
        let head = &self.heads[s];
        if head.ready.get_acquire() == 0 {
            return None;
        }
        // SAFETY: published before the release store of `ready`, never written again.
        unsafe { Some(&(&*head.attrs.get())[..*head.n.get()]) }
    }

    #[inline(always)]
    fn add_u64(&self, base: usize, v: u64) {
        self.cells[base].add(v);
    }
    #[inline(always)]
    fn add_f64(&self, base: usize, v: f64) {
        if !(0.0..f64::INFINITY).contains(&v) {
            self.stats.inc(DropReason::MetricInvalidValue);
            return;
        }
        self.cells[base].update(|b| Some((f64::from_bits(b) + v).to_bits()));
    }
    #[inline(always)]
    fn add_i64(&self, base: usize, v: i64) {
        self.cells[base].add(v as u64);
    }
    #[inline(always)]
    fn set_bits(&self, base: usize, bits: u64) {
        self.cells[base].set(bits);
    }
    #[inline(always)]
    fn record_hist(&self, base: usize, v: f64) {
        if v.is_nan() {
            self.stats.inc(DropReason::MetricInvalidValue);
            return;
        }
        let b = self.bounds.partition_point(|&x| x < v);
        let c = &self.cells[base..base + self.stride];
        c[H_B0 + b].add(1);
        c[H_COUNT].add(1);
        c[H_SUM].update(|s| Some((f64::from_bits(s) + v).to_bits()));
        c[H_MIN].update(|m| if v < f64::from_bits(m) { Some(v.to_bits()) } else { None });
        c[H_MAX].update(|m| if v > f64::from_bits(m) { Some(v.to_bits()) } else { None });
    }
}

// ---------------------------------------------------------------------------------------------
// instruments

macro_rules! instrument {
    ($(#[$m:meta])* $name:ident, $bound:ident, $t:ty, $method:ident, |$i:ident, $b:ident, $v:ident| $body:expr) => {
        impl $name<$t> {
            $(#[$m])*
            #[inline]
            pub fn $method(&self, $v: $t, attributes: &[KeyValue]) {
                let $i = &*self.0;
                let $b = $i.slot(attributes);
                $body
            }
            /// Resolve the attribute set once; the handle records without hashing.
            pub fn bind(&self, attributes: &[KeyValue]) -> $bound<$t> {
                $bound(self.0.clone(), self.0.slot(attributes), PhantomData)
            }
        }
        impl $bound<$t> {
            #[inline]
            pub fn $method(&self, $v: $t) {
                let $i = &*self.0;
                let $b = self.1;
                $body
            }
        }
    };
}

macro_rules! types {
    ($($(#[$m:meta])* $name:ident / $bound:ident),*) => {$(
        $(#[$m])*
        pub struct $name<T>(Arc<Inst>, PhantomData<T>);
        impl<T> Clone for $name<T> { fn clone(&self) -> Self { $name(self.0.clone(), PhantomData) } }
        impl<T> FromInst for $name<T> { fn from_inst(i: Arc<Inst>) -> Self { $name(i, PhantomData) } }
        /// Pre-bound handle (attribute slot resolved at bind time).
        pub struct $bound<T>(Arc<Inst>, usize, PhantomData<T>);
        impl<T> Clone for $bound<T> { fn clone(&self) -> Self { $bound(self.0.clone(), self.1, PhantomData) } }
    )*};
}

types!(
    /// Monotonic sum (`opentelemetry::metrics::Counter`).
    Counter / BoundCounter,
    /// Non-monotonic sum (`opentelemetry::metrics::UpDownCounter`).
    UpDownCounter / BoundUpDownCounter,
    /// Last value (`opentelemetry::metrics::Gauge`).
    Gauge / BoundGauge,
    /// Explicit-bucket histogram with count/sum/min/max (`opentelemetry::metrics::Histogram`).
    Histogram / BoundHistogram
);

instrument!(Counter, BoundCounter, u64, add, |i, b, v| i.add_u64(b, v));
instrument!(
    /// Negative / NaN increments are dropped (`metric_invalid_value`).
    Counter, BoundCounter, f64, add, |i, b, v| i.add_f64(b, v)
);
instrument!(UpDownCounter, BoundUpDownCounter, i64, add, |i, b, v| i.add_i64(b, v));
instrument!(Gauge, BoundGauge, u64, record, |i, b, v| i.set_bits(b, v));
instrument!(Gauge, BoundGauge, i64, record, |i, b, v| i.set_bits(b, v as u64));
instrument!(Gauge, BoundGauge, f64, record, |i, b, v| i.set_bits(b, v.to_bits()));
instrument!(
    /// NaN samples are dropped (`metric_invalid_value`).
    Histogram, BoundHistogram, f64, record, |i, b, v| i.record_hist(b, v)
);

pub trait FromInst {
    #[doc(hidden)]
    fn from_inst(i: Arc<Inst>) -> Self;
}

// ---------------------------------------------------------------------------------------------
// builder / meter / provider

struct Builder {
    name: &'static str,
    description: &'static str,
    unit: &'static str,
    kind: Kind,
    boundaries: Option<Vec<f64>>,
    limit: usize,
}

/// `opentelemetry::metrics::InstrumentBuilder` / `HistogramBuilder`.
pub struct InstrumentBuilder<'a, I> {
    meter: &'a Meter,
    b: Builder,
    _i: PhantomData<I>,
}

impl<'a, I: FromInst> InstrumentBuilder<'a, I> {
    pub fn with_description(mut self, d: impl Into<StringValue>) -> Self {
        self.b.description = d.into().as_str();
        self
    }
    pub fn with_unit(mut self, u: impl Into<StringValue>) -> Self {
        self.b.unit = u.into().as_str();
        self
    }
    /// Explicit bucket boundaries (histograms only; sorted, deduplicated, non-finite removed).
    pub fn with_boundaries(mut self, b: Vec<f64>) -> Self {
        self.b.boundaries = Some(b);
        self
    }
    /// Max distinct attribute sets (non-standard; the official SDK sets this through views).
    pub fn with_cardinality_limit(mut self, n: usize) -> Self {
        self.b.limit = n;
        self
    }
    /// Allocates the instrument (setup time). Building the same name + kind twice in one scope
    /// returns the existing instrument.
    pub fn build(self) -> I {
        I::from_inst(self.meter.register(self.b))
    }
}

/// `opentelemetry::metrics::Meter`.
#[derive(Clone)]
pub struct Meter {
    mp: Arc<MpInner>,
    scope: usize,
}

macro_rules! meter_fn {
    ($($f:ident -> $ty:ident<$t:ty> = $kind:ident),*) => {$(
        pub fn $f(&self, name: impl Into<StringValue>) -> InstrumentBuilder<'_, $ty<$t>> {
            self.builder(name.into().as_str(), Kind::$kind)
        }
    )*};
}

impl Meter {
    fn builder<I>(&self, name: &'static str, kind: Kind) -> InstrumentBuilder<'_, I> {
        InstrumentBuilder {
            meter: self,
            b: Builder {
                name,
                description: "",
                unit: "",
                kind,
                boundaries: None,
                limit: self.mp.default_limit,
            },
            _i: PhantomData,
        }
    }
    meter_fn!(
        u64_counter -> Counter<u64> = CounterU64,
        f64_counter -> Counter<f64> = CounterF64,
        i64_up_down_counter -> UpDownCounter<i64> = UpDownI64,
        u64_gauge -> Gauge<u64> = GaugeU64,
        i64_gauge -> Gauge<i64> = GaugeI64,
        f64_gauge -> Gauge<f64> = GaugeF64,
        f64_histogram -> Histogram<f64> = HistogramF64
    );

    fn register(&self, b: Builder) -> Arc<Inst> {
        let mut insts = lock(&self.mp.insts);
        if let Some(i) = insts.iter().find(|i| i.scope == self.scope && i.name == b.name && i.kind == b.kind) {
            return i.clone();
        }
        let i = Arc::new(Inst::new(&b, self.scope, self.mp.stats.clone()));
        insts.push(i.clone());
        i
    }
}

pub(crate) struct MpInner {
    pub(crate) resource: Mutex<Resource>,
    pub(crate) scopes: Mutex<Vec<InstrumentationScope>>,
    pub(crate) insts: Mutex<Vec<Arc<Inst>>>,
    pub(crate) start_ns: u64,
    pub(crate) stats: Arc<Stats>,
    default_limit: usize,
}

/// `opentelemetry_sdk::metrics::SdkMeterProvider` (no readers/views: export is pull-based through
/// [`SdkMeterProvider::export_json`]).
#[derive(Clone)]
pub struct SdkMeterProvider {
    pub(crate) inner: Arc<MpInner>,
}

/// `opentelemetry::metrics::MeterProvider`.
pub trait MeterProvider {
    fn meter(&self, name: &'static str) -> Meter {
        self.meter_with_scope(InstrumentationScope::builder(name).build())
    }
    fn meter_with_scope(&self, scope: InstrumentationScope) -> Meter;
}

impl MeterProvider for SdkMeterProvider {
    fn meter_with_scope(&self, scope: InstrumentationScope) -> Meter {
        let mut s = lock(&self.inner.scopes);
        let idx = match s.iter().position(|x| *x == scope) {
            Some(i) => i,
            None => {
                s.push(scope);
                s.len() - 1
            }
        };
        Meter { mp: self.inner.clone(), scope: idx }
    }
}

impl Default for SdkMeterProvider {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl SdkMeterProvider {
    pub fn builder() -> MeterProviderBuilder {
        MeterProviderBuilder {
            resource: None,
            start_ns: None,
            stats: None,
            limit: DEFAULT_CARDINALITY_LIMIT,
        }
    }
    pub fn set_resource(&self, r: Resource) {
        *lock(&self.inner.resource) = r;
    }
    pub fn update_resource(&self, f: impl FnOnce(&mut Resource)) {
        f(&mut lock(&self.inner.resource))
    }
    pub fn resource(&self) -> Resource {
        lock(&self.inner.resource).clone()
    }
    pub fn stats(&self) -> Arc<Stats> {
        self.inner.stats.clone()
    }
    /// Encode all metrics as an OTLP/HTTP JSON `ExportMetricsServiceRequest` into `out`
    /// (cleared first, capacity reused). Off the audio thread.
    pub fn export_json(&self, out: &mut Vec<u8>) {
        self.export_json_at(crate::time::now_ns(), out)
    }
    pub fn export_json_at(&self, now_ns: u64, out: &mut Vec<u8>) {
        out.clear();
        crate::otlp::metrics(self, now_ns, out)
    }
}

pub struct MeterProviderBuilder {
    resource: Option<Resource>,
    start_ns: Option<u64>,
    stats: Option<Arc<Stats>>,
    limit: usize,
}

impl MeterProviderBuilder {
    pub fn with_resource(mut self, r: Resource) -> Self {
        self.resource = Some(r);
        self
    }
    /// Start of the cumulative interval (default: now).
    pub fn with_start_time(mut self, ns: u64) -> Self {
        self.start_ns = Some(ns);
        self
    }
    /// Share drop counters (e.g. with the tracer provider, so span drops are exported too).
    pub fn with_stats(mut self, s: Arc<Stats>) -> Self {
        self.stats = Some(s);
        self
    }
    /// Default per-instrument cardinality limit.
    pub fn with_cardinality_limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }
    pub fn build(self) -> SdkMeterProvider {
        SdkMeterProvider {
            inner: Arc::new(MpInner {
                resource: Mutex::new(self.resource.unwrap_or_default()),
                scopes: Mutex::new(Vec::new()),
                insts: Mutex::new(Vec::new()),
                start_ns: self.start_ns.unwrap_or_else(crate::time::now_ns),
                stats: self.stats.unwrap_or_default(),
                default_limit: self.limit,
            }),
        }
    }
}

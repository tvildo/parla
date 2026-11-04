//! Traces: `SdkTracerProvider` -> `Tracer` -> `Span`, mirroring `opentelemetry::trace`.
//!
//! A `Span` is a plain stack value (fixed arrays for attributes and events, no allocation).
//! `end()` (or drop) copies it into the provider's bounded lock-free ring (Vyukov MPMC queue);
//! when the ring is full the span is discarded and `span_queue_full` is counted. The exporter
//! drains the ring off the audio thread. Attributes / events beyond the fixed capacity are
//! counted in the span's `droppedAttributesCount` / `droppedEventsCount` (OTLP fields).

use crate::cell::Cell64;
use crate::common::{DropReason, InstrumentationScope, KeyValue, Stats, StringValue};
use crate::metrics::lock;
use crate::resource::Resource;
use crate::time::{now_ns, IdGen};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex};

pub const MAX_SPAN_ATTRS: usize = 8;
pub const MAX_EVENTS: usize = 4;
pub const MAX_EVENT_ATTRS: usize = 3;
pub const DEFAULT_SPAN_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct TraceId(pub [u8; 16]);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct SpanId(pub [u8; 8]);

impl TraceId {
    pub const INVALID: TraceId = TraceId([0; 16]);
    pub fn to_bytes(self) -> [u8; 16] {
        self.0
    }
}
impl SpanId {
    pub const INVALID: SpanId = SpanId([0; 8]);
    pub fn to_bytes(self) -> [u8; 8] {
        self.0
    }
}

/// `opentelemetry::trace::SpanContext` (trace flags reduced to `sampled`, no trace state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanContext {
    trace_id: TraceId,
    span_id: SpanId,
    sampled: bool,
}

impl SpanContext {
    pub const NONE: SpanContext = SpanContext { trace_id: TraceId::INVALID, span_id: SpanId::INVALID, sampled: false };
    pub fn new(trace_id: TraceId, span_id: SpanId, sampled: bool) -> Self {
        SpanContext { trace_id, span_id, sampled }
    }
    pub fn trace_id(&self) -> TraceId {
        self.trace_id
    }
    pub fn span_id(&self) -> SpanId {
        self.span_id
    }
    pub fn is_sampled(&self) -> bool {
        self.sampled
    }
    pub fn is_valid(&self) -> bool {
        self.trace_id != TraceId::INVALID && self.span_id != SpanId::INVALID
    }
}

/// Explicit parent context (`opentelemetry::Context` reduced to an optional span context; there
/// is no implicit "current" context: pass it explicitly).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Context {
    span: Option<SpanContext>,
}

impl Context {
    pub fn new() -> Self {
        Context { span: None }
    }
    /// `TraceContextExt::with_remote_span_context`.
    pub fn with_remote_span_context(self, sc: SpanContext) -> Self {
        Context { span: Some(sc) }
    }
    /// Shorthand: context whose parent is `span`.
    pub fn from_span(span: &Span) -> Self {
        Context { span: Some(span.d.ctx) }
    }
    pub fn span_context(&self) -> Option<&SpanContext> {
        self.span.as_ref()
    }
}

/// `opentelemetry::trace::SpanKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Internal = 1,
    Server = 2,
    Client = 3,
    Producer = 4,
    Consumer = 5,
}

/// `opentelemetry::trace::Status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Unset,
    Error { description: &'static str },
    Ok,
}

impl Status {
    pub fn error(description: impl Into<StringValue>) -> Self {
        Status::Error { description: description.into().as_str() }
    }
}

/// `opentelemetry_sdk::trace::Sampler` subset. Children always follow their parent's decision
/// (parent-based); roots use the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    AlwaysOn,
    AlwaysOff,
    TraceIdRatioBased(f64),
}

impl Sampler {
    fn threshold(self) -> u64 {
        match self {
            Sampler::AlwaysOn => u64::MAX,
            Sampler::AlwaysOff => 0,
            Sampler::TraceIdRatioBased(r) if r >= 1.0 => u64::MAX,
            Sampler::TraceIdRatioBased(r) if r > 0.0 => (r * 18446744073709551616.0) as u64,
            Sampler::TraceIdRatioBased(_) => 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Event {
    pub(crate) name: &'static str,
    pub(crate) time: u64,
    attrs: [MaybeUninit<KeyValue>; MAX_EVENT_ATTRS],
    pub(crate) n: u8,
    pub(crate) dropped: u8,
}

// Fixed arrays are left uninitialized beyond their count (no init code / stores on span start).
const UNINIT_KV: MaybeUninit<KeyValue> = MaybeUninit::uninit();
const UNINIT_EV: MaybeUninit<Event> = MaybeUninit::uninit();

#[inline(always)]
fn init_slice<T>(a: &[MaybeUninit<T>], n: usize) -> &[T] {
    // SAFETY: callers keep the first `n` entries initialized
    unsafe { core::slice::from_raw_parts(a.as_ptr() as *const T, n) }
}

impl Event {
    pub(crate) fn attributes(&self) -> &[KeyValue] {
        init_slice(&self.attrs, self.n as usize)
    }
}

/// A finished (or in-flight) span record. `Copy`, fixed size.
#[derive(Clone, Copy)]
pub struct SpanData {
    pub(crate) name: &'static str,
    pub(crate) ctx: SpanContext,
    pub(crate) parent: SpanId,
    pub(crate) scope: u32,
    pub(crate) kind: SpanKind,
    pub(crate) status: Status,
    pub(crate) start: u64,
    pub(crate) end: u64,
    attrs: [MaybeUninit<KeyValue>; MAX_SPAN_ATTRS],
    pub(crate) n_attrs: u8,
    pub(crate) n_events: u8,
    pub(crate) dropped_attrs: u16,
    pub(crate) dropped_events: u16,
    events: [MaybeUninit<Event>; MAX_EVENTS],
}

impl SpanData {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn span_context(&self) -> &SpanContext {
        &self.ctx
    }
    pub fn parent_span_id(&self) -> SpanId {
        self.parent
    }
    pub fn attributes(&self) -> &[KeyValue] {
        init_slice(&self.attrs, self.n_attrs as usize)
    }
    pub(crate) fn events(&self) -> &[Event] {
        init_slice(&self.events, self.n_events as usize)
    }
    pub fn start_time(&self) -> u64 {
        self.start
    }
    pub fn end_time(&self) -> u64 {
        self.end
    }
    fn push_attr(&mut self, kv: KeyValue) {
        let n = self.n_attrs as usize;
        // SAFETY: first n initialized
        let init = unsafe { core::slice::from_raw_parts_mut(self.attrs.as_mut_ptr() as *mut KeyValue, n) };
        if let Some(a) = init.iter_mut().find(|a| a.key == kv.key) {
            a.value = kv.value;
        } else if n < MAX_SPAN_ATTRS {
            self.attrs[n] = MaybeUninit::new(kv);
            self.n_attrs += 1;
        } else {
            self.dropped_attrs = self.dropped_attrs.saturating_add(1);
        }
    }
}

// --- bounded MPMC ring (Dmitry Vyukov) --------------------------------------------------------

struct RSlot {
    seq: AtomicUsize,
    data: UnsafeCell<MaybeUninit<SpanData>>,
}

pub(crate) struct Ring {
    slots: Vec<RSlot>,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}
// SAFETY: slot data is only accessed by the thread that won the head/tail CAS for it, ordered by
// the per-slot sequence number (acquire/release).
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub(crate) fn new(cap: usize) -> Ring {
        let cap = cap.max(2).next_power_of_two();
        let mut slots = Vec::with_capacity(cap);
        for i in 0..cap {
            slots.push(RSlot { seq: AtomicUsize::new(i), data: UnsafeCell::new(MaybeUninit::uninit()) });
        }
        Ring {
            slots,
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    pub(crate) fn capacity(&self) -> usize {
        self.mask + 1
    }
    pub(crate) fn push(&self, d: &SpanData) -> bool {
        let mut pos = self.tail.load(Relaxed);
        loop {
            let s = &self.slots[pos & self.mask];
            let seq = s.seq.load(Acquire);
            let dif = seq.wrapping_sub(pos) as isize;
            if dif == 0 {
                match self.tail.compare_exchange_weak(pos, pos.wrapping_add(1), Relaxed, Relaxed) {
                    Ok(_) => {
                        unsafe { (*s.data.get()).write(*d) };
                        s.seq.store(pos.wrapping_add(1), Release);
                        return true;
                    }
                    Err(p) => pos = p,
                }
            } else if dif < 0 {
                return false;
            } else {
                pos = self.tail.load(Relaxed);
            }
        }
    }
    pub(crate) fn pop(&self) -> Option<SpanData> {
        let mut pos = self.head.load(Relaxed);
        loop {
            let s = &self.slots[pos & self.mask];
            let seq = s.seq.load(Acquire);
            let dif = seq.wrapping_sub(pos.wrapping_add(1)) as isize;
            if dif == 0 {
                match self.head.compare_exchange_weak(pos, pos.wrapping_add(1), Relaxed, Relaxed) {
                    Ok(_) => {
                        let d = unsafe { (*s.data.get()).assume_init() };
                        s.seq.store(pos.wrapping_add(self.mask + 1), Release);
                        return Some(d);
                    }
                    Err(p) => pos = p,
                }
            } else if dif < 0 {
                return None;
            } else {
                pos = self.head.load(Relaxed);
            }
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.tail.load(Relaxed).wrapping_sub(self.head.load(Relaxed)).min(self.capacity())
    }
}

// --- provider / tracer / span -------------------------------------------------------------------

pub(crate) struct TpInner {
    pub(crate) resource: Mutex<Resource>,
    pub(crate) scopes: Mutex<Vec<InstrumentationScope>>,
    pub(crate) ring: Ring,
    threshold: Cell64,
    enabled: Cell64,
    ids: IdGen,
    pub(crate) stats: Arc<Stats>,
}

/// `opentelemetry_sdk::trace::SdkTracerProvider` (no processors: export is pull-based through
/// [`SdkTracerProvider::export_json`]).
#[derive(Clone)]
pub struct SdkTracerProvider {
    pub(crate) inner: Arc<TpInner>,
}

/// `opentelemetry::trace::TracerProvider`.
pub trait TracerProvider {
    fn tracer(&self, name: &'static str) -> Tracer {
        self.tracer_with_scope(InstrumentationScope::builder(name).build())
    }
    fn tracer_with_scope(&self, scope: InstrumentationScope) -> Tracer;
}

impl TracerProvider for SdkTracerProvider {
    fn tracer_with_scope(&self, scope: InstrumentationScope) -> Tracer {
        let mut s = lock(&self.inner.scopes);
        let idx = match s.iter().position(|x| *x == scope) {
            Some(i) => i,
            None => {
                s.push(scope);
                s.len() - 1
            }
        };
        Tracer { tp: self.inner.clone(), scope: idx as u32 }
    }
}

impl Default for SdkTracerProvider {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl SdkTracerProvider {
    pub fn builder() -> TracerProviderBuilder {
        TracerProviderBuilder {
            resource: None,
            sampler: Sampler::AlwaysOn,
            cap: DEFAULT_SPAN_CAPACITY,
            seed: None,
            stats: None,
            enabled: true,
        }
    }
    pub fn set_sampler(&self, s: Sampler) {
        self.inner.threshold.set(s.threshold())
    }
    /// Disabled: new spans are non-recording (contexts still propagate).
    pub fn set_enabled(&self, on: bool) {
        self.inner.enabled.set(on as u64)
    }
    pub fn reseed(&self, seed: u64) {
        self.inner.ids.reseed(seed)
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
    /// Spans dropped because the ring was full.
    pub fn dropped_spans(&self) -> u64 {
        self.inner.stats.get(DropReason::SpanQueueFull)
    }
    /// Finished spans waiting for export.
    pub fn pending(&self) -> usize {
        self.inner.ring.len()
    }
    pub fn capacity(&self) -> usize {
        self.inner.ring.capacity()
    }
    /// Move up to `max` finished spans into `out` (appends).
    pub fn drain_into(&self, out: &mut Vec<SpanData>, max: usize) -> usize {
        let mut n = 0;
        while n < max {
            match self.inner.ring.pop() {
                Some(d) => out.push(d),
                None => break,
            }
            n += 1;
        }
        n
    }
    /// Drain up to `max` spans and encode them as an OTLP/HTTP JSON `ExportTraceServiceRequest`
    /// into `out` (cleared first). Returns the number of spans (0: `out` left empty).
    /// `scratch` is a reusable buffer. Off the audio thread.
    pub fn export_json(&self, scratch: &mut Vec<SpanData>, max: usize, out: &mut Vec<u8>) -> usize {
        scratch.clear();
        out.clear();
        let n = self.drain_into(scratch, max);
        if n > 0 {
            crate::otlp::spans(self, scratch, out);
        }
        n
    }
}

pub struct TracerProviderBuilder {
    resource: Option<Resource>,
    sampler: Sampler,
    cap: usize,
    seed: Option<u64>,
    stats: Option<Arc<Stats>>,
    enabled: bool,
}

impl TracerProviderBuilder {
    pub fn with_resource(mut self, r: Resource) -> Self {
        self.resource = Some(r);
        self
    }
    pub fn with_sampler(mut self, s: Sampler) -> Self {
        self.sampler = s;
        self
    }
    /// Ring capacity in spans (rounded up to a power of two). Memory: capacity x size_of::<SpanData>().
    pub fn with_span_capacity(mut self, n: usize) -> Self {
        self.cap = n;
        self
    }
    /// Deterministic ids (tests); default: `time::seed` or time/address entropy.
    pub fn with_id_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
    pub fn with_stats(mut self, s: Arc<Stats>) -> Self {
        self.stats = Some(s);
        self
    }
    pub fn with_enabled(mut self, on: bool) -> Self {
        self.enabled = on;
        self
    }
    pub fn build(self) -> SdkTracerProvider {
        SdkTracerProvider {
            inner: Arc::new(TpInner {
                resource: Mutex::new(self.resource.unwrap_or_default()),
                scopes: Mutex::new(Vec::new()),
                ring: Ring::new(self.cap),
                threshold: Cell64::new(self.sampler.threshold()),
                enabled: Cell64::new(self.enabled as u64),
                ids: IdGen::new(self.seed.unwrap_or_else(crate::time::default_seed)),
                stats: self.stats.unwrap_or_default(),
            }),
        }
    }
}

/// `opentelemetry::trace::Tracer` (concrete type, inherent methods).
#[derive(Clone)]
pub struct Tracer {
    tp: Arc<TpInner>,
    scope: u32,
}

impl Tracer {
    /// New root span.
    #[inline]
    pub fn start(&self, name: &'static str) -> Span {
        self.span_builder(name).start(self)
    }
    /// Child of the span in `cx` (root if `cx` is empty).
    #[inline]
    pub fn start_with_context(&self, name: &'static str, cx: &Context) -> Span {
        self.span_builder(name).start_with_context(self, cx)
    }
    pub fn span_builder(&self, name: &'static str) -> SpanBuilder {
        SpanBuilder::from_name(name)
    }

    fn build(&self, b: SpanBuilder, cx: &Context) -> Span {
        let t = &*self.tp;
        let (trace_id, parent, sampled) = match cx.span {
            Some(p) if p.is_valid() => (p.trace_id, p.span_id, p.sampled),
            _ => {
                let hi = t.ids.nonzero();
                let lo = t.ids.next();
                let mut id = [0u8; 16];
                id[..8].copy_from_slice(&hi.to_be_bytes());
                id[8..].copy_from_slice(&lo.to_be_bytes());
                let th = t.threshold.get();
                (TraceId(id), SpanId::INVALID, th == u64::MAX || lo < th)
            }
        };
        let span_id = SpanId(t.ids.nonzero().to_be_bytes());
        let recording = sampled && t.enabled.get() != 0;
        let mut d = SpanData {
            name: b.name,
            ctx: SpanContext { trace_id, span_id, sampled },
            parent,
            scope: self.scope,
            kind: b.kind,
            status: Status::Unset,
            start: 0,
            end: 0,
            attrs: [UNINIT_KV; MAX_SPAN_ATTRS],
            n_attrs: 0,
            n_events: 0,
            dropped_attrs: 0,
            dropped_events: 0,
            events: [UNINIT_EV; MAX_EVENTS],
        };
        if recording {
            d.start = match b.start {
                Some(s) => s,
                None => now_ns(),
            };
            for kv in init_slice(&b.attrs, b.n) {
                d.push_attr(*kv);
            }
            d.dropped_attrs = d.dropped_attrs.saturating_add(b.dropped);
        }
        Span { d, tp: if recording { Some(self.tp.clone()) } else { None } }
    }
}

/// `opentelemetry::trace::SpanBuilder` subset (start time in ns since epoch, not `SystemTime`).
pub struct SpanBuilder {
    name: &'static str,
    kind: SpanKind,
    start: Option<u64>,
    attrs: [MaybeUninit<KeyValue>; MAX_SPAN_ATTRS],
    n: usize,
    dropped: u16,
}

impl SpanBuilder {
    pub fn from_name(name: &'static str) -> Self {
        SpanBuilder { name, kind: SpanKind::Internal, start: None, attrs: [UNINIT_KV; MAX_SPAN_ATTRS], n: 0, dropped: 0 }
    }
    pub fn with_kind(mut self, k: SpanKind) -> Self {
        self.kind = k;
        self
    }
    pub fn with_start_time(mut self, ns: u64) -> Self {
        self.start = Some(ns);
        self
    }
    pub fn with_attributes<I: IntoIterator<Item = KeyValue>>(mut self, kvs: I) -> Self {
        for kv in kvs {
            if self.n < MAX_SPAN_ATTRS {
                self.attrs[self.n] = MaybeUninit::new(kv);
                self.n += 1;
            } else {
                self.dropped = self.dropped.saturating_add(1);
            }
        }
        self
    }
    pub fn start(self, tracer: &Tracer) -> Span {
        tracer.build(self, &Context::new())
    }
    pub fn start_with_context(self, tracer: &Tracer, cx: &Context) -> Span {
        tracer.build(self, cx)
    }
}

/// `opentelemetry::trace::Span` (concrete type). Ends on `end()` or drop.
pub struct Span {
    d: SpanData,
    tp: Option<Arc<TpInner>>,
}

impl Span {
    pub fn span_context(&self) -> &SpanContext {
        &self.d.ctx
    }
    pub fn is_recording(&self) -> bool {
        self.tp.is_some()
    }
    #[inline]
    pub fn set_attribute(&mut self, kv: KeyValue) {
        if self.tp.is_some() {
            self.d.push_attr(kv)
        }
    }
    pub fn set_attributes<I: IntoIterator<Item = KeyValue>>(&mut self, kvs: I) {
        for kv in kvs {
            self.set_attribute(kv)
        }
    }
    pub fn add_event<I: IntoIterator<Item = KeyValue>>(&mut self, name: &'static str, attributes: I) {
        if self.tp.is_some() {
            self.add_event_with_timestamp(name, now_ns(), attributes)
        }
    }
    pub fn add_event_with_timestamp<I: IntoIterator<Item = KeyValue>>(&mut self, name: &'static str, ns: u64, attributes: I) {
        if self.tp.is_none() {
            return;
        }
        let d = &mut self.d;
        if d.n_events as usize >= MAX_EVENTS {
            d.dropped_events = d.dropped_events.saturating_add(1);
            return;
        }
        let mut e = Event { name, time: ns, attrs: [UNINIT_KV; MAX_EVENT_ATTRS], n: 0, dropped: 0 };
        for kv in attributes {
            if (e.n as usize) < MAX_EVENT_ATTRS {
                e.attrs[e.n as usize] = MaybeUninit::new(kv);
                e.n += 1;
            } else {
                e.dropped = e.dropped.saturating_add(1);
            }
        }
        d.events[d.n_events as usize] = MaybeUninit::new(e);
        d.n_events += 1;
    }
    pub fn set_status(&mut self, s: Status) {
        // Ok is final; Error overrides Unset (spec)
        if self.d.status != Status::Ok {
            self.d.status = s;
        }
    }
    pub fn update_name(&mut self, name: &'static str) {
        self.d.name = name;
    }
    #[inline]
    pub fn end(&mut self) {
        if self.tp.is_some() {
            self.end_with_timestamp(now_ns())
        }
    }
    pub fn end_with_timestamp(&mut self, ns: u64) {
        if let Some(tp) = self.tp.take() {
            self.d.end = ns;
            if !tp.ring.push(&self.d) {
                tp.stats.inc(DropReason::SpanQueueFull);
            }
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.end()
    }
}

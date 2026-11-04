//! Attribute types (`Key`, `Value`, `KeyValue`), instrumentation scope, drop statistics.
//!
//! All attribute types are `Copy` and hold `&'static str`, so recording never allocates.
//! `From<String>` leaks the string (interning for setup-time values such as browser name or a
//! model sha256): never build attributes from fresh `String`s on the audio thread.

use crate::cell::Cell64;

pub(crate) fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Attribute key (`opentelemetry::Key`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key(&'static str);

impl Key {
    pub const fn from_static_str(s: &'static str) -> Self {
        Key(s)
    }
    pub fn new(s: impl Into<Key>) -> Self {
        s.into()
    }
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}
impl From<&'static str> for Key {
    fn from(s: &'static str) -> Self {
        Key(s)
    }
}
impl From<String> for Key {
    /// Leaks (setup-time only).
    fn from(s: String) -> Self {
        Key(leak(s))
    }
}

/// String attribute value (`opentelemetry::StringValue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringValue(&'static str);

impl StringValue {
    pub const fn from_static_str(s: &'static str) -> Self {
        StringValue(s)
    }
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}
impl From<&'static str> for StringValue {
    fn from(s: &'static str) -> Self {
        StringValue(s)
    }
}
impl From<String> for StringValue {
    /// Leaks (setup-time only).
    fn from(s: String) -> Self {
        StringValue(leak(s))
    }
}

/// Attribute value (`opentelemetry::Value` without `Array`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    I64(i64),
    F64(f64),
    String(StringValue),
}

macro_rules! from_int {
    ($($t:ty),*) => {$(impl From<$t> for Value { fn from(v: $t) -> Self { Value::I64(v as i64) } })*};
}
from_int!(i64, i32, i16, i8, u32, u16, u8);
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F64(v as f64)
    }
}
impl From<&'static str> for Value {
    fn from(v: &'static str) -> Self {
        Value::String(StringValue(v))
    }
}
impl From<String> for Value {
    /// Leaks (setup-time only).
    fn from(v: String) -> Self {
        Value::String(StringValue(leak(v)))
    }
}
impl From<StringValue> for Value {
    fn from(v: StringValue) -> Self {
        Value::String(v)
    }
}

/// Key/value pair (`opentelemetry::KeyValue`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyValue {
    pub key: Key,
    pub value: Value,
}

impl KeyValue {
    pub fn new<K: Into<Key>, V: Into<Value>>(key: K, value: V) -> Self {
        KeyValue { key: key.into(), value: value.into() }
    }
    pub(crate) const EMPTY: KeyValue = KeyValue { key: Key(""), value: Value::Bool(false) };
}

#[inline(always)]
fn fnv(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[inline(always)]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Order-independent 64-bit hash of an attribute set; never 0 (0 marks a free slot).
#[inline]
pub(crate) fn attrs_hash(attrs: &[KeyValue]) -> u64 {
    let mut acc: u64 = 0x5bd1e995;
    for kv in attrs {
        let mut h = fnv(0xcbf29ce484222325, kv.key.0.as_bytes());
        h = match kv.value {
            Value::Bool(b) => fnv(h ^ 0x11, &[b as u8]),
            Value::I64(i) => fnv(h ^ 0x22, &i.to_le_bytes()),
            Value::F64(f) => fnv(h ^ 0x33, &f.to_bits().to_le_bytes()),
            Value::String(s) => fnv(h ^ 0x44, s.0.as_bytes()),
        };
        acc = acc.wrapping_add(mix(h));
    }
    mix(acc) | 1
}

/// Instrumentation scope (`opentelemetry::InstrumentationScope`, name + version only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstrumentationScope {
    pub(crate) name: &'static str,
    pub(crate) version: Option<&'static str>,
}

impl InstrumentationScope {
    pub fn builder(name: impl Into<StringValue>) -> InstrumentationScopeBuilder {
        InstrumentationScopeBuilder(InstrumentationScope { name: name.into().0, version: None })
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn version(&self) -> Option<&'static str> {
        self.version
    }
}

pub struct InstrumentationScopeBuilder(InstrumentationScope);
impl InstrumentationScopeBuilder {
    pub fn with_version(mut self, v: impl Into<StringValue>) -> Self {
        self.0.version = Some(v.into().0);
        self
    }
    pub fn build(self) -> InstrumentationScope {
        self.0
    }
}

/// Reasons for dropped data, exported as `vi_otel.dropped{reason}` (scope `vi_otel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum DropReason {
    /// Span ring buffer full: span discarded.
    SpanQueueFull = 0,
    /// Attribute set beyond an instrument's cardinality limit: recorded into the overflow series
    /// (`otel.metric.overflow=true`), as in the OTel SDK spec.
    MetricCardinality = 1,
    /// NaN histogram sample, negative/NaN counter increment: ignored.
    MetricInvalidValue = 2,
    /// Attribute set longer than `metrics::MAX_ATTRS`: truncated.
    MetricAttributesTruncated = 3,
}
pub(crate) const N_DROP: usize = 4;
pub(crate) const DROP_NAMES: [&str; N_DROP] =
    ["span_queue_full", "metric_cardinality", "metric_invalid_value", "metric_attributes_truncated"];

/// Drop counters, shared by a meter provider and a tracer provider (`with_stats`).
pub struct Stats(pub(crate) [Cell64; N_DROP]);

impl Default for Stats {
    fn default() -> Self {
        Stats([Cell64::new(0), Cell64::new(0), Cell64::new(0), Cell64::new(0)])
    }
}

impl Stats {
    #[inline(always)]
    pub(crate) fn inc(&self, r: DropReason) {
        self.0[r as usize].add(1);
    }
    pub fn get(&self, r: DropReason) -> u64 {
        self.0[r as usize].get()
    }
    pub fn total(&self) -> u64 {
        self.0.iter().map(|c| c.get()).sum()
    }
}

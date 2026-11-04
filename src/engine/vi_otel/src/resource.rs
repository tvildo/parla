//! `Resource` (`opentelemetry_sdk::Resource`): attributes describing the emitting entity.

use crate::common::{Key, KeyValue, StringValue, Value};

pub const SDK_NAME: &str = "vi_otel";
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    attrs: Vec<KeyValue>,
}

impl Default for Resource {
    fn default() -> Self {
        Resource::builder().build()
    }
}

impl Resource {
    /// Builder pre-filled with `service.name=unknown_service` and the `telemetry.sdk.*` attributes.
    pub fn builder() -> ResourceBuilder {
        let mut r = Resource { attrs: Vec::new() };
        r.set(KeyValue::new("service.name", "unknown_service"));
        r.set(KeyValue::new("telemetry.sdk.name", SDK_NAME));
        r.set(KeyValue::new("telemetry.sdk.language", "rust"));
        r.set(KeyValue::new("telemetry.sdk.version", SDK_VERSION));
        ResourceBuilder(r)
    }
    /// Builder without any default attribute.
    pub fn builder_empty() -> ResourceBuilder {
        ResourceBuilder(Resource { attrs: Vec::new() })
    }
    /// Insert or replace (by key).
    pub fn set(&mut self, kv: KeyValue) {
        match self.attrs.iter_mut().find(|a| a.key == kv.key || a.key.as_str() == kv.key.as_str()) {
            Some(a) => a.value = kv.value,
            None => self.attrs.push(kv),
        }
    }
    pub fn get(&self, key: &str) -> Option<Value> {
        self.attrs.iter().find(|a| a.key.as_str() == key).map(|a| a.value)
    }
    pub fn iter(&self) -> impl Iterator<Item = &KeyValue> {
        self.attrs.iter()
    }
    pub fn len(&self) -> usize {
        self.attrs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
    pub(crate) fn attrs(&self) -> &[KeyValue] {
        &self.attrs
    }
}

pub struct ResourceBuilder(Resource);

impl ResourceBuilder {
    pub fn with_service_name(mut self, name: impl Into<StringValue>) -> Self {
        self.0.set(KeyValue { key: Key::from_static_str("service.name"), value: Value::String(name.into()) });
        self
    }
    pub fn with_attribute(mut self, kv: KeyValue) -> Self {
        self.0.set(kv);
        self
    }
    pub fn with_attributes<I: IntoIterator<Item = KeyValue>>(mut self, kvs: I) -> Self {
        for kv in kvs {
            self.0.set(kv);
        }
        self
    }
    pub fn build(self) -> Resource {
        self.0
    }
}

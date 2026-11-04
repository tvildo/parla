//! Process-wide providers (`opentelemetry::global`). The global providers are created lazily on
//! first use (so instruments built before any configuration are real, not no-ops) and share one
//! `Stats`. Resource attributes can be updated at any time (`update_resource`).
//! With the `c-abi` feature the global tracer starts disabled (telemetry is opt-in; the host
//! enables it with `otel_set_enabled(1)`).

use crate::metrics::{Meter, MeterProvider, SdkMeterProvider};
use crate::resource::Resource;
use crate::trace::{SdkTracerProvider, Tracer, TracerProvider, DEFAULT_SPAN_CAPACITY};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::OnceLock;

static MP: OnceLock<SdkMeterProvider> = OnceLock::new();
static TP: OnceLock<SdkTracerProvider> = OnceLock::new();
static SPAN_CAP: AtomicUsize = AtomicUsize::new(DEFAULT_SPAN_CAPACITY);

/// Install the global meter provider. Returns false (and drops `p`) if one is already in use.
pub fn set_meter_provider(p: SdkMeterProvider) -> bool {
    MP.set(p).is_ok()
}

/// Install the global tracer provider. Returns false if one is already in use.
pub fn set_tracer_provider(p: SdkTracerProvider) -> bool {
    TP.set(p).is_ok()
}

/// Ring capacity for the lazily created global tracer provider (before its first use).
pub fn set_default_span_capacity(n: usize) -> bool {
    SPAN_CAP.store(n, Relaxed);
    TP.get().is_none()
}

pub fn meter_provider() -> &'static SdkMeterProvider {
    MP.get_or_init(SdkMeterProvider::default)
}

pub fn tracer_provider() -> &'static SdkTracerProvider {
    TP.get_or_init(|| {
        SdkTracerProvider::builder()
            .with_stats(meter_provider().stats())
            .with_span_capacity(SPAN_CAP.load(Relaxed))
            .with_enabled(!cfg!(feature = "c-abi"))
            .build()
    })
}

/// `global::meter(name)`.
pub fn meter(name: &'static str) -> Meter {
    meter_provider().meter(name)
}

/// `global::tracer(name)`.
pub fn tracer(name: &'static str) -> Tracer {
    tracer_provider().tracer(name)
}

/// Replace the resource of both global providers.
pub fn set_resource(r: Resource) {
    meter_provider().set_resource(r.clone());
    tracer_provider().set_resource(r);
}

/// Modify the resource of both global providers.
pub fn update_resource(f: impl Fn(&mut Resource)) {
    meter_provider().update_resource(&f);
    tracer_provider().update_resource(&f);
}

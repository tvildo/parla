//! vi_otel: tiny std-only OpenTelemetry-compatible telemetry (metrics + traces) for the vi
//! noise-suppression engine, native and wasm32. See AGENTS.md for the design, the real-time
//! rules and the mapping to the official `opentelemetry` / `opentelemetry_sdk` crates.
//!
//! ```ignore
//! use vi_otel::{global, KeyValue};
//! let meter = global::meter("vi.engine");
//! let dur = meter.f64_histogram("vi.engine.process.duration").with_unit("us")
//!     .with_boundaries(vec![50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]).build();
//! let dur_gtcrn = dur.bind(&[KeyValue::new("vi.model", "gtcrn")]);   // setup
//! dur_gtcrn.record(123.0);                                             // audio thread
//! let mut span = global::tracer("vi.engine").start("vi.model.load");
//! span.set_attribute(KeyValue::new("vi.model", "gtcrn"));
//! span.end();
//! let mut out = Vec::new();
//! global::meter_provider().export_json(&mut out);                      // exporter thread
//! ```

mod cell;
pub mod common;
pub mod global;
pub mod json;
pub mod metrics;
pub mod otlp;
pub mod resource;
pub mod time;
pub mod trace;

#[cfg(feature = "c-abi")]
pub mod c_abi;
#[cfg(feature = "native-http")]
pub mod http;

pub use common::{DropReason, InstrumentationScope, Key, KeyValue, Stats, StringValue, Value};
pub use resource::Resource;
pub use trace::Context;

//! The telemetry module of the browser host (web/plugins/vi_otel.wasm): vi_otel's C ABI
//! (`otel_*`, see rust/vi_otel/src/c_abi.rs), separate from the engine modules so the engine
//! bundles keep their size and telemetry costs nothing until a user opts in.

#[doc(hidden)]
pub use vi_otel::c_abi as _c_abi;

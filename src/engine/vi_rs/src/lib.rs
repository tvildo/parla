//! vi_rs — top crate of the vi workspace: plugin registry (cargo features), `host::Host` (on/off +
//! strength, voice activity, frame parts: the layer every host uses), generic C ABI for wasm
//! (wasm_api.rs, any wasm runtime), manifest loader, `vi_cli`. Engines are plugins:
//!
//!   id 0 `passthrough` (vi_engine, always)   id 1 reserved (an earlier hand-written engine)
//!   id 2 `graph`  (feature "graph", vi_graph: generic model packages, models/FORMAT.md)
//!   id 3 `dfn`    (feature "dfn", vi_dfn: DeepFilterNet3)   id 4 `rnnoise` (feature "rnnoise", vi_rnnoise)
//! Kind ids are stable across builds; a build without a plugin reports it via `kind_available`.
//! Default features = all plugins (the bundle); per-plugin wasm modules: --no-default-features
//! --features <plugin>.

pub mod host;
#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use vi_audio::json;
pub use vi_engine::{EngineInfo, ModelCore, NoiseEngine};
pub use vi_kernels::{call, Backend};
pub use vi_kernels as kernels;

pub const KINDS: [&str; 5] = ["passthrough", "reserved", "graph", "dfn", "rnnoise"];

pub fn kind_id(name: &str) -> Option<u32> {
    KINDS.iter().position(|k| *k == name).map(|i| i as u32)
}

/// Is the plugin for `kind` compiled into this build?
pub fn kind_available(kind: u32) -> bool {
    match kind {
        0 => true,
        2 => cfg!(feature = "graph"),
        3 => cfg!(feature = "dfn"),
        4 => cfg!(feature = "rnnoise"),
        _ => false,
    }
}

/// Create an engine. `config` = JSON text of the engine config (may be empty), `weights` = blob.
pub fn create_engine(kind: u32, config: &str, weights: &[u8]) -> Result<Box<dyn NoiseEngine>, String> {
    let cfg = if config.trim().is_empty() { json::Value::Object(vec![]) } else { json::parse(config)? };
    let _ = (&cfg, weights);
    match kind {
        0 => Ok(Box::new(vi_engine::passthrough::Passthrough::new(&cfg))),
        #[cfg(feature = "graph")]
        2 => vi_graph::create(&cfg, weights),
        #[cfg(feature = "dfn")]
        3 => vi_dfn::create(&cfg, weights),
        #[cfg(feature = "rnnoise")]
        4 => vi_rnnoise::create(&cfg, weights),
        k if (k as usize) < KINDS.len() => Err(format!("plugin {} not compiled into this build", KINDS[k as usize])),
        _ => Err(format!("unknown engine kind {kind}")),
    }
}

/// A parsed models/<name>/manifest.json.
pub struct Manifest {
    pub name: String,
    pub engine: String,
    /// weights path resolved relative to the manifest directory (None if the engine has none)
    pub weights: Option<std::path::PathBuf>,
    /// JSON text of the "config" object, passed verbatim to the engine
    pub config: String,
    pub raw: json::Value,
}

impl Manifest {
    pub fn load(path: &std::path::Path) -> Result<Manifest, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw = json::parse(&text)?;
        let s = |k: &str| raw.get(k).and_then(|v| v.as_str()).map(String::from);
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        // packages (engine "graph"): config = {"pipeline": .., "graph": <graph.json>, ..config}
        let mut cfg: Vec<(String, json::Value)> = match raw.get("config") {
            Some(json::Value::Object(v)) => v.clone(),
            _ => vec![],
        };
        if let Some(g) = s("graph") {
            let gp = dir.join(g);
            let gt = std::fs::read_to_string(&gp).map_err(|e| format!("{}: {e}", gp.display()))?;
            cfg.push(("graph".into(), json::parse(&gt)?));
            if let Some(p) = raw.get("pipeline") {
                cfg.push(("pipeline".into(), p.clone()));
            }
        }
        Ok(Manifest {
            name: s("name").unwrap_or_default(),
            engine: s("engine").ok_or("manifest: missing engine")?,
            weights: s("weights").map(|w| dir.join(w)),
            config: json::Value::Object(cfg).to_string(),
            raw,
        })
    }

    /// Set/override one top-level config key (e.g. "frames_per_call").
    pub fn set_config(&mut self, key: &str, v: json::Value) {
        let mut c = match json::parse(&self.config) {
            Ok(json::Value::Object(o)) => o,
            _ => vec![],
        };
        c.retain(|(k, _)| k != key);
        c.push((key.to_string(), v));
        self.config = json::Value::Object(c).to_string();
    }

    pub fn create(&self) -> Result<Box<dyn NoiseEngine>, String> {
        let kind = kind_id(&self.engine).ok_or(format!("unknown engine kind {}", self.engine))?;
        let w = match &self.weights {
            Some(p) => std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?,
            None => Vec::new(),
        };
        create_engine(kind, &self.config, &w)
    }
}

#[cfg(test)]
mod stages_test;

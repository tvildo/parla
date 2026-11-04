//! Plugin `graph`: generic model-package interpreter (models/FORMAT.md). The operator graph runs
//! as a `ModelCore`; the declarative vi_audio front/back stages come from the manifest pipeline.
//!
//! config JSON: {"pipeline": {...}, "graph": {...}, "frames_per_call": optional override}

pub mod graph;
pub mod ops;

use graph::{numel, Graph};
use vi_engine::driver::Pipeline;
use vi_engine::json::Value;
#[cfg(not(target_arch = "wasm32"))]
use vi_engine::acc_add;
use vi_engine::{ModelCore, NoiseEngine};
use vi_kernels::Backend;

/// The operator graph as a ModelCore: input = pipeline.input tensor, output = pipeline.output.
pub struct GraphCore {
    pub g: Graph,
    gin: usize,
    gout: usize,
    name: String,
}

impl GraphCore {
    pub fn new(gj: &Value, weights: &[u8], t: usize, input: &str, output: &str) -> Result<GraphCore, String> {
        let g = Graph::load(gj, weights, t, &[input, output])?;
        let gin = g.id(input).ok_or("pipeline.input not a graph tensor")?;
        let gout = g.id(output).ok_or("pipeline.output not a graph tensor")?;
        if !g.inputs.iter().any(|(_, i)| *i == gin) {
            return Err("pipeline.input must be a graph input".into());
        }
        let name = gj.get("name").and_then(|v| v.as_str()).unwrap_or("graph").to_string();
        Ok(GraphCore { g, gin, gout, name })
    }
    pub fn input_shape(&self) -> &[usize] {
        &self.g.tensors[self.gin].shape
    }
    pub fn output_len(&self) -> usize {
        numel(&self.g.tensors[self.gout].shape)
    }
}

impl ModelCore for GraphCore {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn input(&mut self) -> &mut [f32] {
        &mut self.g.tensors[self.gin].data
    }
    fn output(&self) -> &[f32] {
        &self.g.tensors[self.gout].data
    }
    fn run(&mut self) {
        self.g.run();
    }
    fn reset(&mut self) {
        self.g.reset();
    }
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let id = self.g.id(name)?;
        Some((self.g.tensors[id].shape.clone(), self.g.tensors[id].data.clone()))
    }
    fn set_scalar(&mut self, scalar: bool) {
        self.g.backend = if scalar { Backend::Scalar } else { Backend::detect() };
    }
    fn backend_name(&self) -> &'static str {
        self.g.backend.name()
    }
    fn set_stages(&mut self, n: usize) -> usize {
        self.g.set_stages(n)
    }
    fn run_stage(&mut self, s: usize) {
        self.g.run_stage(s);
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn run_profiled(&mut self, acc: &mut Vec<(String, f64)>) {
        let n = self.g.n_nodes();
        let mut per = vec![0f64; n];
        self.g.run_profiled(&mut per);
        for i in 0..n {
            let k = self.g.op_names[i].clone();
            acc_add(acc, &k, per[i]);
        }
    }
}

/// Engine for a model package config {"pipeline", "graph", "frames_per_call"?}.
pub fn create(cfg: &Value, weights: &[u8]) -> Result<Box<dyn NoiseEngine>, String> {
    let p = cfg.get("pipeline").ok_or("graph engine: config.pipeline missing")?;
    let gj = cfg.get("graph").ok_or("graph engine: config.graph missing")?;
    let t = cfg.num_or("frames_per_call", p.num_or("frames_per_call", 1.0)).max(1.0) as usize;
    let s = |k: &str| p.get(k).and_then(|v| v.as_str()).ok_or(format!("pipeline.{k} missing"));
    let core = GraphCore::new(gj, weights, t, s("input")?, s("output")?)?;
    Ok(Box::new(Pipeline::new(p, t, Box::new(core), "graph")?))
}

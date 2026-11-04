//! Pipeline driver: vi_audio front-end -> ModelCore -> vi_audio back-end, from a manifest
//! pipeline spec (models/FORMAT.md):
//!   {"sample_rate", "hop", "frames_per_call", "graph_latency", "front": {...}, "back": {...}}
//! frame_size = frames_per_call * hop; latency = front + back DSP latency + graph_latency.

use crate::json::Value;
#[cfg(not(target_arch = "wasm32"))]
use crate::acc_add;
use crate::{EngineInfo, ModelCore, NoiseEngine};
use vi_audio::stages::{BackEnd, FrontEnd};

pub struct Pipeline {
    pub sr: u32,
    pub hop: usize,
    pub t: usize,
    pub latency: usize,
    pub front: FrontEnd,
    pub back: BackEnd,
    pub core: Box<dyn ModelCore>,
    kind: &'static str,
    stages: usize,
}

impl Pipeline {
    /// `t` = frames per call the core was built for.
    pub fn new(p: &Value, t: usize, core: Box<dyn ModelCore>, kind: &'static str) -> Result<Pipeline, String> {
        let sr = p.num_or("sample_rate", 16000.0) as u32;
        let hop = p.num_or("hop", 0.0) as usize;
        if hop == 0 {
            return Err("pipeline.hop missing".into());
        }
        let lookup = |n: &str| core.tensor(n);
        let front = FrontEnd::new(p.get("front").ok_or("pipeline.front missing")?, hop, t, &lookup)?;
        let mut back = BackEnd::new(p.get("back").ok_or("pipeline.back missing")?, &front, &lookup)?;
        let mut core = core;
        if back.kind == vi_audio::stages::BackKind::Raw {
            back.raw_len = core.output().len();
            if back.raw_len > t * hop {
                return Err("raw back-end: model output larger than the frame".into());
            }
        }
        if core.input().len() != front.out_len() {
            return Err(format!("model input ({} values) does not match front-end output ({} values)", core.input().len(), front.out_len()));
        }
        if core.output().len() != back.in_len() {
            return Err(format!("model output ({} values) does not match back-end input ({} values)", core.output().len(), back.in_len()));
        }
        let latency = front.latency() + back.latency() + p.num_or("graph_latency", 0.0) as usize;
        Ok(Pipeline { sr, hop, t, latency, front, back, core, kind, stages: 1 })
    }
}

impl NoiseEngine for Pipeline {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: format!("{}:{}", self.kind, self.core.name()),
            sample_rate: self.sr,
            frame_size: self.t * self.hop,
            latency_samples: self.latency,
            in_channels: 1,
            out_channels: 1,
        }
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.front.process(input, self.core.input());
        self.core.run();
        self.back.process(self.core.output(), &self.front.spec, output);
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn profile_frame(&mut self, input: &[f32], output: &mut [f32], acc: &mut Vec<(String, f64)>) {
        let t0 = std::time::Instant::now();
        self.front.process(input, self.core.input());
        acc_add(acc, "(front)", t0.elapsed().as_secs_f64());
        self.core.run_profiled(acc);
        let t1 = std::time::Instant::now();
        self.back.process(self.core.output(), &self.front.spec, output);
        acc_add(acc, "(back)", t1.elapsed().as_secs_f64());
    }
    fn reset(&mut self) {
        self.core.reset();
        self.front.reset();
        self.back.reset();
    }
    fn frame_size(&self) -> usize {
        self.t * self.hop
    }
    fn set_scalar(&mut self, scalar: bool) {
        self.core.set_scalar(scalar);
    }
    fn backend_name(&self) -> &'static str {
        self.core.backend_name()
    }
    fn set_stages(&mut self, n: usize) -> usize {
        self.stages = self.core.set_stages(n.max(1));
        self.stages
    }
    // front-end in the first part, back-end in the last, the core's parts in between
    fn process_stage(&mut self, s: usize, input: &[f32], output: &mut [f32]) {
        if s == 0 {
            self.front.process(input, self.core.input());
        }
        self.core.run_stage(s);
        if s + 1 == self.stages {
            self.back.process(self.core.output(), &self.front.spec, output);
        }
    }
}

//! Engine kind `passthrough`: output = input, zero latency. Baseline for host overhead / latency.
//! config: {"sample_rate": 24000, "frame_size": 40}

use crate::{json::Value, EngineInfo, NoiseEngine};

pub struct Passthrough {
    sr: u32,
    fs: usize,
}

impl Passthrough {
    pub fn new(cfg: &Value) -> Passthrough {
        Passthrough { sr: cfg.num_or("sample_rate", 24000.0) as u32, fs: cfg.num_or("frame_size", 40.0) as usize }
    }
}

impl NoiseEngine for Passthrough {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "passthrough".into(),
            sample_rate: self.sr,
            frame_size: self.fs,
            latency_samples: 0,
            in_channels: 1,
            out_channels: 1,
        }
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        output[..self.fs].copy_from_slice(&input[..self.fs]);
    }
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        output[..input.len()].copy_from_slice(input);
    }
    fn reset(&mut self) {}
    fn frame_size(&self) -> usize {
        self.fs
    }
}

//! vi_engine — the engine contract shared by all model plugins.
//!
//! * `NoiseEngine`: what hosts drive (mono f32 frames in/out at the engine rate). Created by the
//!   plugin registry in the top crate (`vi_rs::create_engine`).
//! * `ModelCore`: the model-specific part only (one call = one block of model input -> output
//!   tensor). `driver::Pipeline` wraps any ModelCore with the declarative vi_audio front/back
//!   stages from a manifest pipeline spec, so a plugin with standard DSP needs no audio code.
//! * Plugins with custom DSP (RNNoise band features, DeepFilterNet ERB/deep filtering) implement
//!   `NoiseEngine` themselves and compose vi_audio stages + their own stages.

pub mod driver;
pub mod passthrough;

pub use vi_audio::json;

/// Static description of an engine instance. Audio is mono f32 at `sample_rate`.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub name: String,
    pub sample_rate: u32,
    /// samples per `process_frame` call
    pub frame_size: usize,
    /// algorithmic delay of output vs input in samples (excluding frame buffering by the host)
    pub latency_samples: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

pub trait NoiseEngine {
    fn info(&self) -> EngineInfo;
    /// exactly `frame_size` samples in and out
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]);
    fn reset(&mut self);
    /// cheap accessor (no allocation) for the realtime path
    fn frame_size(&self) -> usize {
        self.info().frame_size
    }
    /// any multiple of `frame_size`; engines may batch internally (must equal frame-by-frame output)
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let fs = self.frame_size();
        for (i, o) in input.chunks_exact(fs).zip(output.chunks_exact_mut(fs)) {
            self.process_frame(i, o);
        }
    }
    /// force scalar kernels (benchmarking); default no-op
    fn set_scalar(&mut self, _scalar: bool) {}
    /// one frame with per-op timing accumulated into `acc` (op name -> seconds); default: no-op
    fn profile_frame(&mut self, input: &[f32], output: &mut [f32], acc: &mut Vec<(String, f64)>) {
        let _ = acc;
        self.process_frame(input, output);
    }
    /// name of the kernel backend in use
    fn backend_name(&self) -> &'static str {
        "scalar"
    }
    /// Realtime smoothing: split one frame's work into up to `n` sequential parts so a host can run
    /// one part per audio callback instead of the whole frame in one. Returns the number of parts
    /// the engine will use (1 = not splittable). Then, per frame, the host calls
    /// `process_stage(s, input, output)` for s = 0..parts with the SAME input/output slices left
    /// untouched in between; the result equals `process_frame` bit-exactly.
    fn set_stages(&mut self, _n: usize) -> usize {
        1
    }
    fn process_stage(&mut self, s: usize, input: &[f32], output: &mut [f32]) {
        if s == 0 {
            self.process_frame(input, output);
        }
    }    /// the model's own speech probability for the last frame (0..1); < 0 = the model has none
    fn speech_prob(&self) -> f32 {
        -1.0
    }
}

/// Model-specific compute block driven by `driver::Pipeline`.
pub trait ModelCore {
    fn name(&self) -> String;
    /// input buffer the front-end fills (length = front-end out_len)
    fn input(&mut self) -> &mut [f32];
    /// output buffer the back-end reads after `run`
    fn output(&self) -> &[f32];
    fn run(&mut self);
    fn reset(&mut self);
    /// named constant tensors (windows / filterbanks referenced by the pipeline spec)
    fn tensor(&self, _name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        None
    }
    fn set_scalar(&mut self, _scalar: bool) {}
    fn backend_name(&self) -> &'static str {
        "scalar"
    }
    fn run_profiled(&mut self, acc: &mut Vec<(String, f64)>) {
        let _ = acc;
        self.run();
    }
    /// `run` split into up to `n` sequential parts (see `NoiseEngine::set_stages`); returns the count
    fn set_stages(&mut self, _n: usize) -> usize {
        1
    }
    fn run_stage(&mut self, s: usize) {
        if s == 0 {
            self.run();
        }
    }
}

/// accumulate `v` seconds under key `k`
pub fn acc_add(acc: &mut Vec<(String, f64)>, k: &str, v: f64) {
    match acc.iter_mut().find(|(n, _)| n == k) {
        Some(e) => e.1 += v,
        None => acc.push((k.to_string(), v)),
    }
}

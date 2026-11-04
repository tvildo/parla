//! Host layer shared by every way of running an engine (wasm C ABI in browsers and standalone wasm
//! runtimes, native code). Model-agnostic, wraps any `NoiseEngine` with:
//!   * on/off and strength (`set_mix`): output = g * denoised + (1 - g) * input delayed by exactly
//!     the engine latency, g ramped linearly (default 20 ms) so toggling never clicks or jumps in time;
//!     at g = 1 the output IS the engine output (bit-exact), at g = 0 it IS the delayed input;
//!   * voice activity (`vi_audio::vad`) on the denoised signal (independent of the mix);
//!   * frame parts (`set_stages` / `process_stage`) for realtime smoothing;
//!   * a count of live instances (`live()`, diagnostics).
//! Realtime-safe after `new`: no allocation in process.

use crate::NoiseEngine;
use std::sync::atomic::{AtomicU32, Ordering};
use vi_audio::vad::Vad;

static LIVE: AtomicU32 = AtomicU32::new(0);

/// engine instances alive right now
pub fn live() -> u32 {
    LIVE.load(Ordering::Relaxed)
}

pub struct Host {
    pub e: Box<dyn NoiseEngine>,
    pub vad: Vad,
    sr: u32,
    fs: usize,
    stages: usize,
    // mix: current gain, target, per-sample step; dry ring of `latency` samples
    gain: f32,
    target: f32,
    step: f32,
    ring: Vec<f32>,
    pos: usize,
    _live: Live,
}

/// counts its host in `LIVE` while alive
struct Live;

impl Drop for Live {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Host {
    pub fn new(e: Box<dyn NoiseEngine>) -> Result<Host, String> {
        let info = e.info();
        LIVE.fetch_add(1, Ordering::Relaxed);
        Ok(Host {
            vad: Vad::new(info.sample_rate),
            sr: info.sample_rate,
            fs: info.frame_size,
            stages: 1,
            gain: 1.0,
            target: 1.0,
            step: 0.0,
            ring: vec![0.0; info.latency_samples],
            pos: 0,
            e,
            _live: Live,
        })
    }

    pub fn create(kind: u32, config: &str, weights: &[u8]) -> Result<Host, String> {
        Host::new(crate::create_engine(kind, config, weights)?)
    }

    pub fn frame_size(&self) -> usize {
        self.fs
    }

    /// strength 0..1 (0 = noise cancellation off, 1 = full), reached linearly over `ramp_ms`
    pub fn set_mix(&mut self, strength: f32, ramp_ms: f32) {
        self.target = strength.clamp(0.0, 1.0);
        let n = (ramp_ms.max(0.0) * self.sr as f32 / 1000.0).round();
        if n < 1.0 {
            self.gain = self.target;
            self.step = 0.0;
        } else {
            self.step = (self.target - self.gain).abs() / n;
        }
    }

    pub fn mix(&self) -> (f32, f32) {
        (self.gain, self.target)
    }

    /// any multiple of the frame size (engines may batch internally)
    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        self.e.process(x, y);
        self.post(x, y);
    }

    pub fn set_stages(&mut self, n: usize) -> usize {
        self.stages = self.e.set_stages(n);
        self.stages
    }

    /// part `s` of one frame (same x / y for every part); the mix happens with the last part
    pub fn process_stage(&mut self, s: usize, x: &[f32], y: &mut [f32]) {
        self.e.process_stage(s, x, y);
        if s + 1 == self.stages {
            self.post(x, y);
        }
    }

    pub fn reset(&mut self) {
        self.e.reset();
        self.vad.reset();
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }

    /// [speech probability 0..1, speaking 0/1, level dBFS] of the denoised signal
    pub fn voice(&self) -> [f32; 3] {
        [self.vad.prob, self.vad.speaking as i32 as f32, self.vad.level_db]
    }

    // y: denoised in, final out. The dry ring advances on every sample in every mode, so the
    // original is always aligned when the mix changes.
    fn post(&mut self, x: &[f32], y: &mut [f32]) {
        self.vad.push(y, self.e.speech_prob());
        let n = self.ring.len();
        let steady_full = self.gain == 1.0 && self.target == 1.0;
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let d = if n == 0 {
                *xi
            } else {
                let d = self.ring[self.pos];
                self.ring[self.pos] = *xi;
                self.pos = if self.pos + 1 == n { 0 } else { self.pos + 1 };
                d
            };
            if steady_full {
                continue; // bit-exact engine output
            }
            if self.gain != self.target {
                self.gain = if self.gain < self.target { (self.gain + self.step).min(self.target) } else { (self.gain - self.step).max(self.target) };
            }
            *yi = if self.gain == 0.0 { d } else { self.gain * *yi + (1.0 - self.gain) * d };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;
    use std::path::Path;

    fn signal(n: usize) -> Vec<f32> {
        let mut s = 7u32;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                0.3 * (i as f32 * 0.021).sin() + 0.05 * ((s >> 9) as f32 / (1u32 << 23) as f32 - 0.5)
            })
            .collect()
    }

    fn host(model: &str) -> Host {
        let m = Manifest::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models").join(model).join("manifest.json")).unwrap();
        Host::new(m.create().unwrap()).unwrap()
    }

    fn run(h: &mut Host, x: &[f32], toggles: &[(usize, f32, f32)]) -> Vec<f32> {
        let fs = h.frame_size();
        let mut y = vec![0f32; x.len()];
        for (k, (i, o)) in x.chunks_exact(fs).zip(y.chunks_exact_mut(fs)).enumerate() {
            for &(at, s, ms) in toggles {
                if at == k {
                    h.set_mix(s, ms);
                }
            }
            h.process(i, o);
        }
        y
    }

    #[test]
    fn full_strength_is_bit_exact_engine_output() {
        for model in ["gtcrn", "rnnoise", "deepfilternet3"] {
            let mut h = host(model);
            let mut e = host(model).e;
            let fs = h.frame_size();
            let x = signal(fs * 120);
            let y = run(&mut h, &x, &[]);
            let mut r = vec![0f32; x.len()];
            for (i, o) in x.chunks_exact(fs).zip(r.chunks_exact_mut(fs)) {
                e.process_frame(i, o);
            }
            assert!(y.iter().zip(&r).all(|(a, b)| a.to_bits() == b.to_bits()), "{model}");
        }
    }

    #[test]
    fn off_is_input_delayed_by_latency() {
        for model in ["gtcrn", "rnnoise", "deepfilternet3", "spectral"] {
            let mut h = host(model);
            let lat = h.e.info().latency_samples;
            let fs = h.frame_size();
            let x = signal(fs * 80);
            let y = run(&mut h, &x, &[(0, 0.0, 0.0)]);
            for i in lat..x.len() {
                assert_eq!(y[i].to_bits(), x[i - lat].to_bits(), "{model} at {i}");
            }
        }
    }

    #[test]
    fn toggling_never_clicks() {
        // the mix of two aligned signals moves at most by the ramp step x their difference, so the
        // output's sample-to-sample jump stays within what the two signals themselves do
        for model in ["gtcrn", "rnnoise", "deepfilternet3", "dtln", "nsnet2"] {
            let mut on = host(model);
            let mut off = host(model);
            let mut t = host(model);
            let fs = on.frame_size();
            let x = signal(fs * 200);
            let a = run(&mut on, &x, &[]);
            let b = run(&mut off, &x, &[(0, 0.0, 0.0)]);
            let y = run(&mut t, &x, &[(40, 0.0, 20.0), (90, 1.0, 20.0), (130, 0.3, 20.0), (131, 1.0, 20.0)]);
            let jump = |v: &[f32], i: usize| (v[i] - v[i - 1]).abs();
            let worst = (1..x.len()).map(|i| jump(&y, i) - jump(&a, i).max(jump(&b, i))).fold(0f32, f32::max);
            assert!(worst < 0.01, "{model}: toggle adds a jump of {worst}");
            // after the ramps, back to exactly the engine output
            let tail = 140 * fs;
            assert!(y[tail..].iter().zip(&a[tail..]).all(|(p, q)| p.to_bits() == q.to_bits()), "{model}: not back to full");
        }
    }

    #[test]
    fn staged_equals_process() {
        let mut a = host("deepfilternet3");
        let mut b = host("deepfilternet3");
        let parts = b.set_stages(3);
        let fs = a.frame_size();
        let x = signal(fs * 60);
        let mut ya = vec![0f32; x.len()];
        let mut yb = vec![0f32; x.len()];
        a.set_mix(0.4, 10.0);
        b.set_mix(0.4, 10.0);
        for ((i, oa), ob) in x.chunks_exact(fs).zip(ya.chunks_exact_mut(fs)).zip(yb.chunks_exact_mut(fs)) {
            a.process(i, oa);
            for s in 0..parts {
                b.process_stage(s, i, ob);
            }
        }
        assert!(ya.iter().zip(&yb).all(|(p, q)| p.to_bits() == q.to_bits()));
        assert_eq!(a.voice()[1], b.voice()[1]);
    }

    #[test]
    fn hosts_are_counted() {
        let h = host("passthrough");
        assert!(live() >= 1); // other tests run in parallel: only a lower bound
        drop(h);
    }
}

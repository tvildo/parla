//! Voice activity on the engine's DENOISED output (any model): is someone speaking, and how loud.
//! Noise the model removed doesn't count, so this follows the voice the model keeps.
//!
//! Per 10 ms block: level L (dBFS); an adaptive floor F tracks the quiet level (follows drops
//! within a block, rises at 3 dB/s); speech evidence = L above max(F + 9 dB, -60 dBFS). A model's
//! own speech probability (RNNoise's VAD output), when given, is combined with it. Decision with
//! 30 ms attack and 250 ms hangover, so a UI (animated mic) doesn't flicker between syllables.
//! Realtime-safe: no allocation after `new`. f32 only (f64 libm would add ~KBs to the wasm bundle).

pub struct Vad {
    block: usize,
    acc: f32,
    n: usize,
    floor: f32,
    on_count: usize,
    hang: usize,
    hang_blocks: usize,
    attack_blocks: usize,
    pub level_db: f32,
    pub prob: f32,
    pub speaking: bool,
}

impl Vad {
    pub fn new(sample_rate: u32) -> Vad {
        Vad {
            block: (sample_rate as usize / 100).max(1),
            acc: 0.0,
            n: 0,
            floor: -60.0,
            on_count: 0,
            hang: 0,
            hang_blocks: 25,
            attack_blocks: 3,
            level_db: -120.0,
            prob: 0.0,
            speaking: false,
        }
    }

    pub fn reset(&mut self) {
        let sr = self.block as u32 * 100;
        *self = Vad::new(sr);
    }

    /// feed output samples; `model_prob` < 0 = the model has no speech probability
    pub fn push(&mut self, y: &[f32], model_prob: f32) {
        for &v in y {
            self.acc += v * v;
            self.n += 1;
            if self.n == self.block {
                let l = 10.0 * (self.acc / self.n as f32 + 1e-12).log10();
                self.acc = 0.0;
                self.n = 0;
                self.block_done(l, model_prob);
            }
        }
    }

    fn block_done(&mut self, l: f32, model_prob: f32) {
        self.level_db = l;
        // floor: down immediately, up slowly (3 dB/s) so long speech doesn't lift it much
        self.floor = if l < self.floor { l } else { self.floor + 0.03 };
        let thr = (self.floor + 9.0).max(-60.0);
        let e = (0.5 + (l - thr) / 12.0).clamp(0.0, 1.0); // energy evidence 0..1 (ramp over +-6 dB)
        let p = if model_prob >= 0.0 { (e * model_prob.clamp(0.0, 1.0)).sqrt() } else { e };
        self.prob = 0.7 * self.prob + 0.3 * p;
        if p > 0.5 {
            self.on_count += 1;
            if self.on_count >= self.attack_blocks {
                self.speaking = true;
                self.hang = self.hang_blocks;
            }
        } else {
            self.on_count = 0;
            if self.hang > 0 {
                self.hang -= 1;
            } else {
                self.speaking = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (i as f32 * 0.07).sin()).collect()
    }

    #[test]
    fn silence_noise_then_speech_then_hangover() {
        let mut v = Vad::new(16000);
        v.push(&vec![0.0; 16000], -1.0);
        assert!(!v.speaking);
        // steady low noise floor: not speech
        v.push(&tone(16000, 0.001), -1.0);
        assert!(!v.speaking, "floor {:.1} level {:.1}", v.floor, v.level_db);
        v.push(&tone(1600, 0.2), -1.0);
        assert!(v.speaking);
        assert!(v.level_db > -20.0);
        // stays on through the hangover, then off
        v.push(&tone(1600, 0.001), -1.0);
        assert!(v.speaking);
        v.push(&tone(8000, 0.001), -1.0);
        assert!(!v.speaking);
    }

    #[test]
    fn model_probability_vetoes() {
        let mut v = Vad::new(16000);
        v.push(&tone(16000, 0.001), 0.0);
        v.push(&tone(3200, 0.2), 0.0); // loud but the model says no speech
        assert!(!v.speaking);
        v.push(&tone(3200, 0.2), 1.0);
        assert!(v.speaking);
    }
}

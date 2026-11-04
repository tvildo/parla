//! vi_audio — the shared audio shell used by every vi model plugin (std only):
//!   fft      mixed-radix FFT, real FFT / inverse, windows
//!   stages   pipeline front-ends (time / frames / stft + features + filterbank) and back-ends
//!            (waveform / ola / istft from spectrum, magnitude mask, complex mask; WOLA), mix
//!   resample streaming polyphase resampler (48 kHz <-> model rate) for native / Node hosts
//!   json     minimal JSON (manifests, pipeline specs)
//!   vad      voice activity on the denoised output (animated mic, telemetry)

pub mod fft;
pub mod json;
pub mod resample;
pub mod stages;
pub mod vad;

pub use fft::{window, Rfft};

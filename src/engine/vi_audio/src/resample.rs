//! Streaming rational (L/M) polyphase resampler, Kaiser-windowed sinc FIR — same design and
//! coefficients as web/resampler.js (32 taps per phase, beta 8.6, cutoff 0.9 * Nyquist of the
//! lower rate). For native / Node hosts (48 kHz <-> model rate); allocation-free in steady state.

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn bessel0(x: f64) -> f64 {
    let (mut s, mut t) = (1.0, 1.0);
    for k in 1..50 {
        t *= (x / (2.0 * k as f64)).powi(2);
        s += t;
        if t < 1e-12 * s {
            break;
        }
    }
    s
}

pub struct Resampler {
    l: usize,
    m: usize,
    taps: usize,
    ph: Vec<f32>,
    buf: Vec<f32>,
    n: usize,
    u: usize,
    pub delay_seconds: f64,
    identity: bool,
}

impl Resampler {
    pub fn new(fs_in: usize, fs_out: usize, taps: usize, max_block: usize) -> Resampler {
        let g = gcd(fs_in, fs_out);
        let (l, m) = (fs_out / g, fs_in / g);
        if l == 1 && m == 1 {
            return Resampler { l, m, taps, ph: vec![], buf: vec![], n: 0, u: 0, delay_seconds: 0.0, identity: true };
        }
        let k = taps * l;
        let fc = 0.5 / l.max(m) as f64 * 0.9;
        let (beta, c) = (8.6, (k as f64 - 1.0) / 2.0);
        let i0b = bessel0(beta);
        let mut ph = vec![0f32; k];
        for kk in 0..k {
            let x = kk as f64 - c;
            let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x) };
            let w = bessel0(beta * (1.0 - (x / c).powi(2)).max(0.0).sqrt()) / i0b;
            let (p, j) = (kk % l, kk / l);
            ph[p * taps + j] = (l as f64 * sinc * w) as f32;
        }
        Resampler { l, m, taps, ph, buf: vec![0.0; taps - 1 + max_block], n: taps - 1, u: (taps - 1) * l, delay_seconds: c / (fs_in * l) as f64, identity: false }
    }

    /// upper bound of output samples for `n_in` input samples
    pub fn max_out(&self, n_in: usize) -> usize {
        n_in * self.l / self.m + 2
    }

    /// x -> out (returns number of samples written)
    pub fn process(&mut self, x: &[f32], out: &mut [f32]) -> usize {
        if self.identity {
            out[..x.len()].copy_from_slice(x);
            return x.len();
        }
        let (l, m, t) = (self.l, self.m, self.taps);
        self.buf[self.n..self.n + x.len()].copy_from_slice(x);
        self.n += x.len();
        let mut u = self.u;
        let mut o = 0;
        loop {
            let i = u / l;
            if i >= self.n {
                break;
            }
            let p = u - i * l;
            let base = p * t;
            let mut s = 0f32;
            for j in 0..t {
                s += self.ph[base + j] * self.buf[i - j];
            }
            out[o] = s;
            o += 1;
            u += m;
        }
        let drop = self.n - (t - 1);
        self.buf.copy_within(drop..self.n, 0);
        self.n = t - 1;
        self.u = u - drop * l;
        o
    }

    pub fn reset(&mut self) {
        if self.identity {
            return;
        }
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.n = self.taps - 1;
        self.u = (self.taps - 1) * self.l;
    }
}

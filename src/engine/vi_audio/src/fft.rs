//! FFT + windows for vi_audio (std only, allocation-free after init):
//! mixed-radix complex FFT (Stockham, radix 2/3/4/5 + generic), real FFT / inverse real FFT
//! helpers, analysis/synthesis windows.

use std::f64::consts::PI;

#[derive(Clone, Copy, Default, Debug)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

#[inline(always)]
fn cmul(a: C32, b: C32) -> C32 {
    C32 { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}

struct Stage {
    p: usize,
    m: usize,
    s: usize,
    tw: Vec<C32>, // [m][p]: w_q^t = exp(-2 pi i q t / n_stage)
}

/// Forward complex FFT plan of size n (unnormalized, e^{-2 pi i k n / N}).
pub struct Fft {
    pub n: usize,
    stages: Vec<Stage>,
    root: Vec<C32>, // generic radix: exp(-2 pi i r / p) tables, concatenated per stage
    work: Vec<C32>,
}

fn factor(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    while n % 4 == 0 {
        f.push(4);
        n /= 4;
    }
    for p in [2usize, 3, 5] {
        while n % p == 0 {
            f.push(p);
            n /= p;
        }
    }
    let mut p = 7;
    while n > 1 {
        while n % p == 0 {
            f.push(p);
            n /= p;
        }
        p += 2;
    }
    f
}

impl Fft {
    pub fn new(n: usize) -> Fft {
        let mut stages = Vec::new();
        let mut len = n;
        let mut s = 1;
        let mut root = Vec::new();
        for p in factor(n) {
            let m = len / p;
            let mut tw = Vec::with_capacity(m * p);
            for q in 0..m {
                for t in 0..p {
                    let a = -2.0 * PI * (q * t) as f64 / len as f64;
                    tw.push(C32 { re: a.cos() as f32, im: a.sin() as f32 });
                }
            }
            stages.push(Stage { p, m, s, tw });
            len = m;
            s *= p;
        }
        for st in &stages {
            for r in 0..st.p {
                let a = -2.0 * PI * r as f64 / st.p as f64;
                root.push(C32 { re: a.cos() as f32, im: a.sin() as f32 });
            }
        }
        Fft { n, stages, root, work: vec![C32::default(); n] }
    }

    /// In-place forward transform of `x` (len n).
    pub fn forward(&mut self, x: &mut [C32]) {
        let n = self.n;
        if n <= 1 {
            return;
        }
        let mut work = std::mem::take(&mut self.work);
        let mut in_x = true; // current data in x?
        let mut ro = 0;
        for st in &self.stages {
            let (src, dst): (&[C32], &mut [C32]) = if in_x { (&*x, &mut work[..]) } else { (&work[..], &mut *x) };
            // SAFETY of aliasing: src and dst are different buffers
            let src: &[C32] = unsafe { std::slice::from_raw_parts(src.as_ptr(), n) };
            let (p, m, s) = (st.p, st.m, st.s);
            let rt = &self.root[ro..ro + p];
            for q in 0..m {
                let tw = &st.tw[q * p..(q + 1) * p];
                for k in 0..s {
                    let a = |r: usize| src[k + s * (q + r * m)];
                    let o = |t: usize| k + s * (p * q + t);
                    match p {
                        2 => {
                            let (a0, a1) = (a(0), a(1));
                            dst[o(0)] = C32 { re: a0.re + a1.re, im: a0.im + a1.im };
                            dst[o(1)] = cmul(C32 { re: a0.re - a1.re, im: a0.im - a1.im }, tw[1]);
                        }
                        4 => {
                            let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));
                            let t0 = C32 { re: a0.re + a2.re, im: a0.im + a2.im };
                            let t1 = C32 { re: a0.re - a2.re, im: a0.im - a2.im };
                            let t2 = C32 { re: a1.re + a3.re, im: a1.im + a3.im };
                            let t3 = C32 { re: a1.re - a3.re, im: a1.im - a3.im };
                            // -i * t3
                            let t3j = C32 { re: t3.im, im: -t3.re };
                            dst[o(0)] = C32 { re: t0.re + t2.re, im: t0.im + t2.im };
                            dst[o(1)] = cmul(C32 { re: t1.re + t3j.re, im: t1.im + t3j.im }, tw[1]);
                            dst[o(2)] = cmul(C32 { re: t0.re - t2.re, im: t0.im - t2.im }, tw[2]);
                            dst[o(3)] = cmul(C32 { re: t1.re - t3j.re, im: t1.im - t3j.im }, tw[3]);
                        }
                        _ => {
                            for t in 0..p {
                                let mut acc = C32::default();
                                for r in 0..p {
                                    let w = rt[(r * t) % p];
                                    let v = cmul(a(r), w);
                                    acc.re += v.re;
                                    acc.im += v.im;
                                }
                                dst[o(t)] = if t == 0 { acc } else { cmul(acc, tw[t]) };
                            }
                        }
                    }
                }
            }
            ro += p;
            in_x = !in_x;
        }
        if !in_x {
            x.copy_from_slice(&work[..n]);
        }
        self.work = work;
    }
}

/// Real FFT helper: forward rfft (n -> n/2+1 complex as interleaved re,im) and inverse.
pub struct Rfft {
    pub n: usize,
    fft: Fft,
    buf: Vec<C32>,
}

impl Rfft {
    pub fn new(n: usize) -> Rfft {
        Rfft { n, fft: Fft::new(n), buf: vec![C32::default(); n] }
    }
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }
    /// x: n real samples -> out: (n/2+1)*2 floats (re, im), unnormalized.
    pub fn forward(&mut self, x: &[f32], out: &mut [f32]) {
        for (b, &v) in self.buf.iter_mut().zip(x) {
            *b = C32 { re: v, im: 0.0 };
        }
        self.fft.forward(&mut self.buf);
        for k in 0..self.bins() {
            out[2 * k] = self.buf[k].re;
            out[2 * k + 1] = self.buf[k].im;
        }
    }
    /// spec: (n/2+1)*2 floats -> out: n real samples, scaled 1/n (imag of DC/Nyquist ignored).
    pub fn inverse(&mut self, spec: &[f32], out: &mut [f32]) {
        let n = self.n;
        let nb = self.bins();
        // ifft(X) = conj(fft(conj(X))) / n ; build conj of the Hermitian-extended spectrum
        for k in 0..nb {
            self.buf[k] = C32 { re: spec[2 * k], im: -spec[2 * k + 1] };
        }
        self.buf[0].im = 0.0;
        if n % 2 == 0 {
            self.buf[n / 2].im = 0.0;
        }
        for k in nb..n {
            let j = n - k;
            self.buf[k] = C32 { re: spec[2 * j], im: spec[2 * j + 1] };
        }
        self.fft.forward(&mut self.buf);
        let s = 1.0 / n as f32;
        for i in 0..n {
            out[i] = self.buf[i].re * s;
        }
    }
}

/// Window by name (periodic variants match torch.*_window defaults).
pub fn window(name: &str, n: usize) -> Result<Vec<f32>, String> {
    let k = |i: usize| i as f64;
    let nn = n as f64;
    let w: Vec<f64> = match name {
        "none" | "rect" => vec![1.0; n],
        "hann" => (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * k(i) / nn).cos()).collect(),
        "hann_sym" => (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * k(i) / (nn - 1.0)).cos()).collect(),
        "sqrt_hann" => (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * k(i) / nn).cos()).sqrt()).collect(),
        "sqrt_hann_sym" => (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * k(i) / (nn - 1.0)).cos()).max(0.0).sqrt()).collect(),
        "hamming" => (0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * k(i) / nn).cos()).collect(),
        "vorbis" => (0..n).map(|i| (PI / 2.0 * (PI * (k(i) + 0.5) / nn).sin().powi(2)).sin()).collect(),
        _ => return Err(format!("unknown window {name}")),
    };
    Ok(w.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_matches_dft() {
        for n in [1usize, 2, 3, 4, 5, 8, 12, 16, 20, 30, 49, 160, 320, 480, 512, 960] {
            let x: Vec<C32> = (0..n).map(|i| C32 { re: ((i * 7 % 13) as f32) - 6.0, im: ((i * 3 % 5) as f32) - 2.0 }).collect();
            let mut y = x.clone();
            Fft::new(n).forward(&mut y);
            for k in 0..n {
                let (mut re, mut im) = (0f64, 0f64);
                for (j, v) in x.iter().enumerate() {
                    let a = -2.0 * PI * ((j * k) % n) as f64 / n as f64;
                    re += v.re as f64 * a.cos() - v.im as f64 * a.sin();
                    im += v.re as f64 * a.sin() + v.im as f64 * a.cos();
                }
                let err = ((y[k].re as f64 - re).abs()).max((y[k].im as f64 - im).abs());
                assert!(err < 1e-3 * (n as f64).max(1.0), "n={n} k={k} err={err}");
            }
        }
    }
    #[test]
    fn rfft_roundtrip() {
        for n in [320usize, 480, 512] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
            let mut r = Rfft::new(n);
            let mut s = vec![0.0; (n / 2 + 1) * 2];
            let mut y = vec![0.0; n];
            r.forward(&x, &mut s);
            r.inverse(&s, &mut y);
            for i in 0..n {
                assert!((x[i] - y[i]).abs() < 1e-5);
            }
        }
    }
}

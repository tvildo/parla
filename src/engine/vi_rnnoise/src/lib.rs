//! Plugin `rnnoise`: RNNoise v0.1 (Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time
//! Full-Band Speech Enhancement", MMSP 2018) — our implementation of the published algorithm;
//! weights from models/rnnoise (tools/make_rnnoise.py, int8 arrays / 256).
//!
//! Custom DSP stages (not expressible as a generic pipeline):
//!   * analysis: 960-pt FFT (scaled 1/N) of [previous | current] 480-sample frame, Vorbis
//!     power-complementary window; 22 triangular bands (eband5ms x 4 bins)
//!   * pitch: 2x decimation + 4th-order LPC whitening, coarse (4x) / fine (2x) normalized xcorr
//!     search, octave-error ("doubling") removal with continuity bias
//!   * 42 features: DCT cepstrum of log band energies (+ 2 deltas, 6 coeffs each), DCT of pitch
//!     correlation per band, pitch period, spectral variability over 8 frames
//!   * net: dense(42->24 tanh) -> GRU 24 (relu) -> dense VAD; GRU 48 on [d, vad, f]; GRU 96 on
//!     [vad, noise, f] -> dense 22 sigmoid band gains
//!   * pitch comb filter (per-band, energy-normalized), gain smoothing (max(g, 0.6 g_prev)),
//!     band-gain interpolation, inverse FFT, windowed overlap-add
//! 48 kHz, frame 480, latency 480 samples (10 ms). Input/output scaled to int16 range.

use vi_audio::fft::Rfft;
use vi_engine::json::Value;
use vi_engine::{EngineInfo, NoiseEngine};
use vi_kernels::util::{sigmoid, tanhf};

const FRAME: usize = 480;
const WIN: usize = 960;
const FREQ: usize = 481;
const NB: usize = 22;
const CEPS_MEM: usize = 8;
const NDC: usize = 6;
const NFEAT: usize = NB + 3 * NDC + 2;
const PMIN: usize = 60;
const PMAX: usize = 768;
const PFRAME: usize = 960;
const PBUF: usize = PMAX + PFRAME;
const EBAND: [usize; NB] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];
const SECOND: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

#[derive(Clone, Copy, PartialEq)]
enum Act {
    Tanh,
    Sigmoid,
    Relu,
}

struct Layer {
    w: Vec<f32>, // [in][g*n]
    r: Vec<f32>, // [n][3n] (GRU)
    b: Vec<f32>,
    ni: usize,
    n: usize,
    act: Act,
}

fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::Tanh => tanhf(x),
        Act::Sigmoid => sigmoid(x),
        Act::Relu => x.max(0.0),
    }
}

impl Layer {
    fn dense(&self, x: &[f32], out: &mut [f32]) {
        let n = self.n;
        out[..n].copy_from_slice(&self.b);
        for (j, &xj) in x[..self.ni].iter().enumerate() {
            let row = &self.w[j * n..(j + 1) * n];
            for i in 0..n {
                out[i] += row[i] * xj;
            }
        }
        for v in &mut out[..n] {
            *v = act(self.act, *v);
        }
    }
    fn gru(&self, x: &[f32], h: &mut [f32], gx: &mut [f32], gh: &mut [f32]) {
        let n = self.n;
        let s = 3 * n;
        gx[..s].copy_from_slice(&self.b);
        for (j, &xj) in x[..self.ni].iter().enumerate() {
            let row = &self.w[j * s..(j + 1) * s];
            for i in 0..s {
                gx[i] += row[i] * xj;
            }
        }
        // z, r: recurrent with h
        for v in &mut gh[..2 * n] {
            *v = 0.0;
        }
        for (j, &hj) in h[..n].iter().enumerate() {
            let row = &self.r[j * s..j * s + 2 * n];
            for i in 0..2 * n {
                gh[i] += row[i] * hj;
            }
        }
        for i in 0..2 * n {
            gx[i] = sigmoid(gx[i] + gh[i]);
        }
        // candidate: (r * h) @ R_h
        for v in &mut gh[2 * n..s] {
            *v = 0.0;
        }
        for j in 0..n {
            let rh = gx[n + j] * h[j];
            let row = &self.r[j * s + 2 * n..j * s + s];
            for i in 0..n {
                gh[2 * n + i] += row[i] * rh;
            }
        }
        for i in 0..n {
            let c = act(self.act, gx[2 * n + i] + gh[2 * n + i]);
            let z = gx[i];
            h[i] = z * h[i] + (1.0 - z) * c;
        }
    }
}

pub struct RnnNoise {
    layers: Vec<Layer>, // input_dense, vad_gru, noise_gru, denoise_gru, denoise_output, vad_output
    scale: f32,
    window: [f32; WIN],
    dct_t: [f32; NB * NB],
    fft: Rfft,
    // state
    analysis: [f32; FRAME],
    ceps: [[f32; NB]; CEPS_MEM],
    memid: usize,
    synth: [f32; FRAME],
    pbuf: Vec<f32>,
    last_gain: f32,
    last_period: usize,
    hp: [f32; 2],
    lastg: [f32; NB],
    h_vad: Vec<f32>,
    h_noise: Vec<f32>,
    h_den: Vec<f32>,
    // scratch
    buf: Vec<f32>,
    xs: Vec<f32>,  // X [481][2]
    ps: Vec<f32>,  // P
    pl: Vec<f32>,  // decimated pitch buffer
    gx: Vec<f32>,
    gh: Vec<f32>,
    cat: Vec<f32>,
    dout: Vec<f32>,
    xcorr: Vec<f32>,
    yyl: Vec<f32>,
    pub vad: f32,
    // stage A -> B hand-over (band energies) and the set_stages part count
    ex: [f32; NB],
    stages: usize,
}

fn band_energy(x: &[f32], out: &mut [f32; NB]) {
    *out = [0.0; NB];
    for i in 0..NB - 1 {
        let (a, n) = (EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2);
        for j in 0..n {
            let fr = j as f32 / n as f32;
            let k = a + j;
            let t = x[2 * k] * x[2 * k] + x[2 * k + 1] * x[2 * k + 1];
            out[i] += (1.0 - fr) * t;
            out[i + 1] += fr * t;
        }
    }
    out[0] *= 2.0;
    out[NB - 1] *= 2.0;
}

fn band_corr(x: &[f32], p: &[f32], out: &mut [f32; NB]) {
    *out = [0.0; NB];
    for i in 0..NB - 1 {
        let (a, n) = (EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2);
        for j in 0..n {
            let fr = j as f32 / n as f32;
            let k = a + j;
            let t = x[2 * k] * p[2 * k] + x[2 * k + 1] * p[2 * k + 1];
            out[i] += (1.0 - fr) * t;
            out[i + 1] += fr * t;
        }
    }
    out[0] *= 2.0;
    out[NB - 1] *= 2.0;
}

/// per-bin gain from band values (bins >= 400 stay 0, as in the reference algorithm)
fn interp_gain(b: &[f32; NB], g: &mut [f32; FREQ]) {
    *g = [0.0; FREQ];
    for i in 0..NB - 1 {
        let (a, n) = (EBAND[i] << 2, (EBAND[i + 1] - EBAND[i]) << 2);
        for j in 0..n {
            let fr = j as f32 / n as f32;
            g[a + j] = (1.0 - fr) * b[i] + fr * b[i + 1];
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len().min(b.len()) {
        s += a[i] * b[i];
    }
    s
}

fn find_best_pitch(xc: &[f32], y: &[f32], len: usize, maxp: usize) -> [usize; 2] {
    let mut bn = [-1f32, -1f32];
    let mut bd = [0f32, 0f32];
    let mut bp = [0usize, 1usize];
    let mut syy = 1f32;
    for j in 0..len {
        syy += y[j] * y[j];
    }
    for i in 0..maxp {
        if xc[i] > 0.0 {
            let x16 = xc[i] * 1e-12;
            let num = x16 * x16;
            if num * bd[1] > bn[1] * syy {
                if num * bd[0] > bn[0] * syy {
                    bn[1] = bn[0];
                    bd[1] = bd[0];
                    bp[1] = bp[0];
                    bn[0] = num;
                    bd[0] = syy;
                    bp[0] = i;
                } else {
                    bn[1] = num;
                    bd[1] = syy;
                    bp[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
    bp
}

fn pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

impl RnnNoise {
    pub fn new(cfg: &Value, weights: &[u8]) -> Result<RnnNoise, String> {
        if weights.len() % 4 != 0 {
            return Err("rnnoise: weights length".into());
        }
        let wf: Vec<f32> = weights.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let names = ["input_dense", "vad_gru", "noise_gru", "denoise_gru", "denoise_output", "vad_output"];
        let ls = match cfg.get("layers") {
            Some(Value::Array(a)) => a,
            _ => return Err("rnnoise: config.layers missing".into()),
        };
        let mut layers = Vec::new();
        for (k, nm) in names.iter().enumerate() {
            let l = ls.iter().find(|l| l.get("name").and_then(|v| v.as_str()) == Some(nm)).ok_or(format!("rnnoise: layer {nm} missing"))?;
            let ni = l.num_or("inputs", 0.0) as usize;
            let n = l.num_or("neurons", 0.0) as usize;
            let act = match l.get("activation").and_then(|v| v.as_str()).unwrap_or("") {
                "tanh" => Act::Tanh,
                "sigmoid" => Act::Sigmoid,
                "relu" => Act::Relu,
                o => return Err(format!("rnnoise: activation {o}")),
            };
            let t = l.get("tensors").ok_or("rnnoise: tensors")?;
            let get = |kind: &str, size: usize| -> Result<Vec<f32>, String> {
                let e = t.get(kind).ok_or(format!("rnnoise: {nm}.{kind} missing"))?;
                let off = e.num_or("offset", 0.0) as usize;
                if e.num_or("size", 0.0) as usize != size || off + size > wf.len() {
                    return Err(format!("rnnoise: {nm}.{kind} size"));
                }
                Ok(wf[off..off + size].to_vec())
            };
            let gru = k >= 1 && k <= 3;
            let g = if gru { 3 } else { 1 };
            layers.push(Layer {
                w: get("weights", ni * n * g)?,
                r: if gru { get("recurrent_weights", n * n * 3)? } else { vec![] },
                b: get("bias", n * g)?,
                ni,
                n,
                act,
            });
        }
        let mut window = [0f32; WIN];
        for i in 0..FRAME {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / FRAME as f64).sin();
            let v = (0.5 * std::f64::consts::PI * s * s).sin() as f32;
            window[i] = v;
            window[WIN - 1 - i] = v;
        }
        let mut dct_t = [0f32; NB * NB];
        for i in 0..NB {
            for j in 0..NB {
                let mut v = ((i as f64 + 0.5) * j as f64 * std::f64::consts::PI / NB as f64).cos();
                if j == 0 {
                    v *= 0.5f64.sqrt();
                }
                dct_t[i * NB + j] = v as f32;
            }
        }
        let (nv, nn, nd) = (layers[1].n, layers[2].n, layers[3].n);
        if layers[2].ni != layers[0].n + nv + NFEAT || layers[3].ni != nv + nn + NFEAT || layers[0].ni != NFEAT || layers[4].n != NB {
            return Err("rnnoise: layer sizes inconsistent with the feature layout".into());
        }
        let mx = 3 * nd.max(nn).max(nv);
        Ok(RnnNoise {
            scale: cfg.num_or("input_scale", 32768.0) as f32,
            layers,
            window,
            dct_t,
            fft: Rfft::new(WIN),
            analysis: [0.0; FRAME],
            ceps: [[0.0; NB]; CEPS_MEM],
            memid: 0,
            synth: [0.0; FRAME],
            pbuf: vec![0.0; PBUF],
            last_gain: 0.0,
            last_period: 0,
            hp: [0.0; 2],
            lastg: [0.0; NB],
            h_vad: vec![0.0; nv],
            h_noise: vec![0.0; nn],
            h_den: vec![0.0; nd],
            buf: vec![0.0; WIN],
            xs: vec![0.0; 2 * FREQ],
            ps: vec![0.0; 2 * FREQ],
            pl: vec![0.0; PBUF / 2],
            gx: vec![0.0; mx],
            gh: vec![0.0; mx],
            cat: vec![0.0; 256],
            dout: vec![0.0; 128],
            xcorr: vec![0.0; PMAX],
            yyl: vec![0.0; PMAX / 2 + 1],
            vad: 0.0,
            ex: [0.0; NB],
            stages: 1,
        })
    }

    fn dct(&self, x: &[f32; NB], out: &mut [f32; NB]) {
        let s = (2.0f64 / 22.0).sqrt() as f32;
        for i in 0..NB {
            let mut sum = 0f32;
            for j in 0..NB {
                sum += x[j] * self.dct_t[j * NB + i];
            }
            out[i] = sum * s;
        }
    }

    /// windowed 960-pt forward transform scaled 1/960 -> out [481][2]
    fn fwd(&mut self, src: &[f32], which_p: bool) {
        for i in 0..WIN {
            self.buf[i] = src[i] * self.window[i];
        }
        let dst = if which_p { &mut self.ps } else { &mut self.xs };
        self.fft.forward(&self.buf, dst);
        let s = 1.0 / WIN as f32;
        for v in dst.iter_mut() {
            *v *= s;
        }
    }

    fn pitch_downsample(&mut self) {
        let x = &self.pbuf;
        let n = PBUF >> 1;
        let xl = &mut self.pl;
        for i in 1..n {
            xl[i] = 0.5 * (0.5 * (x[2 * i - 1] + x[2 * i + 1]) + x[2 * i]);
        }
        xl[0] = 0.5 * (0.5 * x[1] + x[0]);
        let mut ac = [0f32; 5];
        for k in 0..5 {
            ac[k] = dot(&xl[..n - k], &xl[k..n]);
        }
        ac[0] *= 1.0001;
        for i in 1..5 {
            let f = 0.008 * i as f32;
            ac[i] -= ac[i] * f * f;
        }
        // Levinson-Durbin, order 4, 30 dB early exit
        let mut a = [0f32; 4];
        let mut err = ac[0];
        if ac[0] != 0.0 {
            for i in 0..4 {
                let mut rr = 0f32;
                for j in 0..i {
                    rr += a[j] * ac[i - j];
                }
                rr += ac[i + 1];
                let r = -rr / err;
                a[i] = r;
                for j in 0..(i + 1) >> 1 {
                    let (t1, t2) = (a[j], a[i - 1 - j]);
                    a[j] = t1 + r * t2;
                    a[i - 1 - j] = t2 + r * t1;
                }
                err -= r * r * err;
                if err < 0.001 * ac[0] {
                    break;
                }
            }
        }
        let mut tmp = 1f32;
        for v in a.iter_mut() {
            tmp *= 0.9;
            *v *= tmp;
        }
        let c1 = 0.8f32;
        let l2 = [a[0] + 0.8, a[1] + c1 * a[0], a[2] + c1 * a[1], a[3] + c1 * a[2], c1 * a[3]];
        let mut mem = [0f32; 5];
        for i in 0..n {
            let xi = xl[i];
            let s = xi + l2[0] * mem[0] + l2[1] * mem[1] + l2[2] * mem[2] + l2[3] * mem[3] + l2[4] * mem[4];
            mem[4] = mem[3];
            mem[3] = mem[2];
            mem[2] = mem[1];
            mem[1] = mem[0];
            mem[0] = xi;
            xl[i] = s;
        }
    }

    fn pitch_search(&mut self) -> usize {
        let (len, maxp) = (PFRAME, PMAX - 3 * PMIN);
        let y = &self.pl;
        let xl = &self.pl[PMAX >> 1..];
        let lag = len + maxp;
        let mut x4 = [0f32; PFRAME >> 2];
        for j in 0..len >> 2 {
            x4[j] = xl[2 * j];
        }
        let mut y4 = [0f32; (PFRAME + PMAX) >> 2];
        for j in 0..lag >> 2 {
            y4[j] = y[2 * j];
        }
        let xc = &mut self.xcorr;
        for i in 0..maxp >> 2 {
            xc[i] = dot(&x4[..len >> 2], &y4[i..i + (len >> 2)]);
        }
        let bp = find_best_pitch(xc, &y4, len >> 2, maxp >> 2);
        for i in 0..maxp >> 1 {
            xc[i] = 0.0;
            let d0 = (i as isize - 2 * bp[0] as isize).abs();
            let d1 = (i as isize - 2 * bp[1] as isize).abs();
            if d0 > 2 && d1 > 2 {
                continue;
            }
            xc[i] = dot(&xl[..len >> 1], &y[i..i + (len >> 1)]).max(-1.0);
        }
        let bp = find_best_pitch(xc, y, len >> 1, maxp >> 1);
        let mut off = 0isize;
        if bp[0] > 0 && bp[0] < (maxp >> 1) - 1 {
            let (a, b, c) = (xc[bp[0] - 1], xc[bp[0]], xc[bp[0] + 1]);
            if (c - a) > 0.7 * (b - a) {
                off = 1;
            } else if (a - c) > 0.7 * (b - c) {
                off = -1;
            }
        }
        (2 * bp[0] as isize - off) as usize
    }

    fn remove_doubling(&mut self, t0_in: usize) -> (usize, f32) {
        let x = &self.pl;
        let minp0 = PMIN;
        let maxp = PMAX / 2;
        let minp = PMIN / 2;
        let mut t0 = t0_in / 2;
        let prevp = self.last_period / 2;
        let n = PFRAME / 2;
        let b = maxp;
        if t0 >= maxp {
            t0 = maxp - 1;
        }
        let mut t = t0;
        let xb = &x[b..b + n];
        let xx = dot(xb, xb);
        let mut xy = dot(xb, &x[b - t0..b - t0 + n]);
        let yyl = &mut self.yyl;
        yyl[0] = xx;
        let mut yy = xx;
        for i in 1..=maxp {
            yy = yy + x[b - i] * x[b - i] - x[b + n - i] * x[b + n - i];
            yyl[i] = yy.max(0.0);
        }
        yy = yyl[t0];
        let (mut bxy, mut byy) = (xy, yy);
        let g0 = pitch_gain(xy, xx, yy);
        let mut g = g0;
        for k in 2..=15usize {
            let t1 = (2 * t0 + k) / (2 * k);
            if t1 < minp {
                break;
            }
            let t1b = if k == 2 { if t1 + t0 > maxp { t0 } else { t0 + t1 } } else { (2 * SECOND[k] * t0 + k) / (2 * k) };
            xy = dot(xb, &x[b - t1..b - t1 + n]);
            let xy2 = dot(xb, &x[b - t1b..b - t1b + n]);
            xy = 0.5 * (xy + xy2);
            yy = 0.5 * (yyl[t1] + yyl[t1b]);
            let g1 = pitch_gain(xy, xx, yy);
            let dp = (t1 as isize - prevp as isize).abs();
            let cont = if dp <= 1 {
                self.last_gain
            } else if dp <= 2 && 5 * k * k < t0 {
                0.5 * self.last_gain
            } else {
                0.0
            };
            let mut thresh = (0.7 * g0 - cont).max(0.3);
            if t1 < 3 * minp {
                thresh = (0.85 * g0 - cont).max(0.4);
            } else if t1 < 2 * minp {
                thresh = (0.9 * g0 - cont).max(0.5);
            }
            if g1 > thresh {
                bxy = xy;
                byy = yy;
                t = t1;
                g = g1;
            }
        }
        bxy = bxy.max(0.0);
        let mut pg = if byy <= bxy { 1.0 } else { bxy / (byy + 1.0) };
        let mut xc = [0f32; 3];
        for k in 0..3 {
            let lag = t + k - 1;
            xc[k] = dot(xb, &x[b - lag..b - lag + n]);
        }
        let off: isize = if (xc[2] - xc[0]) > 0.7 * (xc[1] - xc[0]) {
            1
        } else if (xc[0] - xc[2]) > 0.7 * (xc[1] - xc[2]) {
            -1
        } else {
            0
        };
        if pg > g {
            pg = g;
        }
        let t0out = ((2 * t) as isize + off).max(minp0 as isize) as usize;
        (t0out, pg)
    }

    fn frame(&mut self, inp: &[f32], out: &mut [f32]) {
        self.frame_a(inp);
        self.frame_b(out);
    }

    /// stage A: high-pass, analysis, band energies, pitch search
    fn frame_a(&mut self, inp: &[f32]) {
        // high-pass biquad b = [-2, 1], a = [-1.99599, 0.99600] (products in f64, state f32)
        let mut x = [0f32; FRAME];
        let (mut m0, mut m1) = (self.hp[0], self.hp[1]);
        for i in 0..FRAME {
            let xi = inp[i] * self.scale;
            let yi = xi + m0;
            m0 = (m1 as f64 + (-2.0 * xi as f64 - (-1.99599f32) as f64 * yi as f64)) as f32;
            m1 = (1.0 * xi as f64 - 0.99600f32 as f64 * yi as f64) as f32;
            x[i] = yi;
        }
        self.hp = [m0, m1];
        // analysis
        let mut win = [0f32; WIN];
        win[..FRAME].copy_from_slice(&self.analysis);
        win[FRAME..].copy_from_slice(&x);
        self.analysis = x;
        self.fwd(&win, false);
        let mut ex = [0f32; NB];
        band_energy(&self.xs, &mut ex);
        self.pbuf.copy_within(FRAME.., 0);
        self.pbuf[PBUF - FRAME..].copy_from_slice(&x);
        self.pitch_downsample();
        let pi = PMAX - self.pitch_search();
        let (pi, g) = self.remove_doubling(pi);
        self.last_period = pi;
        self.last_gain = g;
        self.ex = ex;
    }

    /// stage B: pitch features, network, gains, synthesis (uses the stage A results in self)
    fn frame_b(&mut self, out: &mut [f32]) {
        let (pi, ex) = (self.last_period, self.ex);
        let mut p = [0f32; WIN];
        p.copy_from_slice(&self.pbuf[PBUF - WIN - pi..PBUF - pi]);
        self.fwd(&p, true);
        let mut ep = [0f32; NB];
        band_energy(&self.ps, &mut ep);
        let mut exp = [0f32; NB];
        band_corr(&self.xs, &self.ps, &mut exp);
        for i in 0..NB {
            exp[i] /= (0.001 + ex[i] * ep[i]).sqrt();
        }
        let mut feat = [0f32; NFEAT];
        let mut t = [0f32; NB];
        self.dct(&exp, &mut t);
        feat[NB + 2 * NDC..NB + 3 * NDC].copy_from_slice(&t[..NDC]);
        feat[NB + 2 * NDC] -= 1.3;
        feat[NB + 2 * NDC + 1] -= 0.9;
        feat[NB + 3 * NDC] = 0.01 * (pi as f32 - 300.0);
        let mut ly = [0f32; NB];
        let (mut logmax, mut follow, mut e) = (-2f32, -2f32, 0f32);
        for i in 0..NB {
            let v = (1e-2 + ex[i]).log10();
            let v = (logmax - 7.0).max((follow - 1.5).max(v));
            ly[i] = v;
            logmax = logmax.max(v);
            follow = (follow - 1.5).max(v);
            e += ex[i];
        }
        if e >= 0.04 {
            let mut c = [0f32; NB];
            self.dct(&ly, &mut c);
            c[0] -= 12.0;
            c[1] -= 4.0;
            self.ceps[self.memid] = c;
            let c1 = self.ceps[(self.memid + CEPS_MEM - 1) % CEPS_MEM];
            let c2 = self.ceps[(self.memid + CEPS_MEM - 2) % CEPS_MEM];
            feat[..NB].copy_from_slice(&c);
            for i in 0..NDC {
                feat[i] = c[i] + c1[i] + c2[i];
                feat[NB + i] = c[i] - c2[i];
                feat[NB + NDC + i] = c[i] - 2.0 * c1[i] + c2[i];
            }
            self.memid = (self.memid + 1) % CEPS_MEM;
            let mut sv = 0f32;
            for i in 0..CEPS_MEM {
                let mut mind = 1e15f32;
                for j in 0..CEPS_MEM {
                    if j == i {
                        continue;
                    }
                    let mut d = 0f32;
                    for k in 0..NB {
                        let q = self.ceps[i][k] - self.ceps[j][k];
                        d += q * q;
                    }
                    mind = mind.min(d);
                }
                sv += mind;
            }
            feat[NB + 3 * NDC + 1] = sv / CEPS_MEM as f32 - 2.1;
            // network
            let mut gains = [0f32; NB];
            {
                let (l, gx, gh, cat, dout) = (&self.layers, &mut self.gx, &mut self.gh, &mut self.cat, &mut self.dout);
                l[0].dense(&feat, dout);
                let nd0 = l[0].n;
                l[1].gru(&dout[..nd0], &mut self.h_vad, gx, gh);
                let mut vad = [0f32; 1];
                l[5].dense(&self.h_vad, &mut vad);
                self.vad = vad[0];
                let nv = self.h_vad.len();
                cat[..nd0].copy_from_slice(&dout[..nd0]);
                cat[nd0..nd0 + nv].copy_from_slice(&self.h_vad);
                cat[nd0 + nv..nd0 + nv + NFEAT].copy_from_slice(&feat);
                l[2].gru(cat, &mut self.h_noise, gx, gh);
                let nn = self.h_noise.len();
                cat[..nv].copy_from_slice(&self.h_vad);
                cat[nv..nv + nn].copy_from_slice(&self.h_noise);
                cat[nv + nn..nv + nn + NFEAT].copy_from_slice(&feat);
                l[3].gru(cat, &mut self.h_den, gx, gh);
                l[4].dense(&self.h_den, &mut gains);
            }
            // pitch filter
            let mut r = [0f32; NB];
            for i in 0..NB {
                let v = if exp[i] > gains[i] {
                    1.0
                } else {
                    exp[i] * exp[i] * (1.0 - gains[i] * gains[i]) / (0.001 + gains[i] * gains[i] * (1.0 - exp[i] * exp[i]))
                };
                r[i] = v.max(0.0).min(1.0).sqrt() * (ex[i] / (1e-8 + ep[i])).sqrt();
            }
            let mut rf = [0f32; FREQ];
            interp_gain(&r, &mut rf);
            for k in 0..FREQ {
                self.xs[2 * k] += rf[k] * self.ps[2 * k];
                self.xs[2 * k + 1] += rf[k] * self.ps[2 * k + 1];
            }
            let mut ne = [0f32; NB];
            band_energy(&self.xs, &mut ne);
            let mut nrm = [0f32; NB];
            for i in 0..NB {
                nrm[i] = (ex[i] / (1e-8 + ne[i])).sqrt();
            }
            let mut nf = [0f32; FREQ];
            interp_gain(&nrm, &mut nf);
            for i in 0..NB {
                gains[i] = gains[i].max(0.6 * self.lastg[i]);
            }
            self.lastg = gains;
            let mut gf = [0f32; FREQ];
            interp_gain(&gains, &mut gf);
            for k in 0..FREQ {
                let m = nf[k] * gf[k];
                self.xs[2 * k] *= m;
                self.xs[2 * k + 1] *= m;
            }
        }
        // synthesis: x = N * irfft(X), window, overlap-add
        let mut y = [0f32; WIN];
        self.fft.inverse(&self.xs, &mut y);
        for i in 0..WIN {
            y[i] *= WIN as f32 * self.window[i];
        }
        let inv = 1.0 / self.scale;
        for i in 0..FRAME {
            out[i] = (y[i] + self.synth[i]) * inv;
        }
        self.synth.copy_from_slice(&y[FRAME..]);
    }
}

impl NoiseEngine for RnnNoise {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "rnnoise:v0.1".into(), sample_rate: 48000, frame_size: FRAME, latency_samples: FRAME, in_channels: 1, out_channels: 1 }
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.frame(&input[..FRAME], &mut output[..FRAME]);
    }
    fn reset(&mut self) {
        self.analysis = [0.0; FRAME];
        self.ceps = [[0.0; NB]; CEPS_MEM];
        self.memid = 0;
        self.synth = [0.0; FRAME];
        self.pbuf.iter_mut().for_each(|v| *v = 0.0);
        self.last_gain = 0.0;
        self.last_period = 0;
        self.hp = [0.0; 2];
        self.lastg = [0.0; NB];
        for h in [&mut self.h_vad, &mut self.h_noise, &mut self.h_den] {
            h.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    fn frame_size(&self) -> usize {
        FRAME
    }
    // 2 parts: A = analysis + pitch search, B = features + network + synthesis
    fn set_stages(&mut self, n: usize) -> usize {
        self.stages = n.clamp(1, 2);
        self.stages
    }
    fn process_stage(&mut self, s: usize, input: &[f32], output: &mut [f32]) {
        match (self.stages, s) {
            (1, 0) => self.frame(&input[..FRAME], &mut output[..FRAME]),
            (_, 0) => self.frame_a(&input[..FRAME]),
            (_, 1) => self.frame_b(&mut output[..FRAME]),
            _ => {}
        }
    }
    fn speech_prob(&self) -> f32 {
        self.vad
    }
}

/// Plugin entry: config = manifest.config ({"layers": [...], "input_scale"}), weights = weights.bin.
pub fn create(cfg: &Value, weights: &[u8]) -> Result<Box<dyn NoiseEngine>, String> {
    Ok(Box::new(RnnNoise::new(cfg, weights)?))
}

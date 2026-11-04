//! Plugin `dfn`: DeepFilterNet3 (Schroeter et al., Interspeech 2023). The three published network
//! stages (encoder, ERB decoder, DF decoder) run as one streaming vi_graph graph
//! (tools/make_dfn.py); the DSP around it is ours (paper + published constants in manifest.config):
//!   * analysis: Vorbis window over [prev hop | hop], rFFT 960, x 1/960 (wnorm)
//!   * features: ERB band power (mean over band bins) -> 10 log10(p + 1e-10) -> exponential mean
//!     norm (alpha) -> /40; complex bins < nb_df -> exponential unit norm (|X| EMA, / sqrt(state))
//!   * stage gating by the predicted local SNR (< min: zero gains, > max_erb: bypass, > max_df:
//!     gains only, else gains + deep filtering)
//!   * ERB gains applied to the frame `conv_lookahead` hops back (rolling buffer of
//!     df_order + conv_lookahead frames); deep filtering: order-df_order complex FIR over the
//!     noisy frames t-(df_order-1)..t for bins < nb_df (df_lookahead future frames of the target)
//!   * synthesis: irFFT (unnormalized), window, overlap-add; silent input (mean square < 1e-7 for
//!     more than 5 hops) outputs zeros and freezes state.
//! Latency = hop * (1 + conv_lookahead) = 1440 samples (30 ms) at 48 kHz.

use vi_audio::fft::Rfft;
use vi_engine::json::Value;
#[cfg(not(target_arch = "wasm32"))]
use vi_engine::acc_add;
use vi_engine::{EngineInfo, NoiseEngine};
use vi_graph::graph::Graph;
use vi_kernels::Backend;

pub struct Dfn {
    g: Graph,
    t_erb: usize,
    t_spec: usize,
    t_m: usize,
    t_coefs: usize,
    t_lsnr: usize,
    n: usize,
    hop: usize,
    nf: usize,
    nb_erb: usize,
    nb_df: usize,
    order: usize,
    la: usize,
    erb: Vec<usize>,
    alpha: f32,
    min_db: f32,
    max_erb_db: f32,
    max_df_db: f32,
    mn0: Vec<f32>,
    un0: Vec<f32>,
    win: Vec<f32>,
    wnorm: f32,
    fft: Rfft,
    // state
    amem: Vec<f32>,
    smem: Vec<f32>,
    mn: Vec<f32>,
    un: Vec<f32>,
    ry: Vec<f32>, // ring [df_order + la][nf*2]
    rx: Vec<f32>, // ring [max(df_order, la)][nf*2]
    ny: usize,
    nx: usize,
    py: usize, // index of the oldest frame in ry
    px: usize,
    skip: usize,
    // scratch
    buf: Vec<f32>,
    spec: Vec<f32>,
    out: Vec<f32>,
    pub lsnr: f32,
    // set_stages parts; whether the frame in flight is processed (not skipped as silent)
    stages: usize,
    active: bool,
}

fn lin(a: f32, b: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| a + i as f32 * (b - a) / (n - 1) as f32).collect()
}

impl Dfn {
    pub fn new(cfg: &Value, weights: &[u8]) -> Result<Dfn, String> {
        let gj = cfg.get("graph").ok_or("dfn: config.graph missing")?;
        let g = Graph::load(gj, weights, 1, &["feat_erb", "feat_spec", "m", "coefs", "lsnr"])?;
        let u = |k: &str| cfg.num_or(k, 0.0) as usize;
        let f = |k: &str, d: f64| cfg.num_or(k, d) as f32;
        let (n, hop, nb_erb, nb_df, order, la) = (u("fft_size"), u("hop_size"), u("nb_erb"), u("nb_df"), u("df_order"), u("conv_lookahead"));
        if n != 2 * hop || nb_erb == 0 || nb_df == 0 || order == 0 {
            return Err("dfn: unsupported config (fft_size must be 2*hop_size)".into());
        }
        let erb: Vec<usize> = match cfg.get("erb_widths") {
            Some(Value::Array(a)) => a.iter().map(|v| v.as_f64().unwrap_or(0.0) as usize).collect(),
            _ => return Err("dfn: erb_widths missing".into()),
        };
        let nf = n / 2 + 1;
        if erb.len() != nb_erb || erb.iter().sum::<usize>() != nf {
            return Err("dfn: erb_widths inconsistent".into());
        }
        let id = |s: &str| g.id(s).ok_or(format!("dfn: graph tensor {s} missing"));
        let (t_erb, t_spec, t_m, t_coefs, t_lsnr) = (id("feat_erb")?, id("feat_spec")?, id("m")?, id("coefs")?, id("lsnr")?);
        if g.tensors[t_erb].data.len() != nb_erb || g.tensors[t_spec].data.len() != 2 * nb_df || g.tensors[t_m].data.len() != nb_erb || g.tensors[t_coefs].data.len() != nb_df * order * 2 {
            return Err("dfn: graph I/O shapes do not match the config".into());
        }
        let pair = |k: &str, d: [f32; 2]| -> [f32; 2] {
            match cfg.get(k) {
                Some(Value::Array(a)) if a.len() == 2 => [a[0].as_f64().unwrap_or(d[0] as f64) as f32, a[1].as_f64().unwrap_or(d[1] as f64) as f32],
                _ => d,
            }
        };
        let mi = pair("mean_norm_init", [-60.0, -90.0]);
        let ui = pair("unit_norm_init", [0.001, 0.0001]);
        let h = n / 2;
        let win: Vec<f32> = (0..n)
            .map(|i| {
                let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / h as f64).sin();
                (0.5 * std::f64::consts::PI * s * s).sin() as f32
            })
            .collect();
        let ny = order + la;
        let nx = order.max(la);
        let mut d = Dfn {
            g,
            t_erb,
            t_spec,
            t_m,
            t_coefs,
            t_lsnr,
            n,
            hop,
            nf,
            nb_erb,
            nb_df,
            order,
            la,
            erb,
            alpha: f("norm_alpha", 0.99),
            min_db: f("min_db_thresh", -10.0),
            max_erb_db: f("max_db_erb_thresh", 30.0),
            max_df_db: f("max_db_df_thresh", 20.0),
            mn0: lin(mi[0], mi[1], nb_erb),
            un0: lin(ui[0], ui[1], nb_df),
            win,
            wnorm: 1.0 / ((n * n) as f32 / (2 * hop) as f32),
            fft: Rfft::new(n),
            amem: vec![0.0; n - hop],
            smem: vec![0.0; n - hop],
            mn: vec![],
            un: vec![],
            ry: vec![0.0; ny * nf * 2],
            rx: vec![0.0; nx * nf * 2],
            ny,
            nx,
            py: 0,
            px: 0,
            skip: 0,
            buf: vec![0.0; n],
            spec: vec![0.0; nf * 2],
            out: vec![0.0; nf * 2],
            lsnr: 0.0,
            stages: 1,
            active: false,
        };
        d.reset();
        Ok(d)
    }

    /// analysis + features into the graph inputs; false = silent input, the frame outputs zeros
    fn pre(&mut self, x: &[f32]) -> bool {
        let (n, hop, nf) = (self.n, self.hop, self.nf);
        let ms = x.iter().map(|v| v * v).sum::<f32>() / hop as f32;
        if ms < 1e-7 {
            self.skip += 1;
        } else {
            self.skip = 0;
        }
        if self.skip > 5 {
            return false;
        }
        // analysis
        let h0 = n - hop;
        for i in 0..h0 {
            self.buf[i] = self.amem[i] * self.win[i];
        }
        for i in 0..hop {
            self.buf[h0 + i] = x[i] * self.win[h0 + i];
        }
        self.amem.copy_within(hop.., 0);
        let l = self.amem.len();
        self.amem[l - hop..].copy_from_slice(&x[..hop]);
        self.fft.forward(&self.buf, &mut self.spec);
        for v in self.spec.iter_mut() {
            *v *= self.wnorm;
        }
        // push into rolling buffers (overwrite oldest)
        let fl = nf * 2;
        self.ry[self.py * fl..(self.py + 1) * fl].copy_from_slice(&self.spec);
        self.py = (self.py + 1) % self.ny;
        self.rx[self.px * fl..(self.px + 1) * fl].copy_from_slice(&self.spec);
        self.px = (self.px + 1) % self.nx;
        // features
        let a = self.alpha;
        {
            let fe = &mut self.g.tensors[self.t_erb].data;
            let mut k = 0;
            for (b, &w) in self.erb.iter().enumerate() {
                let inv = 1.0 / w as f32;
                let mut s = 0f32;
                for j in k..k + w {
                    s += (self.spec[2 * j] * self.spec[2 * j] + self.spec[2 * j + 1] * self.spec[2 * j + 1]) * inv;
                }
                k += w;
                let e = (s + 1e-10).log10() * 10.0;
                self.mn[b] = e * (1.0 - a) + self.mn[b] * a;
                fe[b] = (e - self.mn[b]) / 40.0;
            }
        }
        {
            let nd = self.nb_df;
            let fs = &mut self.g.tensors[self.t_spec].data;
            for j in 0..nd {
                let (re, im) = (self.spec[2 * j], self.spec[2 * j + 1]);
                let mag = (re * re + im * im).sqrt();
                self.un[j] = mag * (1.0 - a) + self.un[j] * a;
                let sq = self.un[j].sqrt();
                fs[j] = re / sq;
                fs[nd + j] = im / sq;
            }
        }
        true
    }

    fn frame(&mut self, x: &[f32], y: &mut [f32], acc: Option<&mut Vec<(String, f64)>>) {
        if !self.pre(x) {
            y[..self.hop].iter_mut().for_each(|v| *v = 0.0);
            return;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = acc;
        #[cfg(target_arch = "wasm32")]
        self.g.run();
        #[cfg(not(target_arch = "wasm32"))]
        match acc {
            Some(acc) => {
                let nn = self.g.n_nodes();
                let mut per = vec![0f64; nn];
                self.g.run_profiled(&mut per);
                for i in 0..nn {
                    let k = self.g.op_names[i].clone();
                    acc_add(acc, &k, per[i]);
                }
            }
            None => self.g.run(),
        }
        self.post(y);
    }

    /// stage gating, ERB gains, deep filtering, synthesis (graph outputs -> y)
    fn post(&mut self, y: &mut [f32]) {
        let (n, hop, fl) = (self.n, self.hop, self.nf * 2);
        self.lsnr = self.g.tensors[self.t_lsnr].data[0];
        let (gains, df) = if self.lsnr < self.min_db {
            (Some(false), false) // zero gains
        } else if self.lsnr > self.max_erb_db {
            (None, false)
        } else if self.lsnr > self.max_df_db {
            (Some(true), false)
        } else {
            (Some(true), true)
        };
        // target frame = ry[order-1] counted from the oldest
        let ti = (self.py + self.order - 1) % self.ny;
        if let Some(use_m) = gains {
            let m = &self.g.tensors[self.t_m].data;
            let fr = &mut self.ry[ti * fl..(ti + 1) * fl];
            let mut k = 0;
            for (b, &w) in self.erb.iter().enumerate() {
                let gb = if use_m { m[b] } else { 0.0 };
                for j in k..k + w {
                    fr[2 * j] *= gb;
                    fr[2 * j + 1] *= gb;
                }
                k += w;
            }
        }
        self.out.copy_from_slice(&self.ry[ti * fl..(ti + 1) * fl]);
        if df {
            let nd = self.nb_df;
            let c = &self.g.tensors[self.t_coefs].data; // [nd][order][2]
            for j in 0..nd {
                let (mut sr, mut si) = (0f32, 0f32);
                for k in 0..self.order {
                    let fi = (self.px + self.nx - self.order + k) % self.nx; // k-th of the last `order` frames
                    let (xr, xi) = (self.rx[fi * fl + 2 * j], self.rx[fi * fl + 2 * j + 1]);
                    let (cr, ci) = (c[(j * self.order + k) * 2], c[(j * self.order + k) * 2 + 1]);
                    sr += xr * cr - xi * ci;
                    si += xr * ci + xi * cr;
                }
                self.out[2 * j] = sr;
                self.out[2 * j + 1] = si;
            }
        }
        // synthesis
        self.fft.inverse(&self.out, &mut self.buf);
        let sc = n as f32;
        for i in 0..n {
            self.buf[i] *= sc * self.win[i];
        }
        for i in 0..hop {
            y[i] = self.buf[i] + self.smem[i];
        }
        self.smem.copy_from_slice(&self.buf[hop..]);
    }
}

impl NoiseEngine for Dfn {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "dfn:deepfilternet3".into(), sample_rate: 48000, frame_size: self.hop, latency_samples: self.hop * (1 + self.la), in_channels: 1, out_channels: 1 }
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.frame(input, output, None);
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn profile_frame(&mut self, input: &[f32], output: &mut [f32], acc: &mut Vec<(String, f64)>) {
        self.frame(input, output, Some(acc));
    }
    fn reset(&mut self) {
        self.g.reset();
        self.amem.iter_mut().for_each(|v| *v = 0.0);
        self.smem.iter_mut().for_each(|v| *v = 0.0);
        self.ry.iter_mut().for_each(|v| *v = 0.0);
        self.rx.iter_mut().for_each(|v| *v = 0.0);
        self.py = 0;
        self.px = 0;
        self.skip = 0;
        self.mn = self.mn0.clone();
        self.un = self.un0.clone();
    }
    fn frame_size(&self) -> usize {
        self.hop
    }
    fn set_scalar(&mut self, scalar: bool) {
        self.g.backend = if scalar { Backend::Scalar } else { Backend::detect() };
    }
    fn backend_name(&self) -> &'static str {
        self.g.backend.name()
    }
    fn set_stages(&mut self, n: usize) -> usize {
        self.stages = self.g.set_stages(n.max(1));
        self.stages
    }
    // DSP front in the first part, DSP back in the last, the graph's parts in between
    fn process_stage(&mut self, s: usize, input: &[f32], output: &mut [f32]) {
        if s == 0 {
            self.active = self.pre(input);
        }
        if self.active {
            self.g.run_stage(s);
        }
        if s + 1 == self.stages {
            if self.active {
                self.post(output);
            } else {
                output[..self.hop].iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
}

/// Plugin entry: config = manifest.config + "graph" (graph.json), weights = weights.bin.
pub fn create(cfg: &Value, weights: &[u8]) -> Result<Box<dyn NoiseEngine>, String> {
    Ok(Box::new(Dfn::new(cfg, weights)?))
}

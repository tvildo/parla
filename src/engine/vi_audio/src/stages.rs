//! Front-end / back-end stages of the model pipeline (models/FORMAT.md "pipeline"):
//!   FrontEnd: time | frames (+window) | stft (+window, features, optional filterbank)
//!   BackEnd:  waveform | ola (+synthesis window) | istft from complex spectrum / magnitude mask /
//!             complex mask (noisy STFT, mask delay), optional WOLA normalization.
//! Shared by every plugin (generic graph packages and custom-DSP plugins). Allocation-free after
//! construction; arithmetic identical to the former vi_rs pipeline.

use crate::fft::{window, Rfft};
use crate::json::Value;

/// Lookup of named constant tensors (windows / filterbanks stored in a model package).
pub type TensorLookup<'a> = &'a dyn Fn(&str) -> Option<(Vec<usize>, Vec<f32>)>;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum FrontKind {
    Time,
    Frames,
    Stft,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Feat {
    Complex,
    Mag,
    Pow,
    LogPow,
    Log10Pow,
    LogMag,
    Log10Mag,
    Log1pMag,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum BackKind {
    /// analysis models (scorers): the graph output is copied to the start of the output frame,
    /// rest zero (e.g. DNSMOS: 3 scores per 144160-sample frame)
    Raw,
    Waveform,
    Ola,
    Spec,
    MagMask,
    CplxMask,
}

/// Window from a spec value: name | {"values": [...]} | {"tensor": name}.
pub fn window_spec(v: Option<&Value>, n: usize, lookup: TensorLookup) -> Result<Vec<f32>, String> {
    let w = match v {
        None | Some(Value::Null) => vec![1.0; n],
        Some(Value::Str(s)) => window(s, n)?,
        Some(o @ Value::Object(_)) => {
            if let Some(Value::Array(a)) = o.get("values") {
                a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()
            } else {
                let t = o.get("tensor").and_then(|v| v.as_str()).ok_or("window: tensor or values")?;
                lookup(t).ok_or(format!("window tensor {t} not found"))?.1
            }
        }
        _ => return Err("bad window spec".into()),
    };
    if w.len() != n {
        return Err(format!("window length {} != {n}", w.len()));
    }
    Ok(w)
}

pub struct FrontEnd {
    pub kind: FrontKind,
    pub feat: Feat,
    pub hop: usize,
    pub t: usize,
    pub flen: usize,
    pub nfft: usize,
    log_eps: f32,
    log_max: bool,
    wa: Vec<f32>,
    fb: Option<(Vec<f32>, usize, usize)>, // [bins][bands]
    hist: Vec<f32>,
    fbuf: Vec<f32>,
    /// noisy spectrum of the current call, [T][F][2] (stft front-ends)
    pub spec: Vec<f32>,
    tmp: Vec<f32>,
    rfft: Option<Rfft>,
}

impl FrontEnd {
    pub fn new(fr: &Value, hop: usize, t: usize, lookup: TensorLookup) -> Result<FrontEnd, String> {
        let kind = match fr.get("type").and_then(|v| v.as_str()).unwrap_or("time") {
            "time" => FrontKind::Time,
            "frames" => FrontKind::Frames,
            "stft" => FrontKind::Stft,
            o => return Err(format!("unknown front-end {o}")),
        };
        let nfft = fr.num_or("n_fft", fr.num_or("frame_len", hop as f64)) as usize;
        let flen = if kind == FrontKind::Time { hop } else { fr.num_or("frame_len", nfft as f64) as usize };
        if flen > nfft || flen < hop {
            return Err("frame_len must satisfy hop <= frame_len <= n_fft".into());
        }
        let wa = if kind == FrontKind::Time { vec![] } else { window_spec(fr.get("window"), flen, lookup)? };
        let feat = match fr.get("feature").and_then(|v| v.as_str()).unwrap_or("complex") {
            "complex" => Feat::Complex,
            "magnitude" => Feat::Mag,
            "power" => Feat::Pow,
            "log_power" => Feat::LogPow,
            "log10_power" => Feat::Log10Pow,
            "log_magnitude" => Feat::LogMag,
            "log10_magnitude" => Feat::Log10Mag,
            "log1p_magnitude" => Feat::Log1pMag,
            o => return Err(format!("unknown feature {o}")),
        };
        let nb = nfft / 2 + 1;
        let fb = match fr.get("filterbank").and_then(|v| v.as_str()) {
            Some(n) => {
                let (sh, d) = lookup(n).ok_or("filterbank tensor not found")?;
                if sh.len() != 2 || sh[0] != nb {
                    return Err(format!("filterbank must be [{nb}][bands], got {sh:?}"));
                }
                Some((d, sh[0], sh[1]))
            }
            None => None,
        };
        Ok(FrontEnd {
            kind,
            feat,
            hop,
            t,
            flen,
            nfft,
            log_eps: fr.num_or("log_eps", 1e-12) as f32,
            log_max: fr.get("log_floor").and_then(|v| v.as_str()) == Some("max"),
            wa,
            fb,
            hist: vec![0.0; flen],
            fbuf: vec![0.0; nfft],
            spec: vec![0.0; t * nb * 2],
            tmp: vec![0.0; nb],
            rfft: if kind == FrontKind::Stft { Some(Rfft::new(nfft)) } else { None },
        })
    }

    pub fn bins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// values written per call
    pub fn out_len(&self) -> usize {
        match self.kind {
            FrontKind::Time => self.t * self.hop,
            FrontKind::Frames => self.t * self.flen,
            FrontKind::Stft => {
                if self.feat == Feat::Complex {
                    self.t * self.bins() * 2
                } else {
                    self.t * self.fb.as_ref().map_or(self.bins(), |f| f.2)
                }
            }
        }
    }

    /// algorithmic delay of the framing (samples)
    pub fn latency(&self) -> usize {
        if self.kind == FrontKind::Time { 0 } else { self.flen - self.hop }
    }

    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|x| *x = 0.0);
    }

    /// x: T*hop new samples -> dst: out_len() values
    pub fn process(&mut self, x: &[f32], dst: &mut [f32]) {
        let (t, hop, flen) = (self.t, self.hop, self.flen);
        let nb = self.bins();
        if self.kind == FrontKind::Time {
            dst[..t * hop].copy_from_slice(&x[..t * hop]);
            return;
        }
        for f in 0..t {
            self.hist.copy_within(hop.., 0);
            self.hist[flen - hop..].copy_from_slice(&x[f * hop..(f + 1) * hop]);
            if self.kind == FrontKind::Frames {
                let d = &mut dst[f * flen..(f + 1) * flen];
                for i in 0..flen {
                    d[i] = self.hist[i] * self.wa[i];
                }
                continue;
            }
            for i in 0..flen {
                self.fbuf[i] = self.hist[i] * self.wa[i];
            }
            for v in &mut self.fbuf[flen..] {
                *v = 0.0;
            }
            let sp = &mut self.spec[f * nb * 2..(f + 1) * nb * 2];
            self.rfft.as_mut().unwrap().forward(&self.fbuf, sp);
            if self.feat == Feat::Complex {
                dst[f * nb * 2..(f + 1) * nb * 2].copy_from_slice(sp);
                continue;
            }
            let use_mag = matches!(self.feat, Feat::Mag | Feat::LogMag | Feat::Log10Mag | Feat::Log1pMag);
            for k in 0..nb {
                let p = sp[2 * k] * sp[2 * k] + sp[2 * k + 1] * sp[2 * k + 1];
                self.tmp[k] = if use_mag { p.sqrt() } else { p };
            }
            let (eps, lmax, feat) = (self.log_eps, self.log_max, self.feat);
            let tf = |v: f32| -> f32 {
                let w = if lmax { v.max(eps) } else { v + eps };
                match feat {
                    Feat::LogPow | Feat::LogMag => w.ln(),
                    Feat::Log10Pow | Feat::Log10Mag => w.log10(),
                    Feat::Log1pMag => v.ln_1p(),
                    _ => v,
                }
            };
            match &self.fb {
                Some((m, bins, bands)) => {
                    let o = &mut dst[f * bands..(f + 1) * bands];
                    for b in 0..*bands {
                        let mut s = 0f32;
                        for k in 0..*bins {
                            s += self.tmp[k] * m[k * bands + b];
                        }
                        o[b] = tf(s);
                    }
                }
                None => {
                    for k in 0..nb {
                        dst[f * nb + k] = tf(self.tmp[k]);
                    }
                }
            }
        }
    }
}

pub struct BackEnd {
    pub kind: BackKind,
    pub hop: usize,
    pub t: usize,
    pub blen: usize,
    nfft: usize,
    ws: Vec<f32>,
    norm: Vec<f32>,
    ola: Vec<f32>,
    ybuf: Vec<f32>,
    zspec: Vec<f32>,
    specq: Vec<f32>,
    pub md: usize,
    qpos: usize,
    rfft: Option<Rfft>,
    /// Raw back-end: number of model output values (set by the driver)
    pub raw_len: usize,
}

impl BackEnd {
    pub fn new(bk: &Value, front: &FrontEnd, lookup: TensorLookup) -> Result<BackEnd, String> {
        let (hop, t, nfft) = (front.hop, front.t, front.nfft);
        let nb = nfft / 2 + 1;
        let kind = match bk.get("type").and_then(|v| v.as_str()).unwrap_or("waveform") {
            "waveform" => BackKind::Waveform,
            "raw" => BackKind::Raw,
            "ola" => BackKind::Ola,
            "istft" => match bk.get("source").and_then(|v| v.as_str()).unwrap_or("complex_spectrum") {
                "complex_spectrum" => BackKind::Spec,
                "mag_mask" => BackKind::MagMask,
                "complex_mask" => BackKind::CplxMask,
                o => return Err(format!("unknown istft source {o}")),
            },
            o => return Err(format!("unknown back-end {o}")),
        };
        let spectral = matches!(kind, BackKind::Spec | BackKind::MagMask | BackKind::CplxMask);
        if spectral && front.kind != FrontKind::Stft {
            return Err("istft back-end needs an stft front-end".into());
        }
        let simple = matches!(kind, BackKind::Waveform | BackKind::Raw);
        let blen = if simple { hop } else { bk.num_or("frame_len", front.flen as f64) as usize };
        let ws = if simple { vec![] } else { window_spec(bk.get("window"), blen, lookup)? };
        let mut norm = vec![1.0f32; blen.max(1)];
        if !simple && bk.get("ola_norm").and_then(|v| v.as_str()) == Some("wola") {
            let wa2: Vec<f32> = if front.wa.len() == blen { front.wa.clone() } else { vec![1.0; blen] };
            let mut env = vec![0f64; blen];
            let mut s = -(blen as isize);
            while s < blen as isize {
                for i in 0..blen as isize {
                    let j = i - s;
                    if j >= 0 && j < blen as isize {
                        env[i as usize] += (wa2[j as usize] * ws[j as usize]) as f64;
                    }
                }
                s += hop as isize;
            }
            norm = env.iter().map(|&v| v as f32).collect();
        }
        let md = bk.num_or("mask_delay", 0.0) as usize;
        Ok(BackEnd {
            kind,
            hop,
            t,
            blen,
            nfft,
            ws,
            norm,
            ola: vec![0.0; blen + hop],
            ybuf: vec![0.0; nfft.max(blen)],
            zspec: vec![0.0; nb * 2],
            specq: vec![0.0; md.max(1) * nb * 2],
            md,
            qpos: 0,
            rfft: if spectral { Some(Rfft::new(nfft)) } else { None },
            raw_len: 0,
        })
    }

    /// values consumed per call
    pub fn in_len(&self) -> usize {
        let nb = self.nfft / 2 + 1;
        match self.kind {
            BackKind::Raw => self.raw_len,
            BackKind::Waveform => self.t * self.hop,
            BackKind::Ola => self.t * self.blen,
            BackKind::Spec | BackKind::CplxMask => self.t * nb * 2,
            BackKind::MagMask => self.t * nb,
        }
    }

    /// extra delay added by the back-end (mask delay frames)
    pub fn latency(&self) -> usize {
        if matches!(self.kind, BackKind::MagMask | BackKind::CplxMask) { self.md * self.hop } else { 0 }
    }

    pub fn reset(&mut self) {
        self.ola.iter_mut().for_each(|x| *x = 0.0);
        self.specq.iter_mut().for_each(|x| *x = 0.0);
        self.qpos = 0;
    }

    /// y: model output (in_len values), noisy: front-end spectrum [T][F][2] -> out: T*hop samples
    pub fn process(&mut self, y: &[f32], noisy: &[f32], out: &mut [f32]) {
        let (t, hop, blen, nfft) = (self.t, self.hop, self.blen, self.nfft);
        let nb = nfft / 2 + 1;
        if self.kind == BackKind::Waveform {
            out[..t * hop].copy_from_slice(&y[..t * hop]);
            return;
        }
        if self.kind == BackKind::Raw {
            let n = self.raw_len.min(t * hop);
            out[..n].copy_from_slice(&y[..n]);
            out[n..t * hop].iter_mut().for_each(|v| *v = 0.0);
            return;
        }
        for f in 0..t {
            if self.kind == BackKind::Ola {
                self.ybuf[..blen].copy_from_slice(&y[f * blen..(f + 1) * blen]);
            } else {
                let z = &mut self.zspec;
                if self.kind == BackKind::Spec {
                    z.copy_from_slice(&y[f * nb * 2..(f + 1) * nb * 2]);
                } else {
                    let cur = &noisy[f * nb * 2..(f + 1) * nb * 2];
                    if self.md == 0 {
                        z.copy_from_slice(cur);
                    } else {
                        let q = &mut self.specq[self.qpos * nb * 2..(self.qpos + 1) * nb * 2];
                        for k in 0..nb * 2 {
                            z[k] = q[k];
                            q[k] = cur[k];
                        }
                        self.qpos = (self.qpos + 1) % self.md;
                    }
                    if self.kind == BackKind::MagMask {
                        let m = &y[f * nb..(f + 1) * nb];
                        for k in 0..nb {
                            z[2 * k] *= m[k];
                            z[2 * k + 1] *= m[k];
                        }
                    } else {
                        let m = &y[f * nb * 2..(f + 1) * nb * 2];
                        for k in 0..nb {
                            let (a, b) = (z[2 * k], z[2 * k + 1]);
                            let (c, d) = (m[2 * k], m[2 * k + 1]);
                            z[2 * k] = a * c - b * d;
                            z[2 * k + 1] = a * d + b * c;
                        }
                    }
                }
                self.rfft.as_mut().unwrap().inverse(&self.zspec, &mut self.ybuf[..nfft]);
            }
            for i in 0..blen {
                self.ola[i] += self.ybuf[i] * self.ws[i];
            }
            let o = &mut out[f * hop..(f + 1) * hop];
            for i in 0..hop {
                o[i] = self.ola[i] / self.norm[i];
            }
            self.ola.copy_within(hop.., 0);
            let l = self.ola.len();
            for v in &mut self.ola[l - hop..] {
                *v = 0.0;
            }
        }
    }
}

/// out = wet * a + (1 - wet) * b
pub fn mix(a: &[f32], b: &[f32], wet: f32, out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = wet * a[i] + (1.0 - wet) * b[i];
    }
}

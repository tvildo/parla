//! Op registry for the graph runtime: `prepare(op, attrs, inputs)` validates shapes, computes
//! output shapes, pre-packs constant weights / index maps and returns a kernel object.
//! Semantics: ONNX (opset 17) restricted to f32 + static shapes; see models/FORMAT.md.
//!
//! Kernels avoid generics (wasm size) and never allocate in `run`.

use crate::graph::{numel, Attrs, In, Op, Prepared, Tensors};
use vi_kernels::{call, Backend};

type R<T> = Result<T, String>;

fn strides(s: &[usize]) -> Vec<usize> {
    let mut st = vec![1; s.len()];
    for i in (0..s.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * s[i + 1];
    }
    st
}

fn norm_axis(a: i64, nd: usize) -> R<usize> {
    let r = if a < 0 { a + nd as i64 } else { a };
    if r < 0 || r as usize >= nd.max(1) {
        return Err(format!("axis {a} out of range for rank {nd}"));
    }
    Ok(r as usize)
}

fn need<'a>(ins: &'a [Option<In<'a>>], i: usize) -> R<&'a In<'a>> {
    ins.get(i).and_then(|x| x.as_ref()).ok_or(format!("missing input {i}"))
}

fn cdata<'a>(ins: &'a [Option<In<'a>>], i: usize, what: &str) -> R<&'a [f32]> {
    need(ins, i)?.data.ok_or(format!("{what} (input {i}) must be constant"))
}

fn ok(shapes: Vec<Vec<usize>>, op: impl Op + 'static) -> R<Prepared> {
    Ok(Prepared { out_shapes: shapes, op: Box::new(op) })
}

// =================================================================== registry

pub fn prepare(op: &str, a: &Attrs, ins: &[Option<In>], nout: usize) -> R<Prepared> {
    match op {
        "Relu" | "Sigmoid" | "Tanh" | "Sqrt" | "Abs" | "Exp" | "Log" | "Neg" | "Reciprocal" | "Softplus" | "LeakyRelu"
        | "Elu" | "HardSigmoid" | "Clip" | "Identity" | "Cast" => unary(op, a, ins),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" | "PRelu" => binary(op, ins),
        "MatMul" => matmul(ins),
        "Gemm" => gemm(a, ins),
        "Conv" => conv(a, ins),
        "StreamConv" => stream_conv(a, ins),
        "ConvTranspose" => conv_transpose(a, ins),
        "GRU" | "LSTM" => rnn(op, a, ins),
        "ChannelNorm" | "LayerNormalization" | "InstanceNormalization" | "BatchNormalization" => norm(op, a, ins),
        "ReduceMean" | "ReduceSum" | "ReduceMax" => reduce(op, a, ins),
        "Concat" => concat(a, ins),
        "Split" => split(a, ins, nout),
        "Slice" | "Gather" | "Transpose" | "Pad" | "Expand" | "Tile" => gather_like(op, a, ins),
        "ScatterND" => scatter_nd(ins),
        "Reshape" | "Squeeze" | "Unsqueeze" | "Flatten" => reshape(op, a, ins),
        "Rfft" | "Irfft" => fft(op, a, ins),
        "MaxPool" => maxpool(a, ins),
        _ => Err(format!("unsupported op {op}")),
    }
}

// =================================================================== unary

#[derive(Clone, Copy, PartialEq)]
enum U {
    Relu,
    Sigmoid,
    Tanh,
    Sqrt,
    Abs,
    Exp,
    Log,
    Neg,
    Recip,
    Softplus,
    Leaky,
    Elu,
    HardSig,
    Clip,
    Copy,
}

struct Unary {
    k: U,
    n: usize,
    p0: f32,
    p1: f32,
}

use vi_kernels::util::sigmoid;

fn unary(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let (k, p0, p1) = match op {
        "Relu" => (U::Relu, 0.0, 0.0),
        "Sigmoid" => (U::Sigmoid, 0.0, 0.0),
        "Tanh" => (U::Tanh, 0.0, 0.0),
        "Sqrt" => (U::Sqrt, 0.0, 0.0),
        "Abs" => (U::Abs, 0.0, 0.0),
        "Exp" => (U::Exp, 0.0, 0.0),
        "Log" => (U::Log, 0.0, 0.0),
        "Neg" => (U::Neg, 0.0, 0.0),
        "Reciprocal" => (U::Recip, 0.0, 0.0),
        "Softplus" => (U::Softplus, 0.0, 0.0),
        "LeakyRelu" => (U::Leaky, a.float("alpha", 0.01)?, 0.0),
        "Elu" => (U::Elu, a.float("alpha", 1.0)?, 0.0),
        "HardSigmoid" => (U::HardSig, a.float("alpha", 0.2)?, a.float("beta", 0.5)?),
        "Clip" => (U::Clip, a.float("min", f32::NEG_INFINITY)?, a.float("max", f32::INFINITY)?),
        _ => (U::Copy, 0.0, 0.0),
    };
    if op == "Clip" && (ins.len() > 1 && ins[1].is_some() || ins.len() > 2 && ins[2].is_some()) {
        return Err("Clip min/max must be attributes".into());
    }
    ok(vec![x.shape.to_vec()], Unary { k, n: numel(x.shape), p0, p1 })
}

impl Op for Unary {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let x = &ts.d(ins[0])[..self.n];
        let y = &mut ts.dm(outs[0])[..self.n];
        apply_unary(self.k, x, y, self.p0, self.p1, be);
    }
}

fn apply_unary(k: U, x: &[f32], y: &mut [f32], p0: f32, p1: f32, be: Backend) {
    match k {
        U::Copy => y.copy_from_slice(x),
        U::Relu | U::Leaky => {
            y.copy_from_slice(x);
            let n = y.len();
            let nv = n - n % 8;
            if k == U::Relu {
                call!(be, relu(y.as_mut_ptr(), nv));
                for v in &mut y[nv..] {
                    *v = v.max(0.0);
                }
            } else {
                call!(be, leaky(y.as_mut_ptr(), nv, p0));
                for v in &mut y[nv..] {
                    *v = v.max(0.0) + p0 * v.min(0.0);
                }
            }
        }
        U::Sigmoid => {
            y.copy_from_slice(x);
            ku::sigmoid_inplace(y)
        }
        U::Tanh => {
            y.copy_from_slice(x);
            ku::tanh_inplace(y)
        }
        U::Sqrt => y.iter_mut().zip(x).for_each(|(o, &v)| *o = v.sqrt()),
        U::Abs => y.iter_mut().zip(x).for_each(|(o, &v)| *o = v.abs()),
        U::Exp => y.iter_mut().zip(x).for_each(|(o, &v)| *o = v.exp()),
        U::Log => y.iter_mut().zip(x).for_each(|(o, &v)| *o = v.ln()),
        U::Neg => y.iter_mut().zip(x).for_each(|(o, &v)| *o = -v),
        U::Recip => y.iter_mut().zip(x).for_each(|(o, &v)| *o = 1.0 / v),
        U::Softplus => y.iter_mut().zip(x).for_each(|(o, &v)| *o = if v > 20.0 { v } else { v.exp().ln_1p() }),
        U::Elu => y.iter_mut().zip(x).for_each(|(o, &v)| *o = if v > 0.0 { v } else { p0 * (v.exp() - 1.0) }),
        U::HardSig => y.iter_mut().zip(x).for_each(|(o, &v)| *o = (p0 * v + p1).max(0.0).min(1.0)),
        U::Clip => y.iter_mut().zip(x).for_each(|(o, &v)| *o = v.max(p0).min(p1)),
    }
}

/// Fused activation epilogue (attrs `activation`, `act_alpha` set by the load-time fusion pass).
#[derive(Clone, Copy)]
pub struct Act {
    k: Option<U>,
    alpha: f32,
}

fn act_of(a: &Attrs) -> R<Act> {
    let k = match a.str("activation", "") {
        "" => None,
        "Relu" => Some(U::Relu),
        "LeakyRelu" => Some(U::Leaky),
        "Sigmoid" => Some(U::Sigmoid),
        "Tanh" => Some(U::Tanh),
        o => return Err(format!("unsupported fused activation {o}")),
    };
    Ok(Act { k, alpha: a.float("act_alpha", 0.01)? })
}

impl Act {
    #[inline]
    fn apply(&self, y: &mut [f32], be: Backend) {
        let Some(k) = self.k else { return };
        let n = y.len();
        match k {
            U::Relu | U::Leaky => {
                let nv = n - n % 8;
                if k == U::Relu {
                    call!(be, relu(y.as_mut_ptr(), nv));
                    y[nv..].iter_mut().for_each(|v| *v = v.max(0.0));
                } else {
                    let a = self.alpha;
                    call!(be, leaky(y.as_mut_ptr(), nv, a));
                    y[nv..].iter_mut().for_each(|v| *v = v.max(0.0) + a * v.min(0.0));
                }
            }
            U::Sigmoid => ku::sigmoid_inplace(y),
            _ => ku::tanh_inplace(y),
        }
    }
}

// =================================================================== binary (broadcasting)

use vi_kernels::util::{self as ku, B};

/// Collapsed broadcast iteration plan (<= 6 dims after collapsing).
struct BPlan {
    dims: Vec<usize>,
    sa: Vec<usize>,
    sb: Vec<usize>,
    n: usize,
}

pub fn bcast(a: &[usize], b: &[usize]) -> R<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut o = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        o[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return Err(format!("cannot broadcast {a:?} with {b:?}"));
        };
    }
    Ok(o)
}

fn bstrides(s: &[usize], out: &[usize]) -> Vec<usize> {
    let n = out.len();
    let st = strides(s);
    (0..n)
        .map(|i| if i + s.len() >= n { let j = i + s.len() - n; if s[j] == 1 { 0 } else { st[j] } } else { 0 })
        .collect()
}

fn bplan(a: &[usize], b: &[usize]) -> R<(Vec<usize>, BPlan)> {
    let out = bcast(a, b)?;
    let (sa0, sb0) = (bstrides(a, &out), bstrides(b, &out));
    let (mut dims, mut sa, mut sb) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..out.len() {
        if out[i] == 1 {
            continue;
        }
        if let Some(&ld) = dims.last() {
            let k = dims.len() - 1;
            // merge with previous (outer) dim if contiguous
            if sa[k] == sa0[i] * out[i] && sb[k] == sb0[i] * out[i] {
                dims[k] = ld * out[i];
                sa[k] = sa0[i];
                sb[k] = sb0[i];
                continue;
            }
        }
        dims.push(out[i]);
        sa.push(sa0[i]);
        sb.push(sb0[i]);
    }
    if dims.is_empty() {
        dims.push(1);
        sa.push(0);
        sb.push(0);
    }
    let n = numel(&out);
    Ok((out, BPlan { dims, sa, sb, n }))
}

struct Binary {
    k: B,
    p: BPlan,
}

fn binary(op: &str, ins: &[Option<In>]) -> R<Prepared> {
    let (a, b) = (need(ins, 0)?, need(ins, 1)?);
    let mut k = match op {
        "Add" => B::Add,
        "Sub" => B::Sub,
        "Mul" => B::Mul,
        "Div" => B::Div,
        "Pow" => B::Pow,
        "Max" => B::Max,
        "Min" => B::Min,
        _ => B::PRelu,
    };
    if k == B::Pow {
        if let Some(d) = b.data {
            if d.len() == 1 && d[0] == 2.0 {
                k = B::Square;
            } else if d.len() == 1 && d[0] == 0.5 {
                k = B::SqrtPow;
            }
        }
    }
    let (out, p) = bplan(a.shape, b.shape)?;
    ok(vec![out], Binary { k, p })
}

fn binary_run(k: B, p: &BPlan, a: &[f32], b: &[f32], o: &mut [f32]) {
    ku::binary(k, &p.dims, &p.sa, &p.sb, p.n, a, b, o);
}

impl Op for Binary {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        binary_run(self.k, &self.p, ts.d(ins[0]), ts.d(ins[1]), ts.dm(outs[0]));
    }
}

// =================================================================== MatMul / Gemm

struct MatMul {
    m: usize,
    k: usize,
    n: usize,
    batches: Vec<(usize, usize, usize)>,
}

fn matmul(ins: &[Option<In>]) -> R<Prepared> {
    let (a, b) = (need(ins, 0)?, need(ins, 1)?);
    let (mut sa, mut sb) = (a.shape.to_vec(), b.shape.to_vec());
    let (a1, b1) = (sa.len() == 1, sb.len() == 1);
    if a1 {
        sa.insert(0, 1);
    }
    if b1 {
        sb.push(1);
    }
    if sa.len() < 2 || sb.len() < 2 {
        return Err("MatMul rank".into());
    }
    let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
    let (k2, n) = (sb[sb.len() - 2], sb[sb.len() - 1]);
    if k != k2 {
        return Err(format!("MatMul inner dims {sa:?} x {sb:?}"));
    }
    let ba = &sa[..sa.len() - 2];
    let bb = &sb[..sb.len() - 2];
    let bo = bcast(ba, bb)?;
    let nb = numel(&bo);
    let (sta, stb) = (bstrides(ba, &bo), bstrides(bb, &bo));
    let bost = strides(&bo);
    let mut batches = Vec::with_capacity(nb);
    for i in 0..nb {
        let (mut oa, mut ob) = (0, 0);
        for d in 0..bo.len() {
            let c = (i / bost[d]) % bo[d];
            oa += c * sta[d];
            ob += c * stb[d];
        }
        batches.push((oa * m * k, ob * k * n, i * m * n));
    }
    let mut out = bo.clone();
    if !a1 {
        out.push(m);
    }
    if !b1 {
        out.push(n);
    }
    ok(vec![out], MatMul { m, k, n, batches })
}

impl Op for MatMul {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (a, b, o) = (ts.d(ins[0]).as_ptr(), ts.d(ins[1]).as_ptr(), ts.dm(outs[0]).as_mut_ptr());
        let (m, k, n) = (self.m, self.k, self.n);
        for &(oa, ob, oo) in &self.batches {
            unsafe {
                call!(be, gemm_ld(b.add(ob), n, n, k, a.add(oa), k, m, o.add(oo), n));
            }
        }
    }
}

struct GemmOp {
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    wt: Vec<f32>, // [k][n] packed B (if constant)
    b_const: bool,
    trans_b: bool,
    alpha: f32,
    bias: Vec<f32>, // [m][n] beta*C broadcast, or empty
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

fn gemm(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let (x, w) = (need(ins, 0)?, need(ins, 1)?);
    if x.shape.len() != 2 || w.shape.len() != 2 {
        return Err("Gemm needs 2-D inputs".into());
    }
    let ta = a.int("transA", 0)? != 0;
    let tb = a.int("transB", 0)? != 0;
    let (m, k) = if ta { (x.shape[1], x.shape[0]) } else { (x.shape[0], x.shape[1]) };
    let (k2, n) = if tb { (w.shape[1], w.shape[0]) } else { (w.shape[0], w.shape[1]) };
    if k != k2 {
        return Err("Gemm inner dims".into());
    }
    let alpha = a.float("alpha", 1.0)?;
    let beta = a.float("beta", 1.0)?;
    let mut wt = Vec::new();
    if let Some(d) = w.data {
        wt = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                wt[i * n + j] = if tb { d[j * k + i] } else { d[i * n + j] };
            }
        }
    }
    let mut bias = Vec::new();
    if let Some(Some(c)) = ins.get(2) {
        let cd = c.data.ok_or("Gemm C must be constant")?;
        let (_, p) = bplan(c.shape, &[m, n])?;
        bias = vec![0.0; m * n];
        let zeros = vec![0.0; m * n];
        binary_run(B::Add, &p, cd, &zeros, &mut bias);
        bias.iter_mut().for_each(|v| *v *= beta);
    }
    let bc = w.data.is_some();
    ok(
        vec![vec![m, n]],
        GemmOp { m, k, n, trans_a: ta, wt, b_const: bc, trans_b: tb, alpha, bias, scratch_a: if ta { vec![0.0; m * k] } else { vec![] }, scratch_b: if !bc { vec![0.0; k * n] } else { vec![] } },
    )
}

impl Op for GemmOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (m, k, n) = (self.m, self.k, self.n);
        let mut ap = ts.d(ins[0]).as_ptr();
        if self.trans_a {
            let x = ts.d(ins[0]);
            for i in 0..m {
                for j in 0..k {
                    self.scratch_a[i * k + j] = x[j * m + i];
                }
            }
            ap = self.scratch_a.as_ptr();
        }
        let wp = if self.b_const {
            self.wt.as_ptr()
        } else {
            let w = ts.d(ins[1]);
            for i in 0..k {
                for j in 0..n {
                    self.scratch_b[i * n + j] = if self.trans_b { w[j * k + i] } else { w[i * n + j] };
                }
            }
            self.scratch_b.as_ptr()
        };
        let o = ts.dm(outs[0]);
        call!(be, gemm_ld(wp, n, n, k, ap, k, m, o.as_mut_ptr(), n));
        if self.alpha != 1.0 {
            o.iter_mut().for_each(|v| *v *= self.alpha);
        }
        if !self.bias.is_empty() {
            o.iter_mut().zip(&self.bias).for_each(|(v, b)| *v += b);
        }
    }
}

// =================================================================== Conv

struct ConvOp {
    act: Act,
    g: usize,
    cig: usize,
    cog: usize,
    kc: usize,
    p: usize,
    hw_in: usize,
    batch: usize,
    cin: usize,
    form_a: bool,
    depthwise: bool,
    direct: bool, // 1x1 stride 1 no pad: col == x
    w: Vec<f32>,  // form A: [g][kc][cog] (transposed); form B / depthwise: [g][cog][kc]
    bias: Vec<f32>,
    idx: Vec<u32>, // im2col: form A [p][kc] / form B [kc][p], index within one group's input block
    col: Vec<f32>,
    tmp: Vec<f32>, // form A output [p][cog]
    /// stride-1 form-B im2col by shifted row copies: (kh, kw, dh, dw, pt, pl, h, w, oh, ow)
    shift: Option<[usize; 10]>,
    chunk_rows: usize,
}

const PADI: u32 = u32::MAX;

/// im2col for stride-1 convs, layout [ci*kh*kw][oh*ow], by contiguous row copies (no index table)
fn im2col_shift(xg: &[f32], col: &mut [f32], cig: usize, s: &[usize; 10], oy0: usize, oy1: usize) {
    let [kh, kw, dh, dw, pt, pl, h, w, _oh, ow] = *s;
    let p = (oy1 - oy0) * ow;
    let mut kk = 0;
    for ci in 0..cig {
        let plane = &xg[ci * h * w..(ci + 1) * h * w];
        for i in 0..kh {
            for j in 0..kw {
                let row = &mut col[kk * p..(kk + 1) * p];
                let sx = j * dw; // input column = ox + sx - pl
                let lo = pl.saturating_sub(sx).min(ow);
                let hi = (w + pl).saturating_sub(sx).min(ow).max(lo);
                for oy in oy0..oy1 {
                    let dst = &mut row[(oy - oy0) * ow..(oy - oy0 + 1) * ow];
                    let y = (oy + i * dh) as isize - pt as isize;
                    if y < 0 || y >= h as isize {
                        dst.iter_mut().for_each(|v| *v = 0.0);
                        continue;
                    }
                    let src = &plane[y as usize * w..(y as usize + 1) * w];
                    dst[..lo].iter_mut().for_each(|v| *v = 0.0);
                    if hi > lo {
                        let s0 = lo + sx - pl;
                        dst[lo..hi].copy_from_slice(&src[s0..s0 + (hi - lo)]);
                    }
                    dst[hi..].iter_mut().for_each(|v| *v = 0.0);
                }
                kk += 1;
            }
        }
    }
}

fn spatial(a: &Attrs, nd: usize, ks: &[usize]) -> R<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if a.str("auto_pad", "NOTSET") != "NOTSET" {
        return Err("auto_pad not supported (resolve in converter)".into());
    }
    let st = a.ints("strides")?.unwrap_or(vec![1; nd]).iter().map(|&v| v as usize).collect::<Vec<_>>();
    let dl = a.ints("dilations")?.unwrap_or(vec![1; nd]).iter().map(|&v| v as usize).collect::<Vec<_>>();
    let pd = a.ints("pads")?.unwrap_or(vec![0; 2 * nd]).iter().map(|&v| v as usize).collect::<Vec<_>>();
    if let Some(k) = a.ints("kernel_shape")? {
        if k.iter().map(|&v| v as usize).collect::<Vec<_>>() != ks {
            return Err("kernel_shape mismatch".into());
        }
    }
    if st.len() != nd || dl.len() != nd || pd.len() != 2 * nd {
        return Err("strides/dilations/pads rank".into());
    }
    Ok((st, dl, pd))
}

fn conv(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    if a.int("channels_last", 0)? != 0 {
        return conv_nlc(a, ins);
    }
    let x = need(ins, 0)?;
    let w = need(ins, 1)?;
    let wd = cdata(ins, 1, "Conv weight")?;
    let nd = w.shape.len() - 2;
    if !(nd == 1 || nd == 2) || x.shape.len() != nd + 2 {
        return Err(format!("Conv: x {:?} w {:?}", x.shape, w.shape));
    }
    let (st, dl, pd) = spatial(a, nd, &w.shape[2..])?;
    let g = a.int("group", 1)? as usize;
    let (batch, cin) = (x.shape[0], x.shape[1]);
    let co = w.shape[0];
    if cin % g != 0 || co % g != 0 || w.shape[1] != cin / g {
        return Err(format!("Conv groups: x {:?} w {:?} g {g}", x.shape, w.shape));
    }
    // normalize to 2-D
    let (h, wi) = if nd == 1 { (1, x.shape[2]) } else { (x.shape[2], x.shape[3]) };
    let (kh, kw) = if nd == 1 { (1, w.shape[2]) } else { (w.shape[2], w.shape[3]) };
    let (sh, sw) = if nd == 1 { (1, st[0]) } else { (st[0], st[1]) };
    let (dh, dw) = if nd == 1 { (1, dl[0]) } else { (dl[0], dl[1]) };
    let (pt, pl, pb, pr) = if nd == 1 { (0, pd[0], 0, pd[1]) } else { (pd[0], pd[1], pd[2], pd[3]) };
    let hp = h + pt + pb;
    let wp = wi + pl + pr;
    if hp < dh * (kh - 1) + 1 || wp < dw * (kw - 1) + 1 {
        return Err("Conv input smaller than kernel".into());
    }
    let oh = (hp - dh * (kh - 1) - 1) / sh + 1;
    let ow = (wp - dw * (kw - 1) - 1) / sw + 1;
    let (cig, cog) = (cin / g, co / g);
    let kc = cig * kh * kw;
    let p = oh * ow;
    let mut out = vec![batch, co];
    if nd == 2 {
        out.push(oh);
    }
    out.push(ow);
    let bias = match ins.get(2) {
        Some(Some(_)) => cdata(ins, 2, "Conv bias")?.to_vec(),
        _ => vec![0.0; co],
    };
    let depthwise = cig == 1 && cog == 1;
    // Toeplitz path: kernel taller than the input (e.g. a mask conv sliding over channels):
    // dense [cig*h][cog*oh] matrix, one GEMM per column, no im2col
    if kw == 1 && sw == 1 && g == 1 && h < kh && cig * h * cog * oh <= 4 * cog * oh * kc {
        let (mr, mc) = (cig * h, cog * oh);
        let mut m = vec![0.0; mr * mc];
        for co_ in 0..cog {
            for ci in 0..cig {
                for ho in 0..oh {
                    for i in 0..kh {
                        let hin = (ho * sh + i * dh) as isize - pt as isize;
                        if hin >= 0 && (hin as usize) < h {
                            m[(ci * h + hin as usize) * mc + co_ * oh + ho] += wd[(co_ * cig + ci) * kh + i];
                        }
                    }
                }
            }
        }
        return ok(vec![out], ToeplitzConv { act: act_of(a)?, mr, mc, w: wi, batch, m, bias, xt: vec![0.0; wi * mr], tmp: vec![0.0; wi * mc], oh });
    }
    if kh == 1 && sh == 1 && g == 1 && cig == 1 && wi < kw && wi * cog * ow <= 4 * cog * ow * kc {
        let mc = cog * ow;
        let mut m = vec![0.0; wi * mc];
        for co_ in 0..cog {
            for o in 0..ow {
                for j in 0..kw {
                    let xin = (o * sw + j * dw) as isize - pl as isize;
                    if xin >= 0 && (xin as usize) < wi {
                        m[xin as usize * mc + co_ * ow + o] += wd[co_ * kw + j];
                    }
                }
            }
        }
        return ok(vec![out], ToeplitzW { act: act_of(a)?, wi, mc, ow, cog, rows: h, batch, m, bias, tmp: vec![0.0; h * mc] });
    }
    let direct = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt + pl + pb + pr == 0;
    let cost_a = p * kc * (cog / 8 + cog % 8);
    let cost_b = cog * kc * (p / 8 + p % 8);
    // large stride-1 convs always take the chunked form-B path (bounded scratch: no [P][kc] im2col /
    // index table, which reached ~0.7 GB per layer for DNSMOS-size spectrograms)
    let big_b = p >= 256 && sh == 1 && sw == 1 && !direct;
    let form_a = !depthwise && cost_a < cost_b && !big_b;
    let shift = if !depthwise && !form_a && !direct && sh == 1 && sw == 1 { Some([kh, kw, dh, dw, pt, pl, h, wi, oh, ow]) } else { None };
    // im2col index table (tap order: ci, i, j)
    let idx_at = |pp: usize, kk: usize| -> u32 {
        let (oy, ox) = (pp / ow, pp % ow);
        let ci = kk / (kh * kw);
        let (i, j) = ((kk / kw) % kh, kk % kw);
        let y = (oy * sh + i * dh) as isize - pt as isize;
        let xx = (ox * sw + j * dw) as isize - pl as isize;
        if y < 0 || y >= h as isize || xx < 0 || xx >= wi as isize {
            PADI
        } else {
            (ci * h * wi + y as usize * wi + xx as usize) as u32
        }
    };
    let mut idx = Vec::new();
    if (!direct || depthwise) && shift.is_none() {
        if depthwise {
            idx = (0..p * kh * kw).map(|t| idx_at(t / (kh * kw), t % (kh * kw))).collect();
        } else if form_a {
            idx = (0..p * kc).map(|t| idx_at(t / kc, t % kc)).collect();
        } else {
            idx = (0..kc * p).map(|t| idx_at(t % p, t / p)).collect();
        }
    }
    let wv = if form_a {
        let mut t = vec![0.0; g * kc * cog];
        for gi in 0..g {
            for c in 0..cog {
                for k in 0..kc {
                    t[gi * kc * cog + k * cog + c] = wd[(gi * cog + c) * kc + k];
                }
            }
        }
        t
    } else {
        wd.to_vec()
    };
    // shift path: scratch for a block of output rows only (~4096 positions)
    let chunk_rows = (4096 / ow.max(1)).max(1).min(oh);
    let col = if (direct && !depthwise) || depthwise { vec![] } else if shift.is_some() { vec![0.0; kc * chunk_rows * ow] } else { vec![0.0; kc * p] };
    let tmp = if form_a && p > 1 { vec![0.0; p * cog] } else { vec![] };
    ok(vec![out], ConvOp { act: act_of(a)?, g, cig, cog, kc, p, hw_in: h * wi, batch, cin, form_a, depthwise, direct, w: wv, bias, idx, col, tmp, shift, chunk_rows })
}

impl Op for ConvOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        self.run_inner(ts, ins, outs, be);
        let n = self.batch * self.g * self.cog * self.p;
        self.act.apply(&mut ts.dm(outs[0])[..n], be);
    }
}

impl ConvOp {
    fn run_inner(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let xall = ts.d(ins[0]);
        let yall = ts.dm(outs[0]);
        let (g, cig, cog, kc, p, hw) = (self.g, self.cig, self.cog, self.kc, self.p, self.hw_in);
        for nb in 0..self.batch {
            let x = &xall[nb * self.cin * hw..(nb + 1) * self.cin * hw];
            let y = &mut yall[nb * g * cog * p..(nb + 1) * g * cog * p];
            if self.depthwise {
                let kk = kc;
                for c in 0..g {
                    let xc = &x[c * hw..(c + 1) * hw];
                    let wc = &self.w[c * kk..(c + 1) * kk];
                    let b = self.bias[c];
                    for pp in 0..p {
                        let mut acc = 0f32;
                        let ix = &self.idx[pp * kk..(pp + 1) * kk];
                        for k in 0..kk {
                            let i = ix[k];
                            if i != PADI {
                                acc += wc[k] * xc[i as usize];
                            }
                        }
                        y[c * p + pp] = acc + b;
                    }
                }
                continue;
            }
            for gi in 0..g {
                let xg = &x[gi * cig * hw..(gi + 1) * cig * hw];
                if let Some(sp) = self.shift {
                    // form B, stride 1: im2col + GEMM in blocks of output rows (bounded scratch,
                    // cache-friendly; per-element arithmetic identical to the unblocked form)
                    let (oh, ow) = (sp[8], sp[9]);
                    let yg = &mut y[gi * cog * p..(gi + 1) * cog * p];
                    let wr = self.w[gi * cog * kc..].as_ptr();
                    let mut oy0 = 0;
                    while oy0 < oh {
                        let oy1 = (oy0 + self.chunk_rows).min(oh);
                        let len = (oy1 - oy0) * ow;
                        im2col_shift(xg, &mut self.col[..kc * len], cig, &sp, oy0, oy1);
                        call!(be, gemm_ld(self.col.as_ptr(), len, len, kc, wr, kc, cog, yg[oy0 * ow..].as_mut_ptr(), p));
                        oy0 = oy1;
                    }
                    for c in 0..cog {
                        let b = self.bias[gi * cog + c];
                        if b != 0.0 {
                            yg[c * p..(c + 1) * p].iter_mut().for_each(|v| *v += b);
                        }
                    }
                    continue;
                }
                // build columns
                let colp: *const f32 = if self.direct {
                    xg.as_ptr() // [kc=cig][p] == form B layout; form A only chosen when p == 1 here
                } else {
                    for (c, &i) in self.col.iter_mut().zip(self.idx.iter()) {
                        *c = if i == PADI { 0.0 } else { xg[i as usize] };
                    }
                    self.col.as_ptr()
                };
                let yg = &mut y[gi * cog * p..(gi + 1) * cog * p];
                if self.form_a {
                    let wt = self.w[gi * kc * cog..].as_ptr();
                    if self.direct && p > 1 {
                        // x is [kc][p] (NCL): frame stride 1, k stride p
                        call!(be, gemm_s(wt, cog, cog, kc, colp, 1, p, p, self.tmp.as_mut_ptr(), cog));
                        for c in 0..cog {
                            let b = self.bias[gi * cog + c];
                            for pp in 0..p {
                                yg[c * p + pp] = self.tmp[pp * cog + c] + b;
                            }
                        }
                    } else if p == 1 {
                        call!(be, gemm_ld(wt, cog, cog, kc, colp, kc, 1, yg.as_mut_ptr(), cog));
                        for c in 0..cog {
                            yg[c] += self.bias[gi * cog + c];
                        }
                    } else {
                        call!(be, gemm_ld(wt, cog, cog, kc, colp, kc, p, self.tmp.as_mut_ptr(), cog));
                        for c in 0..cog {
                            let b = self.bias[gi * cog + c];
                            for pp in 0..p {
                                yg[c * p + pp] = self.tmp[pp * cog + c] + b;
                            }
                        }
                    }
                } else {
                    let wr = self.w[gi * cog * kc..].as_ptr();
                    call!(be, gemm_ld(colp, p, p, kc, wr, kc, cog, yg.as_mut_ptr(), p));
                    for c in 0..cog {
                        let b = self.bias[gi * cog + c];
                        if b != 0.0 {
                            yg[c * p..(c + 1) * p].iter_mut().for_each(|v| *v += b);
                        }
                    }
                }
            }
        }
    }
}

struct ToeplitzConv {
    act: Act,
    mr: usize,
    mc: usize,
    w: usize,
    batch: usize,
    m: Vec<f32>,
    bias: Vec<f32>,
    xt: Vec<f32>,
    tmp: Vec<f32>,
    oh: usize,
}

impl Op for ToeplitzConv {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (mr, mc, w) = (self.mr, self.mc, self.w);
        let xall = ts.d(ins[0]);
        let yall = ts.dm(outs[0]);
        for nb in 0..self.batch {
            let x = &xall[nb * mr * w..(nb + 1) * mr * w];
            let y = &mut yall[nb * mc * w..(nb + 1) * mc * w];
            if w == 1 {
                call!(be, gemm_ld(self.m.as_ptr(), mc, mc, mr, x.as_ptr(), mr, 1, y.as_mut_ptr(), mc));
            } else {
                for r in 0..mr {
                    for c in 0..w {
                        self.xt[c * mr + r] = x[r * w + c];
                    }
                }
                call!(be, gemm_ld(self.m.as_ptr(), mc, mc, mr, self.xt.as_ptr(), mr, w, self.tmp.as_mut_ptr(), mc));
                for r in 0..mc {
                    for c in 0..w {
                        y[r * w + c] = self.tmp[c * mc + r];
                    }
                }
            }
            for (k, &b) in self.bias.iter().enumerate() {
                if b != 0.0 {
                    y[k * self.oh * w..(k + 1) * self.oh * w].iter_mut().for_each(|v| *v += b);
                }
            }
        }
        let n = self.batch * self.mc * self.w;
        self.act.apply(&mut ts.dm(outs[0])[..n], be);
    }
}

/// Conv2d with kh == 1, cin == 1 and a kernel wider than the input: per input row one GEMM
/// with the dense [w][cog*ow] Toeplitz matrix.
struct ToeplitzW {
    act: Act,
    wi: usize,
    mc: usize,
    ow: usize,
    cog: usize,
    rows: usize,
    batch: usize,
    m: Vec<f32>,
    bias: Vec<f32>,
    tmp: Vec<f32>,
}

impl Op for ToeplitzW {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (wi, mc, ow, cog, rows) = (self.wi, self.mc, self.ow, self.cog, self.rows);
        let xall = ts.d(ins[0]);
        let yall = ts.dm(outs[0]);
        for nb in 0..self.batch {
            let x = &xall[nb * rows * wi..];
            let y = &mut yall[nb * cog * rows * ow..(nb + 1) * cog * rows * ow];
            if cog == 1 {
                call!(be, gemm_ld(self.m.as_ptr(), mc, mc, wi, x.as_ptr(), wi, rows, y.as_mut_ptr(), mc));
            } else {
                call!(be, gemm_ld(self.m.as_ptr(), mc, mc, wi, x.as_ptr(), wi, rows, self.tmp.as_mut_ptr(), mc));
                for r in 0..rows {
                    for c in 0..cog {
                        y[(c * rows + r) * ow..(c * rows + r + 1) * ow].copy_from_slice(&self.tmp[r * mc + c * ow..r * mc + (c + 1) * ow]);
                    }
                }
            }
            for c in 0..cog {
                let b = self.bias[c];
                if b != 0.0 {
                    y[c * rows * ow..(c + 1) * rows * ow].iter_mut().for_each(|v| *v += b);
                }
            }
        }
        let n = self.batch * self.cog * self.rows * self.ow;
        self.act.apply(&mut ts.dm(outs[0])[..n], be);
    }
}

// =================================================================== channels-last 1-D Conv

/// Conv1d on [N, L, C] (attr channels_last=1); weights packed [g][k*cig][cog] (tap-major).
struct ConvNlc {
    act: Act,
    batch: usize,
    l: usize,
    c: usize,
    lo: usize,
    cout: usize,
    g: usize,
    cig: usize,
    cog: usize,
    kc: usize,
    direct: bool,
    depthwise: bool,
    k: usize,
    w: Vec<f32>,
    bias: Vec<f32>,
    rows: Vec<u32>, // [lo][k] input row index or PADI
    col: Vec<f32>,
}

fn conv_nlc(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let w = need(ins, 1)?;
    let wd = cdata(ins, 1, "Conv weight")?;
    if x.shape.len() != 3 || w.shape.len() != 3 {
        return Err("channels_last Conv: 1-D only".into());
    }
    let (st, dl, pd) = spatial(a, 1, &w.shape[2..])?;
    let g = a.int("group", 1)? as usize;
    let (batch, l, c) = (x.shape[0], x.shape[1], x.shape[2]);
    let (cout, k) = (w.shape[0], w.shape[2]);
    if c % g != 0 || cout % g != 0 || w.shape[1] != c / g {
        return Err("Conv groups".into());
    }
    let (s, d) = (st[0], dl[0]);
    let lp = l + pd[0] + pd[1];
    if lp < d * (k - 1) + 1 {
        return Err("Conv input smaller than kernel".into());
    }
    let lo = (lp - d * (k - 1) - 1) / s + 1;
    let (cig, cog) = (c / g, cout / g);
    let kc = k * cig;
    let depthwise = cig == 1 && cog == 1;
    let direct = k == 1 && s == 1 && pd[0] + pd[1] == 0 && g == 1;
    let mut rows = vec![PADI; lo * k];
    for t in 0..lo {
        for j in 0..k {
            let r = (t * s + j * d) as isize - pd[0] as isize;
            if r >= 0 && (r as usize) < l {
                rows[t * k + j] = r as u32;
            }
        }
    }
    let wv = if depthwise {
        let mut t = vec![0.0; k * c];
        for ch in 0..c {
            for j in 0..k {
                t[j * c + ch] = wd[ch * k + j];
            }
        }
        t
    } else {
        let mut t = vec![0.0; g * kc * cog];
        for gi in 0..g {
            for co_ in 0..cog {
                for ci in 0..cig {
                    for j in 0..k {
                        t[gi * kc * cog + (j * cig + ci) * cog + co_] = wd[((gi * cog + co_) * cig + ci) * k + j];
                    }
                }
            }
        }
        t
    };
    let bias = match ins.get(2) {
        Some(Some(_)) => cdata(ins, 2, "Conv bias")?.to_vec(),
        _ => vec![0.0; cout],
    };
    let col = if direct || depthwise { vec![] } else { vec![0.0; lo * kc] };
    ok(vec![vec![batch, lo, cout]], ConvNlc { act: act_of(a)?, batch, l, c, lo, cout, g, cig, cog, kc, direct, depthwise, k, w: wv, bias, rows, col })
}

impl Op for ConvNlc {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        self.run_inner(ts, ins, outs, be);
        let n = self.batch * self.lo * self.cout;
        self.act.apply(&mut ts.dm(outs[0])[..n], be);
    }
}

impl ConvNlc {
    fn run_inner(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (l, c, lo, cout, k) = (self.l, self.c, self.lo, self.cout, self.k);
        let (cig, cog, kc) = (self.cig, self.cog, self.kc);
        let xall = ts.d(ins[0]);
        let yall = ts.dm(outs[0]);
        for nb in 0..self.batch {
            let x = &xall[nb * l * c..(nb + 1) * l * c];
            let y = &mut yall[nb * lo * cout..(nb + 1) * lo * cout];
            if self.depthwise {
                for t in 0..lo {
                    let o = &mut y[t * c..(t + 1) * c];
                    o.copy_from_slice(&self.bias);
                    for j in 0..k {
                        let r = self.rows[t * k + j];
                        if r == PADI {
                            continue;
                        }
                        ku::madd(o, &self.w[j * c..(j + 1) * c], &x[r as usize * c..(r as usize + 1) * c]);
                    }
                }
                continue;
            }
            for gi in 0..self.g {
                let (inp, is) = if self.direct {
                    (x.as_ptr(), c)
                } else {
                    for t in 0..lo {
                        for j in 0..k {
                            let r = self.rows[t * k + j];
                            let dst = &mut self.col[t * kc + j * cig..t * kc + (j + 1) * cig];
                            if r == PADI {
                                dst.iter_mut().for_each(|v| *v = 0.0);
                            } else {
                                dst.copy_from_slice(&x[r as usize * c + gi * cig..r as usize * c + (gi + 1) * cig]);
                            }
                        }
                    }
                    (self.col.as_ptr(), kc)
                };
                let wt = self.w[gi * kc * cog..].as_ptr();
                call!(be, gemm_ld(wt, cog, cog, kc, inp, is, lo, y[gi * cog..].as_mut_ptr(), cout));
            }
            if self.bias.iter().any(|&b| b != 0.0) {
                for t in 0..lo {
                    for (v, b) in y[t * cout..(t + 1) * cout].iter_mut().zip(&self.bias) {
                        *v += b;
                    }
                }
            }
        }
    }
}

// =================================================================== StreamConv (fused causal conv + history state)

/// Conv1d over concat(history[N], x) with the history kept in a time-major ring buffer.
struct StreamConv {
    act: Act,
    nlc: bool,
    c: usize,
    l: usize,
    n: usize,
    cap: usize,
    pos: usize, // ring slot of the next sample to write
    k: usize,
    d: usize,
    s: usize,
    to: usize,
    g: usize,
    cig: usize,
    cog: usize,
    depthwise: bool,
    zero_bias: bool,
    w: Vec<f32>, // depthwise: [k][c] ; general: [g][cig*k][cog]
    bias: Vec<f32>,
    ring: Vec<f32>, // [cap][c]
    col: Vec<f32>,
    tmp: Vec<f32>, // [to][cout]
}

fn stream_conv(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let w = need(ins, 1)?;
    let wd = cdata(ins, 1, "StreamConv weight")?;
    if x.shape.len() != 3 || x.shape[0] != 1 || w.shape.len() != 3 {
        return Err("StreamConv: 1-D, batch 1 only".into());
    }
    let nlc = a.int("channels_last", 0)? != 0;
    let (c, l) = if nlc { (x.shape[2], x.shape[1]) } else { (x.shape[1], x.shape[2]) };
    let n = a.int("history", 0)? as usize;
    let k = w.shape[2];
    let d = a.ints("dilations")?.map_or(1, |v| v[0] as usize);
    let s = a.ints("strides")?.map_or(1, |v| v[0] as usize);
    let g = a.int("group", 1)? as usize;
    let cout = w.shape[0];
    if c % g != 0 || w.shape[1] != c / g {
        return Err("StreamConv groups".into());
    }
    let keff = d * (k - 1) + 1;
    if n + l < keff {
        return Err("StreamConv: history + input shorter than kernel".into());
    }
    let to = (n + l - keff) / s + 1;
    let (cig, cog) = (c / g, cout / g);
    let depthwise = cig == 1 && cog == 1;
    let wv = if depthwise {
        let mut t = vec![0.0; k * c];
        for ch in 0..c {
            for j in 0..k {
                t[j * c + ch] = wd[ch * k + j];
            }
        }
        t
    } else {
        let kc = cig * k;
        let mut t = vec![0.0; g * kc * cog];
        for gi in 0..g {
            for co_ in 0..cog {
                for kk in 0..kc {
                    t[gi * kc * cog + kk * cog + co_] = wd[(gi * cog + co_) * kc + kk];
                }
            }
        }
        t
    };
    let bias = match ins.get(2) {
        Some(Some(_)) => cdata(ins, 2, "bias")?.to_vec(),
        _ => vec![0.0; cout],
    };
    let cap = n + l;
    ok(
        vec![if nlc { vec![1, to, cout] } else { vec![1, cout, to] }],
        StreamConv { act: act_of(a)?, nlc, c, l, n, cap, pos: 0, k, d, s, to, g, cig, cog, depthwise, zero_bias: bias.iter().all(|&b| b == 0.0), w: wv, bias, ring: vec![0.0; cap * c], col: vec![0.0; to * cig * k], tmp: vec![0.0; to * cout] },
    )
}

impl Op for StreamConv {
    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (c, l, n, cap) = (self.c, self.l, self.n, self.cap);
        let x = ts.d(ins[0]);
        let base = self.pos;
        for t in 0..l {
            let r = &mut self.ring[((base + t) % cap) * c..((base + t) % cap + 1) * c];
            if self.nlc {
                r.copy_from_slice(&x[t * c..(t + 1) * c]);
            } else {
                for ch in 0..c {
                    r[ch] = x[ch * l + t];
                }
            }
        }
        // concat index i -> ring slot (base + cap - n + i) % cap
        let slot = |i: usize| (base + cap - n + i) % cap;
        let cout = self.g * self.cog;
        let y = ts.dm(outs[0]);
        let to = self.to;
        let dst: &mut [f32] = if to == 1 || self.nlc { &mut y[..to * cout] } else { &mut self.tmp[..] };
        if self.depthwise && self.k == 3 && c % 8 == 0 && self.zero_bias {
            for t in 0..to {
                let o = dst[t * c..].as_mut_ptr();
                let r0 = self.ring[slot(t * self.s) * c..].as_ptr();
                let r1 = self.ring[slot(t * self.s + self.d) * c..].as_ptr();
                let r2 = self.ring[slot(t * self.s + 2 * self.d) * c..].as_ptr();
                let w = self.w.as_ptr();
                call!(be, dwconv(r0, r1, r2, w, w.add(c), w.add(2 * c), o, c));
            }
        } else if self.depthwise {
            for t in 0..to {
                let o = &mut dst[t * c..(t + 1) * c];
                o.copy_from_slice(&self.bias);
                for j in 0..self.k {
                    ku::madd(o, &self.w[j * c..(j + 1) * c], &self.ring[slot(t * self.s + j * self.d) * c..][..c]);
                }
            }
        } else {
            let (cig, cog, k) = (self.cig, self.cog, self.k);
            let kc = cig * k;
            for gi in 0..self.g {
                for t in 0..to {
                    for j in 0..k {
                        let r = &self.ring[slot(t * self.s + j * self.d) * c + gi * cig..][..cig];
                        for ci in 0..cig {
                            self.col[t * kc + ci * k + j] = r[ci];
                        }
                    }
                }
                let wt = self.w[gi * kc * cog..].as_ptr();
                let op = dst[gi * cog..].as_mut_ptr();
                call!(be, gemm_ld(wt, cog, cog, kc, self.col.as_ptr(), kc, to, op, cout));
                for t in 0..to {
                    for co_ in 0..cog {
                        dst[t * cout + gi * cog + co_] += self.bias[gi * cog + co_];
                    }
                }
            }
        }
        if to > 1 && !self.nlc {
            for t in 0..to {
                for co_ in 0..cout {
                    y[co_ * to + t] = self.tmp[t * cout + co_];
                }
            }
        }
        self.pos = (base + l) % cap;
        self.act.apply(&mut y[..to * cout], be);
    }
}

// =================================================================== ConvTranspose

struct ConvTOp {
    nlc: bool,
    g: usize,
    cig: usize,
    cog: usize,
    kk: usize, // kh*kw
    p: usize,  // input positions
    po: usize, // output positions per channel
    batch: usize,
    w: Vec<f32>, // [cin][cog*kk] (ONNX layout, contiguous per group)
    bias: Vec<f32>,
    dst: Vec<u32>, // [p][kk] -> output position or PADI (cropped)
    xt: Vec<f32>,  // [p][cig]
    z: Vec<f32>,   // [p][cog*kk]
}

fn conv_transpose(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let w = need(ins, 1)?;
    let wd = cdata(ins, 1, "ConvTranspose weight")?;
    let nd = w.shape.len() - 2;
    if !(nd == 1 || nd == 2) || x.shape.len() != nd + 2 {
        return Err("ConvTranspose rank".into());
    }
    let (st, dl, pd) = spatial(a, nd, &w.shape[2..])?;
    let op = a.ints("output_padding")?.unwrap_or(vec![0; nd]);
    if a.has("output_shape") {
        return Err("ConvTranspose output_shape not supported".into());
    }
    let g = a.int("group", 1)? as usize;
    let nlc = a.int("channels_last", 0)? != 0;
    if nlc && nd != 1 {
        return Err("channels_last ConvTranspose: 1-D only".into());
    }
    let (batch, cin) = (x.shape[0], if nlc { x.shape[2] } else { x.shape[1] });
    if w.shape[0] != cin || cin % g != 0 {
        return Err("ConvTranspose weight/groups".into());
    }
    let cog = w.shape[1];
    let (h, wi) = if nd == 1 { (1, if nlc { x.shape[1] } else { x.shape[2] }) } else { (x.shape[2], x.shape[3]) };
    let (kh, kw) = if nd == 1 { (1, w.shape[2]) } else { (w.shape[2], w.shape[3]) };
    let (sh, sw) = if nd == 1 { (1, st[0]) } else { (st[0], st[1]) };
    let (dh, dw) = if nd == 1 { (1, dl[0]) } else { (dl[0], dl[1]) };
    let (pt, pl, pb, pr) = if nd == 1 { (0, pd[0], 0, pd[1]) } else { (pd[0], pd[1], pd[2], pd[3]) };
    let (oph, opw) = if nd == 1 { (0, op[0] as usize) } else { (op[0] as usize, op[1] as usize) };
    let fh = (h - 1) * sh + dh * (kh - 1) + 1 + oph;
    let fw = (wi - 1) * sw + dw * (kw - 1) + 1 + opw;
    if fh < pt + pb || fw < pl + pr {
        return Err("ConvTranspose pads too large".into());
    }
    let (oh, ow) = (fh - pt - pb, fw - pl - pr);
    let kk = kh * kw;
    let p = h * wi;
    let mut dst = vec![PADI; p * kk];
    for iy in 0..h {
        for ix in 0..wi {
            for i in 0..kh {
                for j in 0..kw {
                    let y = (iy * sh + i * dh) as isize - pt as isize;
                    let xx = (ix * sw + j * dw) as isize - pl as isize;
                    if y >= 0 && (y as usize) < oh && xx >= 0 && (xx as usize) < ow {
                        dst[(iy * wi + ix) * kk + i * kw + j] = (y as usize * ow + xx as usize) as u32;
                    }
                }
            }
        }
    }
    let co = cog * g;
    let bias = match ins.get(2) {
        Some(Some(_)) => cdata(ins, 2, "ConvTranspose bias")?.to_vec(),
        _ => vec![0.0; co],
    };
    let mut out = vec![batch, co];
    if nd == 2 {
        out.push(oh);
    }
    out.push(ow);
    if nlc {
        out = vec![batch, ow, co];
    }
    let cig = cin / g;
    ok(vec![out], ConvTOp { nlc, g, cig, cog, kk, p, po: oh * ow, batch, w: wd.to_vec(), bias, dst, xt: vec![0.0; p * cig], z: vec![0.0; p * cog * kk] })
}

impl Op for ConvTOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (g, cig, cog, kk, p, po) = (self.g, self.cig, self.cog, self.kk, self.p, self.po);
        let xall = ts.d(ins[0]);
        let yall = ts.dm(outs[0]);
        let m = cog * kk;
        if self.nlc {
            let (cin, co) = (g * cig, g * cog);
            for nb in 0..self.batch {
                let x = &xall[nb * p * cin..(nb + 1) * p * cin];
                let y = &mut yall[nb * po * co..(nb + 1) * po * co];
                for r in 0..po {
                    y[r * co..(r + 1) * co].copy_from_slice(&self.bias);
                }
                for gi in 0..g {
                    let wt = self.w[gi * cig * m..].as_ptr();
                    call!(be, gemm_ld(wt, m, m, cig, x[gi * cig..].as_ptr(), cin, p, self.z.as_mut_ptr(), m));
                    for pp in 0..p {
                        let zr = &self.z[pp * m..(pp + 1) * m];
                        let dr = &self.dst[pp * kk..(pp + 1) * kk];
                        for kq in 0..kk {
                            let d = dr[kq];
                            if d == PADI {
                                continue;
                            }
                            let yr = &mut y[d as usize * co + gi * cog..d as usize * co + (gi + 1) * cog];
                            for c in 0..cog {
                                yr[c] += zr[c * kk + kq];
                            }
                        }
                    }
                }
            }
            return;
        }
        for nb in 0..self.batch {
            let x = &xall[nb * g * cig * p..];
            let y = &mut yall[nb * g * cog * po..(nb + 1) * g * cog * po];
            for c in 0..g * cog {
                let b = self.bias[c];
                y[c * po..(c + 1) * po].iter_mut().for_each(|v| *v = b);
            }
            for gi in 0..g {
                let xg = &x[gi * cig * p..(gi + 1) * cig * p];
                for ci in 0..cig {
                    for pp in 0..p {
                        self.xt[pp * cig + ci] = xg[ci * p + pp];
                    }
                }
                let wt = self.w[gi * cig * m..].as_ptr();
                call!(be, gemm_ld(wt, m, m, cig, self.xt.as_ptr(), cig, p, self.z.as_mut_ptr(), m));
                let yg = &mut y[gi * cog * po..(gi + 1) * cog * po];
                for pp in 0..p {
                    let zr = &self.z[pp * m..(pp + 1) * m];
                    let dr = &self.dst[pp * kk..(pp + 1) * kk];
                    for c in 0..cog {
                        let yc = &mut yg[c * po..(c + 1) * po];
                        let zc = &zr[c * kk..(c + 1) * kk];
                        for k in 0..kk {
                            let d = dr[k];
                            if d != PADI {
                                yc[d as usize] += zc[k];
                            }
                        }
                    }
                }
            }
        }
    }
}

// =================================================================== GRU / LSTM

struct Rnn {
    lstm: bool,
    lbr: bool,
    dirs: usize,
    s: usize,
    bt: usize,
    i: usize,
    h: usize,
    wt: Vec<f32>, // [dirs][I][G*H]
    rt: Vec<f32>, // [dirs][H][G*H]
    wb: Vec<f32>, // [dirs][G*H]
    rb: Vec<f32>, // [dirs][G*H]
    gx: Vec<f32>, // [S*Bt][G*H]
    gh: Vec<f32>, // [Bt][G*H]
    hv: Vec<f32>, // [Bt][H]
    cv: Vec<f32>,
    rh: Vec<f32>,
    has_h0: bool,
    has_c0: bool,
}

fn rnn(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let lstm = op == "LSTM";
    let gn = if lstm { 4 } else { 3 };
    let x = need(ins, 0)?;
    let w = cdata(ins, 1, "W")?;
    let r = cdata(ins, 2, "R")?;
    let hs = a.int("hidden_size", 0)? as usize;
    let dirs = match a.str("direction", "forward") {
        "forward" => 1,
        "bidirectional" => 2,
        _ => return Err("RNN direction reverse not supported".into()),
    };
    if a.int("layout", 0)? != 0 || a.has("activations") || a.has("clip") || a.int("input_forget", 0)? != 0 {
        return Err("unsupported RNN attributes".into());
    }
    if x.shape.len() != 3 {
        return Err(format!("RNN X must be [S,B,I], got {:?}", x.shape));
    }
    let (s, bt, ni) = (x.shape[0], x.shape[1], x.shape[2]);
    let gh = gn * hs;
    if w.len() != dirs * gh * ni || r.len() != dirs * gh * hs {
        return Err("RNN W/R shape".into());
    }
    if ins.get(4).map_or(false, |v| v.is_some()) {
        return Err("sequence_lens not supported".into());
    }
    if lstm && ins.get(7).map_or(false, |v| v.is_some()) {
        return Err("LSTM peepholes not supported".into());
    }
    let mut wt = vec![0.0; dirs * ni * gh];
    let mut rt = vec![0.0; dirs * hs * gh];
    for d in 0..dirs {
        for g in 0..gh {
            for k in 0..ni {
                wt[d * ni * gh + k * gh + g] = w[(d * gh + g) * ni + k];
            }
            for k in 0..hs {
                rt[d * hs * gh + k * gh + g] = r[(d * gh + g) * hs + k];
            }
        }
    }
    let (mut wb, mut rb) = (vec![0.0; dirs * gh], vec![0.0; dirs * gh]);
    if let Some(Some(_)) = ins.get(3) {
        let b = cdata(ins, 3, "B")?;
        if b.len() != dirs * 2 * gh {
            return Err("RNN B shape".into());
        }
        for d in 0..dirs {
            wb[d * gh..(d + 1) * gh].copy_from_slice(&b[d * 2 * gh..d * 2 * gh + gh]);
            rb[d * gh..(d + 1) * gh].copy_from_slice(&b[d * 2 * gh + gh..(d + 1) * 2 * gh]);
        }
    }
    let lbr = a.int("linear_before_reset", 0)? != 0;
    let has_h0 = ins.get(5).map_or(false, |v| v.is_some());
    let has_c0 = lstm && ins.get(6).map_or(false, |v| v.is_some());
    for (k, nm) in [(5usize, "initial_h"), (6, "initial_c")] {
        if let Some(Some(v)) = ins.get(k) {
            if numel(v.shape) != dirs * bt * hs {
                return Err(format!("RNN {nm} shape {:?}", v.shape));
            }
        }
    }
    let mut outs = vec![vec![s, dirs, bt, hs], vec![dirs, bt, hs]];
    if lstm {
        outs.push(vec![dirs, bt, hs]);
    }
    ok(
        outs,
        Rnn {
            lstm, lbr, dirs, s, bt, i: ni, h: hs, wt, rt, wb, rb,
            gx: vec![0.0; s * bt * gh], gh: vec![0.0; bt * gh], hv: vec![0.0; bt * hs], cv: vec![0.0; bt * hs], rh: vec![0.0; bt * hs],
            has_h0, has_c0,
        },
    )
}

impl Op for Rnn {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let (s, bt, ni, h, dirs) = (self.s, self.bt, self.i, self.h, self.dirs);
        let gn = if self.lstm { 4 } else { 3 };
        let gh = gn * h;
        let x = ts.d(ins[0]);
        let y = ts.dm(outs[0]);
        for d in 0..dirs {
            let wt = self.wt[d * ni * gh..].as_ptr();
            let rt = &self.rt[d * h * gh..(d + 1) * h * gh];
            let wb = &self.wb[d * gh..(d + 1) * gh];
            let rb = &self.rb[d * gh..(d + 1) * gh];
            call!(be, gemm_ld(wt, gh, gh, ni, x.as_ptr(), ni, s * bt, self.gx.as_mut_ptr(), gh));
            if self.has_h0 {
                self.hv.copy_from_slice(&ts.d(ins[5])[d * bt * h..(d + 1) * bt * h]);
            } else {
                self.hv.iter_mut().for_each(|v| *v = 0.0);
            }
            if self.lstm {
                if self.has_c0 {
                    self.cv.copy_from_slice(&ts.d(ins[6])[d * bt * h..(d + 1) * bt * h]);
                } else {
                    self.cv.iter_mut().for_each(|v| *v = 0.0);
                }
            }
            for step in 0..s {
                let t = if d == 0 { step } else { s - 1 - step };
                let gxt = &mut self.gx[t * bt * gh..(t + 1) * bt * gh];
                for b in 0..bt {
                    for (v, c) in gxt[b * gh..(b + 1) * gh].iter_mut().zip(wb) {
                        *v += c;
                    }
                }
                let full = self.lstm || self.lbr;
                let m = if full { gh } else { 2 * h };
                if h <= 32 {
                    for b in 0..bt {
                        ku::matvec_bias(rt, gh, m, &self.hv[b * h..(b + 1) * h], rb, &mut self.gh[b * gh..(b + 1) * gh]);
                    }
                } else {
                    call!(be, gemm_ld(rt.as_ptr(), gh, m, h, self.hv.as_ptr(), h, bt, self.gh.as_mut_ptr(), gh));
                    for b in 0..bt {
                        for j in 0..m {
                            self.gh[b * gh + j] += rb[j];
                        }
                    }
                }
                for b in 0..bt {
                    let gx = &mut gxt[b * gh..(b + 1) * gh];
                    let ghb = &self.gh[b * gh..(b + 1) * gh];
                    let hb = &mut self.hv[b * h..(b + 1) * h];
                    if self.lstm {
                        ku::lstm(gx, ghb, hb, &mut self.cv[b * h..(b + 1) * h]);
                    } else if self.lbr {
                        ku::gru_lbr(gx, ghb, hb);
                    } else {
                        for j in 0..2 * h {
                            gx[j] = sigmoid(gx[j] + ghb[j]);
                        }
                        for j in 0..h {
                            self.rh[b * h + j] = gx[h + j] * hb[j];
                        }
                    }
                }
                if !self.lstm && !self.lbr {
                    let rn = rt[2 * h..].as_ptr();
                    call!(be, gemm_ld(rn, gh, h, h, self.rh.as_ptr(), h, bt, self.gh[2 * h..].as_mut_ptr(), gh));
                    for b in 0..bt {
                        let gx = &gxt[b * gh..(b + 1) * gh];
                        for j in 0..h {
                            let n = ku::tanhf(gx[2 * h + j] + self.gh[b * gh + 2 * h + j] + rb[2 * h + j]);
                            let z = gx[j];
                            self.hv[b * h + j] = (1.0 - z) * n + z * self.hv[b * h + j];
                        }
                    }
                }
                for b in 0..bt {
                    let o = ((t * dirs + d) * bt + b) * h;
                    y[o..o + h].copy_from_slice(&self.hv[b * h..(b + 1) * h]);
                }
            }
            ts.dm(outs[1])[d * bt * h..(d + 1) * bt * h].copy_from_slice(&self.hv);
            if self.lstm && outs.len() > 2 {
                ts.dm(outs[2])[d * bt * h..(d + 1) * bt * h].copy_from_slice(&self.cv);
            }
        }
    }
}

// =================================================================== norms

#[derive(PartialEq, Clone, Copy)]
enum NK {
    Channel,
    Layer,
    Instance,
    Batch,
}

struct NormOp {
    sc: Vec<f32>,
    k: NK,
    outer: usize,
    len: usize,
    inner: usize,
    eps: f32,
    g: Vec<f32>,
    b: Vec<f32>,
}

fn norm(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let nd = s.len();
    let eps = a.float("epsilon", 1e-5)?;
    let get = |i: usize, n: usize, d: f32| -> R<Vec<f32>> {
        match ins.get(i) {
            Some(Some(v)) => {
                let dd = v.data.ok_or("norm parameters must be constant")?;
                if dd.len() != n {
                    return Err(format!("norm parameter {i}: {} values, expected {n}", dd.len()));
                }
                Ok(dd.to_vec())
            }
            _ => Ok(vec![d; n]),
        }
    };
    let (k, outer, len, inner, g, b) = match op {
        "ChannelNorm" => {
            let ax = norm_axis(a.int("axis", 1)?, nd)?;
            let (o, l, i) = (numel(&s[..ax]), s[ax], numel(&s[ax + 1..]));
            (NK::Channel, o, l, i, get(1, l, 1.0)?, get(2, l, 0.0)?)
        }
        "LayerNormalization" => {
            let ax = norm_axis(a.int("axis", -1)?, nd)?;
            let (o, l) = (numel(&s[..ax]), numel(&s[ax..]));
            (NK::Layer, o, l, 1, get(1, l, 1.0)?, get(2, l, 0.0)?)
        }
        "InstanceNormalization" => {
            let c = s[1];
            (NK::Instance, s[0] * c, numel(&s[2..]), 1, get(1, c, 1.0)?, get(2, c, 0.0)?)
        }
        _ => {
            // BatchNormalization (inference): y = (x - mean)/sqrt(var+eps)*scale + bias, per channel (axis 1)
            let c = s[1];
            let sc = get(1, c, 1.0)?;
            let bi = get(2, c, 0.0)?;
            let mu = get(3, c, 0.0)?;
            let var = get(4, c, 1.0)?;
            let gg: Vec<f32> = (0..c).map(|i| sc[i] / (var[i] + eps).sqrt()).collect();
            let bb: Vec<f32> = (0..c).map(|i| bi[i] - mu[i] * gg[i]).collect();
            (NK::Batch, s[0], c, numel(&s[2..]), gg, bb)
        }
    };
    ok(vec![s.to_vec()], NormOp { sc: vec![0.0; 2 * inner], k, outer, len, inner, eps, g, b })
}

impl Op for NormOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend) {
        let x = ts.d(ins[0]);
        let y = ts.dm(outs[0]);
        let (len, inner, eps) = (self.len, self.inner, self.eps);
        match self.k {
            NK::Batch => {
                for o in 0..self.outer {
                    for c in 0..len {
                        let (g, b) = (self.g[c], self.b[c]);
                        let base = (o * len + c) * inner;
                        for i in 0..inner {
                            y[base + i] = x[base + i] * g + b;
                        }
                    }
                }
            }
            NK::Channel if inner == 1 && len % 8 == 0 => {
                for o in 0..self.outer {
                    let p = o * len;
                    call!(be, norm(x[p..].as_ptr(), y[p..].as_mut_ptr(), self.g.as_ptr(), self.b.as_ptr(), len, eps));
                }
            }
            NK::Channel => {
                // reduce over `len` with stride `inner`, vectorized across the contiguous inner
                // axis; per element the arithmetic order of the MIL pattern (sequential over c)
                let n = len as f32;
                let (m, q) = self.sc.split_at_mut(inner);
                for o in 0..self.outer {
                    let xb = &x[o * len * inner..(o + 1) * len * inner];
                    let yb = &mut y[o * len * inner..(o + 1) * len * inner];
                    m.iter_mut().for_each(|v| *v = 0.0);
                    q.iter_mut().for_each(|v| *v = 0.0);
                    for c in 0..len {
                        let r = &xb[c * inner..(c + 1) * inner];
                        for i in 0..inner {
                            m[i] += r[i];
                        }
                    }
                    m.iter_mut().for_each(|v| *v /= n);
                    for c in 0..len {
                        let r = &xb[c * inner..(c + 1) * inner];
                        for i in 0..inner {
                            let d = r[i] - m[i];
                            q[i] += d * d;
                        }
                    }
                    q.iter_mut().for_each(|v| *v = (*v / n + eps).sqrt());
                    for c in 0..len {
                        let (g, b) = (self.g[c], self.b[c]);
                        let r = &xb[c * inner..(c + 1) * inner];
                        let w = &mut yb[c * inner..(c + 1) * inner];
                        for i in 0..inner {
                            w[i] = g * (r[i] - m[i]) / q[i] + b;
                        }
                    }
                }
            }
            NK::Layer | NK::Instance => {
                let n = len as f32;
                for o in 0..self.outer {
                    let xs = &x[o * len..(o + 1) * len];
                    let mean = xs.iter().sum::<f32>() / n;
                    let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                    let inv = 1.0 / (var + eps).sqrt();
                    let ys = &mut y[o * len..(o + 1) * len];
                    if self.k == NK::Layer {
                        for i in 0..len {
                            ys[i] = (xs[i] - mean) * inv * self.g[i] + self.b[i];
                        }
                    } else {
                        let c = o % self.g.len();
                        for i in 0..len {
                            ys[i] = (xs[i] - mean) * inv * self.g[c] + self.b[c];
                        }
                    }
                }
            }
        }
    }
}

// =================================================================== reduce

struct Reduce {
    kind: u8, // 0 mean, 1 sum, 2 max
    map: Vec<u32>,
    nout: usize,
    scale: f32,
}

fn reduce(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let nd = s.len();
    if ins.len() > 1 && ins[1].is_some() {
        return Err("Reduce axes must be an attribute".into());
    }
    let axes: Vec<usize> = match a.ints("axes")? {
        Some(v) => v.iter().map(|&q| norm_axis(q, nd)).collect::<R<_>>()?,
        None => (0..nd).collect(),
    };
    let kd = a.int("keepdims", 1)? != 0;
    let oshape_keep: Vec<usize> = (0..nd).map(|i| if axes.contains(&i) { 1 } else { s[i] }).collect();
    let ost = strides(&oshape_keep);
    let ist = strides(s);
    let n = numel(s);
    let mut map = vec![0u32; n];
    for (f, m) in map.iter_mut().enumerate() {
        let mut o = 0;
        for d in 0..nd {
            let c = (f / ist[d]) % s[d];
            if !axes.contains(&d) {
                o += c * ost[d];
            }
        }
        *m = o as u32;
    }
    let nout = numel(&oshape_keep);
    let cnt = (n / nout.max(1)) as f32;
    let out = if kd { oshape_keep } else { (0..nd).filter(|i| !axes.contains(i)).map(|i| s[i]).collect() };
    let kind = match op {
        "ReduceMean" => 0,
        "ReduceSum" => 1,
        _ => 2,
    };
    ok(vec![out], Reduce { kind, map, nout, scale: 1.0 / cnt })
}

impl Op for Reduce {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let x = ts.d(ins[0]);
        let y = &mut ts.dm(outs[0])[..self.nout];
        let init = if self.kind == 2 { f32::NEG_INFINITY } else { 0.0 };
        y.iter_mut().for_each(|v| *v = init);
        if self.kind == 2 {
            for (i, &m) in self.map.iter().enumerate() {
                let o = &mut y[m as usize];
                *o = o.max(x[i]);
            }
        } else {
            for (i, &m) in self.map.iter().enumerate() {
                y[m as usize] += x[i];
            }
            if self.kind == 0 {
                y.iter_mut().for_each(|v| *v *= self.scale);
            }
        }
    }
}

// =================================================================== data movement

/// out[i] = in[map[i]] (PADI -> fill); maps made of long contiguous runs are executed as
/// block copies (dst, src, len; src PADI = fill).
struct GatherMap {
    map: Vec<u32>,
    runs: Vec<(u32, u32, u32)>,
    fill: f32,
}

fn gather_op(map: Vec<u32>, fill: f32) -> GatherMap {
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for (i, &m) in map.iter().enumerate() {
        if let Some(r) = runs.last_mut() {
            let cont = if r.1 == PADI { m == PADI } else { m != PADI && m == r.1 + r.2 };
            if cont && r.0 + r.2 == i as u32 {
                r.2 += 1;
                continue;
            }
        }
        runs.push((i as u32, m, 1));
    }
    if runs.len() * 8 > map.len() {
        runs.clear();
    }
    GatherMap { map, runs, fill }
}

impl Op for GatherMap {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        if self.runs.is_empty() {
            ku::gather(&self.map, ts.d(ins[0]), ts.dm(outs[0]), self.fill);
            return;
        }
        let (x, y) = (ts.d(ins[0]), ts.dm(outs[0]));
        for &(d, s, l) in &self.runs {
            let (d, l) = (d as usize, l as usize);
            if s == PADI {
                y[d..d + l].iter_mut().for_each(|v| *v = self.fill);
            } else {
                y[d..d + l].copy_from_slice(&x[s as usize..s as usize + l]);
            }
        }
    }
}

fn gather_like(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let nd = s.len();
    let ist = strides(s);
    let mut fill = 0.0;
    let (out, map): (Vec<usize>, Vec<u32>) = match op {
        "Transpose" => {
            let perm: Vec<usize> = match a.ints("perm")? {
                Some(p) => p.iter().map(|&v| v as usize).collect(),
                None => (0..nd).rev().collect(),
            };
            if perm.len() != nd {
                return Err("Transpose perm rank".into());
            }
            let out: Vec<usize> = perm.iter().map(|&p| s[p]).collect();
            let ost = strides(&out);
            let n = numel(&out);
            let map = (0..n)
                .map(|f| {
                    let mut src = 0;
                    for d in 0..nd {
                        src += ((f / ost[d]) % out[d]) * ist[perm[d]];
                    }
                    src as u32
                })
                .collect();
            (out, map)
        }
        "Slice" => {
            if ins.len() > 1 && ins[1].is_some() {
                return Err("Slice starts/ends must be attributes".into());
            }
            let starts = a.ints("starts")?.ok_or("Slice: starts")?;
            let ends = a.ints("ends")?.ok_or("Slice: ends")?;
            let axes = a.ints("axes")?.unwrap_or((0..starts.len() as i64).collect());
            let steps = a.ints("steps")?.unwrap_or(vec![1; starts.len()]);
            let mut beg = vec![0i64; nd];
            let mut stp = vec![1i64; nd];
            let mut out = s.to_vec();
            for k in 0..starts.len() {
                let ax = norm_axis(axes[k], nd)?;
                let dim = s[ax] as i64;
                let st = steps[k];
                if st == 0 {
                    return Err("Slice step 0".into());
                }
                let clampi = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
                let (mut b, mut e) = (starts[k], ends[k]);
                if b < 0 {
                    b += dim;
                }
                if e < 0 {
                    e += dim;
                }
                let (b, e) = if st > 0 { (clampi(b, 0, dim), clampi(e, 0, dim)) } else { (clampi(b, 0, dim - 1), clampi(e, -1, dim - 1)) };
                let len = if st > 0 { ((e - b).max(0) + st - 1) / st } else { ((b - e).max(0) + (-st) - 1) / (-st) };
                beg[ax] = b;
                stp[ax] = st;
                out[ax] = len as usize;
            }
            let ost = strides(&out);
            let n = numel(&out);
            let map = (0..n)
                .map(|f| {
                    let mut src = 0i64;
                    for d in 0..nd {
                        let c = ((f / ost[d]) % out[d]) as i64;
                        src += (beg[d] + c * stp[d]) * ist[d] as i64;
                    }
                    src as u32
                })
                .collect();
            (out, map)
        }
        "Gather" => {
            let idx = cdata(ins, 1, "Gather indices")?;
            let ishape = need(ins, 1)?.shape;
            let ax = norm_axis(a.int("axis", 0)?, nd)?;
            let mut out = s[..ax].to_vec();
            out.extend_from_slice(ishape);
            out.extend_from_slice(&s[ax + 1..]);
            let outer = numel(&s[..ax]);
            let inner = numel(&s[ax + 1..]);
            let mut map = Vec::with_capacity(numel(&out));
            for o in 0..outer {
                for &iv in idx {
                    let mut i = iv as i64;
                    if i < 0 {
                        i += s[ax] as i64;
                    }
                    if i < 0 || i as usize >= s[ax] {
                        return Err("Gather index out of range".into());
                    }
                    for k in 0..inner {
                        map.push(((o * s[ax] + i as usize) * inner + k) as u32);
                    }
                }
            }
            (out, map)
        }
        "Pad" => {
            if ins.len() > 1 && ins[1].is_some() {
                return Err("Pad pads must be an attribute".into());
            }
            let pads = a.ints("pads")?.ok_or("Pad: pads")?;
            if pads.len() != 2 * nd {
                return Err("Pad: pads rank".into());
            }
            let mode = a.str("mode", "constant");
            fill = a.float("value", 0.0)?;
            let out: Vec<usize> = (0..nd).map(|d| (s[d] as i64 + pads[d] + pads[d + nd]) as usize).collect();
            let ost = strides(&out);
            let n = numel(&out);
            let mut map = Vec::with_capacity(n);
            for f in 0..n {
                let mut src = 0i64;
                let mut pad = false;
                for d in 0..nd {
                    let mut c = ((f / ost[d]) % out[d]) as i64 - pads[d];
                    let dim = s[d] as i64;
                    if c < 0 || c >= dim {
                        match mode {
                            "reflect" => {
                                while c < 0 || c >= dim {
                                    if c < 0 {
                                        c = -c;
                                    }
                                    if c >= dim {
                                        c = 2 * (dim - 1) - c;
                                    }
                                }
                            }
                            "edge" => c = c.max(0).min(dim - 1),
                            _ => pad = true,
                        }
                    }
                    src += c * ist[d] as i64;
                }
                map.push(if pad { PADI } else { src as u32 });
            }
            (out, map)
        }
        "Expand" | "Tile" => {
            let out: Vec<usize> = if op == "Expand" {
                let sh: Vec<usize> = a.ints("shape")?.ok_or("Expand: shape")?.iter().map(|&v| v as usize).collect();
                bcast(s, &sh)?
            } else {
                let rep = a.ints("repeats")?.ok_or("Tile: repeats")?;
                (0..nd).map(|d| s[d] * rep[d] as usize).collect()
            };
            let ost = strides(&out);
            let n = numel(&out);
            let off = out.len() - nd;
            let map = (0..n)
                .map(|f| {
                    let mut src = 0;
                    for d in 0..nd {
                        let c = (f / ost[d + off]) % out[d + off];
                        src += (c % s[d]) * ist[d];
                    }
                    src as u32
                })
                .collect();
            (out, map)
        }
        _ => unreachable!(),
    };
    ok(vec![out], gather_op(map, fill))
}

struct ScatterOp {
    runs: Vec<(u32, u32, u32)>, // (dst, src, len)
    n: usize,
}

fn scatter_nd(ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let idx = cdata(ins, 1, "ScatterND indices")?;
    let ishape = need(ins, 1)?.shape;
    let s = x.shape;
    let k = *ishape.last().ok_or("ScatterND indices rank")?;
    let rows = numel(&ishape[..ishape.len() - 1]);
    let inner = numel(&s[k..]);
    let st = strides(s);
    let mut dst = Vec::with_capacity(rows * inner);
    for r in 0..rows {
        let mut base = 0;
        for d in 0..k {
            let mut v = idx[r * k + d] as i64;
            if v < 0 {
                v += s[d] as i64;
            }
            base += v as usize * st[d];
        }
        for j in 0..inner {
            dst.push((base + j) as u32);
        }
    }
    let u = need(ins, 2)?;
    if numel(u.shape) != dst.len() {
        return Err("ScatterND updates shape".into());
    }
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for (i, &d) in dst.iter().enumerate() {
        if let Some(r) = runs.last_mut() {
            if r.0 + r.2 == d && r.1 + r.2 == i as u32 {
                r.2 += 1;
                continue;
            }
        }
        runs.push((d, i as u32, 1));
    }
    ok(vec![s.to_vec()], ScatterOp { runs, n: numel(s) })
}

impl Op for ScatterOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let y = ts.dm(outs[0]);
        y[..self.n].copy_from_slice(&ts.d(ins[0])[..self.n]);
        let u = ts.d(ins[2]);
        for &(d, s, l) in &self.runs {
            y[d as usize..(d + l) as usize].copy_from_slice(&u[s as usize..(s + l) as usize]);
        }
    }
}

struct ConcatOp {
    outer: usize,
    blocks: Vec<usize>, // per input: shape[axis]*inner
    total: usize,
}

fn concat(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let xs: Vec<&In> = ins.iter().filter_map(|v| v.as_ref()).collect();
    if xs.is_empty() || ins.iter().any(|v| v.is_none()) {
        return Err("Concat inputs".into());
    }
    let nd = xs[0].shape.len();
    let ax = norm_axis(a.int("axis", 0)?, nd)?;
    let mut out = xs[0].shape.to_vec();
    out[ax] = 0;
    for x in &xs {
        if x.shape.len() != nd || (0..nd).any(|d| d != ax && x.shape[d] != xs[0].shape[d]) {
            return Err(format!("Concat shape mismatch {:?} vs {:?}", x.shape, xs[0].shape));
        }
        out[ax] += x.shape[ax];
    }
    let inner = numel(&out[ax + 1..]);
    let blocks: Vec<usize> = xs.iter().map(|x| x.shape[ax] * inner).collect();
    let total = blocks.iter().sum();
    ok(vec![out.clone()], ConcatOp { outer: numel(&out[..ax]), blocks, total })
}

impl Op for ConcatOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let y = ts.dm(outs[0]);
        let mut off = 0;
        for (k, &b) in self.blocks.iter().enumerate() {
            if b == 0 {
                continue;
            }
            let x = ts.d(ins[k]);
            for o in 0..self.outer {
                y[o * self.total + off..o * self.total + off + b].copy_from_slice(&x[o * b..(o + 1) * b]);
            }
            off += b;
        }
    }
}

struct SplitOp {
    outer: usize,
    blocks: Vec<usize>,
    total: usize,
}

fn split(a: &Attrs, ins: &[Option<In>], nout: usize) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let ax = norm_axis(a.int("axis", 0)?, s.len())?;
    let sp: Vec<usize> = match a.ints("split")? {
        Some(v) => v.iter().map(|&q| q as usize).collect(),
        None => vec![s[ax] / nout; nout],
    };
    if sp.iter().sum::<usize>() != s[ax] {
        return Err("Split sizes".into());
    }
    let inner = numel(&s[ax + 1..]);
    let outs: Vec<Vec<usize>> = sp
        .iter()
        .map(|&k| {
            let mut o = s.to_vec();
            o[ax] = k;
            o
        })
        .collect();
    ok(outs, SplitOp { outer: numel(&s[..ax]), blocks: sp.iter().map(|k| k * inner).collect(), total: s[ax] * inner })
}

impl Op for SplitOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let x = ts.d(ins[0]);
        let mut off = 0;
        for (k, &b) in self.blocks.iter().enumerate() {
            if k < outs.len() {
                let y = ts.dm(outs[k]);
                for o in 0..self.outer {
                    y[o * b..(o + 1) * b].copy_from_slice(&x[o * self.total + off..o * self.total + off + b]);
                }
            }
            off += b;
        }
    }
}

struct CopyOp {
    n: usize,
}

impl Op for CopyOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        ts.dm(outs[0])[..self.n].copy_from_slice(&ts.d(ins[0])[..self.n]);
    }
}

fn reshape(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let n = numel(s);
    if ins.len() > 1 && ins[1].is_some() {
        return Err(format!("{op}: shape/axes must be attributes"));
    }
    let out: Vec<usize> = match op {
        "Reshape" => {
            let sh = a.ints("shape")?.ok_or("Reshape: shape")?;
            let mut o: Vec<i64> = sh.iter().enumerate().map(|(i, &v)| if v == 0 && a.int("allowzero", 0).unwrap_or(0) == 0 { s[i] as i64 } else { v }).collect();
            let known: i64 = o.iter().filter(|&&v| v >= 0).product();
            if let Some(p) = o.iter().position(|&v| v == -1) {
                o[p] = n as i64 / known.max(1);
            }
            o.iter().map(|&v| v as usize).collect()
        }
        "Flatten" => {
            let ax = a.int("axis", 1)?;
            let ax = if ax < 0 { ax + s.len() as i64 } else { ax } as usize;
            vec![numel(&s[..ax]), numel(&s[ax..])]
        }
        "Squeeze" => {
            let nd = s.len();
            match a.ints("axes")? {
                Some(ax) => {
                    let ax: Vec<usize> = ax.iter().map(|&q| norm_axis(q, nd)).collect::<R<_>>()?;
                    (0..nd).filter(|i| !ax.contains(i)).map(|i| s[i]).collect()
                }
                None => s.iter().copied().filter(|&d| d != 1).collect(),
            }
        }
        _ => {
            // Unsqueeze
            let ax = a.ints("axes")?.ok_or("Unsqueeze: axes")?;
            let nd = s.len() + ax.len();
            let mut axn: Vec<usize> = ax.iter().map(|&q| norm_axis(q, nd)).collect::<R<_>>()?;
            axn.sort_unstable();
            let mut o = s.to_vec();
            for q in axn {
                o.insert(q, 1);
            }
            o
        }
    };
    if numel(&out) != n {
        return Err(format!("{op}: {s:?} -> {out:?} changes element count"));
    }
    ok(vec![out], CopyOp { n })
}

// =================================================================== Rfft / Irfft

struct FftOp {
    inv: bool,
    rows: usize,
    n: usize,
    plan: vi_audio::fft::Rfft,
}

fn fft(op: &str, a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    if op == "Rfft" {
        let n = *s.last().ok_or("Rfft rank")?;
        let mut out = s[..s.len() - 1].to_vec();
        out.push(n / 2 + 1);
        out.push(2);
        ok(vec![out], FftOp { inv: false, rows: numel(&s[..s.len() - 1]), n, plan: vi_audio::fft::Rfft::new(n) })
    } else {
        let n = a.int("n", 0)? as usize;
        if s.len() < 2 || s[s.len() - 1] != 2 || s[s.len() - 2] != n / 2 + 1 {
            return Err("Irfft input must be [..., n/2+1, 2]".into());
        }
        let mut out = s[..s.len() - 2].to_vec();
        out.push(n);
        ok(vec![out], FftOp { inv: true, rows: numel(&s[..s.len() - 2]), n, plan: vi_audio::fft::Rfft::new(n) })
    }
}

impl Op for FftOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let x = ts.d(ins[0]);
        let y = ts.dm(outs[0]);
        let (n, nb) = (self.n, (self.n / 2 + 1) * 2);
        for r in 0..self.rows {
            if self.inv {
                self.plan.inverse(&x[r * nb..(r + 1) * nb], &mut y[r * n..(r + 1) * n]);
            } else {
                self.plan.forward(&x[r * n..(r + 1) * n], &mut y[r * nb..(r + 1) * nb]);
            }
        }
    }
}

// =================================================================== MaxPool (1-D / 2-D)

struct MaxPoolOp {
    map: Vec<u32>, // [out][k] input indices (PADI = padding, ignored)
    k: usize,
    nout: usize,
}

fn maxpool(a: &Attrs, ins: &[Option<In>]) -> R<Prepared> {
    let x = need(ins, 0)?;
    let s = x.shape;
    let nd = s.len() - 2;
    if !(nd == 1 || nd == 2) || a.str("auto_pad", "NOTSET") != "NOTSET" || a.int("ceil_mode", 0)? != 0 {
        return Err("MaxPool: 1-D/2-D, explicit pads, no ceil_mode".into());
    }
    let ks: Vec<usize> = a.ints("kernel_shape")?.ok_or("MaxPool: kernel_shape")?.iter().map(|&v| v as usize).collect();
    let st: Vec<usize> = a.ints("strides")?.map(|v| v.iter().map(|&q| q as usize).collect()).unwrap_or(vec![1; nd]);
    let pd: Vec<usize> = a.ints("pads")?.map(|v| v.iter().map(|&q| q as usize).collect()).unwrap_or(vec![0; 2 * nd]);
    let (h, w) = if nd == 1 { (1, s[2]) } else { (s[2], s[3]) };
    let (kh, kw) = if nd == 1 { (1, ks[0]) } else { (ks[0], ks[1]) };
    let (sh, sw) = if nd == 1 { (1, st[0]) } else { (st[0], st[1]) };
    let (pt, pl, pb, pr) = if nd == 1 { (0, pd[0], 0, pd[1]) } else { (pd[0], pd[1], pd[2], pd[3]) };
    let oh = (h + pt + pb - kh) / sh + 1;
    let ow = (w + pl + pr - kw) / sw + 1;
    let planes = s[0] * s[1];
    let k = kh * kw;
    let mut map = Vec::with_capacity(planes * oh * ow * k);
    for p in 0..planes {
        for i in 0..oh {
            for j in 0..ow {
                for di in 0..kh {
                    for dj in 0..kw {
                        let y = (i * sh + di) as isize - pt as isize;
                        let xx = (j * sw + dj) as isize - pl as isize;
                        map.push(if y < 0 || y >= h as isize || xx < 0 || xx >= w as isize { PADI } else { (p * h * w + y as usize * w + xx as usize) as u32 });
                    }
                }
            }
        }
    }
    let mut out = vec![s[0], s[1]];
    if nd == 2 {
        out.push(oh);
    }
    out.push(ow);
    let nout = planes * oh * ow;
    ok(vec![out], MaxPoolOp { map, k, nout })
}

impl Op for MaxPoolOp {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], _be: Backend) {
        let x = ts.d(ins[0]);
        let y = ts.dm(outs[0]);
        for o in 0..self.nout {
            let mut m = f32::NEG_INFINITY;
            for &i in &self.map[o * self.k..(o + 1) * self.k] {
                if i != PADI {
                    m = m.max(x[i as usize]);
                }
            }
            y[o] = m;
        }
    }
}

//! Compute kernels, instantiated per backend from one macro body.
//!
//! Each backend module defines a vector type `V` of `W` f32 lanes and a handful of
//! primitives (all `#[inline]`, carrying the same `#[target_feature]` attributes as the
//! kernels so LLVM inlines them). `kernels!` then expands the shared kernel code.
//!
//! GEMM convention (all 1x1 convs, encoder, mask Toeplitz, decoder):
//!   out[f*os + j] = sum_{k<n} wt[k*m + j] * inp[f*is + k]      j < m, f < nf
//! `wt` is the transposed weight ([n][m], output channels contiguous) so the kernel is
//! a broadcast-FMA over output lanes. Every output element is accumulated sequentially
//! over k, independent of the tile shape -> frame-by-frame (nf=1) and batched runs
//! produce bit-identical results.
// The scalar backend instantiates `kernels!` with W = 1, where `m % W` is 0 by design.
#![allow(clippy::modulo_one)]

macro_rules! kernels {
    ($(#[$attr:meta])*) => {
        $(#[$attr])*
        unsafe fn tile<const FT: usize, const MV: usize>(
            wt: *const f32, m: usize, n: usize, inp: *const f32, is: usize, ks: usize,
            out: *mut f32, os: usize, f0: usize, j0: usize,
        ) {
            let mut acc = [[vzero(); MV]; FT];
            let wp = wt.add(j0);
            let ip = inp.add(f0 * is);
            for k in 0..n {
                let wrow = wp.add(k * m);
                let mut wv = [vzero(); MV];
                for j in 0..MV {
                    wv[j] = vload(wrow.add(j * W));
                }
                for i in 0..FT {
                    let x = vsplat(*ip.add(i * is + k * ks));
                    for j in 0..MV {
                        acc[i][j] = vfma(acc[i][j], wv[j], x);
                    }
                }
            }
            for i in 0..FT {
                let op = out.add((f0 + i) * os + j0);
                for j in 0..MV {
                    vstore(op.add(j * W), acc[i][j]);
                }
            }
        }

        /// See module docs. Requires m % W == 0 (weight row stride = m).
        $(#[$attr])*
        pub unsafe fn gemm(wt: *const f32, m: usize, n: usize, inp: *const f32, is: usize,
                           nf: usize, out: *mut f32, os: usize) {
            gemm_ld(wt, m, m, n, inp, is, nf, out, os)
        }

        /// General form: weight row stride `ldw` (>= m), any m (columns m - m%W.. done by a
        /// scalar tail, still sequential over k per element -> frame/batch bit-identical).
        $(#[$attr])*
        pub unsafe fn gemm_ld(wt: *const f32, ldw: usize, m: usize, n: usize, inp: *const f32, is: usize,
                              nf: usize, out: *mut f32, os: usize) {
            gemm_s(wt, ldw, m, n, inp, is, 1, nf, out, os)
        }

        /// gemm_ld with input element (f, k) at inp[f*is + k*ks] (e.g. channel-major NCL input).
        $(#[$attr])*
        pub unsafe fn gemm_s(wt: *const f32, ldw: usize, m: usize, n: usize, inp: *const f32, is: usize, ks: usize,
                              nf: usize, out: *mut f32, os: usize) {
            let mm = m - m % W;
            // column chunks: the input-side panel of a chunk stays in cache across all row tiles
            // (per-element accumulation order unchanged -> bit-identical to the unchunked loop)
            const JC: usize = 256;
            let mut jc0 = 0;
            while jc0 < mm {
                let jc1 = if jc0 + JC < mm { jc0 + JC } else { mm };
                let mut f = 0;
                while f + 4 <= nf {
                    let mut j = jc0;
                    while j + MV4 * W <= jc1 {
                        tile::<4, MV4>(wt, ldw, n, inp, is, ks, out, os, f, j);
                        j += MV4 * W;
                    }
                    while j < jc1 {
                        tile::<4, 1>(wt, ldw, n, inp, is, ks, out, os, f, j);
                        j += W;
                    }
                    f += 4;
                }
                while f < nf {
                    let mut j = jc0;
                    while j + MV1 * W <= jc1 {
                        tile::<1, MV1>(wt, ldw, n, inp, is, ks, out, os, f, j);
                        j += MV1 * W;
                    }
                    while j < jc1 {
                        tile::<1, 1>(wt, ldw, n, inp, is, ks, out, os, f, j);
                        j += W;
                    }
                    f += 1;
                }
                jc0 = jc1;
            }
            if mm < m {
                for f in 0..nf {
                    for j in mm..m {
                        let mut acc = 0f32;
                        for k in 0..n {
                            acc += *wt.add(k * ldw + j) * *inp.add(f * is + k * ks);
                        }
                        *out.add(f * os + j) = acc;
                    }
                }
            }
        }

        /// x = max(x, 0)
        $(#[$attr])*
        pub unsafe fn relu(x: *mut f32, n: usize) {
            let z = vzero();
            let mut i = 0;
            while i < n {
                vstore(x.add(i), vmax(vload(x.add(i)), z));
                i += W;
            }
        }

        /// x = x >= 0 ? x : alpha*x   (exact for any alpha: max(x,0) + alpha*min(x,0))
        $(#[$attr])*
        pub unsafe fn leaky(x: *mut f32, n: usize, alpha: f32) {
            let z = vzero();
            let a = vsplat(alpha);
            let mut i = 0;
            while i < n {
                let v = vload(x.add(i));
                vstore(x.add(i), vadd(vmax(v, z), vmul(a, vmin(v, z))));
                i += W;
            }
        }

        /// ChannelNorm over n channels: out = (g*(x-mean)) / sqrt(mean((x-mean)^2) + eps) + b
        /// (same op order as the MIL: sub, square, reduce_mean, add eps, sqrt, mul gamma, real_div, add beta)
        $(#[$attr])*
        pub unsafe fn norm(x: *const f32, out: *mut f32, g: *const f32, b: *const f32, n: usize, eps: f32) {
            let mut s = vzero();
            let mut i = 0;
            while i < n {
                s = vadd(s, vload(x.add(i)));
                i += W;
            }
            let mean = hsum(s) / n as f32;
            let mv = vsplat(mean);
            let mut q = vzero();
            i = 0;
            while i < n {
                let d = vsub(vload(x.add(i)), mv);
                q = vadd(q, vmul(d, d));
                i += W;
            }
            let sd = vsplat((hsum(q) / n as f32 + eps).sqrt());
            i = 0;
            while i < n {
                let d = vsub(vload(x.add(i)), mv);
                vstore(out.add(i), vadd(vdiv(vmul(vload(g.add(i)), d), sd), vload(b.add(i))));
                i += W;
            }
        }

        /// out = w0*x2 + w1*x1 + w2*x0  (depthwise k=3: x2 = t-2d, x1 = t-d, x0 = t)
        $(#[$attr])*
        pub unsafe fn dwconv(x2: *const f32, x1: *const f32, x0: *const f32,
                             w0: *const f32, w1: *const f32, w2: *const f32, out: *mut f32, n: usize) {
            let mut i = 0;
            while i < n {
                let a = vmul(vload(w0.add(i)), vload(x2.add(i)));
                let a = vadd(a, vmul(vload(w1.add(i)), vload(x1.add(i))));
                let a = vadd(a, vmul(vload(w2.add(i)), vload(x0.add(i))));
                vstore(out.add(i), a);
                i += W;
            }
        }

        /// out = a + b
        $(#[$attr])*
        pub unsafe fn add(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
            let mut i = 0;
            while i < n {
                vstore(out.add(i), vadd(vload(a.add(i)), vload(b.add(i))));
                i += W;
            }
        }
    };
}

pub mod scalar {
    pub type V = f32;
    pub const W: usize = 1;
    const MV4: usize = 4;
    const MV1: usize = 8;
    #[inline(always)] fn vzero() -> V { 0.0 }
    #[inline(always)] fn vsplat(x: f32) -> V { x }
    #[inline(always)] unsafe fn vload(p: *const f32) -> V { *p }
    #[inline(always)] unsafe fn vstore(p: *mut f32, v: V) { *p = v }
    #[inline(always)] fn vadd(a: V, b: V) -> V { a + b }
    #[inline(always)] fn vsub(a: V, b: V) -> V { a - b }
    #[inline(always)] fn vmul(a: V, b: V) -> V { a * b }
    #[inline(always)] fn vdiv(a: V, b: V) -> V { a / b }
    #[inline(always)] fn vfma(acc: V, a: V, b: V) -> V { acc + a * b }
    #[inline(always)] fn vmax(a: V, b: V) -> V { if a > b { a } else { b } }
    #[inline(always)] fn vmin(a: V, b: V) -> V { if a < b { a } else { b } }
    #[inline(always)] fn hsum(a: V) -> f32 { a }
    kernels!();
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use core::arch::x86_64::*;
    pub type V = __m256;
    pub const W: usize = 8;
    const MV4: usize = 2; // 4 frames x 16 outputs = 8 accumulators
    const MV1: usize = 8; // 1 frame x 64 outputs = 8 independent FMA chains
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vzero() -> V { _mm256_setzero_ps() }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vsplat(x: f32) -> V { _mm256_set1_ps(x) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vload(p: *const f32) -> V { _mm256_loadu_ps(p) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vstore(p: *mut f32, v: V) { _mm256_storeu_ps(p, v) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vadd(a: V, b: V) -> V { _mm256_add_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vsub(a: V, b: V) -> V { _mm256_sub_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vmul(a: V, b: V) -> V { _mm256_mul_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vdiv(a: V, b: V) -> V { _mm256_div_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vfma(acc: V, a: V, b: V) -> V { _mm256_fmadd_ps(a, b, acc) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vmax(a: V, b: V) -> V { _mm256_max_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn vmin(a: V, b: V) -> V { _mm256_min_ps(a, b) }
    #[inline] #[target_feature(enable = "avx2,fma")] unsafe fn hsum(a: V) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }
    kernels!(#[target_feature(enable = "avx2,fma")]);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod simd128 {
    use core::arch::wasm32::*;
    pub type V = v128;
    pub const W: usize = 4;
    const MV4: usize = 4; // 4 frames x 16 outputs = 16 accumulators (arm64 hosts have 32 v-regs)
    const MV1: usize = 8; // 1 frame x 32 outputs
    #[inline(always)] fn vzero() -> V { f32x4_splat(0.0) }
    #[inline(always)] fn vsplat(x: f32) -> V { f32x4_splat(x) }
    #[inline(always)] unsafe fn vload(p: *const f32) -> V { v128_load(p as *const v128) }
    #[inline(always)] unsafe fn vstore(p: *mut f32, v: V) { v128_store(p as *mut v128, v) }
    #[inline(always)] fn vadd(a: V, b: V) -> V { f32x4_add(a, b) }
    #[inline(always)] fn vsub(a: V, b: V) -> V { f32x4_sub(a, b) }
    #[inline(always)] fn vmul(a: V, b: V) -> V { f32x4_mul(a, b) }
    #[inline(always)] fn vdiv(a: V, b: V) -> V { f32x4_div(a, b) }
    // relaxed-simd build: hardware FMA where available (numerics then match FMA backends)
    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)] fn vfma(acc: V, a: V, b: V) -> V { f32x4_relaxed_madd(a, b, acc) }
    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)] fn vfma(acc: V, a: V, b: V) -> V { f32x4_add(acc, f32x4_mul(a, b)) }
    #[inline(always)] fn vmax(a: V, b: V) -> V { f32x4_pmax(a, b) }
    #[inline(always)] fn vmin(a: V, b: V) -> V { f32x4_pmin(a, b) }
    #[inline(always)] fn hsum(a: V) -> f32 {
        (f32x4_extract_lane::<0>(a) + f32x4_extract_lane::<1>(a))
            + (f32x4_extract_lane::<2>(a) + f32x4_extract_lane::<3>(a))
    }
    kernels!();
}

/// Backend-independent hot loops (plain scalar code that LLVM auto-vectorizes); live in this
/// crate so they are compiled at opt-level 3 even when the rest of the wasm is size-optimized.
pub mod util {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum B {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Max,
        Min,
        PRelu,
        Square,
        SqrtPow,
    }

    #[inline(always)]
    fn bop(k: B, x: f32, y: f32) -> f32 {
        match k {
            B::Add => x + y,
            B::Sub => x - y,
            B::Mul => x * y,
            B::Div => x / y,
            B::Pow => x.powf(y),
            B::Max => x.max(y),
            B::Min => x.min(y),
            B::PRelu => x.max(0.0) + y * x.min(0.0),
            B::Square => x * x,
            B::SqrtPow => x.sqrt(),
        }
    }

    /// Broadcast binary op over a collapsed iteration plan (dims, per-operand strides; <= 8 dims).
    pub fn binary(k: B, dims: &[usize], sa: &[usize], sb: &[usize], n: usize, a: &[f32], b: &[f32], o: &mut [f32]) {
        let nd = dims.len();
        let inner = dims[nd - 1];
        let (ia, ib) = (sa[nd - 1], sb[nd - 1]);
        let mut idx = [0usize; 8];
        let (mut oa, mut ob, mut oo) = (0usize, 0usize, 0usize);
        while oo < n {
            macro_rules! inner_loop {
                ($k:expr) => {{
                    let kk = $k;
                    let oo_ = &mut o[oo..oo + inner];
                    if ia == 1 && ib == 1 {
                        let (x, y) = (&a[oa..oa + inner], &b[ob..ob + inner]);
                        for j in 0..inner { oo_[j] = bop(kk, x[j], y[j]); }
                    } else if ia == 1 && ib == 0 {
                        let (x, y) = (&a[oa..oa + inner], b[ob]);
                        for j in 0..inner { oo_[j] = bop(kk, x[j], y); }
                    } else if ia == 0 && ib == 1 {
                        let (x, y) = (a[oa], &b[ob..ob + inner]);
                        for j in 0..inner { oo_[j] = bop(kk, x, y[j]); }
                    } else {
                        for j in 0..inner { oo_[j] = bop(kk, a[oa + j * ia], b[ob + j * ib]); }
                    }
                }};
            }
            match k {
                B::Add => inner_loop!(B::Add),
                B::Sub => inner_loop!(B::Sub),
                B::Mul => inner_loop!(B::Mul),
                B::Div => inner_loop!(B::Div),
                B::PRelu => inner_loop!(B::PRelu),
                _ => inner_loop!(k),
            }
            oo += inner;
            let mut d = nd - 1;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                oa += sa[d];
                ob += sb[d];
                if idx[d] < dims[d] {
                    break;
                }
                oa -= sa[d] * dims[d];
                ob -= sb[d] * dims[d];
                idx[d] = 0;
            }
        }
    }

    /// expf: Cephes range reduction + degree-6 polynomial (~1-2 ulp), branch-free so loops
    /// vectorize; replaces libm exp in sigmoid/tanh (much faster in wasm).
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        let x = x.max(-87.3).min(88.7);
        let n = (x * core::f32::consts::LOG2_E + 0.5).floor();
        let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
        let r2 = r * r;
        let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 5.000_000_1e-1;
        let e = p * r2 + r + 1.0;
        e * f32::from_bits(((n as i32 + 127) as u32) << 23)
    }
    #[inline(always)]
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + expf(-x))
    }
    /// tanh(x) = 1 - 2 / (exp(2x) + 1)  (abs error ~1e-7)
    #[inline(always)]
    pub fn tanhf(x: f32) -> f32 {
        1.0 - 2.0 / (expf(2.0 * x) + 1.0)
    }
    pub fn sigmoid_inplace(y: &mut [f32]) {
        y.iter_mut().for_each(|v| *v = sigmoid(*v));
    }
    pub fn tanh_inplace(y: &mut [f32]) {
        y.iter_mut().for_each(|v| *v = tanhf(*v));
    }
    /// out[j] = b[j] + sum_k w[k*ldw + j] * x[k]  (j < m): small recurrent mat-vec, k outer / j inner
    pub fn matvec_bias(w: &[f32], ldw: usize, m: usize, x: &[f32], b: &[f32], out: &mut [f32]) {
        let out = &mut out[..m];
        out.copy_from_slice(&b[..m]);
        for (k, &xk) in x.iter().enumerate() {
            let wr = &w[k * ldw..k * ldw + m];
            for j in 0..m {
                out[j] += wr[j] * xk;
            }
        }
    }
    /// y[i] = map[i] == u32::MAX ? fill : x[map[i]]
    pub fn gather(map: &[u32], x: &[f32], y: &mut [f32], fill: f32) {
        for (o, &m) in y.iter_mut().zip(map) {
            *o = if m == u32::MAX { fill } else { x[m as usize] };
        }
    }
    /// o += w * x
    pub fn madd(o: &mut [f32], w: &[f32], x: &[f32]) {
        let n = o.len().min(w.len()).min(x.len());
        let (o, w, x) = (&mut o[..n], &w[..n], &x[..n]);
        for i in 0..n {
            o[i] += w[i] * x[i];
        }
    }
    /// GRU gates (ONNX order z, r, h; linear_before_reset = 1): updates h in place.
    /// gx = W x + Wb (3H), gh = R h + Rb (3H)
    pub fn gru_lbr(gx: &[f32], gh: &[f32], h: &mut [f32]) {
        let n = h.len();
        for j in 0..n {
            let z = sigmoid(gx[j] + gh[j]);
            let r = sigmoid(gx[n + j] + gh[n + j]);
            let c = tanhf(gx[2 * n + j] + r * gh[2 * n + j]);
            h[j] = (1.0 - z) * c + z * h[j];
        }
    }
    /// LSTM gates (ONNX order i, o, f, c): g = W x + R h + b (4H); updates h, c.
    pub fn lstm(gx: &[f32], gh: &[f32], h: &mut [f32], c: &mut [f32]) {
        let n = h.len();
        for j in 0..n {
            let ig = sigmoid(gx[j] + gh[j]);
            let og = sigmoid(gx[n + j] + gh[n + j]);
            let fg = sigmoid(gx[2 * n + j] + gh[2 * n + j]);
            let cc = tanhf(gx[3 * n + j] + gh[3 * n + j]);
            c[j] = fg * c[j] + ig * cc;
            h[j] = og * tanhf(c[j]);
        }
    }
}

// ------------------------------------------------------------------ backend selection + dispatch

/// Kernel backend: scalar, AVX2+FMA (runtime-detected on x86_64), simd128 (compile-time on wasm).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

impl Backend {
    /// Best available backend (runtime-detected on x86_64, compile-time on wasm).
    pub fn detect() -> Backend {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Backend::Avx2;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return Backend::Simd128;
        }
        #[allow(unreachable_code)]
        Backend::Scalar
    }
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2+fma",
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => "simd128",
        }
    }
}

#[macro_export]
macro_rules! call {
    ($be:expr, $f:ident($($a:expr),* $(,)?)) => {
        match $be {
            $crate::Backend::Scalar => unsafe { $crate::scalar::$f($($a),*) },
            #[cfg(target_arch = "x86_64")]
            $crate::Backend::Avx2 => unsafe { $crate::avx2::$f($($a),*) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Backend::Simd128 => unsafe { $crate::simd128::$f($($a),*) },
        }
    };
}



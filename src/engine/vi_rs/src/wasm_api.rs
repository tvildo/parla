//! Generic C ABI for wasm32 (no wasm-bindgen, no imports): usable from browsers (web/vi_engine.js)
//! and any standalone wasm runtime (wasmtime, wasmer, WAMR, Node, Deno, ...). Every handle is a
//! `host::Host` (engine + on/off/strength mix + voice activity + frame parts).
//! Several engine instances may coexist (A/B switching without reload).
//!
//!   vi_alloc(bytes) -> ptr (16-byte aligned) / vi_free(ptr, bytes)
//!   engine_create(kind_id, cfg_ptr, cfg_len, w_ptr, w_len) -> handle (0 on error, see last_error)
//!   engine_info(h, out_ptr: *mut i32[8]) -> [sample_rate, frame_size, latency, in_ch, out_ch, simd_active, 0, 0]
//!   engine_process(h, in_ptr, out_ptr, n_frames)  (n_frames * frame_size samples, mono f32 at sample_rate)
//!   engine_set_mix(h, strength 0..1, ramp_ms): noise cancellation off (0) .. full (1), ramped; the
//!     dry part is the input delayed by exactly the engine latency (no click, no jump in time)
//!   engine_get_mix(h, out_ptr: *mut f32[2]) -> [current gain, target]
//!   engine_reset(h), engine_destroy(h), engine_set_scalar(h, flag)
//!   engine_set_stages(h, n) -> parts; engine_process_stage(h, in_ptr, out_ptr, s): one frame spread
//!     over `parts` calls (same buffers each call; == engine_process of 1 frame) — realtime smoothing
//!   engine_voice(h, out_ptr: *mut f32[3]) -> [speech probability 0..1, speaking 0/1, level dBFS]
//!     of the denoised signal (vi_audio::vad; combined with the model's own VAD if it has one)
//!   engines_live() -> engine instances alive (diagnostics)
//!   simd_build() -> 1 if compiled with simd128
//!   kind_available_c(kind) -> 1 if that plugin is in this module
//!   last_error(out_ptr, cap) -> len  (message of the last failed engine_create)

use crate::host::{self, Host};
use crate::kind_available;
use std::alloc::{alloc as sys_alloc, dealloc, Layout};

type Handle = Host;
static mut LAST_ERR: String = String::new();

#[no_mangle]
pub extern "C" fn vi_alloc(bytes: usize) -> *mut u8 {
    unsafe { sys_alloc(Layout::from_size_align(bytes.max(16), 16).unwrap()) }
}

#[no_mangle]
pub unsafe extern "C" fn vi_free(p: *mut u8, bytes: usize) {
    dealloc(p, Layout::from_size_align(bytes.max(16), 16).unwrap())
}

#[no_mangle]
pub unsafe extern "C" fn engine_create(kind: u32, cfg: *const u8, cfg_len: usize, w: *const u8, w_len: usize) -> *mut Handle {
    let cfg = if cfg_len == 0 { "" } else { std::str::from_utf8(std::slice::from_raw_parts(cfg, cfg_len)).unwrap_or("") };
    let w = if w_len == 0 { &[][..] } else { std::slice::from_raw_parts(w, w_len) };
    match Host::create(kind, cfg, w) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            *std::ptr::addr_of_mut!(LAST_ERR) = e;
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn last_error(out: *mut u8, cap: usize) -> usize {
    let e = &*std::ptr::addr_of!(LAST_ERR);
    let n = e.len().min(cap);
    std::ptr::copy_nonoverlapping(e.as_ptr(), out, n);
    n
}

#[no_mangle]
pub unsafe extern "C" fn engine_info(h: *mut Handle, out: *mut i32) {
    let e = &(*h).e;
    let i = e.info();
    let simd = (e.backend_name() == "simd128") as i32;
    let v = [i.sample_rate as i32, i.frame_size as i32, i.latency_samples as i32, i.in_channels as i32, i.out_channels as i32, simd, 0, 0];
    std::ptr::copy_nonoverlapping(v.as_ptr(), out, 8);
}

#[no_mangle]
pub unsafe extern "C" fn engine_process(h: *mut Handle, inp: *const f32, out: *mut f32, n_frames: usize) {
    let h = &mut *h;
    let n = n_frames * h.frame_size();
    h.process(std::slice::from_raw_parts(inp, n), std::slice::from_raw_parts_mut(out, n));
}

#[no_mangle]
pub unsafe extern "C" fn engine_set_mix(h: *mut Handle, strength: f32, ramp_ms: f32) {
    (*h).set_mix(strength, ramp_ms)
}

#[no_mangle]
pub unsafe extern "C" fn engine_get_mix(h: *mut Handle, out: *mut f32) {
    let (g, t) = (*h).mix();
    std::ptr::copy_nonoverlapping([g, t].as_ptr(), out, 2);
}

#[no_mangle]
pub unsafe extern "C" fn engine_set_stages(h: *mut Handle, n: usize) -> usize {
    (*h).set_stages(n)
}

#[no_mangle]
pub unsafe extern "C" fn engine_process_stage(h: *mut Handle, inp: *const f32, out: *mut f32, s: usize) {
    let h = &mut *h;
    let n = h.frame_size();
    h.process_stage(s, std::slice::from_raw_parts(inp, n), std::slice::from_raw_parts_mut(out, n));
}

#[no_mangle]
pub unsafe extern "C" fn engine_voice(h: *mut Handle, out: *mut f32) {
    std::ptr::copy_nonoverlapping((*h).voice().as_ptr(), out, 3);
}

#[no_mangle]
pub unsafe extern "C" fn engine_reset(h: *mut Handle) {
    (*h).reset()
}

#[no_mangle]
pub unsafe extern "C" fn engine_set_scalar(h: *mut Handle, scalar: i32) {
    (*h).e.set_scalar(scalar != 0)
}

#[no_mangle]
pub unsafe extern "C" fn engine_destroy(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[no_mangle]
pub extern "C" fn engines_live() -> u32 {
    host::live()
}

/// 1 if the plugin for `kind` is compiled into this module (bundle vs per-plugin builds)
#[no_mangle]
pub extern "C" fn kind_available_c(kind: u32) -> i32 {
    kind_available(kind) as i32
}

#[no_mangle]
pub extern "C" fn simd_build() -> i32 {
    cfg!(target_feature = "simd128") as i32
}

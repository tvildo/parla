//! Pluggable clock + id generator.
//!
//! wasm32: there is no clock in wasm32-unknown-unknown, the host pushes wall time with
//! [`set_now_ns`] (C ABI `otel_set_time_ms` / `otel_set_time_ns`, e.g. once per audio quantum).
//! Native: `SystemTime::now()` unless an override was set with [`set_now_ns`] (tests).

use crate::cell::Cell64;

static HOST_NOW: Cell64 = Cell64::new(0);
static SEED: Cell64 = Cell64::new(0);

/// Set the current wall time (ns since Unix epoch). wasm: the only clock. Native: override
/// (0 = back to `SystemTime`).
pub fn set_now_ns(ns: u64) {
    HOST_NOW.set(ns)
}

/// Current wall time in ns since the Unix epoch.
#[inline]
pub fn now_ns() -> u64 {
    let h = HOST_NOW.get();
    #[cfg(target_arch = "wasm32")]
    {
        h
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if h != 0 {
            return h;
        }
        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64,
            Err(_) => 0,
        }
    }
}

/// Process-wide seed for id generators created afterwards (wasm: host passes crypto randomness).
pub fn seed(s: u64) {
    SEED.set(s | 1)
}

pub(crate) fn default_seed() -> u64 {
    let s = SEED.get();
    if s != 0 {
        return s;
    }
    let local = 0u8;
    now_ns() ^ ((&local as *const u8 as usize as u64) << 17) ^ 0x6a09e667f3bcc909
}

/// SplitMix64 over an atomic counter: lock-free, thread-safe, not cryptographic.
pub(crate) struct IdGen(Cell64);

impl IdGen {
    pub(crate) fn new(seed: u64) -> Self {
        IdGen(Cell64::new(seed))
    }
    pub(crate) fn reseed(&self, seed: u64) {
        self.0.set(seed)
    }
    #[inline]
    pub(crate) fn next(&self) -> u64 {
        let mut z = self.0.add(0x9e3779b97f4a7c15).wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    #[inline]
    pub(crate) fn nonzero(&self) -> u64 {
        loop {
            let v = self.next();
            if v != 0 {
                return v;
            }
        }
    }
}

//! 64-bit lock-free cell used for every recorded value (counters, gauges, histogram fields, slot
//! headers). Atomic by default; `single-thread` swaps in a plain `UnsafeCell<u64>`.

#[cfg(not(feature = "single-thread"))]
mod imp {
    use core::sync::atomic::{AtomicU64, Ordering::*};

    #[repr(transparent)]
    pub struct Cell64(AtomicU64);

    impl Cell64 {
        pub const fn new(v: u64) -> Self {
            Self(AtomicU64::new(v))
        }
        #[inline(always)]
        pub fn get(&self) -> u64 {
            self.0.load(Relaxed)
        }
        #[inline(always)]
        pub fn get_acquire(&self) -> u64 {
            self.0.load(Acquire)
        }
        #[inline(always)]
        pub fn set(&self, v: u64) {
            self.0.store(v, Relaxed)
        }
        #[inline(always)]
        pub fn set_release(&self, v: u64) {
            self.0.store(v, Release)
        }
        /// Wrapping add, returns the previous value.
        #[inline(always)]
        pub fn add(&self, v: u64) -> u64 {
            self.0.fetch_add(v, Relaxed)
        }
        #[inline(always)]
        pub fn cas(&self, cur: u64, new: u64) -> bool {
            self.0.compare_exchange(cur, new, AcqRel, Relaxed).is_ok()
        }
        /// CAS loop: `f(cur)` returns the new value or None to leave the cell untouched.
        #[inline(always)]
        pub fn update(&self, f: impl Fn(u64) -> Option<u64>) {
            let mut cur = self.get();
            while let Some(n) = f(cur) {
                match self.0.compare_exchange_weak(cur, n, Relaxed, Relaxed) {
                    Ok(_) => return,
                    Err(c) => cur = c,
                }
            }
        }
    }
}

#[cfg(feature = "single-thread")]
mod imp {
    use core::cell::UnsafeCell;

    #[repr(transparent)]
    pub struct Cell64(UnsafeCell<u64>);
    // SAFETY: the `single-thread` feature is a promise by the host that providers are only used
    // from one thread (e.g. wasm32-unknown-unknown without threads).
    unsafe impl Sync for Cell64 {}

    impl Cell64 {
        pub const fn new(v: u64) -> Self {
            Self(UnsafeCell::new(v))
        }
        #[inline(always)]
        pub fn get(&self) -> u64 {
            unsafe { *self.0.get() }
        }
        #[inline(always)]
        pub fn get_acquire(&self) -> u64 {
            self.get()
        }
        #[inline(always)]
        pub fn set(&self, v: u64) {
            unsafe { *self.0.get() = v }
        }
        #[inline(always)]
        pub fn set_release(&self, v: u64) {
            self.set(v)
        }
        #[inline(always)]
        pub fn add(&self, v: u64) -> u64 {
            let o = self.get();
            self.set(o.wrapping_add(v));
            o
        }
        #[inline(always)]
        pub fn cas(&self, cur: u64, new: u64) -> bool {
            if self.get() == cur {
                self.set(new);
                true
            } else {
                false
            }
        }
        #[inline(always)]
        pub fn update(&self, f: impl Fn(u64) -> Option<u64>) {
            if let Some(n) = f(self.get()) {
                self.set(n)
            }
        }
    }
}

pub(crate) use imp::Cell64;

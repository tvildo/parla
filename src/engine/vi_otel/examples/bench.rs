//! Recording micro-benchmark + allocation check: `cargo run --release -p vi_otel --example bench`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use vi_otel::metrics::{MeterProvider, SdkMeterProvider};
use vi_otel::trace::{SdkTracerProvider, TracerProvider};
use vi_otel::KeyValue;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn bench(name: &str, iters: usize, mut f: impl FnMut(usize)) {
    let mut best = f64::MAX;
    let mut allocs = 0;
    for _ in 0..7 {
        let a0 = ALLOCS.load(Relaxed);
        let t = Instant::now();
        for i in 0..iters {
            f(i);
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        allocs += ALLOCS.load(Relaxed) - a0;
        best = best.min(ns);
    }
    println!("{name:<44} {best:>7.2} ns/op   allocs during timed loops: {allocs}");
}

fn main() {
    let mp = SdkMeterProvider::builder().build();
    let m = mp.meter("bench");
    let model = KeyValue::new("vi.model", "gtcrn");
    let backend = KeyValue::new("vi.backend", "avx2");
    let c = m.u64_counter("c").build();
    let cb = c.bind(&[model, backend]);
    let h = m
        .f64_histogram("h")
        .with_boundaries(vec![10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 20000.0])
        .build();
    let hb = h.bind(&[model]);
    let g = m.f64_gauge("g").build();
    let gb = g.bind(&[]);
    let fc = m.f64_counter("fc").build();
    // warm: claim slots outside the timed loops
    c.add(0, &[model, backend]);
    h.record(1.0, &[model]);
    fc.add(0.0, &[]);

    const N: usize = 20_000_000;
    bench("bound u64 counter add", N, |i| cb.add(black_box(i as u64 & 1)));
    bench("u64 counter add, 2 attrs (hash + probe)", N, |i| c.add(black_box(i as u64 & 1), black_box(&[model, backend])));
    bench("f64 counter add, no attrs", N, |i| fc.add(black_box((i & 3) as f64), &[]));
    bench("bound f64 gauge record", N, |i| gb.record(black_box(i as f64)));
    bench("bound f64 histogram record (16 bounds)", N, |i| hb.record(black_box((i % 4000) as f64)));
    bench("f64 histogram record, 1 attr", N, |i| h.record(black_box((i % 4000) as f64), black_box(&[model])));

    let tp = SdkTracerProvider::builder().with_span_capacity(4096).build();
    let t = tp.tracer("bench");
    let mut scratch = Vec::with_capacity(4096);
    let mut best = f64::MAX;
    let a0 = ALLOCS.load(Relaxed);
    for _ in 0..200 {
        let t0 = Instant::now();
        for _ in 0..4096 {
            let mut s = t.start("s");
            s.set_attribute(model);
            s.add_event("e", []);
            s.end();
        }
        best = best.min(t0.elapsed().as_nanos() as f64 / 4096.0);
        let a1 = ALLOCS.load(Relaxed);
        scratch.clear();
        tp.drain_into(&mut scratch, 4096);
        ALLOCS.store(a1, Relaxed);
    }
    println!(
        "{:<44} {:>7.2} ns/op   allocs during timed loops: {}   (SpanData = {} B)",
        "span start + attr + event + end (ring push)",
        best,
        ALLOCS.load(Relaxed) - a0,
        std::mem::size_of::<vi_otel::trace::SpanData>()
    );

    // encoding cost (off the audio thread)
    let mut out = Vec::new();
    mp.export_json(&mut out);
    let t0 = Instant::now();
    for _ in 0..10_000 {
        mp.export_json(&mut out);
    }
    println!("metrics export_json ({} B)                    {:>7.2} us", out.len(), t0.elapsed().as_nanos() as f64 / 1e7);
    for _ in 0..512 {
        t.start("s").end();
    }
    let t0 = Instant::now();
    let n = tp.export_json(&mut scratch, 512, &mut out);
    println!("spans export_json ({n} spans, {} B)        {:>7.2} us", out.len(), t0.elapsed().as_nanos() as f64 / 1e3);
}

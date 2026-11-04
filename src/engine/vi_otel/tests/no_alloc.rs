//! Recording after setup must not allocate (own test binary: the counting allocator is global).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use vi_otel::metrics::{MeterProvider, SdkMeterProvider};
use vi_otel::trace::{SdkTracerProvider, TracerProvider};
use vi_otel::{Context, KeyValue};

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

#[test]
fn recording_does_not_allocate() {
    let mp = SdkMeterProvider::builder().build();
    let m = mp.meter("t");
    let c = m.u64_counter("c").with_cardinality_limit(4).build();
    let h = m.f64_histogram("h").build();
    let g = m.i64_gauge("g").build();
    let u = m.i64_up_down_counter("u").build();
    let hb = h.bind(&[KeyValue::new("m", "x")]);
    let tp = SdkTracerProvider::builder().with_span_capacity(8).build();
    let t = tp.tracer("t");

    let before = ALLOCS.load(Relaxed);
    for i in 0..1000i64 {
        // includes first-use slot claims and overflow (no allocation either)
        c.add(1, &[KeyValue::new("i", i % 7)]);
        h.record(i as f64, &[]);
        hb.record(1.0);
        g.record(i, &[KeyValue::new("a", true)]);
        u.add(-1, &[]);
        let mut p = t.start("p");
        p.set_attribute(KeyValue::new("k", 1.5));
        let mut ch = t.start_with_context("c", &Context::from_span(&p));
        ch.add_event("e", [KeyValue::new("x", 1i64)]);
        ch.end();
        p.end(); // ring overflows after 8 spans: drop path must not allocate either
    }
    assert_eq!(ALLOCS.load(Relaxed) - before, 0);
    assert!(tp.dropped_spans() > 0);
}

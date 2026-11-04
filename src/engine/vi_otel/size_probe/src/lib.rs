//! Size probe. Baseline: a std cdylib that allocates and indexes (allocator + panic machinery the
//! engine already pays for). `otel`: adds vi_otel's C ABI plus a Rust-side recording export that
//! exercises every instrument kind and spans, like the engine would.

use std::alloc::{alloc, Layout};

#[no_mangle]
pub extern "C" fn probe_alloc(n: usize) -> *mut u8 {
    unsafe { alloc(Layout::from_size_align(n.max(8), 8).unwrap()) }
}

#[no_mangle]
pub extern "C" fn probe_vec(n: usize, i: usize) -> u32 {
    let v: Vec<u32> = (0..n as u32).collect();
    v[i]
}

#[cfg(feature = "otel")]
mod rec {
    use std::sync::OnceLock;
    use vi_otel::metrics::{BoundCounter, BoundHistogram, Gauge, UpDownCounter};
    use vi_otel::{global, KeyValue};

    struct I {
        dur: BoundHistogram<f64>,
        over: BoundCounter<u64>,
        level: Gauge<f64>,
        inst: UpDownCounter<i64>,
    }
    static INST: OnceLock<I> = OnceLock::new();

    fn inst() -> &'static I {
        INST.get_or_init(|| {
            let m = global::meter("vi.engine");
            let model = KeyValue::new("vi.model", "gtcrn");
            I {
                dur: m.f64_histogram("vi.engine.process.duration").with_unit("us")
                    .with_boundaries(vec![25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 2667.0]).build().bind(&[model]),
                over: m.u64_counter("vi.engine.overruns").build().bind(&[model]),
                level: m.f64_gauge("vi.audio.input.level").with_unit("dBFS").build(),
                inst: m.i64_up_down_counter("vi.engine.instances").build(),
            }
        })
    }

    /// what the engine would call once per render quantum
    #[no_mangle]
    pub extern "C" fn probe_record(us: f64, level: f64) {
        let i = inst();
        i.dur.record(us);
        if us > 2667.0 {
            i.over.add(1);
        }
        i.level.record(level, &[]);
    }

    #[no_mangle]
    pub extern "C" fn probe_span(n: u32) {
        inst().inst.add(1, &[]);
        let t = global::tracer("vi.engine");
        let mut s = t.start("vi.model.load");
        s.set_attribute(KeyValue::new("vi.model", "gtcrn"));
        s.set_attribute(KeyValue::new("n", n as i64));
        s.add_event("weights.parsed", []);
        s.end();
    }
}

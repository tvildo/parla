//! Simulated engine telemetry -> OTLP/HTTP JSON -> a real collector.
//!   cargo run --release -p vi_otel --features native-http --example collector_demo -- [host:port] [rounds] [interval_ms]
//!   cargo run --release -p vi_otel --features native-http --example collector_demo -- --forever [host:port] [export_ms]
//! Default mode: 127.0.0.1:4318, 1 round, prints HTTP status + response body of every request
//! (collector validation: all instrument kinds, cardinality overflow, `vi_otel.dropped`, parent/child spans).
//! `--forever`: runs until killed, simulating 5 browser sessions (4 models, 4 browsers) with plausible,
//! slowly drifting distributions for the local Grafana dashboard (`compose/`). ~375 records per
//! session per second, export every `export_ms` (default 10000); low CPU.

use std::time::{Duration, Instant};
use vi_otel::http::OtlpHttpExporter;
use vi_otel::metrics::{
    BoundCounter, BoundGauge, BoundHistogram, BoundUpDownCounter, Counter, Histogram, Meter, MeterProvider, SdkMeterProvider,
};
use vi_otel::trace::{SdkTracerProvider, SpanKind, Status, Tracer, TracerProvider};
use vi_otel::{Context, InstrumentationScope, KeyValue, Resource, Stats};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--forever") {
        let addr = args.get(2).cloned().unwrap_or_else(|| "127.0.0.1:4318".into());
        let export_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
        forever(addr, export_ms);
        return;
    }
    let addr = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1:4318".into());
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let interval: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5000);
    one_shot(addr, rounds, interval);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// standard normal (Box-Muller)
    fn normal(&mut self) -> f64 {
        let u = self.next().max(1e-12);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn lognormal(&mut self, sigma: f64) -> f64 {
        (sigma * self.normal() - 0.5 * sigma * sigma).exp()
    }
}

fn one_shot(addr: String, rounds: usize, interval: u64) {
    let res = Resource::builder()
        .with_service_name("vi-engine")
        .with_attributes([
            KeyValue::new("service.version", "0.1.0"),
            KeyValue::new("vi.engine.build", "demo"),
            KeyValue::new("host.arch", std::env::consts::ARCH),
            KeyValue::new("os.type", std::env::consts::OS),
            KeyValue::new("vi.cpu.simd", "avx2"),
            KeyValue::new("vi.audio.sample_rate", 48000i64),
        ])
        .build();
    let stats = std::sync::Arc::new(Stats::default());
    let mp = SdkMeterProvider::builder().with_resource(res.clone()).with_stats(stats.clone()).build();
    let tp = SdkTracerProvider::builder().with_resource(res).with_stats(stats).build();
    let scope = InstrumentationScope::builder("vi.engine").with_version("0.1.0").build();
    let m = mp.meter_with_scope(scope);
    let t = tp.tracer_with_scope(scope);

    let model = KeyValue::new("vi.model", "gtcrn");
    let dur = m
        .f64_histogram("vi.engine.process.duration")
        .with_description("Engine compute time per render quantum")
        .with_unit("us")
        .with_boundaries(vec![25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 2667.0, 5000.0])
        .build()
        .bind(&[model]);
    let overruns = m.u64_counter("vi.engine.overruns").with_description("Quanta whose processing exceeded the real-time budget").build().bind(&[model]);
    let resets = m.u64_counter("vi.engine.resets").with_description("State resets after non-finite output").build();
    let arena = m.u64_gauge("vi.engine.arena.bytes").with_unit("By").build();
    let supp = m.f64_gauge("vi.audio.suppression.db").with_unit("dB").build();
    let level = m.f64_histogram("vi.audio.input.level").with_unit("dBFS").with_boundaries(vec![-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0]).build();
    let instances = m.i64_up_down_counter("vi.engine.instances").build();

    let mut ex = OtlpHttpExporter::new(addr.clone());
    let mut rng = Rng(0x2545F4914F6CDD1D);
    for round in 0..rounds {
        // model load span with a child compile span
        let mut load = t.span_builder("vi.model.load").with_kind(SpanKind::Internal).with_attributes([model, KeyValue::new("vi.model.sha256", "9f2c...e1")]).start(&t);
        load.add_event("weights.fetched", [KeyValue::new("bytes", 2_097_152i64)]);
        let mut compile = t.start_with_context("vi.wasm.compile", &Context::from_span(&load));
        compile.set_attribute(KeyValue::new("vi.wasm.variant", "simd"));
        std::thread::sleep(Duration::from_millis(3));
        compile.set_status(Status::Ok);
        compile.end();
        load.end();
        instances.add(1, &[]);
        arena.record(1_234_567, &[model]);
        // one second of 48 kHz quanta
        for _ in 0..375 {
            let us = 80.0 + 400.0 * rng.next() * rng.next() * rng.next() * 10.0;
            dur.record(us);
            if us > 2667.0 {
                overruns.add(1);
            }
            level.record(-55.0 + 50.0 * rng.next(), &[]);
        }
        supp.record(-17.25, &[model]);
        if round == 0 {
            resets.add(1, &[model, KeyValue::new("vi.reason", "nan")]);
            // cardinality limit 2, 3 attribute sets -> overflow series + vi_otel.dropped{reason=metric_cardinality}
            let sw = m.u64_counter("vi.webrtc.codec.switches").with_cardinality_limit(2).build();
            for c in ["opus", "red", "pcmu"] {
                sw.add(1, &[KeyValue::new("codec", c)]);
            }
        }
        match ex.export_metrics(&mp) {
            Ok(r) => println!("POST /v1/metrics ({} B) -> {} {}", ex.last_payload().len(), r.status, r.body.trim()),
            Err(e) => println!("POST /v1/metrics failed: {e}"),
        }
        match ex.export_spans(&tp) {
            Ok((n, bytes, Some(r))) => println!("POST /v1/traces ({n} spans, {bytes} B) -> {} {}", r.status, r.body.trim()),
            Ok((_, _, None)) => println!("no spans"),
            Err(e) => println!("POST /v1/traces failed: {e}"),
        }
        if round + 1 < rounds {
            std::thread::sleep(Duration::from_millis(interval));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// --forever: multi-session simulation

const QUANTA_PER_S: usize = 375; // 128 frames at 48 kHz
const BUDGET_US: f64 = 2666.7;
const OPS: [&str; 6] = ["conv", "gru", "matmul", "stft", "istft", "mask"];

struct Model {
    name: &'static str,
    quantum_us: f64, // mean engine time per quantum on a reference desktop
    fps: f64,        // model frames (hops) per second
    weights: i64,
    load_ms: f64,
    arena: u64,
    supp_db: f64, // suppression in noise-only periods
}

const MODELS: [Model; 4] = [
    Model { name: "passthrough", quantum_us: 10.0, fps: 0.0, weights: 0, load_ms: 4.0, arena: 16_384, supp_db: 0.0 },
    Model { name: "spectral", quantum_us: 60.0, fps: 100.0, weights: 65_536, load_ms: 25.0, arena: 262_144, supp_db: 9.0 },
    Model { name: "gtcrn", quantum_us: 420.0, fps: 62.5, weights: 523_264, load_ms: 140.0, arena: 1_966_080, supp_db: 18.0 },
    Model { name: "dtln", quantum_us: 420.0, fps: 100.0, weights: 3_970_000, load_ms: 300.0, arena: 6_291_456, supp_db: 22.0 },
];

struct Browser {
    name: &'static str,
    version: &'static str,
    os: &'static str,
    mobile: bool,
    cores: i64,
    speed: f64, // engine time multiplier vs reference
    relaxed: bool,
    loss: f64,      // packet loss probability
    jitter_ms: f64, // median jitter
}

const SESSIONS: [(Browser, usize); 5] = [
    (Browser { name: "Chrome", version: "141", os: "windows", mobile: false, cores: 16, speed: 1.0, relaxed: true, loss: 0.003, jitter_ms: 6.0 }, 3),
    (Browser { name: "Firefox", version: "143", os: "linux", mobile: false, cores: 8, speed: 1.3, relaxed: false, loss: 0.005, jitter_ms: 8.0 }, 2),
    (Browser { name: "Safari", version: "26.0", os: "macos", mobile: false, cores: 10, speed: 0.85, relaxed: false, loss: 0.002, jitter_ms: 5.0 }, 1),
    (Browser { name: "Chrome Mobile", version: "141", os: "android", mobile: true, cores: 8, speed: 2.1, relaxed: false, loss: 0.02, jitter_ms: 22.0 }, 2),
    (Browser { name: "Edge", version: "141", os: "windows", mobile: false, cores: 12, speed: 1.05, relaxed: true, loss: 0.004, jitter_ms: 7.0 }, 0),
];

struct Session {
    b: &'static Browser,
    m: &'static Model,
    model: KeyValue,
    mp: SdkMeterProvider,
    tp: SdkTracerProvider,
    tracer: Tracer,
    ex: OtlpHttpExporter,
    rng: Rng,
    phase: f64,
    talking: bool,
    frame_credit: f64,
    next_load_s: u64,
    loaded: bool,
    loss_burst: u32,
    process: BoundHistogram<f64>,
    frame: BoundHistogram<f64>,
    op: Histogram<f64>,
    overruns: BoundCounter<u64>,
    underruns: BoundCounter<u64>,
    resets: Counter<u64>,
    arena: BoundGauge<u64>,
    instances: BoundUpDownCounter<i64>,
    load_dur: BoundHistogram<f64>,
    compile_dur: Histogram<f64>,
    in_level: BoundHistogram<f64>,
    out_level: BoundHistogram<f64>,
    supp: BoundHistogram<f64>,
    presence: BoundHistogram<f64>,
    jitter: BoundHistogram<f64>,
    lost: BoundCounter<u64>,
    received: BoundCounter<u64>,
    concealed: BoundCounter<u64>,
}

fn hist(m: &Meter, name: &'static str, unit: &'static str, desc: &'static str, b: &[f64]) -> Histogram<f64> {
    m.f64_histogram(name).with_unit(unit).with_description(desc).with_boundaries(b.to_vec()).build()
}

impl Session {
    fn new(i: usize, b: &'static Browser, m: &'static Model, addr: &str) -> Session {
        let res = Resource::builder()
            .with_service_name("vi-web")
            .with_attributes([
                KeyValue::new("service.version", "0.3.0"),
                KeyValue::new("service.instance.id", ["demo-0", "demo-1", "demo-2", "demo-3", "demo-4"][i]),
                KeyValue::new("vi.engine.build", "demo"),
                KeyValue::new("browser.name", b.name),
                KeyValue::new("browser.version", b.version),
                KeyValue::new("browser.mobile", b.mobile),
                KeyValue::new("os.type", b.os),
                KeyValue::new("host.cpu.count", b.cores),
                KeyValue::new("vi.wasm.simd", true),
                KeyValue::new("vi.wasm.relaxed_simd", b.relaxed),
                KeyValue::new("audio.sample_rate", 48000i64),
            ])
            .build();
        let stats = std::sync::Arc::new(Stats::default());
        let mp = SdkMeterProvider::builder().with_resource(res.clone()).with_stats(stats.clone()).build();
        // the mobile session gets a small span ring so profiling bursts overflow it (vi_otel.dropped)
        let cap = if b.mobile { 16 } else { 128 };
        let tp = SdkTracerProvider::builder().with_resource(res).with_stats(stats).with_span_capacity(cap).with_id_seed(0x9E37_79B9 + i as u64).build();
        let scope = InstrumentationScope::builder("vi.engine").with_version("0.3.0").build();
        let me = mp.meter_with_scope(scope);
        let tracer = tp.tracer_with_scope(scope);
        let model = KeyValue::new("vi.model", m.name);
        let mk = [model];
        let us_bounds = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 2667.0, 3500.0, 5000.0, 10000.0];
        let frame_bounds = [100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 16000.0, 25000.0];
        let db_bounds = [-90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0];
        let rng = Rng(0xD1B5_4A32_D192_ED03 ^ ((i as u64 + 1) * 0x9E37_79B9_7F4A_7C15));
        Session {
            b,
            m,
            model,
            process: hist(&me, "vi.engine.process.duration", "us", "Engine compute time per render quantum", &us_bounds).bind(&mk),
            frame: hist(&me, "vi.engine.frame.duration", "us", "Model compute time per frame (hop)", &frame_bounds).bind(&mk),
            op: hist(&me, "vi.engine.op.duration", "us", "Sampled per-op compute time", &[5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
            overruns: me.u64_counter("vi.engine.overruns").with_unit("{quantum}").with_description("Quanta whose processing exceeded the real-time budget").build().bind(&mk),
            underruns: me.u64_counter("vi.engine.underruns").with_unit("{quantum}").with_description("Render quanta without output ready").build().bind(&mk),
            resets: me.u64_counter("vi.engine.resets").with_description("State resets").build(),
            arena: me.u64_gauge("vi.engine.arena.bytes").with_unit("By").build().bind(&mk),
            instances: me.i64_up_down_counter("vi.engine.instances").build().bind(&mk),
            load_dur: hist(&me, "vi.model.load.duration", "ms", "Model package load (fetch + plan)", &[10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]).bind(&mk),
            compile_dur: hist(&me, "vi.wasm.compile.duration", "ms", "WebAssembly.compile", &[5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
            in_level: hist(&me, "vi.audio.input.level", "dBFS", "Per-quantum input RMS", &db_bounds).bind(&mk),
            out_level: hist(&me, "vi.audio.output.level", "dBFS", "Per-quantum output RMS", &db_bounds).bind(&mk),
            supp: hist(&me, "vi.audio.suppression.db", "dB", "Input minus output energy per quantum", &[0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]).bind(&mk),
            presence: hist(&me, "vi.audio.speech_presence", "1", "Mean model speech mask", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).bind(&mk),
            jitter: hist(&me, "vi.webrtc.jitter", "ms", "getStats jitter", &[2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]).bind(&[]),
            lost: me.u64_counter("vi.webrtc.packets_lost").with_unit("{packet}").build().bind(&[]),
            received: me.u64_counter("vi.webrtc.packets_received").with_unit("{packet}").build().bind(&[]),
            concealed: me.u64_counter("vi.webrtc.concealed_samples").with_unit("{sample}").build().bind(&[]),
            mp,
            tp,
            tracer,
            ex: OtlpHttpExporter::new(addr.to_string()),
            phase: i as f64 * 1.7,
            rng,
            talking: false,
            frame_credit: 0.0,
            next_load_s: 0,
            loaded: false,
            loss_burst: 0,
        }
    }

    fn load_model(&mut self, now_s: u64) {
        let first = !self.loaded;
        self.loaded = true;
        let ns = vi_otel::time::now_ns();
        let fail = !first && self.rng.next() < 0.05;
        let load_ms = self.m.load_ms * self.b.speed.sqrt() * self.rng.lognormal(0.35);
        let compile_ms = (8.0 + 30.0 * self.b.speed) * self.rng.lognormal(0.25);
        let total_ms = load_ms + compile_ms;
        let start = ns - (total_ms * 1e6) as u64;
        let variant = if self.b.relaxed { "relaxed" } else { "simd" };
        let mut load = self
            .tracer
            .span_builder("vi.model.load")
            .with_kind(SpanKind::Internal)
            .with_start_time(start)
            .with_attributes([self.model, KeyValue::new("vi.model.sha256", "9f2c41d0e1"), KeyValue::new("vi.model.bytes", self.m.weights)])
            .start(&self.tracer);
        let mut compile = self.tracer.span_builder("vi.wasm.compile").with_start_time(start).start_with_context(&self.tracer, &Context::from_span(&load));
        compile.set_attribute(KeyValue::new("vi.wasm.variant", variant));
        compile.set_status(Status::Ok);
        compile.end_with_timestamp(start + (compile_ms * 1e6) as u64);
        load.add_event_with_timestamp("weights.fetched", start + ((compile_ms + 0.7 * load_ms) * 1e6) as u64, [KeyValue::new("bytes", self.m.weights)]);
        if fail {
            load.set_status(Status::error("weights fetch failed: 503"));
        } else {
            load.add_event_with_timestamp("graph.planned", ns - 1_000_000, [KeyValue::new("arena.bytes", self.m.arena as i64)]);
            load.set_status(Status::Ok);
        }
        load.end_with_timestamp(ns);
        self.compile_dur.record(compile_ms, &[KeyValue::new("vi.wasm.variant", variant)]);
        self.load_dur.record(total_ms);
        if !first && !fail {
            self.resets.add(1, &[self.model, KeyValue::new("vi.reason", "manual")]);
        }
        self.arena.record(self.m.arena);
        // next reload in 2..6 min (a failed load retries after 20 s)
        self.next_load_s = now_s + if fail { 20 } else { 120 + (self.rng.next() * 240.0) as u64 };
    }

    /// One simulated second of audio.
    fn second(&mut self, t: f64) {
        let m = self.m;
        // slow load drift (other tabs, thermal) + a faster wobble
        let drift = 1.0 + 0.18 * (std::f64::consts::TAU * t / 900.0 + self.phase).sin() + 0.06 * (std::f64::consts::TAU * t / 47.0 + self.phase).sin();
        // speech / pause Markov chain
        let stay = if self.talking { 0.88 } else { 0.8 };
        if self.rng.next() > stay {
            self.talking = !self.talking;
        }
        let noise_floor = -50.0 + 6.0 * (std::f64::consts::TAU * t / 300.0 + self.phase).sin();
        let mut spike = 0usize;
        for q in 0..QUANTA_PER_S {
            if spike == 0 && self.rng.next() < 0.0003 {
                spike = 2 + (self.rng.next() * 12.0) as usize; // GC pause / throttled tab
            }
            let mut us = m.quantum_us * self.b.speed * drift * self.rng.lognormal(0.25);
            if spike > 0 {
                us *= 2.5 + 2.0 * self.rng.next();
                spike -= 1;
            }
            self.process.record(us);
            if us > BUDGET_US {
                self.overruns.add(1);
                if self.rng.next() < 0.35 {
                    self.underruns.add(1);
                }
            } else if self.rng.next() < 0.00005 {
                self.underruns.add(1);
            }
            if m.fps > 0.0 {
                self.frame_credit += m.fps / QUANTA_PER_S as f64;
                while self.frame_credit >= 1.0 {
                    self.frame_credit -= 1.0;
                    self.frame.record(us * QUANTA_PER_S as f64 / m.fps * self.rng.lognormal(0.1));
                }
                if q % 32 == 0 {
                    let k = (self.rng.next() * OPS.len() as f64) as usize % OPS.len();
                    let share = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05][k];
                    self.op.record(us * share * self.rng.lognormal(0.3), &[self.model, KeyValue::new("vi.op", OPS[k])]);
                }
            }
            let input = if self.talking { -26.0 + 6.0 * self.rng.normal() } else { noise_floor + 3.0 * self.rng.normal() };
            let input = input.clamp(-90.0, -0.5);
            let supp = if m.supp_db == 0.0 {
                (0.2 * self.rng.normal()).abs()
            } else {
                (m.supp_db * if self.talking { 0.3 } else { 1.0 } + 2.0 * self.rng.normal()).max(0.0)
            };
            self.in_level.record(input);
            self.out_level.record((input - supp).max(-90.0));
            self.supp.record(supp);
            if m.fps > 0.0 && q % 8 == 0 {
                let p = if self.talking { 0.7 + 0.25 * self.rng.next() } else { 0.05 + 0.15 * self.rng.next() };
                self.presence.record(p);
            }
        }
        // WebRTC getStats once per second: 50 Opus packets (20 ms)
        if self.loss_burst == 0 && self.rng.next() < 0.01 {
            self.loss_burst = 2 + (self.rng.next() * 6.0) as u32;
        }
        let p_loss = if self.loss_burst > 0 {
            self.loss_burst -= 1;
            (self.b.loss * 8.0).min(0.3)
        } else {
            self.b.loss
        };
        let mut lost = 0u64;
        for _ in 0..50 {
            if self.rng.next() < p_loss {
                lost += 1;
            }
        }
        self.lost.add(lost);
        self.received.add(50 - lost);
        self.concealed.add(lost * 960 + (self.rng.next() * 40.0) as u64);
        self.jitter.record(self.b.jitter_ms * self.rng.lognormal(0.45) * if lost > 1 { 2.0 } else { 1.0 });
        if self.rng.next() < 1.0 / 1800.0 {
            self.resets.add(1, &[self.model, KeyValue::new("vi.reason", "nan")]);
        }
    }

    /// Every minute: a burst of sampled per-op profile spans (overflows the mobile ring).
    fn profile_burst(&mut self) {
        let parent = self.tracer.span_builder("vi.engine.profile").with_attributes([self.model]).start(&self.tracer);
        let cx = Context::from_span(&parent);
        for k in 0..24 {
            let mut s = self.tracer.start_with_context("vi.engine.op", &cx);
            s.set_attribute(KeyValue::new("vi.op", OPS[k % OPS.len()]));
            s.end();
        }
        drop(parent);
    }

    fn export(&mut self, verbose: bool) {
        let m = self.ex.export_metrics(&self.mp);
        let t = self.ex.export_spans(&self.tp);
        let ok_m = matches!(&m, Ok(r) if r.status == 200);
        let ok_t = match &t {
            Ok((_, _, Some(r))) => r.status == 200,
            Ok((_, _, None)) => true,
            Err(_) => false,
        };
        if verbose || !ok_m || !ok_t {
            let ms = match &m {
                Ok(r) => format!("{}", r.status),
                Err(e) => format!("err {e}"),
            };
            let ts = match &t {
                Ok((n, _, Some(r))) => format!("{} ({n} spans)", r.status),
                Ok((_, _, None)) => "-".into(),
                Err(e) => format!("err {e}"),
            };
            println!("  {:<14} {:<15} metrics {ms} traces {ts}", self.b.name, self.m.name);
        }
    }
}

fn forever(addr: String, export_ms: u64) {
    println!("collector_demo --forever -> {addr}, export every {export_ms} ms, {} sessions", SESSIONS.len());
    let mut sessions: Vec<Session> = SESSIONS.iter().enumerate().map(|(i, (b, m))| Session::new(i, b, &MODELS[*m], &addr)).collect();
    // first loads staggered after the first export, so Prometheus sees them as increases
    for (i, s) in sessions.iter_mut().enumerate() {
        s.next_load_s = 12 + 7 * i as u64;
        s.instances.add(1);
        s.arena.record(s.m.arena);
    }
    let t0 = Instant::now();
    let mut sec: u64 = 0;
    let mut last_export = Instant::now();
    let mut exports: u64 = 0;
    loop {
        sec += 1;
        for s in sessions.iter_mut() {
            s.second(sec as f64);
            if sec >= s.next_load_s {
                s.load_model(sec);
            }
            if sec % 60 == 0 {
                s.profile_burst();
            }
        }
        if last_export.elapsed() >= Duration::from_millis(export_ms) {
            last_export = Instant::now();
            let verbose = exports % 30 == 0; // status table every ~5 min
            if verbose {
                println!("t={}s export #{exports}", t0.elapsed().as_secs());
            }
            for s in sessions.iter_mut() {
                s.export(verbose);
            }
            exports += 1;
        }
        // keep simulated time aligned with wall time
        let target = t0 + Duration::from_secs(sec);
        let now = Instant::now();
        if target > now {
            std::thread::sleep(target - now);
        }
    }
}

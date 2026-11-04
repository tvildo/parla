//! vi_cli — manifest-driven runner / benchmark for vi_rs engines (raw f32 mono files).
//!
//!   vi_cli run <manifest.json> <in.f32> <out.f32> [--frame] [--scalar] [--reps N] [--T frames_per_call]
//!   vi_cli all <manifest.json> <in.f32> <out_prefix> [--reps N]
//!       runs {best SIMD, scalar} x {block, frame}; writes <prefix>_<simd|scalar>_<block|frame>.f32
//!
//! Input is zero-padded to a multiple of the engine frame size; output has the same length,
//! delayed by the engine latency. Each timing line is followed by a machine-readable
//! `BENCH {json}` line (collected by bench scripts).

use std::path::Path;
use std::time::Instant;
use vi_rs::{Backend, Manifest, NoiseEngine};

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(p: &str, x: &[f32]) {
    let mut b = Vec::with_capacity(x.len() * 4);
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(p, b).unwrap_or_else(|e| panic!("write {p}: {e}"));
}

fn run(e: &mut dyn NoiseEngine, x: &[f32], y: &mut [f32], frame: bool, reps: usize) -> (f64, f64) {
    let fs = e.frame_size();
    let mut best = f64::MAX;
    let mut max_frame = 0f64;
    for _ in 0..reps {
        e.reset();
        let t = Instant::now();
        if frame {
            for (i, o) in x.chunks_exact(fs).zip(y.chunks_exact_mut(fs)) {
                let tf = Instant::now();
                e.process_frame(i, o);
                max_frame = max_frame.max(tf.elapsed().as_secs_f64());
            }
        } else {
            e.process(x, y);
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, max_frame)
}

fn report(model: &str, e: &dyn NoiseEngine, frame: bool, n: usize, (best, maxf): (f64, f64)) {
    let info = e.info();
    let audio_s = n as f64 / info.sample_rate as f64;
    let frames = (n / info.frame_size) as f64;
    let us_frame = 1e6 * best / frames;
    let mode = if frame { "frame" } else { "block" };
    println!(
        "{model:18} {:9} {mode}  {:8.2} ms/s-audio  RTF {:7.4} ({:6.1}x realtime)  {:7.2} us/frame({})  max {:7.1} us",
        e.backend_name(),
        1e3 * best / audio_s,
        best / audio_s,
        audio_s / best,
        us_frame,
        info.frame_size,
        maxf * 1e6
    );
    println!(
        "BENCH {{\"runner\":\"native\",\"model\":\"{model}\",\"backend\":\"{}\",\"mode\":\"{mode}\",\"rtf\":{:.6},\"us_per_frame\":{:.3},\"frame_size\":{},\"sample_rate\":{},\"max_frame_us\":{:.1},\"us_per_40\":{:.3}}}",
        e.backend_name(),
        best / audio_s,
        us_frame,
        info.frame_size,
        info.sample_rate,
        maxf * 1e6,
        us_frame * 40.0 / info.frame_size as f64
    );
}

/// `vi_cli tensor <graph.json> <weights.bin> <in_name> <out_name> <in.f32> <out.f32> [--hop N]`:
/// slide a whole-graph model (e.g. the DNSMOS scorer package) over an f32 file: one execution per
/// segment of numel(in) samples every `hop` samples; writes all outputs concatenated.
#[cfg(feature = "graph")]
fn tensor_mode(args: &[String]) {
    let gj = vi_rs::json::parse(&std::fs::read_to_string(&args[2]).expect("graph")).expect("graph json");
    let w = std::fs::read(&args[3]).expect("weights");
    let mut g = vi_graph::graph::Graph::load(&gj, &w, 1, &[&args[4], &args[5]]).unwrap_or_else(|e| panic!("{e}"));
    let (ti, to) = (g.id(&args[4]).expect("input"), g.id(&args[5]).expect("output"));
    let (tb, ab) = g.mem_bytes();
    eprintln!("graph memory: tensors {:.1} MB, arena {:.1} MB, nodes {}", tb as f64 / 1e6, ab as f64 / 1e6, g.n_nodes());
    // stream the input (no whole-file buffering): keep a window of `seg` samples, advance by `hop`
    use std::io::{Read, Write};
    let seg = g.tensors[ti].data.len();
    let hop = args.iter().position(|a| a == "--hop").map(|i| args[i + 1].parse().unwrap()).unwrap_or(seg);
    let mut rd = std::io::BufReader::new(std::fs::File::open(&args[6]).expect("input"));
    let mut wr = std::io::BufWriter::new(std::fs::File::create(&args[7]).expect("output"));
    let mut win: Vec<f32> = Vec::with_capacity(seg + hop);
    let mut bytes = vec![0u8; 4 * hop.max(1)];
    let t = Instant::now();
    let mut n = 0;
    let mut eof = false;
    loop {
        while win.len() < seg && !eof {
            let want = (seg - win.len()).min(hop.max(1)) * 4;
            let mut got = 0;
            while got < want {
                let r = rd.read(&mut bytes[got..want]).expect("read");
                if r == 0 {
                    eof = true;
                    break;
                }
                got += r;
            }
            win.extend(bytes[..got - got % 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        if win.len() < seg {
            break;
        }
        g.tensors[ti].data.copy_from_slice(&win[..seg]);
        g.run();
        for v in &g.tensors[to].data {
            wr.write_all(&v.to_le_bytes()).expect("write");
        }
        n += 1;
        win.drain(..hop.min(win.len()));
    }
    wr.flush().expect("flush");
    println!("tensor: {n} executions of {seg} samples in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
}

/// `vi_cli batch <manifest> <pairs.txt> [--T n] [--scalar]`: every "<in.f32> <out.f32>" line is
/// processed from a fresh engine state (reset), input zero-padded by `latency` samples (+ to a
/// frame multiple) so the output covers the whole input; frame-by-frame streaming path.
/// Prints one BATCH {json} line (load ms, total audio s, compute s, us per 10 ms of audio).
fn batch_mode(args: &[String]) {
    let mut man = Manifest::load(Path::new(&args[2])).unwrap_or_else(|e| panic!("{e}"));
    if let Some(i) = args.iter().position(|a| a == "--T") {
        man.set_config("frames_per_call", vi_rs::json::Value::Num(args[i + 1].parse().unwrap()));
    }
    let t0 = Instant::now();
    let mut e = man.create().unwrap_or_else(|e| panic!("{e}"));
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    e.set_scalar(args.iter().any(|a| a == "--scalar"));
    let info = e.info();
    let fs = info.frame_size;
    let list = std::fs::read_to_string(&args[3]).expect("pairs");
    let (mut audio, mut comp, mut nfiles) = (0f64, 0f64, 0);
    for line in list.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 2 {
            continue;
        }
        let mut x = read_f32(p[0]);
        let n0 = x.len();
        let n = (n0 + info.latency_samples + fs - 1) / fs * fs;
        x.resize(n, 0.0);
        let mut y = vec![0f32; n];
        e.reset();
        let t = Instant::now();
        for (i, o) in x.chunks_exact(fs).zip(y.chunks_exact_mut(fs)) {
            e.process_frame(i, o);
        }
        comp += t.elapsed().as_secs_f64();
        audio += n as f64 / info.sample_rate as f64;
        nfiles += 1;
        write_f32(p[1], &y);
    }
    println!(
        "BATCH {{\"model\":\"{}\",\"backend\":\"{}\",\"sample_rate\":{},\"frame_size\":{},\"latency\":{},\"files\":{},\"load_ms\":{:.2},\"audio_s\":{:.3},\"compute_s\":{:.4},\"us_per_10ms\":{:.2}}}",
        man.name, e.backend_name(), info.sample_rate, fs, info.latency_samples, nfiles, load_ms, audio, comp, 1e6 * comp / (audio * 100.0)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 3 && args[1] == "batch" {
        batch_mode(&args);
        return;
    }
    #[cfg(feature = "graph")]
    if args.len() > 7 && args[1] == "tensor" {
        tensor_mode(&args);
        return;
    }
    if args.len() < 5 || !(args[1] == "run" || args[1] == "all") {
        eprintln!("usage: vi_cli run <manifest.json> <in.f32> <out.f32> [--frame] [--scalar] [--reps N]\n       vi_cli all <manifest.json> <in.f32> <out_prefix> [--reps N]");
        std::process::exit(2);
    }
    let flag = |f: &str| args.iter().any(|a| a == f);
    let opt = |f: &str| args.iter().position(|a| a == f).map(|i| args[i + 1].clone());
    let reps: usize = opt("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);

    let t = Instant::now();
    let mut man = Manifest::load(Path::new(&args[2])).unwrap_or_else(|e| panic!("{e}"));
    if let Some(t) = opt("--T") {
        man.set_config("frames_per_call", vi_rs::json::Value::Num(t.parse().unwrap()));
    }
    let mut e = man.create().unwrap_or_else(|e| panic!("{e}"));
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let info = e.info();
    let mut x = read_f32(&args[3]);
    let n0 = x.len();
    let fs = info.frame_size;
    x.resize(n0 + (fs - n0 % fs) % fs, 0.0);
    let mut y = vec![0.0f32; x.len()];
    println!(
        "vi_cli: model {} ({}), {} Hz, frame {}, latency {}, {} samples ({:.2} s), load {:.1} ms, best backend {}, reps {}",
        man.name, info.name, info.sample_rate, fs, info.latency_samples, n0,
        n0 as f64 / info.sample_rate as f64, load_ms, Backend::detect().name(), reps
    );

    if args.iter().any(|a| a == "--profile") {
        let mut acc = Vec::new();
        e.reset();
        let t = Instant::now();
        for (i, o) in x.chunks_exact(fs).zip(y.chunks_exact_mut(fs)) {
            e.profile_frame(i, o, &mut acc);
        }
        let tot = t.elapsed().as_secs_f64();
        let nfr = (x.len() / fs) as f64;
        acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("profile: {:.1} us/frame total", 1e6 * tot / nfr);
        for (k, v) in &acc {
            println!("  {k:24} {:9.2} us/frame  {:5.1}%", 1e6 * v / nfr, 100.0 * v / tot);
        }
        return;
    }
    // --stages N: one frame split into parts (realtime smoothing); time per part = balance check
    if let Some(n) = opt("--stages") {
        let parts = e.set_stages(n.parse().unwrap());
        let mut per = vec![0f64; parts];
        e.reset();
        for (i, o) in x.chunks_exact(fs).zip(y.chunks_exact_mut(fs)) {
            for (s, p) in per.iter_mut().enumerate() {
                let t = Instant::now();
                e.process_stage(s, i, o);
                *p += t.elapsed().as_secs_f64();
            }
        }
        let nfr = (x.len() / fs) as f64;
        let tot: f64 = per.iter().sum();
        let row: Vec<String> = per.iter().map(|p| format!("{:.1} us ({:.0}%)", 1e6 * p / nfr, 100.0 * p / tot)).collect();
        println!("stages: {} -> {parts} parts per frame: {}; max/mean {:.2}", n, row.join(" | "),
                 per.iter().cloned().fold(0.0, f64::max) / (tot / parts as f64));
        return;
    }
    if args[1] == "all" {
        let prefix = &args[4];
        let mut outs: Vec<(String, Vec<f32>)> = Vec::new();
        for scalar in [false, true] {
            e.set_scalar(scalar);
            for frame in [false, true] {
                let r = run(e.as_mut(), &x, &mut y, frame, reps);
                report(&man.name, e.as_ref(), frame, x.len(), r);
                let p = format!("{prefix}_{}_{}.f32", if scalar { "scalar" } else { "simd" }, if frame { "frame" } else { "block" });
                write_f32(&p, &y);
                outs.push((p, y.clone()));
            }
        }
        for (p, o) in &outs[1..] {
            let d = o.iter().zip(&outs[0].1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("max|{} - {}| = {:e}", p, outs[0].0, d);
        }
        return;
    }
    e.set_scalar(flag("--scalar"));
    let frame = flag("--frame");
    let r = run(e.as_mut(), &x, &mut y, frame, reps);
    report(&man.name, e.as_ref(), frame, x.len(), r);
    write_f32(&args[4], &y);
}

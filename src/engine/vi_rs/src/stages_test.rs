// Realtime smoothing contract: a frame run as `set_stages(n)` parts via `process_stage` equals
// `process_frame` bit-exactly, for every model package in models/ (and keeps doing so over many
// frames, so streaming state is carried correctly). Needs `--features full`.
use std::path::Path;
use crate::Manifest;

fn signal(n: usize) -> Vec<f32> {
    // speech-ish tone bursts + noise, with a silent stretch (exercises DFN3's silence skip)
    let mut s = 1u32;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (s >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let silent = (n / 3..n / 3 + n / 8).contains(&i);
            if silent { 0.0 } else { 0.3 * (i as f32 * 0.031).sin() * ((i / 4000) % 2) as f32 + 0.05 * noise }
        })
        .collect()
}

#[test]
fn staged_equals_unstaged_for_every_package() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models");
    let mut checked = 0;
    for d in std::fs::read_dir(&root).unwrap() {
        let m = d.unwrap().path().join("manifest.json");
        if !m.exists() {
            continue;
        }
        let man = Manifest::load(&m).unwrap();
        let mut a = match man.create() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip {}: {e}", m.display());
                continue;
            }
        };
        let fs = a.frame_size();
        let frames = (2 * a.info().sample_rate as usize / fs).clamp(50, 400);
        let x = signal(frames * fs);
        let mut ya = vec![0f32; x.len()];
        for (i, o) in x.chunks_exact(fs).zip(ya.chunks_exact_mut(fs)) {
            a.process_frame(i, o);
        }
        for want in [2, 3, 4] {
            let mut b = man.create().unwrap();
            let parts = b.set_stages(want);
            assert!(parts >= 1 && parts <= want, "{}: {parts} parts for {want}", man.name);
            let mut yb = vec![0f32; x.len()];
            for (i, o) in x.chunks_exact(fs).zip(yb.chunks_exact_mut(fs)) {
                for s in 0..parts {
                    b.process_stage(s, i, o);
                }
            }
            let bad = ya.iter().zip(&yb).position(|(p, q)| p.to_bits() != q.to_bits());
            assert!(bad.is_none(), "{} with {parts} parts differs at sample {:?}", man.name, bad);
            eprintln!("{:24} {want} -> {parts} parts: bit-exact over {frames} frames", man.name);
        }
        checked += 1;
    }
    assert!(checked >= 7, "only {checked} packages checked");
}

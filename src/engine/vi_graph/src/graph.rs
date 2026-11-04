//! Generic operator-graph runtime for vi model packages (models/FORMAT.md).
//!
//! Load: parse graph JSON + raw f32 weights, resolve symbolic dims (`T` = frames per call),
//! run each op's prepare/shape function (validates shapes, pre-packs weights, precomputes index
//! maps and scratch), fold nodes whose inputs are all constant, allocate every tensor once.
//! Run: execute the remaining nodes in order, then swap state buffers (`in <- out`).
//! No allocation after load.

use crate::ops;
use vi_engine::json::Value;
use vi_kernels::Backend;

/// Tiny string-keyed map (FNV hash + linear scan): load-time only, much smaller in wasm than
/// std's HashMap (SipHash + hashbrown).
pub struct SMap<V> {
    v: Vec<(u64, String, V)>,
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl<V> SMap<V> {
    pub fn new() -> Self {
        SMap { v: Vec::new() }
    }
    pub fn get(&self, k: &str) -> Option<&V> {
        let h = fnv(k);
        self.v.iter().find(|e| e.0 == h && e.1 == k).map(|e| &e.2)
    }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut V> {
        let h = fnv(k);
        self.v.iter_mut().find(|e| e.0 == h && e.1 == k).map(|e| &mut e.2)
    }
    pub fn contains_key(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn insert(&mut self, k: String, val: V) {
        match self.get_mut(&k) {
            Some(e) => *e = val,
            None => self.v.push((fnv(&k), k, val)),
        }
    }
}

impl<V: Default> SMap<V> {
    pub fn entry_or_default(&mut self, k: &str) -> &mut V {
        if !self.contains_key(k) {
            self.v.push((fnv(k), k.to_string(), V::default()));
        }
        self.get_mut(k).unwrap()
    }
}

pub struct Tensor {
    pub shape: Vec<usize>,
    /// own storage (constants, graph inputs/outputs, states); empty for arena-planned tensors
    pub data: Vec<f32>,
    /// offset into the shared arena (usize::MAX = uses `data`)
    pub off: usize,
    pub len: usize,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor { shape, data, off: usize::MAX, len }
    }
}

pub fn numel(s: &[usize]) -> usize {
    s.iter().product()
}

/// Raw view of all tensors for op kernels. Graph tensors are SSA (an op never writes one of its
/// own inputs), so handing out `&` for inputs and `&mut` for outputs at the same time is sound.
pub struct Tensors {
    p: *mut Tensor,
    n: usize,
    a: *mut f32,
}

impl Tensors {
    #[inline]
    pub fn d(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.n);
        unsafe {
            let t = &*self.p.add(i);
            if t.off == usize::MAX { &t.data } else { std::slice::from_raw_parts(self.a.add(t.off), t.len) }
        }
    }
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn dm(&self, i: usize) -> &mut [f32] {
        debug_assert!(i < self.n);
        unsafe {
            let t = &mut *self.p.add(i);
            if t.off == usize::MAX { &mut t.data } else { std::slice::from_raw_parts_mut(self.a.add(t.off), t.len) }
        }
    }
}

pub trait Op {
    fn run(&mut self, ts: &Tensors, ins: &[usize], outs: &[usize], be: Backend);
    /// clear internal streaming state (ops that own state, e.g. fused StreamConv)
    fn reset(&mut self) {}
}

/// Static info about one node input at prepare time.
pub struct In<'a> {
    pub shape: &'a [usize],
    /// Some if the input is constant (initializer or folded)
    pub data: Option<&'a [f32]>,
}

pub struct Prepared {
    pub out_shapes: Vec<Vec<usize>>,
    pub op: Box<dyn Op>,
}

/// Node attributes with symbolic-dim evaluation.
pub struct Attrs<'a> {
    pub v: Option<&'a Value>,
    pub t: usize,
}

pub fn eval_dim(v: &Value, t: usize) -> Result<i64, String> {
    match v {
        Value::Num(x) => Ok(*x as i64),
        Value::Str(s) => {
            let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
            let mut tot = 0i64;
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() {
                let mut sign = 1;
                if b[i] == b'+' || b[i] == b'-' {
                    if b[i] == b'-' {
                        sign = -1;
                    }
                    i += 1;
                }
                let st = i;
                while i < b.len() && b[i] != b'+' && b[i] != b'-' {
                    i += 1;
                }
                let term = &s[st..i];
                let val = if term == "T" {
                    t as i64
                } else if let Some(k) = term.strip_suffix("*T") {
                    k.parse::<i64>().map_err(|_| format!("bad dim {s}"))? * t as i64
                } else if let Some(k) = term.strip_prefix("T*") {
                    k.parse::<i64>().map_err(|_| format!("bad dim {s}"))? * t as i64
                } else {
                    term.parse::<i64>().map_err(|_| format!("bad dim {s}"))?
                };
                tot += sign * val;
            }
            Ok(tot)
        }
        _ => Err("bad dim (expected int or affine string)".into()),
    }
}

impl<'a> Attrs<'a> {
    fn get(&self, k: &str) -> Option<&'a Value> {
        self.v.and_then(|v| v.get(k))
    }
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn int(&self, k: &str, d: i64) -> Result<i64, String> {
        match self.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => eval_dim(v, self.t).map_err(|e| format!("attr {k}: {e}")),
        }
    }
    pub fn ints(&self, k: &str) -> Result<Option<Vec<i64>>, String> {
        match self.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Array(a)) => a.iter().map(|v| eval_dim(v, self.t)).collect::<Result<Vec<_>, _>>().map(Some),
            Some(v) => Ok(Some(vec![eval_dim(v, self.t)?])),
        }
    }
    pub fn float(&self, k: &str, d: f32) -> Result<f32, String> {
        match self.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(Value::Num(x)) => Ok(*x as f32),
            Some(Value::Str(s)) if s == "inf" => Ok(f32::INFINITY),
            Some(Value::Str(s)) if s == "-inf" => Ok(f32::NEG_INFINITY),
            Some(_) => Err(format!("attr {k}: expected number")),
        }
    }
    pub fn str(&self, k: &str, d: &'a str) -> &'a str {
        self.get(k).and_then(|v| v.as_str()).unwrap_or(d)
    }
}

struct Node {
    ins: Vec<usize>,
    outs: Vec<usize>,
    op: Box<dyn Op>,
}

pub struct Graph {
    pub tensors: Vec<Tensor>,
    arena: Vec<f32>,
    pub names: SMap<usize>,
    nodes: Vec<Node>,
    pub inputs: Vec<(String, usize)>,
    pub outputs: Vec<(String, usize)>,
    pub states: Vec<(usize, usize)>,
    pub backend: Backend,
    pub n_folded: usize,
    pub op_names: Vec<String>,
    /// node index boundaries of the `set_stages` parts (empty = one part)
    splits: Vec<usize>,
}

pub const NONE: usize = usize::MAX;

fn shape_of(v: Option<&Value>, t: usize) -> Result<Vec<usize>, String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .map(|d| eval_dim(d, t).and_then(|x| if x < 0 { Err(format!("negative dim {x}")) } else { Ok(x as usize) }))
            .collect(),
        _ => Err("missing shape".into()),
    }
}

impl Graph {
    /// `keep`: tensor names the caller reads/writes directly (never arena-planned).
    pub fn load(g: &Value, weights: &[u8], t: usize, keep: &[&str]) -> Result<Graph, String> {
        if g.get("format").and_then(|v| v.as_str()) != Some("vi-graph") {
            return Err("graph: format must be \"vi-graph\"".into());
        }
        if weights.len() % 4 != 0 {
            return Err("weights.bin length not a multiple of 4".into());
        }
        let nw = weights.len() / 4;
        let wf = |o: usize, n: usize| -> Result<Vec<f32>, String> {
            if o + n > nw {
                return Err(format!("initializer out of range ({o}+{n} > {nw})"));
            }
            Ok(weights[4 * o..4 * (o + n)].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut is_const: Vec<bool> = Vec::new();
        let mut names: SMap<usize> = SMap::new();
        fn add(names: &mut SMap<usize>, name: &str, t: Tensor, c: bool, tensors: &mut Vec<Tensor>, is_const: &mut Vec<bool>) -> Result<usize, String> {
            if names.contains_key(name) {
                return Err(format!("tensor {name} defined twice"));
            }
            tensors.push(t);
            is_const.push(c);
            names.insert(name.to_string(), tensors.len() - 1);
            Ok(tensors.len() - 1)
        }
        if let Some(Value::Object(inits)) = g.get("initializers") {
            for (name, d) in inits {
                let shape = shape_of(d.get("shape"), t)?;
                let n = numel(&shape);
                let data = if let Some(Value::Array(v)) = d.get("value") {
                    let v: Vec<f32> = v.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                    if v.len() != n {
                        return Err(format!("initializer {name}: {} values for shape {shape:?}", v.len()));
                    }
                    v
                } else {
                    wf(d.num_or("offset", -1.0).max(0.0) as usize, n)?
                };
                add(&mut names, name, Tensor::new(shape, data), true, &mut tensors, &mut is_const)?;
            }
        }
        let mut inputs = Vec::new();
        if let Some(Value::Array(a)) = g.get("inputs") {
            for i in a {
                let name = i.get("name").and_then(|v| v.as_str()).ok_or("input without name")?;
                let shape = shape_of(i.get("shape"), t).map_err(|e| format!("input {name}: {e}"))?;
                let n = numel(&shape);
                let id = add(&mut names, name, Tensor::new(shape, vec![0.0; n]), false, &mut tensors, &mut is_const)?;
                inputs.push((name.to_string(), id));
            }
        }
        let (node_list_f, fused_states) = fuse_stream_conv(g);
        let node_list_f = fuse_activation(g, node_list_f);
        let mut nodes = Vec::new();
        let mut n_folded = 0;
        let mut op_names = Vec::new();
        let be = Backend::detect();
        let node_list = &node_list_f;
        for (ni, nd) in node_list.iter().enumerate() {
            let op = nd.get("op").and_then(|v| v.as_str()).ok_or(format!("node {ni}: missing op"))?;
            let nname = nd.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let ctx = |e: String| format!("node {ni} ({op} {nname}): {e}");
            let mut ins = Vec::new();
            if let Some(Value::Array(a)) = nd.get("inputs") {
                for v in a {
                    let s = v.as_str().unwrap_or("");
                    if s.is_empty() {
                        ins.push(NONE);
                    } else {
                        ins.push(*names.get(s).ok_or_else(|| ctx(format!("unknown input tensor {s}")))?);
                    }
                }
            }
            let out_names: Vec<&str> = match nd.get("outputs") {
                Some(Value::Array(a)) => a.iter().map(|v| v.as_str().unwrap_or("")).collect(),
                _ => vec![],
            };
            let attrs = Attrs { v: nd.get("attrs"), t };
            let prep = {
                let infos: Vec<Option<In>> = ins
                    .iter()
                    .map(|&i| {
                        if i == NONE {
                            None
                        } else {
                            Some(In { shape: &tensors[i].shape, data: if is_const[i] { Some(&tensors[i].data[..]) } else { None } })
                        }
                    })
                    .collect();
                ops::prepare(op, &attrs, &infos, out_names.len()).map_err(ctx)?
            };
            if prep.out_shapes.len() < out_names.iter().filter(|s| !s.is_empty()).count() {
                return Err(ctx("too many outputs".into()));
            }
            let all_const = ins.iter().all(|&i| i == NONE || is_const[i]);
            let mut outs = Vec::new();
            for (k, on) in out_names.iter().enumerate() {
                let shape = prep.out_shapes.get(k).cloned().unwrap_or_default();
                let n = numel(&shape);
                if on.is_empty() {
                    // unnamed optional output: still needs a buffer
                    // runtime outputs get storage only after memory planning (no load-time peak)
                    tensors.push(if all_const { Tensor::new(shape, vec![0.0; n]) } else { Tensor { shape, data: Vec::new(), off: usize::MAX, len: n } });
                    is_const.push(all_const);
                    outs.push(tensors.len() - 1);
                } else {
                    let t = if all_const { Tensor::new(shape, vec![0.0; n]) } else { Tensor { shape, data: Vec::new(), off: usize::MAX, len: n } };
                    outs.push(add(&mut names, on, t, all_const, &mut tensors, &mut is_const).map_err(ctx)?);
                }
            }
            let mut opb = prep.op;
            if all_const {
                let ts = Tensors { p: tensors.as_mut_ptr(), n: tensors.len(), a: std::ptr::null_mut() };
                opb.run(&ts, &ins, &outs, be);
                n_folded += 1;
            } else {
                op_names.push(op.to_string());
                nodes.push(Node { ins, outs, op: opb });
            }
        }
        let find = |n: &str| names.get(n).copied().ok_or(format!("unknown tensor {n}"));
        let mut outputs = Vec::new();
        if let Some(Value::Array(a)) = g.get("outputs") {
            for o in a {
                let n = o.get("name").and_then(|v| v.as_str()).ok_or("output without name")?;
                outputs.push((n.to_string(), find(n)?));
            }
        }
        let mut states = Vec::new();
        if let Some(Value::Array(a)) = g.get("states") {
            for s in a {
                if fused_states.iter().any(|f| Some(f.as_str()) == s.get("in").and_then(|v| v.as_str())) {
                    continue; // owned by a fused StreamConv op
                }
                let i = find(s.get("in").and_then(|v| v.as_str()).ok_or("state without in")?)?;
                let o = find(s.get("out").and_then(|v| v.as_str()).ok_or("state without out")?)?;
                if tensors[i].shape != tensors[o].shape {
                    return Err(format!("state shape mismatch {:?} vs {:?}", tensors[i].shape, tensors[o].shape));
                }
                if is_const[o] {
                    return Err("state output is constant".into());
                }
                states.push((i, o));
            }
        }
        // free constant data no runtime node reads (pre-packed copies live in the ops)
        let mut used = vec![false; tensors.len()];
        for n in &nodes {
            for &i in &n.ins {
                if i != NONE {
                    used[i] = true;
                }
            }
        }
        for &(_, i) in &outputs {
            used[i] = true;
        }
        for (i, t) in tensors.iter_mut().enumerate() {
            if is_const[i] && !used[i] {
                t.data = Vec::new();
            }
        }
        // ---- static memory planning: intermediates share one arena (liveness-based slot reuse)
        let nt = tensors.len();
        let mut pinned = vec![false; nt];
        for (i, c) in is_const.iter().enumerate() {
            pinned[i] = *c;
        }
        for &(_, i) in inputs.iter().chain(outputs.iter()) {
            pinned[i] = true;
        }
        for &(i, o) in &states {
            pinned[i] = true;
            pinned[o] = true;
        }
        for k in keep {
            if let Some(&i) = names.get(*k) {
                pinned[i] = true;
            }
        }
        let mut last = vec![usize::MAX; nt];
        for (k, n) in nodes.iter().enumerate() {
            for &i in &n.ins {
                if i != NONE {
                    last[i] = k;
                }
            }
        }
        let mut slots: Vec<(usize, bool)> = Vec::new(); // (size, busy)
        let mut slot_of = vec![usize::MAX; nt];
        for (k, n) in nodes.iter().enumerate() {
            for &o in &n.outs {
                if pinned[o] {
                    continue;
                }
                let len = tensors[o].len;
                let mut best = usize::MAX;
                for (si, &(sz, busy)) in slots.iter().enumerate() {
                    if !busy && sz >= len && (best == usize::MAX || sz < slots[best].0) {
                        best = si;
                    }
                }
                if best == usize::MAX {
                    // grow the largest free slot if any, else add one
                    if let Some(si) = (0..slots.len()).filter(|&si| !slots[si].1).max_by_key(|&si| slots[si].0) {
                        slots[si].0 = len;
                        best = si;
                    } else {
                        slots.push((len, false));
                        best = slots.len() - 1;
                    }
                }
                slots[best].1 = true;
                slot_of[o] = best;
            }
            let free = |i: usize, slots: &mut Vec<(usize, bool)>| {
                if slot_of[i] != usize::MAX {
                    slots[slot_of[i]].1 = false;
                }
            };
            for &i in &n.ins {
                if i != NONE && last[i] == k && !pinned[i] {
                    free(i, &mut slots);
                }
            }
            for &o in &n.outs {
                if last[o] == usize::MAX && !pinned[o] {
                    free(o, &mut slots);
                }
            }
        }
        let mut offs = Vec::with_capacity(slots.len());
        let mut total = 0;
        for &(sz, _) in &slots {
            offs.push(total);
            total += (sz + 15) & !15; // 64-byte aligned slots
        }
        for i in 0..nt {
            if slot_of[i] != usize::MAX {
                tensors[i].off = offs[slot_of[i]];
                tensors[i].data = Vec::new();
            }
        }
        // pinned runtime tensors (graph outputs, state outputs, caller-kept) get their own buffer
        for i in 0..nt {
            if slot_of[i] == usize::MAX && tensors[i].data.len() != tensors[i].len && !is_const[i] {
                tensors[i].data = vec![0.0; tensors[i].len];
            }
        }
        let arena = vec![0.0; total];
        Ok(Graph { tensors, arena, names, nodes, inputs, outputs, states, backend: be, n_folded, op_names, splits: Vec::new() })
    }

    pub fn id(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// bytes held by tensors (constants + pinned) and the arena (op-internal packed weights /
    /// scratch not included)
    pub fn mem_bytes(&self) -> (usize, usize) {
        (self.tensors.iter().map(|t| t.data.len() * 4).sum(), self.arena.len() * 4)
    }

    pub fn reset(&mut self) {
        for n in self.nodes.iter_mut() {
            n.op.reset();
        }
        for &(i, o) in &self.states {
            self.tensors[i].data.iter_mut().for_each(|x| *x = 0.0);
            self.tensors[o].data.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Execute once (inputs already written into their tensors).
    pub fn run(&mut self) {
        self.run_nodes(0, self.nodes.len());
        self.swap_states();
    }

    /// Split the node list into up to `n` contiguous parts of about equal estimated cost. Returns
    /// the part count. Cost of a node = memory touched (inputs incl. weights + outputs) +
    /// multiply-adds when it has a weight tensor: outputs x weights / (the weight dim that is the
    /// output channel dim), i.e. GEMM T*K*N, conv out_positions*Cout*Cin*k.
    // (plain loops on purpose: iterator adapters here cost ~3 KB of wasm)
    pub fn set_stages(&mut self, n: usize) -> usize {
        let mut io = Vec::new();
        for &(_, i) in self.inputs.iter() {
            io.push(i);
        }
        for &(_, i) in self.outputs.iter() {
            io.push(i);
        }
        for &(a, b) in self.states.iter() {
            io.push(a);
            io.push(b);
        }
        let mut cost = Vec::with_capacity(self.nodes.len());
        let mut total = 0.0;
        for (k, nd) in self.nodes.iter().enumerate() {
            let mut c = 1.0;
            for &i in nd.ins.iter().chain(nd.outs.iter()) {
                if i != NONE {
                    c += numel(&self.tensors[i].shape) as f64;
                }
            }
            // weights: constant inputs with storage that aren't graph I/O
            let (mut wmax, mut wsum, mut wbig) = (0usize, 0f64, NONE);
            for &i in nd.ins.iter() {
                if i != NONE && !self.tensors[i].data.is_empty() && !io.contains(&i) {
                    let m = numel(&self.tensors[i].shape);
                    if m > 1 {
                        wsum += m as f64;
                        if m > wmax {
                            wmax = m;
                            wbig = i;
                        }
                    }
                }
            }
            if wbig != NONE && !nd.outs.is_empty() {
                let os = &self.tensors[nd.outs[0]].shape;
                let on = numel(os) as f64;
                let op = self.op_names[k].as_str();
                c += if op == "GRU" || op == "LSTM" || op == "RNN" {
                    // recurrent: every (step, batch) applies all gate weights (W and R)
                    on / (*os.last().unwrap_or(&1)).max(1) as f64 * wsum
                } else {
                    let mut oc = 0;
                    for &d in self.tensors[wbig].shape.iter() {
                        if d > 1 && os.contains(&d) {
                            oc = d;
                            break;
                        }
                    }
                    if oc > 0 { on * wmax as f64 / oc as f64 } else { wmax as f64 }
                };
            }
            total += c;
            cost.push(c);
        }
        let n = n.clamp(1, self.nodes.len().max(1));
        let mut splits = vec![0];
        let mut acc = 0.0;
        for (k, c) in cost.iter().enumerate() {
            // close a part once its share is reached (half of the next node may still go in)
            if splits.len() < n && acc + 0.5 * c >= total * splits.len() as f64 / n as f64 && k > *splits.last().unwrap() {
                splits.push(k);
            }
            acc += c;
        }
        splits.push(self.nodes.len());
        self.splits = splits;
        self.splits.len() - 1
    }

    /// Part `s` of `set_stages` (all parts in order == `run`).
    pub fn run_stage(&mut self, s: usize) {
        if self.splits.len() < 2 {
            if s == 0 {
                self.run();
            }
            return;
        }
        if s + 1 >= self.splits.len() {
            return;
        }
        self.run_nodes(self.splits[s], self.splits[s + 1]);
        if s + 2 == self.splits.len() {
            self.swap_states();
        }
    }

    fn run_nodes(&mut self, a: usize, b: usize) {
        let be = self.backend;
        let ts = Tensors { p: self.tensors.as_mut_ptr(), n: self.tensors.len(), a: self.arena.as_mut_ptr() };
        for n in self.nodes[a..b].iter_mut() {
            n.op.run(&ts, &n.ins, &n.outs, be);
        }
    }

    fn swap_states(&mut self) {
        for &(i, o) in &self.states {
            let a = std::mem::take(&mut self.tensors[i].data);
            let b = std::mem::replace(&mut self.tensors[o].data, a);
            self.tensors[i].data = b;
        }
    }

    /// Per-node timing (profiling helper for vi_cli --profile): accumulates ns per node.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_profiled(&mut self, acc: &mut [f64]) {
        let be = self.backend;
        let ts = Tensors { p: self.tensors.as_mut_ptr(), n: self.tensors.len(), a: self.arena.as_mut_ptr() };
        for (k, n) in self.nodes.iter_mut().enumerate() {
            let t0 = std::time::Instant::now();
            n.op.run(&ts, &n.ins, &n.outs, be);
            acc[k] += t0.elapsed().as_secs_f64();
        }
        for &(i, o) in &self.states {
            let a = std::mem::take(&mut self.tensors[i].data);
            let b = std::mem::replace(&mut self.tensors[o].data, a);
            self.tensors[i].data = b;
        }
    }
}

/// Load-time rewrite: `Y = Concat([S_in, X], axis=-1)`, `S_out = Slice(Y, starts=[-N], axes=[-1])`,
/// `Z = Conv(Y, W, B)` (1-D) where (S_in, S_out) is a state pair and Y has no other consumer
/// -> `Z = StreamConv(X, W, B; history=N)` whose kernel keeps the history in a ring buffer
/// (no per-call copy of the whole history). Returns the new node list + fused state inputs.
fn fuse_stream_conv(g: &Value) -> (Vec<Value>, Vec<String>) {
    let nodes: Vec<Value> = match g.get("nodes") {
        Some(Value::Array(a)) => a.clone(),
        _ => return (vec![], vec![]),
    };
    let mut st: SMap<String> = SMap::new();
    if let Some(Value::Array(a)) = g.get("states") {
        for s in a {
            if let (Some(i), Some(o)) = (s.get("in").and_then(|v| v.as_str()), s.get("out").and_then(|v| v.as_str())) {
                st.insert(i.to_string(), o.to_string());
            }
        }
    }
    let ins_of = |n: &Value| -> Vec<String> {
        match n.get("inputs") {
            Some(Value::Array(a)) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(),
            _ => vec![],
        }
    };
    let outs_of = |n: &Value| -> Vec<String> {
        match n.get("outputs") {
            Some(Value::Array(a)) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(),
            _ => vec![],
        }
    };
    let op_of = |n: &Value| n.get("op").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let mut cons: SMap<Vec<(usize, usize)>> = SMap::new();
    for (k, n) in nodes.iter().enumerate() {
        for (j, i) in ins_of(n).iter().enumerate() {
            cons.entry_or_default(i).push((k, j));
        }
    }
    let mut gout = Vec::new();
    if let Some(Value::Array(a)) = g.get("outputs") {
        for o in a {
            if let Some(n) = o.get("name").and_then(|v| v.as_str()) {
                gout.push(n.to_string());
            }
        }
    }
    let attr_ints = |n: &Value, k: &str| -> Option<Vec<i64>> {
        match n.get("attrs").and_then(|a| a.get(k)) {
            Some(Value::Array(a)) => a.iter().map(|v| v.as_f64().map(|x| x as i64)).collect(),
            _ => None,
        }
    };
    let mut drop = vec![false; nodes.len()];
    let mut repl: Vec<Option<Value>> = vec![None; nodes.len()];
    let mut fused = Vec::new();
    for (k, n) in nodes.iter().enumerate() {
        if op_of(n) != "Concat" {
            continue;
        }
        let ci = ins_of(n);
        let co = outs_of(n);
        if ci.len() != 2 || co.len() != 1 || !st.contains_key(&ci[0]) || gout.contains(&co[0]) {
            continue;
        }
        let ax = n.get("attrs").and_then(|a| a.get("axis")).and_then(|v| v.as_f64()).unwrap_or(0.0) as i64;
        let y = &co[0];
        let users = match cons.get(y) {
            Some(u) if u.len() == 2 => u.clone(),
            _ => continue,
        };
        let (mut conv, mut sl) = (None, None);
        for &(ui, uj) in &users {
            match op_of(&nodes[ui]).as_str() {
                "Conv" if uj == 0 => conv = Some(ui),
                "Slice" => sl = Some(ui),
                _ => {}
            }
        }
        let (Some(cv), Some(sv)) = (conv, sl) else { continue };
        let nlc = nodes[cv].get("attrs").and_then(|a| a.get("channels_last")).and_then(|v| v.as_f64()).unwrap_or(0.0) != 0.0;
        let tax: [i64; 2] = if nlc { [1, -2] } else { [2, -1] };
        if !tax.contains(&ax) {
            continue;
        }
        let so = outs_of(&nodes[sv]);
        if so.len() != 1 || Some(&so[0]) != st.get(&ci[0]) {
            continue;
        }
        let (Some(starts), Some(ends)) = (attr_ints(&nodes[sv], "starts"), attr_ints(&nodes[sv], "ends")) else { continue };
        let axes = attr_ints(&nodes[sv], "axes").unwrap_or(vec![0]);
        if starts.len() != 1 || starts[0] >= 0 || ends[0] < i32::MAX as i64 || !tax.contains(&axes[0]) {
            continue;
        }
        if attr_ints(&nodes[sv], "steps").map_or(false, |s| s != vec![1]) {
            continue;
        }
        let ks = attr_ints(&nodes[cv], "kernel_shape").unwrap_or_default();
        let pads = attr_ints(&nodes[cv], "pads").unwrap_or(vec![0, 0]);
        if ks.len() != 1 || pads.iter().any(|&p| p != 0) {
            continue;
        }
        // build the fused node
        let mut cins = ins_of(&nodes[cv]);
        cins[0] = ci[1].clone();
        let mut attrs = match nodes[cv].get("attrs") {
            Some(Value::Object(o)) => o.clone(),
            _ => vec![],
        };
        attrs.push(("history".into(), Value::Num((-starts[0]) as f64)));
        let mut nn = match &nodes[cv] {
            Value::Object(o) => o.clone(),
            _ => continue,
        };
        for e in nn.iter_mut() {
            match e.0.as_str() {
                "op" => e.1 = Value::Str("StreamConv".into()),
                "inputs" => e.1 = Value::Array(cins.iter().map(|s| Value::Str(s.clone())).collect()),
                "attrs" => e.1 = Value::Object(attrs.clone()),
                _ => {}
            }
        }
        if !nn.iter().any(|e| e.0 == "attrs") {
            nn.push(("attrs".into(), Value::Object(attrs)));
        }
        drop[k] = true;
        drop[sv] = true;
        repl[cv] = Some(Value::Object(nn));
        fused.push(ci[0].clone());
    }
    let mut out = Vec::with_capacity(nodes.len());
    for (k, n) in nodes.into_iter().enumerate() {
        if drop[k] {
            continue;
        }
        out.push(repl[k].take().unwrap_or(n));
    }
    (out, fused)
}

/// Load-time rewrite: `Y = Act(Conv(...))` with a single consumer -> Conv with attrs
/// `activation` / `act_alpha` applied in place as an epilogue (Relu, LeakyRelu, Sigmoid, Tanh).
fn fuse_activation(g: &Value, nodes: Vec<Value>) -> Vec<Value> {
    let s = |v: &Value, k: &str| -> Vec<String> {
        match v.get(k) {
            Some(Value::Array(a)) => a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
            _ => vec![],
        }
    };
    let mut protected: Vec<String> = Vec::new();
    if let Some(Value::Array(a)) = g.get("outputs") {
        for o in a {
            if let Some(n) = o.get("name").and_then(|v| v.as_str()) {
                protected.push(n.into());
            }
        }
    }
    if let Some(Value::Array(a)) = g.get("states") {
        for o in a {
            if let Some(n) = o.get("out").and_then(|v| v.as_str()) {
                protected.push(n.into());
            }
        }
    }
    let mut cnt: SMap<usize> = SMap::new();
    for n in &nodes {
        for i in s(n, "inputs") {
            *cnt.entry_or_default(&i) += 1;
        }
    }
    let mut prod: SMap<usize> = SMap::new();
    for (k, n) in nodes.iter().enumerate() {
        for o in s(n, "outputs") {
            prod.insert(o, k);
        }
    }
    let mut nodes = nodes;
    let mut drop = vec![false; nodes.len()];
    for k in 0..nodes.len() {
        let op = nodes[k].get("op").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if !matches!(op.as_str(), "Relu" | "LeakyRelu" | "Sigmoid" | "Tanh") {
            continue;
        }
        let ins = s(&nodes[k], "inputs");
        let outs = s(&nodes[k], "outputs");
        if ins.len() != 1 || outs.len() != 1 || cnt.get(&ins[0]) != Some(&1) || protected.contains(&ins[0]) {
            continue;
        }
        let Some(&pk) = prod.get(&ins[0]) else { continue };
        let pop = nodes[pk].get("op").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(pop, "Conv" | "StreamConv") || nodes[pk].get("attrs").and_then(|a| a.get("activation")).is_some() || drop[pk] {
            continue;
        }
        let alpha = nodes[k].get("attrs").and_then(|a| a.get("alpha")).and_then(|v| v.as_f64()).unwrap_or(0.01);
        if let Value::Object(o) = &mut nodes[pk] {
            if !o.iter().any(|e| e.0 == "attrs") {
                o.push(("attrs".into(), Value::Object(vec![])));
            }
            for e in o.iter_mut() {
                match e.0.as_str() {
                    "attrs" => {
                        if let Value::Object(a) = &mut e.1 {
                            a.push(("activation".into(), Value::Str(op.clone())));
                            a.push(("act_alpha".into(), Value::Num(alpha)));
                        }
                    }
                    "outputs" => e.1 = Value::Array(vec![Value::Str(outs[0].clone())]),
                    _ => {}
                }
            }
        }
        drop[k] = true;
    }
    nodes.into_iter().enumerate().filter(|(k, _)| !drop[*k]).map(|(_, n)| n).collect()
}

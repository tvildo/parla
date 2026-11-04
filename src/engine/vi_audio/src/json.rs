//! Minimal JSON parser (std only) for manifests / engine configs.

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, k: &str) -> Option<&Value> {
        match self {
            Value::Object(v) => v.iter().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Value::Num(x) = self { Some(*x) } else { None }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }
    pub fn num_or(&self, k: &str, d: f64) -> f64 {
        self.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
    }
}

pub fn parse(s: &str) -> Result<Value, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i != b.len() {
        return Err(format!("json: trailing data at {i}"));
    }
    Ok(v)
}

fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Value, String> {
    ws(b, i);
    let err = |i: usize| format!("json: unexpected input at {i}");
    match b.get(*i).ok_or("json: unexpected end")? {
        b'{' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Value::Object(v));
            }
            loop {
                ws(b, i);
                let Value::Str(k) = value(b, i)? else { return Err(err(*i)) };
                ws(b, i);
                if b.get(*i) != Some(&b':') { return Err(err(*i)); }
                *i += 1;
                v.push((k, value(b, i)?));
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => { *i += 1; return Ok(Value::Object(v)); }
                    _ => return Err(err(*i)),
                }
            }
        }
        b'[' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Value::Array(v));
            }
            loop {
                v.push(value(b, i)?);
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => { *i += 1; return Ok(Value::Array(v)); }
                    _ => return Err(err(*i)),
                }
            }
        }
        b'"' => {
            *i += 1;
            let mut s = String::new();
            loop {
                let c = *b.get(*i).ok_or("json: unterminated string")?;
                *i += 1;
                match c {
                    b'"' => return Ok(Value::Str(s)),
                    b'\\' => {
                        let e = *b.get(*i).ok_or("json: bad escape")?;
                        *i += 1;
                        match e {
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            b'r' => s.push('\r'),
                            b'b' => s.push('\u{8}'),
                            b'f' => s.push('\u{c}'),
                            b'u' => {
                                let h = std::str::from_utf8(&b[*i..*i + 4]).map_err(|_| err(*i))?;
                                let cp = u32::from_str_radix(h, 16).map_err(|_| err(*i))?;
                                s.push(char::from_u32(cp).unwrap_or('?'));
                                *i += 4;
                            }
                            other => s.push(other as char),
                        }
                    }
                    _ => {
                        // copy UTF-8 bytes through
                        let start = *i - 1;
                        let mut end = *i;
                        while end < b.len() && b[end] != b'"' && b[end] != b'\\' {
                            end += 1;
                        }
                        s.push_str(std::str::from_utf8(&b[start..end]).map_err(|_| err(start))?);
                        *i = end;
                    }
                }
            }
        }
        b't' if b[*i..].starts_with(b"true") => { *i += 4; Ok(Value::Bool(true)) }
        b'f' if b[*i..].starts_with(b"false") => { *i += 5; Ok(Value::Bool(false)) }
        b'n' if b[*i..].starts_with(b"null") => { *i += 4; Ok(Value::Null) }
        _ => {
            let start = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            parse_num(&b[start..*i]).map(Value::Num).ok_or_else(|| err(start))
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Num(x) => write!(f, "{x}"),
            Value::Str(s) => {
                write!(f, "\"")?;
                for c in s.chars() {
                    match c {
                        '"' => write!(f, "\\\"")?,
                        '\\' => write!(f, "\\\\")?,
                        '\n' => write!(f, "\\n")?,
                        c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
                        c => write!(f, "{c}")?,
                    }
                }
                write!(f, "\"")
            }
            Value::Array(v) => {
                write!(f, "[")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{x}")?;
                }
                write!(f, "]")
            }
            Value::Object(v) => {
                write!(f, "{{")?;
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{}:{x}", Value::Str(k.clone()))?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Small decimal parser (no std dec2flt tables -> smaller wasm). Mantissa digits are accumulated
/// exactly in u64 (first 19 significant digits), the decimal exponent is applied with exact powers
/// of ten up to 1e22 (one rounding step) -> correctly rounded for typical values; values are used
/// as f32 (weights/attrs) where this is exact.
pub fn parse_num(b: &[u8]) -> Option<f64> {
    let mut i = 0;
    let neg = b.first() == Some(&b'-');
    if neg || b.first() == Some(&b'+') {
        i += 1;
    }
    let (mut m, mut exp, mut nd, mut any) = (0u64, 0i32, 0, false);
    while i < b.len() && b[i].is_ascii_digit() {
        any = true;
        if nd < 19 {
            m = m * 10 + (b[i] - b'0') as u64;
            if m > 0 {
                nd += 1;
            }
        } else {
            exp += 1;
        }
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            any = true;
            if nd < 19 {
                m = m * 10 + (b[i] - b'0') as u64;
                if m > 0 {
                    nd += 1;
                }
                exp -= 1;
            }
            i += 1;
        }
    }
    if !any {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        let en = b.get(i) == Some(&b'-');
        if en || b.get(i) == Some(&b'+') {
            i += 1;
        }
        let mut e = 0i32;
        let mut ed = false;
        while i < b.len() && b[i].is_ascii_digit() {
            e = (e * 10 + (b[i] - b'0') as i32).min(100000);
            i += 1;
            ed = true;
        }
        if !ed {
            return None;
        }
        exp += if en { -e } else { e };
    }
    if i != b.len() {
        return None;
    }
    const P: [f64; 23] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22];
    let mut v = m as f64;
    let mut e = exp;
    while e > 22 {
        v *= 1e22;
        e -= 22;
    }
    while e < -22 {
        v /= 1e22;
        e += 22;
    }
    v = if e >= 0 { v * P[e as usize] } else { v / P[(-e) as usize] };
    Some(if neg { -v } else { v })
}

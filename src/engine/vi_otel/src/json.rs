//! Minimal JSON writer without `core::fmt`: integers, hex, escaped strings and doubles are
//! written digit by digit (keeps wasm small and the encoder allocation-free beyond `Vec` growth).

pub struct W<'a>(pub &'a mut Vec<u8>);

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Exact powers of ten representable in f64.
const P10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
    1e20, 1e21, 1e22,
];

fn scale10(mut x: f64, mut p: i32) -> f64 {
    while p > 22 {
        x *= 1e22;
        p -= 22;
    }
    while p < -22 {
        x /= 1e22;
        p += 22;
    }
    if p >= 0 {
        x * P10[p as usize]
    } else {
        x / P10[(-p) as usize]
    }
}

impl<'a> W<'a> {
    #[inline]
    pub fn raw(&mut self, s: &[u8]) {
        self.0.extend_from_slice(s)
    }
    #[inline]
    pub fn ch(&mut self, c: u8) {
        self.0.push(c)
    }

    /// Quoted, escaped JSON string.
    pub fn str(&mut self, s: &str) {
        self.ch(b'"');
        let b = s.as_bytes();
        let mut start = 0;
        for (i, &c) in b.iter().enumerate() {
            if c == b'"' || c == b'\\' || c < 0x20 {
                self.raw(&b[start..i]);
                match c {
                    b'"' => self.raw(b"\\\""),
                    b'\\' => self.raw(b"\\\\"),
                    b'\n' => self.raw(b"\\n"),
                    b'\r' => self.raw(b"\\r"),
                    b'\t' => self.raw(b"\\t"),
                    _ => self.raw(&[b'\\', b'u', b'0', b'0', HEX[(c >> 4) as usize], HEX[(c & 15) as usize]]),
                }
                start = i + 1;
            }
        }
        self.raw(&b[start..]);
        self.ch(b'"');
    }

    pub fn u64(&mut self, mut v: u64) {
        let mut buf = [0u8; 20];
        let mut i = 20;
        loop {
            i -= 1;
            buf[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.raw(&buf[i..]);
    }
    pub fn i64(&mut self, v: i64) {
        if v < 0 {
            self.ch(b'-');
        }
        self.u64(v.unsigned_abs())
    }
    /// 64-bit integers are JSON strings in the protobuf JSON mapping.
    pub fn u64_str(&mut self, v: u64) {
        self.ch(b'"');
        self.u64(v);
        self.ch(b'"');
    }
    pub fn i64_str(&mut self, v: i64) {
        self.ch(b'"');
        self.i64(v);
        self.ch(b'"');
    }
    /// Lowercase hex string (trace / span ids).
    pub fn hex(&mut self, b: &[u8]) {
        self.ch(b'"');
        for &x in b {
            self.ch(HEX[(x >> 4) as usize]);
            self.ch(HEX[(x & 15) as usize]);
        }
        self.ch(b'"');
    }

    /// Double: integers below 2^53 exactly, otherwise 15 significant digits (trailing zeros
    /// trimmed), scientific notation outside 1e-5..1e15. NaN / ±Infinity as the protobuf JSON
    /// strings "NaN" / "Infinity" / "-Infinity".
    pub fn f64(&mut self, x: f64) {
        if x.is_nan() {
            return self.raw(b"\"NaN\"");
        }
        if x.is_infinite() {
            return self.raw(if x > 0.0 { b"\"Infinity\"" } else { b"\"-Infinity\"" });
        }
        if x == 0.0 {
            return self.ch(b'0');
        }
        let mut x = x;
        if x < 0.0 {
            self.ch(b'-');
            x = -x;
        }
        if x < 9007199254740992.0 && (x as u64) as f64 == x {
            return self.u64(x as u64);
        }
        // decimal exponent estimate from the binary exponent, then fix up
        let e2 = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let mut k = (e2 as f64 * core::f64::consts::LOG10_2) as i32 - if e2 < 0 { 1 } else { 0 };
        let mut m = 0u64;
        for _ in 0..24 {
            let y = scale10(x, 14 - k);
            if y >= 1e15 {
                k += 1;
                continue;
            }
            if y < 1e14 {
                k -= 1;
                continue;
            }
            m = (y + 0.5) as u64;
            if m >= 1_000_000_000_000_000 {
                m /= 10;
                k += 1;
            }
            break;
        }
        // 15 digits of m, value = d.dddd x 10^k
        let mut d = [0u8; 15];
        let mut t = m;
        for i in (0..15).rev() {
            d[i] = b'0' + (t % 10) as u8;
            t /= 10;
        }
        let mut nd = 15;
        while nd > 1 && d[nd - 1] == b'0' {
            nd -= 1;
        }
        let d = &d[..nd];
        if (-5..15).contains(&k) {
            if k < 0 {
                self.raw(b"0.");
                for _ in 0..(-k - 1) {
                    self.ch(b'0');
                }
                self.raw(d);
            } else {
                let int = (k + 1) as usize;
                if nd <= int {
                    self.raw(d);
                    for _ in nd..int {
                        self.ch(b'0');
                    }
                } else {
                    self.raw(&d[..int]);
                    self.ch(b'.');
                    self.raw(&d[int..]);
                }
            }
        } else {
            self.ch(d[0]);
            if nd > 1 {
                self.ch(b'.');
                self.raw(&d[1..]);
            }
            self.ch(b'e');
            self.i64(k as i64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::W;

    fn f(x: f64) -> String {
        let mut v = Vec::new();
        W(&mut v).f64(x);
        String::from_utf8(v).unwrap()
    }

    #[test]
    fn doubles() {
        assert_eq!(f(0.0), "0");
        assert_eq!(f(5.0), "5");
        assert_eq!(f(-42.0), "-42");
        assert_eq!(f(0.1), "0.1");
        assert_eq!(f(2.5), "2.5");
        assert_eq!(f(-0.125), "-0.125");
        assert_eq!(f(1234.5678), "1234.5678");
        assert_eq!(f(1e-7), "1e-7");
        assert_eq!(f(1.5e-7), "1.5e-7");
        assert_eq!(f(0.00001), "0.00001");
        assert_eq!(f(1e20), "1e20");
        assert_eq!(f(1.25e300), "1.25e300");
        assert_eq!(f(f64::NAN), "\"NaN\"");
        assert_eq!(f(f64::NEG_INFINITY), "\"-Infinity\"");
        assert_eq!(f(1.0 / 3.0), "0.333333333333333");
        assert_eq!(f(123_456_789_012_345.6), "123456789012346");
        assert_eq!(f(1e23), "1e23");
        assert_eq!(f(9.999_999_999_999_997e22), "1e23"); // 15-digit rounding carries into the exponent
    }

    #[test]
    fn doubles_roundtrip_15_digits() {
        // every printed value must parse back within 15-significant-digit precision
        let mut s = 0x1234_5678_9abc_def1u64;
        for _ in 0..20000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits(s);
            if !x.is_normal() {
                continue;
            }
            let txt = f(x);
            let y: f64 = txt.parse().unwrap();
            let rel = if x == 0.0 { y.abs() } else { ((y - x) / x).abs() };
            assert!(rel < 1e-14, "{x:e} -> {txt} ({rel:e})");
        }
    }

    #[test]
    fn strings_and_ints() {
        let mut v = Vec::new();
        let mut w = W(&mut v);
        w.str("a\"b\\c\nd\u{1}é");
        w.ch(b' ');
        w.i64_str(i64::MIN);
        w.ch(b' ');
        w.u64_str(u64::MAX);
        w.ch(b' ');
        w.hex(&[0x00, 0xab, 0x0f]);
        assert_eq!(
            String::from_utf8(v).unwrap(),
            "\"a\\\"b\\\\c\\nd\\u0001é\" \"-9223372036854775808\" \"18446744073709551615\" \"00ab0f\""
        );
    }
}

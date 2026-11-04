//! Minimal OTLP/HTTP JSON transport over `std::net::TcpStream` (feature `native-http`).
//! Plain HTTP/1.1, one connection per request (`Connection: close`), no TLS, no redirects:
//! meant for a local collector (`127.0.0.1:4318`). Runs on an exporter thread, never the audio
//! thread.

use crate::metrics::SdkMeterProvider;
use crate::trace::{SdkTracerProvider, SpanData};
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub body: String,
}

/// POST `body` as `application/json` to `http://{addr}{path}`.
pub fn post(addr: &str, path: &str, body: &[u8], timeout: Duration) -> io::Result<Response> {
    let sa = addr
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no address"))?;
    let mut s = TcpStream::connect_timeout(&sa, timeout)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nUser-Agent: vi_otel/{}\r\n\r\n",
        body.len(),
        crate::resource::SDK_VERSION
    );
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    let mut resp = Vec::new();
    s.read_to_end(&mut resp)?;
    let text = String::from_utf8_lossy(&resp).into_owned();
    // "HTTP/1.1 200 OK"
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad status line"))?;
    let body = match text.find("\r\n\r\n") {
        Some(i) => text[i + 4..].to_string(),
        None => String::new(),
    };
    Ok(Response { status, body })
}

/// Reusable exporter: encodes into owned buffers and POSTs to `/v1/metrics` and `/v1/traces`.
pub struct OtlpHttpExporter {
    pub addr: String,
    pub timeout: Duration,
    pub max_spans_per_request: usize,
    buf: Vec<u8>,
    scratch: Vec<SpanData>,
}

impl OtlpHttpExporter {
    /// `addr` = "host:port" (e.g. "127.0.0.1:4318").
    pub fn new(addr: impl Into<String>) -> Self {
        OtlpHttpExporter {
            addr: addr.into(),
            timeout: Duration::from_secs(5),
            max_spans_per_request: 512,
            buf: Vec::with_capacity(16 << 10),
            scratch: Vec::new(),
        }
    }
    pub fn export_metrics(&mut self, mp: &SdkMeterProvider) -> io::Result<Response> {
        mp.export_json(&mut self.buf);
        post(&self.addr, "/v1/metrics", &self.buf, self.timeout)
    }
    /// Drains all pending spans in batches. Returns (spans sent, bytes sent, last response).
    pub fn export_spans(&mut self, tp: &SdkTracerProvider) -> io::Result<(usize, usize, Option<Response>)> {
        let (mut spans, mut bytes, mut last) = (0, 0, None);
        loop {
            let n = tp.export_json(&mut self.scratch, self.max_spans_per_request, &mut self.buf);
            if n == 0 {
                return Ok((spans, bytes, last));
            }
            spans += n;
            bytes += self.buf.len();
            last = Some(post(&self.addr, "/v1/traces", &self.buf, self.timeout)?);
        }
    }
    /// Last encoded payload (metrics; empty after `export_spans` drained the ring).
    pub fn last_payload(&self) -> &[u8] {
        &self.buf
    }
}

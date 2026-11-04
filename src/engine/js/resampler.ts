// resampler.ts — streaming rational (L/M) polyphase resampler, Kaiser-windowed sinc FIR.
// Pure JS, allocation-free in steady state. Used inside the AudioWorklet (48 kHz <-> engine rate).
//
//   const r = new Resampler(48000, 24000);   // any integer rates
//   const y = r.process(x);                  // Float32Array view (valid until next call)
//   r.delaySeconds                           // group delay of the FIR

function gcd(a: number, b: number): number { while (b) [a, b] = [b, a % b]; return a; }

function bessel0(x: number): number {
  let s = 1, t = 1;
  for (let k = 1; k < 50; k++) { t *= (x / (2 * k)) ** 2; s += t; if (t < 1e-12 * s) break; }
  return s;
}

export class Resampler {
  readonly L: number;
  readonly M: number;
  readonly identity: boolean;
  /** group delay of the FIR, seconds */
  readonly delaySeconds: number = 0;
  private readonly out: Float32Array;
  private ph = new Float32Array(0);
  private T = 0;
  private buf = new Float32Array(0);
  private n = 0;
  private u = 0;

  constructor(fsIn: number, fsOut: number, tapsPerPhase = 32, maxBlock = 4096) {
    const g = gcd(fsIn, fsOut);
    this.L = fsOut / g; this.M = fsIn / g;
    this.identity = this.L === 1 && this.M === 1;
    this.out = new Float32Array(Math.ceil(maxBlock * this.L / this.M) + 4);
    if (this.identity) return;
    const L = this.L, M = this.M, T = tapsPerPhase, K = T * L;
    const fc = 0.5 / Math.max(L, M) * 0.9; // cutoff in cycles/sample at the upsampled rate
    const beta = 8.6, c = (K - 1) / 2, i0b = bessel0(beta);
    // polyphase layout: ph[p*T + j] = h[p + j*L]
    this.ph = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const x = k - c;
      const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
      const w = bessel0(beta * Math.sqrt(Math.max(0, 1 - (x / c) ** 2))) / i0b;
      const p = k % L, j = (k - p) / L;
      this.ph[p * T + j] = L * sinc * w;
    }
    this.T = T;
    this.buf = new Float32Array(T - 1 + maxBlock);
    this.n = T - 1;            // valid samples in buf (history pre-filled with zeros)
    this.u = (T - 1) * L;      // upsampled-time index of the next output, relative to buf[0]
    this.delaySeconds = c / (fsIn * L);
  }

  process(x: Float32Array): Float32Array {
    if (this.identity) { this.out.set(x); return this.out.subarray(0, x.length); }
    const { L, M, T, ph, buf, out } = this;
    buf.set(x, this.n);
    this.n += x.length;
    let u = this.u, o = 0;
    for (;;) {
      const i = Math.floor(u / L);
      if (i >= this.n) break;
      const p = u - i * L, base = p * T;
      let s = 0;
      for (let j = 0; j < T; j++) s += ph[base + j] * buf[i - j];
      out[o++] = s;
      u += M;
    }
    // keep the last T-1 samples as history
    const drop = this.n - (T - 1);
    buf.copyWithin(0, drop, this.n);
    this.n = T - 1;
    this.u = u - drop * L;
    return out.subarray(0, o);
  }

  reset(): void {
    if (this.identity) return;
    this.buf.fill(0); this.n = this.T - 1; this.u = (this.T - 1) * this.L;
  }
}

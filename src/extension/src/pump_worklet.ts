// pump_worklet.ts — isolated mode, page side: a plain-JS AudioWorklet (no WebAssembly, so it runs under
// any page CSP that lets the extension's file load). Each render quantum goes to the engine worker over a
// MessagePort; processed quanta come back and play after a small jitter buffer (TARGET quanta, 2.67 ms
// each at 48 kHz). Buffers circulate (sent, processed, returned, reused): no allocation per quantum after
// warm-up. On a gap the output is silent for that quantum and the buffer re-primes; the count is reported.
// If the worker stops answering (the extension was updated or reloaded mid-call: its worker dies), the
// pump plays the microphone itself after STARVED quanta (~21 ms) instead of silence, until replies return.
const TARGET = 3;
const MAX = 16;
const STARVED = 8;

class ViPump extends AudioWorkletProcessor {
  q: Float32Array[] = [];
  pool: Float32Array[] = [];
  primed = false;
  underruns = 0;
  reported = 0;
  dry = 0;          // quanta in a row without a reply
  ap: MessagePort | null = null;
  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent) => {
      if (e.data && e.data.type === 'port' && e.ports[0]) {
        this.ap = e.ports[0];
        this.ap.onmessage = (ev: MessageEvent<Float32Array>) => {
          this.q.push(ev.data);
          while (this.q.length > MAX) this.pool.push(this.q.shift()!);   // fell behind: drop the oldest
        };
      }
    };
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0][0];
    if (this.ap) {
      const b = this.pool.pop() || new Float32Array(128);
      if (inp) b.set(inp); else b.fill(0);
      this.ap.postMessage(b, [b.buffer]);
    }
    if (this.q.length === 0) this.dry++; else if (this.q.length >= TARGET) this.dry = 0;
    if (this.dry > STARVED) {        // no engine any more: the plain microphone
      if (inp) out.set(inp); else out.fill(0);
      this.primed = false;
      return true;
    }
    if (!this.primed) {
      if (this.q.length < TARGET) { out.fill(0); return true; }
      this.primed = true;
    }
    const b = this.q.shift();
    if (b) {
      out.set(b);
      this.pool.push(b);
    } else {
      out.fill(0);
      this.primed = false;
      this.underruns++;
      if (this.underruns - this.reported >= 10 || this.reported === 0) {
        this.reported = this.underruns;
        this.port.postMessage({ type: 'pump', underruns: this.underruns });
      }
    }
    return true;
  }
}

registerProcessor('vi-pump', ViPump);

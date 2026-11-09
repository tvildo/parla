// echo_gate.ts — the popup shows the user's voice and background noise only, not the other side of the call coming
// back through the speakers. Used by inject.ts on the meter samples (~19 a second, only while a popup is open); the
// audio the call gets is not touched.
//
// On speakers the microphone hears the other side; the browser's echo cancellation removes most of it, and the call
// app's own canceller the rest before sending, but what reaches our engine still holds some of it, and the engine
// keeps voices. Given the level of the incoming call audio (farDb, what the page plays), a sample is an echo when the
// other side talks (the loudest incoming level of the last ~0.4 s, for the echo's delay) and the microphone is no
// louder than their echo usually is (`coupling`: the median of microphone minus incoming level while they talk,
// learned in the call) plus a margin. The user talking over them is louder than that and shows. An echo sample is
// shown as the background noise measured in the user's last pauses, never as speech.

export interface MeterLevels { inDb: number; outDb: number; speaking: boolean }

export class EchoGate {
  /** incoming level above which the other side counts as talking (dBFS) */
  static readonly FAR_ON = -55;
  /** how much louder than their usual echo the microphone must be to count as the user (dB) */
  static readonly MARGIN = 6;
  /** samples of incoming level kept: ~0.4 s at ~19 a second covers the echo's delay and tail */
  static readonly HOLD = 8;
  /** coupling step per sample (dB): a running median, ~6 dB a second */
  static readonly STEP = 0.3;

  private far: number[] = [];
  /** microphone minus incoming level while the other side talks: starts high (assume a loud echo) and learns */
  coupling = 0;
  private floorIn: number | null = null;
  private floorOut: number | null = null;

  step(s: MeterLevels, farDb: number): MeterLevels & { echo: boolean } {
    this.far.push(farDb);
    if (this.far.length > EchoGate.HOLD) this.far.shift();
    const farMax = Math.max(...this.far);
    const farOn = farMax > EchoGate.FAR_ON;
    if (farOn) this.coupling += s.inDb - farMax > this.coupling ? EchoGate.STEP : -EchoGate.STEP;
    const echo = farOn && s.inDb < farMax + this.coupling + EchoGate.MARGIN;
    if (!echo && !s.speaking) {
      this.floorIn = this.floorIn === null ? s.inDb : this.floorIn + 0.1 * (s.inDb - this.floorIn);
      this.floorOut = this.floorOut === null ? s.outDb : this.floorOut + 0.1 * (s.outDb - this.floorOut);
    }
    if (!echo) return { ...s, echo };
    return { inDb: Math.min(s.inDb, this.floorIn ?? s.inDb), outDb: Math.min(s.outDb, this.floorOut ?? s.outDb), speaking: false, echo };
  }
}

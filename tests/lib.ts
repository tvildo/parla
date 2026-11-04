// Shared helpers: the fake microphone (a generated WAV: no recording of anyone) and level measurement.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** A 12 s, 48 kHz mono WAV of steady noise + a 120 Hz hum + a 1 kHz whistle: everything in it is noise to a
 *  speech model, so with suppression on the output is far quieter than the input, off it is the input. */
export function fakeMic(): string {
  // under build/ (ours), not the shared temp folder: another user can't plant or swap the file there
  const dir = path.join(ROOT, 'build', 'test-media');
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, 'fake-mic.wav');
  if (fs.existsSync(file)) return file;
  const rate = 48000, n = rate * 12;
  const pcm = Buffer.alloc(n * 2);
  let s = 12345;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const t = i / rate;
    const v = 0.25 * (s / 2 ** 32 - 0.5) + 0.1 * Math.sin(2 * Math.PI * 120 * t) + 0.05 * Math.sin(2 * Math.PI * 1000 * t);
    pcm.writeInt16LE(Math.round(Math.max(-1, Math.min(1, v)) * 32767), i * 2);
  }
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8); h.write('fmt ', 12);
  h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, pcm]));
  return file;
}

/** RMS of x[a..b) (seconds at 48 kHz) */
export function rms(x: Float32Array, a = 0, b = x.length / 48000): number {
  const lo = Math.floor(a * 48000), hi = Math.min(x.length, Math.floor(b * 48000));
  let e = 0;
  for (let i = lo; i < hi; i++) e += x[i] * x[i];
  return Math.sqrt(e / Math.max(1, hi - lo));
}

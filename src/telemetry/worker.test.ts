// node src/telemetry/worker.test.ts — the receiver without Cloudflare: its fetch handler called directly.
import assert from 'node:assert/strict';
import worker, { clean } from './worker.ts';
import type { Env, CleanEvent } from './worker.ts';

type Logged = { parla: CleanEvent };
const post = async (body: unknown, headers: Record<string, string> = {}) => {
  const logs: Logged[] = [];
  const r = await worker.fetch(new Request('https://t.parla.cc/v1/e', { method: 'POST', body: typeof body === 'string' ? body : JSON.stringify(body), headers }), {}, {}, (x) => logs.push(x as Logged));
  return { status: r.status, logs };
};
const base = { v: 1, ts: 1761500000000, id: 'a1b2c3d4e5f60718293a4b5c', ext: '0.1.0', br: 'chrome', brv: 141, os: 'windows', cores: 16, lang: 'en-US', app: 'meet' };

// a valid batch: one log line per event, only known fields
let r = await post([{ ...base, t: 'call', mode: 'page', model: 'deepfilternet3', strength: 100, dur: '<5m', lat: 33, cost: 640, ip: '1.2.3.4', secret: 'x' },
  { ...base, t: 'error', code: 'start', msg: 'no\u0007wasm' }]);
assert.equal(r.status, 204);
assert.equal(r.logs.length, 2);
assert.equal(r.logs[0].parla.lat, 33);
assert.equal(r.logs[0].parla.ip, undefined, 'unknown fields dropped');
assert.equal(r.logs[0].parla.secret, undefined);
assert.equal(r.logs[1].parla.msg, 'no wasm', 'control characters removed');
// clamped and filtered
assert.equal(clean({ ...base, t: 'call', strength: 900 })!.strength, 100);
assert.equal(clean({ ...base, t: 'call', mode: 'evil' })!.mode, undefined);
assert.equal(clean({ ...base, t: 'nope' }), null, 'unknown type');
assert.equal(clean({ ...base, t: 'call', id: 'not-hex!' }), null, 'bad id');
assert.equal(clean({ ...base, t: 'call', app: 'https://example.com' })!.app, undefined, 'app is a fixed word, never an address');
// diagnostics: a domain only; an older build's `url` (host + path) is cut to its domain
assert.equal(clean({ ...base, t: 'call', domain: 'meet.google.com' })!.domain, 'meet.google.com');
assert.equal(clean({ ...base, t: 'call', domain: 'meet.google.com/abc-defg-hij' })!.domain, undefined, 'a path is refused');
assert.equal(clean({ ...base, t: 'call', url: 'meet.google.com/abc-defg-hij?x=1' })!.domain, 'meet.google.com');
assert.equal(clean({ ...base, t: 'call', url: 'localhost:9911/plain.html' })!.domain, 'localhost');
assert.equal(clean({ ...base, t: 'call', url: 'meet.google.com/abc' })!.url, undefined, 'the url field itself is never stored');
assert.equal(clean({ ...base, t: 'error', code: 'start', msg: 'fetch https://meet.google.com:443/abc?x=1#y failed' })!.msg, 'fetch meet.google.com failed');
// the build: version and commit; anything else in commit is dropped
assert.equal(clean({ ...base, t: 'active', commit: 'e54dc00a1b2c' })!.commit, 'e54dc00a1b2c');
assert.equal(clean({ ...base, t: 'active', commit: 'e54dc00a1b2c-dirty' })!.commit, 'e54dc00a1b2c-dirty');
assert.equal(clean({ ...base, t: 'active', commit: 'https://x' })!.commit, undefined);
// abuse
assert.equal((await post('x'.repeat(9000))).status, 413);
assert.equal((await post('{not json')).status, 400);
assert.equal((await post(Array.from({ length: 21 }, () => ({ ...base, t: 'call' })))).status, 400);
assert.equal((await worker.fetch(new Request('https://t.parla.cc/v1/e'), {}, {})).status, 405);
assert.equal((await worker.fetch(new Request('https://t.parla.cc/other', { method: 'POST' }), {}, {})).status, 404);
// junk in the batch: kept out, the rest logged
r = await post([{ t: 'call' }, 5, null, { ...base, t: 'settings', key: 'strength', value: 70 }]);
assert.equal(r.status, 204);
assert.deepEqual(r.logs.map((l) => l.parla.t), ['settings']);
console.log('worker: all checks passed');

// ---- turning reports down: sample share, daily limit ----
const ev1 = [{ ...base, t: 'active' }];
type Point = { indexes: string[]; blobs: string[]; doubles: number[] };
const call = (env: Env) => worker.fetch(new Request('https://t.parla.cc/v1/e', { method: 'POST', body: JSON.stringify(ev1) }), env, {}, () => {});
// the share to report goes back in every answer
assert.equal((await call({ SAMPLE: '0.25' })).headers.get('x-parla-sample'), '0.25');
assert.equal((await call({})).headers.get('x-parla-sample'), '1');
// SAMPLE 0: accepted (no retries), nothing written
{
  const pts: Point[] = [];
  const r0 = await worker.fetch(new Request('https://t.parla.cc/v1/e', { method: 'POST', body: JSON.stringify(ev1) }), { SAMPLE: '0', AE: { writeDataPoint: (x) => pts.push(x) } }, {}, () => assert.fail('logged'));
  assert.equal(r0.status, 204);
  assert.equal(pts.length, 0);
}
// the daily budget: the real Budget class over an in-memory storage
const { Budget } = await import('./worker.ts');
const mem = new Map<string, number>();
const budget = new Budget({ storage: { get: async (k) => mem.get(k), put: async (k, v) => { mem.set(k, v); }, delete: async (k) => mem.delete(k) } });
const BUDGET = { idFromName: () => 'daily', get: () => ({ fetch: (u: string, init: RequestInit) => budget.fetch(new Request(u, init)) }) };
const points: Point[] = [];
const env: Env = { BUDGET, DAILY_LIMIT: '2', AE: { writeDataPoint: (x) => points.push(x) } };
assert.equal((await call(env)).status, 204);
assert.equal((await call(env)).status, 204);
const over = await call(env);
assert.equal(over.status, 429, 'over the daily limit');
const after = Number(over.headers.get('retry-after'));
assert.ok(after > 0 && after <= 86400);
assert.equal(points.length, 2);
assert.deepEqual(points[0].indexes, [base.id], 'the random ID is the Analytics Engine index');
assert.equal(points[0].blobs[0], 'active');
console.log('worker: sample and daily limit checks passed');

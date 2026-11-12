import { describe, expect, it } from 'vitest';
import { QuadTracker } from '../src/detect/index.ts';
import type { Quad } from '../src/geometry/index.ts';

const base: Quad = [
  { x: 100, y: 100 },
  { x: 300, y: 100 },
  { x: 300, y: 400 },
  { x: 100, y: 400 },
];
const shift = (q: Quad, dx: number, dy = 0): Quad =>
  q.map((p) => ({ x: p.x + dx, y: p.y + dy })) as unknown as Quad;
/** Diagonal of `base` (≈ 360.6 px). */
const DIAG = Math.hypot(200, 300);

describe('QuadTracker', () => {
  it('becomes stable once the window holds enough still observations', () => {
    const tracker = new QuadTracker({ stableFrames: 4, windowMs: 200 });
    const states = Array.from({ length: 6 }, (_, i) =>
      tracker.update(shift(base, i % 2 ? 0.5 : 0), i * 50),
    );
    // Needs 4 observations spanning half the window (100 ms).
    expect(states.map((s) => s.stable)).toEqual([false, false, false, true, true, true]);
    expect(states.map((s) => s.instability)).toEqual([
      'settling',
      'settling',
      'settling',
      null,
      null,
      null,
    ]);
    expect(states[5].stableCount).toBe(3);
    expect(states[5].speed).toBeLessThan(0.01);
  });

  it('reports a steady drift as stable and panning as moving', () => {
    const run = (speed: number) => {
      const tracker = new QuadTracker();
      let state = tracker.update(base, 0);
      for (let i = 1; i <= 20; i++) {
        state = tracker.update(shift(base, (speed * DIAG * i * 50) / 1000), i * 50);
      }
      return state;
    };
    const drift = run(0.05);
    expect(drift.stable).toBe(true);
    expect(drift.speed).toBeCloseTo(0.05, 3);
    const pan = run(0.3);
    expect(pan.stable).toBe(false);
    expect(pan.instability).toBe('moving');
    expect(pan.speed).toBeCloseTo(0.3, 3);
  });

  it('reports scattered detections as jitter, but tolerates single outliers', () => {
    const tracker = new QuadTracker({ outlierFraction: 0 });
    let state = tracker.update(base, 0);
    // ±2.5 % of the diagonal on every frame: scattered.
    for (let i = 1; i <= 12; i++) state = tracker.update(shift(base, (i % 2 ? 1 : -1) * 9), i * 50);
    expect(state.stable).toBe(false);
    expect(state.instability).toBe('jitter');

    const steady = new QuadTracker();
    state = steady.update(base, 0);
    const stable: boolean[] = [];
    for (let i = 1; i <= 30; i++) {
      // Every 5th detection is off by 8 % of the diagonal.
      const quad = i % 5 === 0 ? shift(base, 0.08 * DIAG, 10) : shift(base, (i % 3) * 0.5);
      state = steady.update(quad, i * 50);
      if (i >= 10) stable.push(state.stable);
    }
    expect(stable.every(Boolean)).toBe(true);
  });

  it('smooths towards the observation', () => {
    const tracker = new QuadTracker({ smoothing: 0.5 });
    tracker.update(base, 0);
    tracker.update(base, 33);
    const moved = tracker.update(shift(base, 20), 66);
    expect(moved.quad?.[0].x).toBeCloseTo(110, 9);
    let state = moved;
    for (let i = 0; i < 20; i++) state = tracker.update(shift(base, 20), 100 + i * 33);
    expect(state.quad?.[0].x).toBeCloseTo(120, 3);
    expect(state.stable).toBe(true);
  });

  it('ignores a single far jump and restarts once a second observation confirms it', () => {
    const tracker = new QuadTracker();
    tracker.update(base, 0);
    const outlier = tracker.update(shift(base, 200, 50), 33);
    expect(outlier.quad).toEqual(base);
    expect(outlier.holding).toBe(true);
    expect(outlier.restarted).toBe(false);
    // Back to the document: the outlier is forgotten.
    expect(tracker.update(base, 66).quad).toEqual(base);
    tracker.update(shift(base, 200, 50), 100);
    const jumped = tracker.update(shift(base, 202, 50), 133);
    expect(jumped.quad).toEqual(shift(base, 202, 50));
    expect(jumped.restarted).toBe(true);
    expect(jumped.stable).toBe(false);
    expect(jumped.windowFrames).toBe(1);
  });

  it('holds the quad across short gaps and drops it after the timeout', () => {
    const tracker = new QuadTracker({ holdMs: 100 });
    tracker.update(base, 0);
    const held = tracker.update(null, 80);
    expect(held.holding).toBe(true);
    expect(held.quad).toEqual(base);
    const dropped = tracker.update(null, 150);
    expect(dropped).toMatchObject({ quad: null, stable: false, stableCount: 0, holding: false });
  });

  it('aligns cyclically relabeled corners before smoothing', () => {
    const tracker = new QuadTracker({ smoothing: 0.5 });
    tracker.update(base, 0);
    const relabeled: Quad = [base[1], base[2], base[3], base[0]];
    const state = tracker.update(relabeled, 33);
    expect(state.quad).toEqual(base);
    expect(state.speed).toBe(0);
    expect(state.jitter).toBe(0);
  });
});

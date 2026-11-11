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

describe('QuadTracker', () => {
  it('becomes stable after N still frames', () => {
    const tracker = new QuadTracker({ stableFrames: 5 });
    const states = Array.from({ length: 6 }, (_, i) =>
      tracker.update(shift(base, i % 2 ? 0.5 : 0), i * 33),
    );
    expect(states.map((s) => s.stable)).toEqual([false, false, false, false, true, true]);
    expect(states[5].stableCount).toBe(6);
  });

  it('resets stability on movement and smooths towards the observation', () => {
    const tracker = new QuadTracker({ smoothing: 0.5, stableFrames: 2 });
    tracker.update(base, 0);
    tracker.update(base, 33);
    const moved = tracker.update(shift(base, 20), 66);
    expect(moved.stable).toBe(false);
    expect(moved.stableCount).toBe(1);
    expect(moved.quad?.[0].x).toBeCloseTo(110, 9);
    let state = moved;
    for (let i = 0; i < 20; i++) state = tracker.update(shift(base, 20), 100 + i * 33);
    expect(state.quad?.[0].x).toBeCloseTo(120, 3);
    expect(state.stable).toBe(true);
  });

  it('restarts on large jumps instead of smoothing', () => {
    const tracker = new QuadTracker();
    tracker.update(base, 0);
    const jumped = tracker.update(shift(base, 200, 50), 33);
    expect(jumped.quad).toEqual(shift(base, 200, 50));
    expect(jumped.stableCount).toBe(1);
  });

  it('holds the quad across short gaps and drops it after the timeout', () => {
    const tracker = new QuadTracker({ holdMs: 100 });
    tracker.update(base, 0);
    const held = tracker.update(null, 80);
    expect(held.holding).toBe(true);
    expect(held.quad).toEqual(base);
    const dropped = tracker.update(null, 150);
    expect(dropped).toEqual({ quad: null, stable: false, stableCount: 0, holding: false });
  });

  it('aligns cyclically relabeled corners before smoothing', () => {
    const tracker = new QuadTracker({ smoothing: 0.5 });
    tracker.update(base, 0);
    const relabeled: Quad = [base[1], base[2], base[3], base[0]];
    const state = tracker.update(relabeled, 33);
    expect(state.quad).toEqual(base);
    expect(state.stableCount).toBe(2);
  });
});

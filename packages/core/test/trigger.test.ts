import { describe, expect, it } from 'vitest';
import { CaptureTrigger, type CaptureTriggerStatus } from '../src/controller/index.ts';
import { QuadTracker } from '../src/detect/index.ts';
import type { Quad } from '../src/geometry/index.ts';
import { Random } from './support/random.ts';

/** A4-like document in a 1080×1920 frame (diagonal ≈ 1150 px). */
const doc: Quad = [
  { x: 200, y: 400 },
  { x: 880, y: 420 },
  { x: 900, y: 1340 },
  { x: 180, y: 1320 },
];
const DIAG = Math.hypot(700, 940);
const RATE = 17; // live detections per second ("Neural v4" on an iPhone)

type Motion = (t: number) => { x: number; y: number };

interface Frame {
  quad: Quad | null;
  sharpness: number;
}

/**
 * Plays a synthetic hand-held sequence through tracker and trigger at 17 Hz. `motion` gives the
 * document offset in diagonals at time t (s); every detection gets Gaussian corner noise
 * (`noise` diagonals); `edit` can replace single frames (outliers, misses, blur).
 */
function play(
  seconds: number,
  motion: Motion,
  options: {
    noise?: number;
    seed?: number;
    edit?: (i: number, frame: Frame) => Frame;
  } = {},
): { statuses: CaptureTriggerStatus[]; fireTime: number | null } {
  const random = new Random(options.seed ?? 1);
  const tracker = new QuadTracker();
  const trigger = new CaptureTrigger();
  const statuses: CaptureTriggerStatus[] = [];
  let fireTime: number | null = null;
  const noise = options.noise ?? 0.002;
  for (let i = 0; i * (1000 / RATE) <= seconds * 1000; i++) {
    const t = i * (1000 / RATE);
    const offset = motion(t / 1000);
    const quad = doc.map((p) => ({
      x: p.x + offset.x * DIAG + random.gaussian() * noise * DIAG,
      y: p.y + offset.y * DIAG + random.gaussian() * noise * DIAG,
    })) as unknown as Quad;
    const frame = options.edit?.(i, { quad, sharpness: 100 * (1 + 0.05 * random.gaussian()) }) ?? {
      quad,
      sharpness: 100 * (1 + 0.05 * random.gaussian()),
    };
    const status = trigger.update(tracker.update(frame.quad, t), frame.sharpness, t);
    statuses.push(status);
    if (status.fire && fireTime === null) fireTime = t;
  }
  return { statuses, fireTime };
}

/** A hand trying to hold still: slow sway (1 % at 0.4–0.5 Hz) and tremor (0.1 % at 9 Hz). */
const steady: Motion = (t) => ({
  x: 0.01 * Math.sin(2 * Math.PI * 0.4 * t) + 0.001 * Math.sin(2 * Math.PI * 9 * t),
  y: 0.01 * Math.sin(2 * Math.PI * 0.5 * t + 1) + 0.001 * Math.sin(2 * Math.PI * 9 * t + 2),
});

describe('CaptureTrigger with QuadTracker (synthetic hand-held sequences)', () => {
  it('fires for a steady hand within about a second', () => {
    const { fireTime } = play(3, steady);
    expect(fireTime).not.toBeNull();
    expect(fireTime).toBeLessThan(1200);
  });

  it('fires for a steady hand despite outlier detections, misses and a very sharp frame', () => {
    const { fireTime, statuses } = play(3, steady, {
      edit: (i, frame) => {
        if (i % 7 === 3) return { ...frame, quad: shiftQuad(frame.quad, 0.06) }; // bad corners
        if (i === 9) return { ...frame, quad: shiftQuad(frame.quad, 0.4) }; // a far glitch
        if (i % 11 === 5) return { ...frame, quad: null }; // missed detection
        if (i === 4) return { ...frame, sharpness: 400 }; // one very sharp frame
        return frame;
      },
    });
    expect(fireTime).not.toBeNull();
    expect(fireTime).toBeLessThan(1500);
    expect(statuses.some((s) => s.blocker === 'holding')).toBe(true);
  });

  it('does not fire while panning or walking', () => {
    for (const seed of [1, 2, 3]) {
      const pan = play(5, (t) => ({ x: steady(t).x + 0.3 * t, y: steady(t).y }), { seed });
      expect(pan.fireTime).toBeNull();
      expect(pan.statuses.slice(10).every((s) => s.blocker === 'moving')).toBe(true);
      const walk = play(5, (t) => ({ x: steady(t).x, y: 0.04 * Math.sin(2 * Math.PI * 1.8 * t) }), {
        seed,
      });
      expect(walk.fireTime).toBeNull();
    }
  });

  it('restarts the progress on a re-framing', () => {
    const reframeAt = 0.75; // s, before the first capture
    const { statuses, fireTime } = play(3, (t) => {
      const s = steady(t);
      return t < reframeAt ? s : { x: s.x + 0.35, y: s.y };
    });
    const i = Math.ceil(reframeAt * RATE);
    expect(statuses[i - 1].progress).toBeGreaterThan(0.3);
    expect(statuses[i].progress).toBeLessThan(statuses[i - 1].progress); // outlier: decays
    expect(statuses[i + 1].progress).toBe(0); // confirmed: restart
    expect(fireTime).not.toBeNull();
    // Settling (half the 500-ms window) plus 600 ms of readiness after the re-framing.
    expect((fireTime as number) - reframeAt * 1000).toBeGreaterThan(800);
  });

  it('lets single blurry frames decay the progress instead of resetting it', () => {
    const { statuses, fireTime } = play(3, steady, {
      edit: (i, frame) => (i % 6 === 0 ? { ...frame, sharpness: 30 } : frame),
    });
    const blurry = statuses.findIndex((s, k) => k > 10 && s.blocker === 'blurry');
    expect(statuses[blurry].progress).toBeGreaterThan(0);
    expect(statuses[blurry].progress).toBeLessThan(statuses[blurry - 1].progress);
    expect(fireTime).not.toBeNull();
  });

  it('holds while the document stays blurred', () => {
    const { statuses, fireTime } = play(3, steady, {
      edit: (i, frame) => (i > 8 ? { ...frame, sharpness: 30 + (i % 2) } : frame),
    });
    expect(statuses[14].blocker).toBe('blurry');
    // After 1.5 s the reference adapts to the new level (focus is not judged absolutely).
    expect(fireTime === null || fireTime > 1500).toBe(true);
  });

  it('reports outside blockers and never fires with them', () => {
    const tracker = new QuadTracker();
    const trigger = new CaptureTrigger();
    let status: CaptureTriggerStatus | null = null;
    for (let i = 0; i < 40; i++) {
      status = trigger.update(tracker.update(doc, i * 60), 100, i * 60, 'device-motion');
      expect(status.fire).toBe(false);
    }
    expect(status?.blocker).toBe('device-motion');
    expect(status?.progress).toBe(0);
  });
});

function shiftQuad(quad: Quad | null, by: number): Quad | null {
  return quad && (quad.map((p) => ({ x: p.x + by * DIAG, y: p.y })) as unknown as Quad);
}

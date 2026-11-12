/**
 * Auto-capture policies replayed by the simulator (`autocapture.ts`).
 *
 * - {@link LegacyPolicy}: the policy of the library before the auto-capture rework (commit
 *   b72c9d6), reproduced exactly: the old `QuadTracker` (8 consecutive frames within 1.5 % of the
 *   diagonal of the EMA quad) and the old controller rule (relative sharpness against a peak that
 *   decays by 0.98 per frame ≥ 0.6, 1000 ms of uninterrupted readiness).
 * - {@link CandidatePolicy}: a configurable policy for the design study; each knob switches one
 *   change on or off, so that the changes can be compared one at a time.
 * - {@link LibraryPolicy}: the shipped policy, i.e. the library's `QuadTracker` and
 *   `CaptureTrigger` as the controller uses them.
 */
import { CaptureTrigger, type CaptureTriggerOptions } from '../src/controller/trigger.ts';
import { QuadTracker, type QuadTrackerOptions } from '../src/detect/tracker.ts';
import { distance, maxCornerDistance, type Quad } from '../src/geometry/quad.ts';

/** What a policy gets per analyzed frame (one live detection). */
export interface Tick {
  /** Time since the clip start (ms). */
  t: number;
  /** Index of the analyzed frame in the clip. */
  frame: number;
  /** Accepted detection in frame pixels, or `null`. */
  quad: Quad | null;
  /** Sharpness of the detection's bounding box (whole frame without a detection). */
  sharpness: number;
}

export interface Decision {
  fire: boolean;
  /**
   * Frame to capture when the policy keeps a buffer of analyzed frames; otherwise the still is
   * the video frame at the time of the decision plus the still latency.
   */
  captureFrame?: number;
  /** Tracked quad (fallback crop if the still has no detection). */
  quad: Quad | null;
}

export interface Policy {
  step(tick: Tick): Decision;
}

// ---- legacy ------------------------------------------------------------------------------

/** The `QuadTracker` of commit b72c9d6 (consecutive-frame stability), unchanged. */
class LegacyTracker {
  private quad: Quad | null = null;
  private stableCount = 0;
  private lastSeen = Number.NEGATIVE_INFINITY;
  update(observation: Quad | null, timeMs: number) {
    if (!observation) {
      if (this.quad && timeMs - this.lastSeen <= 500) return this.state(true);
      this.quad = null;
      this.stableCount = 0;
      this.lastSeen = Number.NEGATIVE_INFINITY;
      return this.state(false);
    }
    this.lastSeen = timeMs;
    if (!this.quad) {
      this.quad = observation;
      this.stableCount = 1;
      return this.state(false);
    }
    const aligned = alignCorners(observation, this.quad);
    const diagonal = diagonalOf(this.quad);
    const movement = maxCornerDistance(aligned, this.quad) / diagonal;
    if (movement > 0.2) {
      this.quad = observation;
      this.stableCount = 1;
      return this.state(false);
    }
    this.stableCount = movement <= 0.015 ? this.stableCount + 1 : 1;
    const q = this.quad;
    this.quad = [0, 1, 2, 3].map((i) => ({
      x: q[i].x + 0.5 * (aligned[i].x - q[i].x),
      y: q[i].y + 0.5 * (aligned[i].y - q[i].y),
    })) as unknown as Quad;
    return this.state(false);
  }
  private state(holding: boolean) {
    return { quad: this.quad, stable: this.quad !== null && this.stableCount >= 8, holding };
  }
}

export class LegacyPolicy implements Policy {
  private readonly tracker = new LegacyTracker();
  private peak = 0;
  private since: number | null = null;
  step(tick: Tick): Decision {
    const tracked = this.tracker.update(tick.quad, tick.t);
    this.peak = Math.max(this.peak * 0.98, tick.sharpness);
    const relative = this.peak > 0 ? tick.sharpness / this.peak : 0;
    const ready = tracked.stable && !tracked.holding && relative >= 0.6;
    if (!ready) this.since = null;
    else this.since ??= tick.t;
    return { fire: this.since !== null && tick.t - this.since >= 1000, quad: tracked.quad };
  }
}

// ---- library -------------------------------------------------------------------------------

/**
 * The shipped policy: the library's `QuadTracker` and `CaptureTrigger`, wired as in the
 * controller, including its best-frame buffer: while the progress is in its last `bestFrameMs`,
 * every analyzed frame is also grabbed as a still, and only the sharpest ready one of the last
 * `bestFrameMs` is kept (one slot). On firing, the kept frame is the page.
 */
export class LibraryPolicy implements Policy {
  private readonly tracker: QuadTracker;
  private readonly trigger: CaptureTrigger;
  private readonly bestFrameMs: number;
  private readonly stableMs: number;
  private progress = 0;
  private kept: { t: number; frame: number; sharpness: number } | null = null;
  /** Analyzed frames per blocker (`ready` for ready frames), summed over all instances. */
  static readonly blockers: Record<string, number> = {};
  constructor(
    tracker: QuadTrackerOptions = {},
    trigger: CaptureTriggerOptions = {},
    bestFrameMs = 300,
  ) {
    this.tracker = new QuadTracker(tracker);
    this.trigger = new CaptureTrigger(trigger);
    this.bestFrameMs = bestFrameMs;
    this.stableMs = trigger.stableMs ?? 600;
  }
  step(tick: Tick): Decision {
    const grab = this.bestFrameMs > 0 && this.progress >= 1 - this.bestFrameMs / this.stableMs;
    const tracked = this.tracker.update(tick.quad, tick.t);
    const status = this.trigger.update(tracked, tick.sharpness, tick.t);
    this.progress = status.progress;
    const key = status.blocker ?? 'ready';
    LibraryPolicy.blockers[key] = (LibraryPolicy.blockers[key] ?? 0) + 1;
    if (this.kept && this.kept.t < tick.t - this.bestFrameMs) this.kept = null;
    if (grab && status.ready && (!this.kept || tick.sharpness >= this.kept.sharpness)) {
      this.kept = { t: tick.t, frame: tick.frame, sharpness: tick.sharpness };
    }
    if (!status.ready && status.progress === 0) this.kept = null;
    return {
      fire: status.fire,
      captureFrame: status.fire ? this.kept?.frame : undefined,
      quad: tracked.quad,
    };
  }
}

// ---- candidate (design study) --------------------------------------------------------------

export interface CandidateOptions {
  /** `consecutive`: the legacy tracker rule; `window`: robust spread over a time window. */
  stability: 'consecutive' | 'window' | 'fit';
  /** `fit`: maximum corner speed of a line fit over the window (diagonals per second). */
  maxSpeed: number;
  windowMs: number;
  /** Spread limit (window) or per-frame movement limit (consecutive), fraction of the diagonal. */
  threshold: number;
  /** Minimum observations in the window. */
  minFrames: number;
  /** Share of the window's observations ignored as outliers (largest deviations). */
  outlierFraction: number;
  /** A jump beyond 20 % of the diagonal restarts only if the next observation confirms it. */
  confirmJump: boolean;
  /** Required readiness time (ms). */
  stableMs: number;
  /** Not-ready time that undoes a full progress (ms); 0: reset to 0 on any not-ready frame. */
  decayMs: number;
  /** `peak`: legacy decaying maximum; `percentile`: percentile of the last `sharpWindowMs`. */
  sharpRef: 'peak' | 'percentile';
  sharpWindowMs: number;
  sharpPercentile: number;
  minRelative: number;
  /**
   * After the progress is full: `none` captures now; `wait` waits up to `bestMs` for a frame at
   * least `bestRatio` × the sharpest frame of the last `bestMs` (no buffering); `buffer` captures
   * the sharpest analyzed frame of the last `bestMs` (needs a frame buffer; an upper bound).
   */
  best: 'none' | 'wait' | 'buffer';
  bestMs: number;
  bestRatio: number;
  /** Only frames with a detection enter the sharpness reference. */
  sharpDetectedOnly?: boolean;
  /** Missed detections (holding) count as not ready. */
  holdBreaks: boolean;
}

export const LEGACY_LIKE: CandidateOptions = {
  stability: 'consecutive',
  maxSpeed: 0.1,
  windowMs: 500,
  threshold: 0.015,
  minFrames: 8,
  outlierFraction: 0,
  confirmJump: false,
  stableMs: 1000,
  decayMs: 0,
  sharpRef: 'peak',
  sharpWindowMs: 1500,
  sharpPercentile: 0.75,
  minRelative: 0.6,
  best: 'none',
  bestMs: 300,
  bestRatio: 0.9,
  holdBreaks: true,
};

interface Entry {
  t: number;
  quad: Quad;
  sharpness: number;
  frame: number;
}

export class CandidatePolicy implements Policy {
  private readonly o: CandidateOptions;
  private quad: Quad | null = null;
  private lastSeen = Number.NEGATIVE_INFINITY;
  private window: Entry[] = [];
  private pending: Quad | null = null;
  private consecutive = 0;
  private sharp: { t: number; s: number; frame: number }[] = [];
  private peak = 0;
  private progress = 0;
  private lastT: number | null = null;
  private armedAt: number | null = null;
  private wasReady = false;

  constructor(options: Partial<CandidateOptions> = {}) {
    this.o = { ...LEGACY_LIKE, ...options };
  }

  step(tick: Tick): Decision {
    const o = this.o;
    const { t } = tick;
    let restarted = false;
    let holding = false;
    let stable = false;
    if (!tick.quad) {
      if (this.quad && t - this.lastSeen <= 500) holding = true;
      else {
        if (this.quad) restarted = true;
        this.quad = null;
        this.window = [];
        this.consecutive = 0;
      }
    } else {
      this.lastSeen = t;
      if (!this.quad) {
        this.quad = tick.quad;
        this.window = [{ t, quad: tick.quad, sharpness: tick.sharpness, frame: tick.frame }];
        this.consecutive = 1;
        restarted = true;
      } else {
        const aligned = alignCorners(tick.quad, this.quad);
        const movement = maxCornerDistance(aligned, this.quad) / diagonalOf(this.quad);
        let jump = movement > 0.2;
        if (jump && o.confirmJump) {
          const p = this.pending;
          const confirmed =
            p && maxCornerDistance(alignCorners(tick.quad, p), p) / diagonalOf(p) <= 0.1;
          if (!confirmed) {
            this.pending = tick.quad;
            jump = false;
            holding = true; // the outlier is skipped
          }
        }
        if (jump) {
          this.pending = null;
          this.quad = tick.quad;
          this.window = [{ t, quad: tick.quad, sharpness: tick.sharpness, frame: tick.frame }];
          this.consecutive = 1;
          restarted = true;
        } else if (!holding) {
          this.pending = null;
          this.consecutive = movement <= o.threshold ? this.consecutive + 1 : 1;
          const q = this.quad;
          this.quad = [0, 1, 2, 3].map((i) => ({
            x: q[i].x + 0.5 * (aligned[i].x - q[i].x),
            y: q[i].y + 0.5 * (aligned[i].y - q[i].y),
          })) as unknown as Quad;
          this.window.push({ t, quad: aligned, sharpness: tick.sharpness, frame: tick.frame });
        }
      }
    }
    while (this.window.length > 0 && this.window[0].t <= t - o.windowMs) this.window.shift();
    if (this.quad && !holding) {
      if (o.stability === 'consecutive') stable = this.consecutive >= o.minFrames;
      else stable = windowStable(this.window, t, o);
    }

    // sharpness reference
    this.peak = Math.max(this.peak * 0.98, tick.sharpness);
    if (tick.quad || !o.sharpDetectedOnly)
      this.sharp.push({ t, s: tick.sharpness, frame: tick.frame });
    while (this.sharp.length > 1 && this.sharp[0].t <= t - o.sharpWindowMs) this.sharp.shift();
    const reference =
      o.sharpRef === 'peak'
        ? this.peak
        : percentile(
            this.sharp.map((e) => e.s),
            o.sharpPercentile,
          );
    const relative = reference > 0 ? tick.sharpness / reference : 0;
    const ready = stable && !(holding && o.holdBreaks) && relative >= o.minRelative;

    const dt = this.lastT === null ? 0 : Math.min(250, t - this.lastT);
    this.lastT = t;
    if (restarted || !this.quad) {
      this.progress = 0;
      this.armedAt = null;
    }
    if (ready) this.progress = Math.min(1, this.progress + (this.wasReady ? dt / o.stableMs : 0));
    else {
      this.progress = o.decayMs > 0 ? Math.max(0, this.progress - dt / o.decayMs) : 0;
      if (this.progress < 1) this.armedAt = null;
    }
    // As in the legacy rule, the first ready frame starts the clock.
    this.wasReady = ready;
    if (!(this.progress >= 1 - 1e-9 && ready)) return { fire: false, quad: this.quad };
    this.armedAt ??= t;
    if (o.best === 'none') return { fire: true, quad: this.quad };
    const recent = this.sharp.filter((e) => e.t > t - o.bestMs);
    if (recent.length === 0) recent.push({ t, s: tick.sharpness, frame: tick.frame });
    if (o.best === 'buffer') {
      const best = recent.reduce((a, b) => (b.s > a.s ? b : a));
      return { fire: true, captureFrame: best.frame, quad: this.quad };
    }
    const max = Math.max(...recent.map((e) => e.s));
    const fire = tick.sharpness >= o.bestRatio * max || t - this.armedAt >= o.bestMs;
    return { fire, quad: this.quad };
  }
}

function windowStable(window: Entry[], t: number, o: CandidateOptions): boolean {
  if (window.length < o.minFrames || t - window[0].t < o.windowMs / 2) return false;
  if (o.stability === 'fit') {
    const fit = lineFit(window, o.outlierFraction);
    return fit.speed <= o.maxSpeed && fit.residual <= o.threshold;
  }
  return (
    robustSpread(
      window.map((e) => e.quad),
      o.outlierFraction,
    ) <= o.threshold
  );
}

/**
 * Least-squares line per corner coordinate over time; the worst `fraction` of observations (by
 * residual) are dropped and the fit repeated. Speed: fastest corner (diagonals per second);
 * residual: largest remaining corner residual (diagonals).
 */
export function lineFit(window: Entry[], fraction: number): { speed: number; residual: number } {
  const fitOnce = (entries: Entry[]) => {
    const tMean = entries.reduce((s, e) => s + e.t, 0) / entries.length;
    const tt = entries.reduce((s, e) => s + (e.t - tMean) ** 2, 0) || 1;
    const lines = [0, 1, 2, 3].map((i) => {
      const mx = entries.reduce((s, e) => s + e.quad[i].x, 0) / entries.length;
      const my = entries.reduce((s, e) => s + e.quad[i].y, 0) / entries.length;
      const vx = entries.reduce((s, e) => s + (e.t - tMean) * (e.quad[i].x - mx), 0) / tt;
      const vy = entries.reduce((s, e) => s + (e.t - tMean) * (e.quad[i].y - my), 0) / tt;
      return { mx, my, vx, vy };
    });
    const center = lines.map((l) => ({ x: l.mx, y: l.my })) as unknown as Quad;
    const diagonal = diagonalOf(center);
    const residuals = entries.map(
      (e) =>
        Math.max(
          ...lines.map((l, i) =>
            Math.hypot(
              e.quad[i].x - l.mx - l.vx * (e.t - tMean),
              e.quad[i].y - l.my - l.vy * (e.t - tMean),
            ),
          ),
        ) / diagonal,
    );
    const speed = (Math.max(...lines.map((l) => Math.hypot(l.vx, l.vy))) * 1000) / diagonal;
    return { speed, residuals };
  };
  const first = fitOnce(window);
  const drop = Math.floor(window.length * fraction);
  if (drop === 0) return { speed: first.speed, residual: Math.max(...first.residuals) };
  const order = first.residuals.map((r, i) => [r, i]).sort((a, b) => a[0] - b[0]);
  const kept = order.slice(0, window.length - drop).map(([, i]) => window[i]);
  const second = fitOnce(kept);
  return { speed: second.speed, residual: Math.max(...second.residuals) };
}

/** Deviation from the per-corner median quad, after dropping the largest `fraction`. */
export function robustSpread(quads: Quad[], fraction: number): number {
  const median = [0, 1, 2, 3].map((i) => ({
    x: percentile(
      quads.map((q) => q[i].x),
      0.5,
    ),
    y: percentile(
      quads.map((q) => q[i].y),
      0.5,
    ),
  })) as unknown as Quad;
  const diagonal = diagonalOf(median);
  const deviations = quads
    .map((q) => maxCornerDistance(q, median) / diagonal)
    .sort((a, b) => a - b);
  const drop = Math.floor(deviations.length * fraction);
  return deviations[deviations.length - 1 - drop];
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = p * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function diagonalOf(q: Quad): number {
  return Math.max(distance(q[0], q[2]), distance(q[1], q[3]));
}

function alignCorners(quad: Quad, reference: Quad): Quad {
  let best = quad;
  let bestCost = Number.POSITIVE_INFINITY;
  for (let r = 0; r < 4; r++) {
    const rotated: Quad = [quad[r], quad[(r + 1) % 4], quad[(r + 2) % 4], quad[(r + 3) % 4]];
    let cost = 0;
    for (let i = 0; i < 4; i++) cost += distance(rotated[i], reference[i]);
    if (cost < bestCost) {
      bestCost = cost;
      best = rotated;
    }
  }
  return best;
}

import { distance, maxCornerDistance, type Quad } from '../geometry/quad.ts';

/** Options for {@link QuadTracker}. */
export interface QuadTrackerOptions {
  /** Weight of the newest observation in the per-corner exponential moving average, (0, 1]. Default 0.5. */
  smoothing?: number;
  /**
   * Jitter limit for stability: after a straight-line motion is fitted to every corner over the
   * stability window and the worst observations (`outlierFraction`) are dropped, no remaining
   * corner may be farther than this fraction of the diagonal from its line. Default 0.015.
   */
  stabilityThreshold?: number;
  /**
   * Minimum number of observations in the stability window before the quad can be stable.
   * Default 4. At low detection rates the window keeps the newest `minWindowFrames`
   * observations even if they are older than `windowMs`, up to `2 × windowMs`; so the default
   * works down to about 2–3 detections per second. Missed detections (holding frames) add no
   * observation, so they shrink the window.
   */
  minWindowFrames?: number;
  /**
   * @deprecated Before the window rule, this was the number of consecutive still frames
   * (default 8). It is mapped to `minWindowFrames = min(stableFrames, 4)`, so that existing
   * settings never need more observations than the default; `minWindowFrames` wins if both are
   * given.
   */
  stableFrames?: number;
  /**
   * Length (ms) of the stability window: the observations of this period are judged together,
   * so single outlier frames do not reset stability (see `minWindowFrames` for low rates).
   * Stability needs observations spanning at least half of it. Default 500.
   */
  windowMs?: number;
  /**
   * Speed limit for stability, in quad diagonals per second: the fastest corner of the fitted
   * motion. At 0.1 a frame exposed for 1/30 s is blurred by 0.3 % of the document. The default
   * assumes that a hand trying to hold still drifts by about 0.02–0.05 diagonals/s (not yet
   * measured on phones; see docs/autocapture.md). Default 0.1.
   */
  maxSpeed?: number;
  /**
   * Share of the window's observations ignored as outliers (the largest residuals), [0, 0.5);
   * at least `minWindowFrames` are always kept (no outliers are dropped at 5 detections/s).
   * Default 0.25.
   */
  outlierFraction?: number;
  /** How long (ms) the last quad is held when detections are missing. Default 500. */
  holdMs?: number;
  /**
   * If any corner jumps farther than this fraction of the diagonal, the observation is a
   * re-framing or a different document: once the next observation confirms it (lies within half
   * this distance of it), the tracker restarts from it instead of smoothing towards it. A single
   * far observation is ignored as an outlier (reported as `holding`); it is forgotten after
   * `holdMs`. Default 0.2.
   */
  resetThreshold?: number;
}

/** Snapshot returned by {@link QuadTracker.update}. */
export interface TrackerState {
  /** Smoothed quad, or `null` if nothing is tracked. */
  quad: Quad | null;
  /** True while the quad is stable: slow (`maxSpeed`) and steady (`stabilityThreshold`) over the window. */
  stable: boolean;
  /** Number of consecutive updates with `stable` true. */
  stableCount: number;
  /**
   * True if the latest frame had no detection (or an unconfirmed far jump) and the previous quad
   * is being held.
   */
  holding: boolean;
  /** True on the update that started tracking a new quad (first detection or re-framing). */
  restarted?: boolean;
  /** Fastest corner of the fitted motion in diagonals per second; `null` with too few observations. */
  speed?: number | null;
  /** Largest corner residual around the fitted motion (outliers dropped), in diagonals; `null` with too few observations. */
  jitter?: number | null;
  /** Observations in the stability window. */
  windowFrames?: number;
  /** Time (ms) from the oldest observation in the window to this update. */
  windowSpanMs?: number;
  /**
   * Why a tracked, not held quad is not stable: too few observations yet (`settling`), too fast
   * (`moving`) or scattered (`jitter`); `null` otherwise.
   */
  instability?: 'settling' | 'moving' | 'jitter' | null;
}

interface Observation {
  t: number;
  quad: Quad;
}

/**
 * Temporal filter for per-frame document detections (pure, no DOM).
 *
 * - Smooths each corner with an exponential moving average after aligning the corner order
 *   of the new observation to the tracked quad (detectors may rotate the TL/TR/BR/BL labels of
 *   near-45° quads between frames).
 * - Reports stability over a time window, which is the trigger condition for auto-capture: a
 *   straight-line motion is fitted to every corner over the last `windowMs`, the worst quarter of
 *   the observations is dropped as outliers, and the quad is stable if the fitted motion is slow
 *   (`maxSpeed`) and the rest stays close to it (`stabilityThreshold`). Slow drift of a hand that
 *   tries to hold still passes, panning and walking do not, and a single bad detection does not
 *   reset anything.
 * - Ignores a single far jump (an outlier); two consecutive consistent jumps restart tracking.
 * - Holds the last quad for a short time across missed detections, then drops it.
 */
export class QuadTracker {
  private readonly smoothing: number;
  /** Resolved stability options (see {@link QuadTrackerOptions}). */
  readonly stabilityThreshold: number;
  readonly minWindowFrames: number;
  /** @deprecated Same as {@link QuadTracker.minWindowFrames}. */
  readonly stableFrames: number;
  readonly windowMs: number;
  readonly maxSpeed: number;
  private readonly outlierFraction: number;
  private readonly holdMs: number;
  private readonly resetThreshold: number;
  private quad: Quad | null = null;
  private window: Observation[] = [];
  private pending: { quad: Quad; t: number } | null = null;
  private stableCount = 0;
  private lastSeen = Number.NEGATIVE_INFINITY;

  constructor(options: QuadTrackerOptions = {}) {
    this.smoothing = options.smoothing ?? 0.5;
    this.stabilityThreshold = options.stabilityThreshold ?? 0.015;
    const legacy = options.stableFrames === undefined ? 4 : Math.min(options.stableFrames, 4);
    this.minWindowFrames = Math.max(2, options.minWindowFrames ?? legacy);
    this.stableFrames = this.minWindowFrames;
    this.windowMs = options.windowMs ?? 500;
    this.maxSpeed = options.maxSpeed ?? 0.1;
    this.outlierFraction = Math.min(0.49, Math.max(0, options.outlierFraction ?? 0.25));
    this.holdMs = options.holdMs ?? 500;
    this.resetThreshold = options.resetThreshold ?? 0.2;
  }

  /**
   * Feeds the detection of one frame (or `null` if none) and returns the new state.
   *
   * @param timeMs - Frame timestamp in milliseconds (e.g. `performance.now()`), used for the
   *   stability window and for holding.
   */
  update(observation: Quad | null, timeMs: number): TrackerState {
    if (!observation) {
      if (this.quad && timeMs - this.lastSeen <= this.holdMs) return this.held(timeMs);
      this.reset();
      return this.state(false, false, timeMs);
    }
    this.lastSeen = timeMs;
    if (!this.quad) return this.restart(observation, timeMs);
    const aligned = alignCorners(observation, this.quad);
    const movement = maxCornerDistance(aligned, this.quad) / diagonalOf(this.quad);
    if (movement > this.resetThreshold) {
      const pending =
        this.pending && timeMs - this.pending.t <= this.holdMs ? this.pending.quad : null;
      if (
        pending &&
        maxCornerDistance(alignCorners(observation, pending), pending) / diagonalOf(pending) <=
          this.resetThreshold / 2
      ) {
        return this.restart(observation, timeMs);
      }
      this.pending = { quad: observation, t: timeMs };
      return this.held(timeMs);
    }
    this.pending = null;
    const a = this.smoothing;
    const q = this.quad;
    this.quad = [0, 1, 2, 3].map((i) => ({
      x: q[i].x + a * (aligned[i].x - q[i].x),
      y: q[i].y + a * (aligned[i].y - q[i].y),
    })) as unknown as Quad;
    this.window.push({ t: timeMs, quad: aligned });
    return this.state(false, false, timeMs);
  }

  /** Forgets the tracked quad. */
  reset(): void {
    this.quad = null;
    this.window = [];
    this.pending = null;
    this.stableCount = 0;
    this.lastSeen = Number.NEGATIVE_INFINITY;
  }

  private restart(observation: Quad, timeMs: number): TrackerState {
    this.quad = observation;
    this.window = [{ t: timeMs, quad: observation }];
    this.pending = null;
    this.stableCount = 0;
    return this.state(false, true, timeMs);
  }

  private held(timeMs: number): TrackerState {
    return this.state(true, false, timeMs);
  }

  private state(holding: boolean, restarted: boolean, timeMs: number): TrackerState {
    const window = this.window;
    // Older than windowMs: kept only while needed for minWindowFrames, and never beyond 2×.
    while (
      window.length > 0 &&
      (window[0].t <= timeMs - 2 * this.windowMs ||
        (window[0].t <= timeMs - this.windowMs && window.length > this.minWindowFrames))
    ) {
      window.shift();
    }
    let speed: number | null = null;
    let jitter: number | null = null;
    if (this.quad && window.length >= 2) {
      const fit = fitMotion(window, this.outlierFraction, this.minWindowFrames);
      speed = fit.speed;
      jitter = fit.jitter;
    }
    let instability: TrackerState['instability'] = null;
    if (this.quad && !holding) {
      if (
        speed === null ||
        jitter === null ||
        window.length < this.minWindowFrames ||
        timeMs - window[0].t < this.windowMs / 2
      ) {
        instability = 'settling';
      } else if (speed > this.maxSpeed) instability = 'moving';
      else if (jitter > this.stabilityThreshold) instability = 'jitter';
    }
    const stable = this.quad !== null && !holding && instability === null;
    // Holding frames neither extend nor break a stable run.
    if (!holding) this.stableCount = stable ? this.stableCount + 1 : 0;
    return {
      quad: this.quad,
      stable,
      stableCount: this.quad ? this.stableCount : 0,
      holding,
      restarted,
      speed,
      jitter,
      windowFrames: window.length,
      windowSpanMs: window.length > 0 ? timeMs - window[0].t : 0,
      instability,
    };
  }
}

/**
 * Fits a straight-line motion (least squares over time) to every corner of the observations,
 * drops the `outlierFraction` with the largest residuals (but keeps at least `minKept`, so that
 * a few sparse samples of a fast motion cannot be fitted by dropping one) and fits again. Returns
 * the speed of the fastest corner (diagonals per second) and the largest remaining residual
 * (diagonals).
 */
function fitMotion(
  observations: readonly Observation[],
  outlierFraction: number,
  minKept: number,
): { speed: number; jitter: number } {
  const first = fitLines(observations);
  const drop = Math.max(
    0,
    Math.min(Math.floor(observations.length * outlierFraction), observations.length - minKept),
  );
  if (drop === 0) return { speed: first.speed, jitter: Math.max(...first.residuals) };
  const kept = first.residuals
    .map((r, i) => ({ r, i }))
    .sort((a, b) => a.r - b.r)
    .slice(0, observations.length - drop)
    .map(({ i }) => observations[i]);
  const second = fitLines(kept);
  return { speed: second.speed, jitter: Math.max(...second.residuals) };
}

function fitLines(observations: readonly Observation[]): { speed: number; residuals: number[] } {
  const n = observations.length;
  const tMean = observations.reduce((s, o) => s + o.t, 0) / n;
  const tVar = observations.reduce((s, o) => s + (o.t - tMean) ** 2, 0);
  const lines = [0, 1, 2, 3].map((i) => {
    let mx = 0;
    let my = 0;
    for (const o of observations) {
      mx += o.quad[i].x / n;
      my += o.quad[i].y / n;
    }
    let vx = 0;
    let vy = 0;
    if (tVar > 0) {
      for (const o of observations) {
        vx += ((o.t - tMean) * (o.quad[i].x - mx)) / tVar;
        vy += ((o.t - tMean) * (o.quad[i].y - my)) / tVar;
      }
    }
    return { mx, my, vx, vy };
  });
  const diagonal = diagonalOf(lines.map((l) => ({ x: l.mx, y: l.my })) as unknown as Quad) || 1;
  const residuals = observations.map((o) => {
    const dt = o.t - tMean;
    let worst = 0;
    for (let i = 0; i < 4; i++) {
      const l = lines[i];
      worst = Math.max(
        worst,
        Math.hypot(o.quad[i].x - l.mx - l.vx * dt, o.quad[i].y - l.my - l.vy * dt),
      );
    }
    return worst / diagonal;
  });
  const speed = (Math.max(...lines.map((l) => Math.hypot(l.vx, l.vy))) * 1000) / diagonal;
  return { speed, residuals };
}

function diagonalOf(q: Quad): number {
  return Math.max(distance(q[0], q[2]), distance(q[1], q[3]));
}

/** Cyclic rotation of `quad`'s corners that best matches `reference` corner by corner. */
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

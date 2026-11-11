import { distance, maxCornerDistance, type Quad } from '../geometry/quad.ts';

/** Options for {@link QuadTracker}. */
export interface QuadTrackerOptions {
  /** Weight of the newest observation in the per-corner exponential moving average, (0, 1]. Default 0.5. */
  smoothing?: number;
  /**
   * A frame counts as stable if no corner of the observation is farther than this fraction of
   * the tracked quad's diagonal from the smoothed corner. Default 0.015.
   */
  stabilityThreshold?: number;
  /** Consecutive stable frames required before {@link TrackerState.stable} is true. Default 8. */
  stableFrames?: number;
  /** How long (ms) the last quad is held when detections are missing. Default 500. */
  holdMs?: number;
  /**
   * If any corner jumps farther than this fraction of the diagonal, the tracker restarts from the
   * new observation instead of smoothing towards it (a different document or a re-framing).
   * Default 0.2.
   */
  resetThreshold?: number;
}

/** Snapshot returned by {@link QuadTracker.update}. */
export interface TrackerState {
  /** Smoothed quad, or `null` if nothing is tracked. */
  quad: Quad | null;
  /** True once the quad has been stable for the configured number of frames. */
  stable: boolean;
  /** Current run of consecutive stable frames. */
  stableCount: number;
  /** True if the latest frame had no detection and the previous quad is being held. */
  holding: boolean;
}

/**
 * Temporal filter for per-frame document detections (pure, no DOM).
 *
 * - Smooths each corner with an exponential moving average after aligning the corner order
 *   of the new observation to the tracked quad (detectors may rotate the TL/TR/BR/BL labels of
 *   near-45° quads between frames).
 * - Reports stability when observations stay within a small radius for N consecutive frames,
 *   which is the trigger condition for auto-capture.
 * - Holds the last quad for a short time across missed detections, then drops it.
 */
export class QuadTracker {
  private readonly smoothing: number;
  private readonly stabilityThreshold: number;
  private readonly stableFrames: number;
  private readonly holdMs: number;
  private readonly resetThreshold: number;
  private quad: Quad | null = null;
  private stableCount = 0;
  private lastSeen = Number.NEGATIVE_INFINITY;

  constructor(options: QuadTrackerOptions = {}) {
    this.smoothing = options.smoothing ?? 0.5;
    this.stabilityThreshold = options.stabilityThreshold ?? 0.015;
    this.stableFrames = options.stableFrames ?? 8;
    this.holdMs = options.holdMs ?? 500;
    this.resetThreshold = options.resetThreshold ?? 0.2;
  }

  /**
   * Feeds the detection of one frame (or `null` if none) and returns the new state.
   *
   * @param timeMs - Frame timestamp in milliseconds (e.g. `performance.now()`), used for holding.
   */
  update(observation: Quad | null, timeMs: number): TrackerState {
    if (!observation) {
      if (this.quad && timeMs - this.lastSeen <= this.holdMs) return this.state(true);
      this.reset();
      return this.state(false);
    }
    this.lastSeen = timeMs;
    if (!this.quad) {
      this.quad = observation;
      this.stableCount = 1;
      return this.state(false);
    }
    const aligned = alignCorners(observation, this.quad);
    const diagonal = Math.max(
      distance(this.quad[0], this.quad[2]),
      distance(this.quad[1], this.quad[3]),
    );
    const movement = maxCornerDistance(aligned, this.quad) / diagonal;
    if (movement > this.resetThreshold) {
      this.quad = observation;
      this.stableCount = 1;
      return this.state(false);
    }
    this.stableCount = movement <= this.stabilityThreshold ? this.stableCount + 1 : 1;
    const a = this.smoothing;
    const q = this.quad;
    this.quad = [0, 1, 2, 3].map((i) => ({
      x: q[i].x + a * (aligned[i].x - q[i].x),
      y: q[i].y + a * (aligned[i].y - q[i].y),
    })) as unknown as Quad;
    return this.state(false);
  }

  /** Forgets the tracked quad. */
  reset(): void {
    this.quad = null;
    this.stableCount = 0;
    this.lastSeen = Number.NEGATIVE_INFINITY;
  }

  private state(holding: boolean): TrackerState {
    return {
      quad: this.quad,
      stable: this.quad !== null && this.stableCount >= this.stableFrames,
      stableCount: this.stableCount,
      holding,
    };
  }
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

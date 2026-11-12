import type { TrackerState } from '../detect/tracker.ts';

/** Options for {@link CaptureTrigger}. */
export interface CaptureTriggerOptions {
  /** Time (ms) the document must be ready (stable and sharp) before auto-capture. Default 600. */
  stableMs?: number;
  /**
   * Not-ready time (ms) that empties a full progress: progress decays while not ready instead of
   * resetting, so one bad frame costs only its own duration. A re-framing, a new document or a
   * lost document resets it. 0 resets on every not-ready frame. Default 1000.
   */
  decayMs?: number;
  /** Minimum sharpness relative to the reference (see `sharpnessPercentile`). Default 0.6. */
  minRelativeSharpness?: number;
  /** Period (ms) of the sharpness reference. Default 1500. */
  sharpnessWindowMs?: number;
  /**
   * The sharpness reference is this percentile of the frames with a detection in the last
   * `sharpnessWindowMs` (robust to single very sharp frames, unlike a maximum). Default 0.75.
   */
  sharpnessPercentile?: number;
}

/** Why auto-capture is not ready (the first failing criterion), or `null` if it is. */
export type CaptureBlocker =
  /** Auto-capture is off. */
  | 'disabled'
  /** No document is tracked. */
  | 'no-document'
  /** The last frame had no detection (or an outlier); the previous quad is held. */
  | 'holding'
  /** This document was captured already; waiting for a different one. */
  | 'captured'
  /** Too few observations in the stability window yet. */
  | 'settling'
  /** The quad moves faster than the tracker's `maxSpeed`. */
  | 'moving'
  /** The detections scatter more than the tracker's `stabilityThreshold`. */
  | 'jitter'
  /** The device's rotation rate is above the limit (motion sensor, opt-in). */
  | 'device-motion'
  /** The frame is less sharp than `minRelativeSharpness` × the reference. */
  | 'blurry'
  /**
   * Still `settling` because detections come too slowly to fill the stability window (reported
   * by `ScannerController` instead of `settling` when its measured rate is below the minimum).
   */
  | 'slow-detection';

/** Result of {@link CaptureTrigger.update}. */
export interface CaptureTriggerStatus {
  /** All criteria hold for this frame. */
  ready: boolean;
  /** Progress towards auto-capture in [0, 1]. */
  progress: number;
  /** Capture now (progress complete on a ready frame). */
  fire: boolean;
  blocker: CaptureBlocker | null;
  /** Sharpness of the frame divided by the reference (0 without a reference). */
  relativeSharpness: number;
  /** Sharpness reference (see {@link CaptureTriggerOptions.sharpnessPercentile}). */
  sharpnessReference: number;
}

/**
 * The auto-capture rule (pure, no DOM): turns tracker states and frame sharpness into a progress
 * that fills while the document is ready and decays while it is not, and fires when it is full.
 * Used by `ScannerController`; exported for custom pipelines and simulations.
 */
export class CaptureTrigger {
  /** Resolved options. */
  readonly stableMs: number;
  readonly decayMs: number;
  readonly minRelativeSharpness: number;
  private readonly sharpnessWindowMs: number;
  private readonly sharpnessPercentile: number;
  private sharpness: { t: number; value: number }[] = [];
  private progress = 0;
  private lastTime: number | null = null;
  private wasReady = false;

  constructor(options: CaptureTriggerOptions = {}) {
    this.stableMs = Math.max(1, options.stableMs ?? 600);
    this.decayMs = options.decayMs ?? 1000;
    this.minRelativeSharpness = options.minRelativeSharpness ?? 0.6;
    this.sharpnessWindowMs = options.sharpnessWindowMs ?? 1500;
    this.sharpnessPercentile = options.sharpnessPercentile ?? 0.75;
  }

  /**
   * Feeds one analyzed frame.
   *
   * @param tracked - The tracker state of the frame.
   * @param frameSharpness - Sharpness of the frame (document region).
   * @param timeMs - Frame timestamp (ms).
   * @param blocker - An outside reason not to capture (`disabled`, `captured`,
   *   `device-motion`), checked after the document criteria; progress decays meanwhile.
   */
  update(
    tracked: TrackerState,
    frameSharpness: number,
    timeMs: number,
    blocker: CaptureBlocker | null = null,
  ): CaptureTriggerStatus {
    const dt = this.lastTime === null ? 0 : Math.min(250, Math.max(0, timeMs - this.lastTime));
    this.lastTime = timeMs;
    if (tracked.quad && !tracked.holding) this.sharpness.push({ t: timeMs, value: frameSharpness });
    while (this.sharpness.length > 0 && this.sharpness[0].t <= timeMs - this.sharpnessWindowMs) {
      this.sharpness.shift();
    }
    const reference = percentile(
      this.sharpness.map((s) => s.value),
      this.sharpnessPercentile,
    );
    const relativeSharpness = reference > 0 ? frameSharpness / reference : 0;

    let reason: CaptureBlocker | null = null;
    if (!tracked.quad) reason = 'no-document';
    else if (tracked.holding) reason = 'holding';
    else if (!tracked.stable) reason = tracked.instability ?? 'settling';
    else if (relativeSharpness < this.minRelativeSharpness) reason = 'blurry';
    if (reason === null && blocker) reason = blocker;

    const ready = reason === null;
    if (!tracked.quad || tracked.restarted) this.progress = 0;
    if (ready) {
      // The first ready frame of an empty progress starts the clock; later ones add their interval.
      if (this.wasReady || this.progress > 0) {
        this.progress = Math.min(1, this.progress + dt / this.stableMs);
      }
    } else {
      this.progress = this.decayMs > 0 ? Math.max(0, this.progress - dt / this.decayMs) : 0;
    }
    this.wasReady = ready;
    return {
      ready,
      progress: this.progress,
      fire: ready && this.progress >= 1 - 1e-9,
      blocker: reason,
      relativeSharpness,
      sharpnessReference: reference,
    };
  }

  /** Clears the progress and the sharpness history. */
  reset(): void {
    this.sharpness = [];
    this.progress = 0;
    this.lastTime = null;
    this.wasReady = false;
  }

  /** Empties the progress (e.g. after a capture), keeping the sharpness history. */
  restart(): void {
    this.progress = 0;
    this.wasReady = false;
  }
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = p * (sorted.length - 1);
  const lo = Math.floor(position);
  const hi = Math.ceil(position);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}

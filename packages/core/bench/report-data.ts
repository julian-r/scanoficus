/** Per-frame metrics and their aggregation for the benchmark report and montages. */
import { QuadTracker } from '../src/detect/index.ts';
import { type Quad, quadArea } from '../src/geometry/index.ts';
import { cornerError, smartDocJaccard, visibleFraction, visibleIoU } from './metrics.ts';
import { type ClipResult, type FrameResult, unpackQuad } from './results.ts';

/** Nominal frame interval (ms) for the tracked variant: SmartDoc ≈ 17–20 fps, MIDV 10 fps. */
export function frameIntervalMs(clip: ClipResult): number {
  return clip.dataset === 'smartdoc' ? 50 : 100;
}

export interface FrameMetrics {
  /** SmartDoc Jaccard index in the document's reference frame (0 if nothing was found). */
  ji: number;
  /** IoU of the visible parts in the image (0 if nothing was found). */
  iou: number;
  found: boolean;
  /** Mean corner error / image diagonal, NaN if nothing was found. */
  cornerError: number;
  ms: number;
  /** Fraction of the ground-truth document inside the frame. */
  visible: number;
  /** Number of ground-truth corners inside the frame (0–4). */
  cornersInside: number;
  /** Ground-truth document area as a fraction of the frame area. */
  area: number;
  /** Best SmartDoc JI over the detection and its runner-up candidates. */
  oracleJi: number;
  /** Best visible-region IoU over the detection and its runner-up candidates. */
  oracleIou: number;
  /** Detector confidence (0 if nothing was found). */
  score: number;
}

/** Primary per-frame score: SmartDoc JI for SmartDoc, visible-region IoU for MIDV. */
export function frameScore(clip: ClipResult, frame: FrameResult, size: number): number {
  const truth = unpackQuad(frame.truth);
  const run = frame.runs[size];
  const quad = run.quad ? unpackQuad(run.quad) : null;
  return clip.dataset === 'smartdoc'
    ? smartDocJaccard(truth, quad, clip.reference)
    : visibleIoU(truth, quad, clip.frameSize.width, clip.frameSize.height);
}

/**
 * Metrics of every frame of `clip` at processing size `size`, either per frame or after the
 * library's `QuadTracker` (default options) smoothed the detections over consecutive frames.
 */
export function clipMetrics(
  clip: ClipResult,
  size: number,
  variant: 'frame' | 'tracked',
): FrameMetrics[] {
  const { width, height } = clip.frameSize;
  const tracker = new QuadTracker();
  const interval = frameIntervalMs(clip);
  return clip.frames.map((frame) => {
    const truth = unpackQuad(frame.truth);
    const run = frame.runs[size];
    let quad: Quad | null = run.quad ? unpackQuad(run.quad) : null;
    if (variant === 'tracked') quad = tracker.update(quad, frame.index * interval).quad;
    const ji = smartDocJaccard(truth, quad, clip.reference);
    const iou = visibleIoU(truth, quad, width, height);
    let oracleJi = ji;
    let oracleIou = iou;
    for (const packed of run.candidates) {
      const candidate = unpackQuad(packed);
      oracleJi = Math.max(oracleJi, smartDocJaccard(truth, candidate, clip.reference));
      oracleIou = Math.max(oracleIou, visibleIoU(truth, candidate, width, height));
    }
    return {
      ji,
      iou,
      found: quad !== null,
      cornerError: quad ? cornerError(truth, quad, width, height) : Number.NaN,
      ms: run.ms,
      visible: visibleFraction(truth, width, height),
      cornersInside: truth.filter((p) => p.x >= 0 && p.y >= 0 && p.x <= width && p.y <= height)
        .length,
      area: quadArea(truth) / (width * height),
      oracleJi,
      oracleIou,
      score: run.score,
    };
  });
}

export interface Aggregate {
  frames: number;
  /** Mean SmartDoc JI. */
  ji: number;
  /** Mean visible-region IoU. */
  iou: number;
  found: number;
  iou90: number;
  iou95: number;
  /** Over found frames. */
  cornerMean: number;
  cornerP90: number;
  msMean: number;
  msP95: number;
}

export function aggregate(metrics: readonly FrameMetrics[]): Aggregate {
  const n = metrics.length;
  const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : Number.NaN);
  const corners = metrics.filter((m) => m.found).map((m) => m.cornerError);
  const ms = metrics.map((m) => m.ms);
  return {
    frames: n,
    ji: mean(metrics.map((m) => m.ji)),
    iou: mean(metrics.map((m) => m.iou)),
    found: metrics.filter((m) => m.found).length / n,
    iou90: metrics.filter((m) => m.iou >= 0.9).length / n,
    iou95: metrics.filter((m) => m.iou >= 0.95).length / n,
    cornerMean: mean(corners),
    cornerP90: quantile(corners, 0.9),
    msMean: mean(ms),
    msP95: quantile(ms, 0.95),
  };
}

export function quantile(values: readonly number[], q: number): number {
  if (!values.length) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

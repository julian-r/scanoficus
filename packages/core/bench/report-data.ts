/** Per-frame metrics, their aggregation and table formatting for the benchmark reports. */
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

/**
 * Primary per-frame score (higher is better): SmartDoc JI for SmartDoc, visible-region IoU for
 * MIDV, and for the no-document negatives 1 minus the confidence of the (false) detection.
 */
export function frameScore(clip: ClipResult, frame: FrameResult, size: number): number {
  const run = frame.runs[size];
  if (!frame.truth) return run.quad ? 1 - run.score : 1;
  const truth = unpackQuad(frame.truth);
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
    if (!frame.truth) throw new Error(`${clip.dataset}/${clip.id} has no ground-truth quads`);
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

/** Outcome of the detector on a no-document frame: any returned quad is a false positive. */
export interface NegativeMetrics {
  found: boolean;
  /** Confidence of the returned quad (0 if none). */
  score: number;
  ms: number;
}

export function negativeMetrics(clip: ClipResult, size: number): NegativeMetrics[] {
  return clip.frames.map((frame) => {
    const run = frame.runs[size];
    return { found: run.quad !== null, score: run.quad ? run.score : 0, ms: run.ms };
  });
}

/** Share of negative frames with a returned quad of confidence ≥ `threshold`. */
export function falsePositiveRate(metrics: readonly NegativeMetrics[], threshold = 0): number {
  return metrics.filter((m) => m.found && m.score >= threshold).length / metrics.length;
}

export const pct = (v: number, digits = 1) =>
  Number.isFinite(v) ? `${(100 * v).toFixed(digits)} %` : '–';
export const num = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : '–');

/** Markdown table; the first column is left-aligned, the others right-aligned. */
export function table(header: string[], rows: string[][]): string {
  const align = header.map((_, i) => (i === 0 ? '---' : '---:'));
  return [header, align, ...rows].map((r) => `| ${r.join(' | ')} |`).join('\n');
}

export const METRIC_HEADER = [
  'frames',
  'JI',
  'found',
  'IoU',
  'IoU≥0.9',
  'IoU≥0.95',
  'corner err',
  'p90 err',
  'ms',
  'p95 ms',
];

export function metricCells(a: Aggregate): string[] {
  return [
    String(a.frames),
    num(a.ji),
    pct(a.found),
    num(a.iou),
    pct(a.iou90),
    pct(a.iou95),
    pct(a.cornerMean, 2),
    pct(a.cornerP90, 2),
    num(a.msMean, 1),
    num(a.msP95, 1),
  ];
}

/**
 * Per-frame accuracy metrics of the real-data benchmark.
 *
 * The SmartDoc Jaccard index follows the official ICDAR 2015 SmartDoc Challenge 1 evaluation
 * (`eval_seg.py` in github.com/jchazalon/smartdoc15-ch1-eval; Burie et al., "ICDAR2015
 * Competition on Smartphone Document Capture and OCR (SmartDoc)", ICDAR 2015): both quads are
 * projected into the document's reference frame by the homography that maps the ground-truth
 * quad onto the reference rectangle (model size), and the Jaccard index (IoU) of the projected
 * detection and the reference rectangle is taken. A frame without a detection, or whose
 * projected detection self-intersects, scores 0. The dataset score is the mean over all frames.
 */
import {
  applyHomography,
  clipConvexPolygon,
  convexIoU,
  homographyFromQuads,
  type Point,
  type Quad,
  signedArea,
} from '../src/geometry/index.ts';
import { meanCornerError } from '../test/support/metrics.ts';

/** SmartDoc Challenge 1 Jaccard index of `detected` against `truth` (see module docs). */
export function smartDocJaccard(
  truth: Quad,
  detected: Quad | null,
  reference: { width: number; height: number },
): number {
  if (!detected) return 0;
  const { width, height } = reference;
  const target: Point[] = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
  const h = homographyFromQuads(truth, target);
  if (!h) return 0;
  const projected = detected.map((p) => applyHomography(h, p));
  if (!projected.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y))) return 0;
  if (isSelfIntersectingQuad(projected)) return 0;
  const areaTarget = width * height;
  const areaTest = Math.abs(signedArea(projected));
  if (areaTest === 0) return 0;
  // Sutherland–Hodgman against the convex target is exact for any simple subject polygon.
  const areaInter = Math.min(
    Math.abs(signedArea(clipConvexPolygon(projected, target))),
    areaTarget,
    areaTest,
  );
  return areaInter / (areaTest + areaTarget - areaInter);
}

/** True if the closed polygon `a b c d` has crossing opposite edges (a "bow tie"). */
export function isSelfIntersectingQuad(q: readonly Point[]): boolean {
  return segmentsCross(q[0], q[1], q[2], q[3]) || segmentsCross(q[1], q[2], q[3], q[0]);
}

function segmentsCross(a: Point, b: Point, c: Point, d: Point): boolean {
  const orient = (p: Point, q: Point, r: Point) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  const d1 = orient(c, d, a);
  const d2 = orient(c, d, b);
  const d3 = orient(a, b, c);
  const d4 = orient(a, b, d);
  return d1 * d2 < 0 && d3 * d4 < 0;
}

/**
 * Image-space IoU of the parts of both quads that are visible in the `width × height` frame
 * (ground-truth corners may lie outside the frame). 0 without a detection.
 */
export function visibleIoU(truth: Quad, detected: Quad | null, width: number, height: number) {
  if (!detected) return 0;
  const frame: Point[] = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
  return convexIoU(clipConvexPolygon(truth, frame), clipConvexPolygon(detected, frame));
}

/** Fraction of the ground-truth document area that lies inside the frame. */
export function visibleFraction(truth: Quad, width: number, height: number): number {
  const frame: Point[] = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
  const area = Math.abs(signedArea(truth));
  return area > 0 ? Math.abs(signedArea(clipConvexPolygon(truth, frame))) / area : 0;
}

/** Mean corner distance (best cyclic relabeling) as a fraction of the image diagonal. */
export function cornerError(truth: Quad, detected: Quad, width: number, height: number): number {
  return meanCornerError(truth, detected) / Math.hypot(width, height);
}

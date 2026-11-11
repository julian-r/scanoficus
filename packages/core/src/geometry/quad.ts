/** A 2D point in continuous image coordinates (pixel `(i, j)` is centered at `(i + .5, j + .5)`). */
export interface Point {
  readonly x: number;
  readonly y: number;
}

/**
 * A quadrilateral given by its four corners in the order top-left, top-right, bottom-right,
 * bottom-left (clockwise on screen, where y points down). Use {@link orderQuad} to normalize
 * arbitrary corner sets.
 */
export type Quad = readonly [tl: Point, tr: Point, br: Point, bl: Point];

/** Euclidean distance between two points. */
export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Orders four points as top-left, top-right, bottom-right, bottom-left.
 *
 * The points are first sorted by angle around their centroid, which yields a valid clockwise
 * cycle for any convex quad regardless of rotation (unlike the common "min/max of x+y and x−y"
 * trick, which can pick the same point twice for rotated quads). The cycle is then rotated so
 * that the edge `tl → tr` is the edge whose midpoint is highest in the image. For a quad
 * rotated by exactly 45° the choice between the two upper edges is arbitrary but deterministic.
 */
export function orderQuad(points: readonly Point[]): Quad {
  if (points.length !== 4) throw new RangeError(`Expected 4 points, got ${points.length}`);
  const cx = (points[0].x + points[1].x + points[2].x + points[3].x) / 4;
  const cy = (points[0].y + points[1].y + points[2].y + points[3].y) / 4;
  // Increasing atan2 with y pointing down is clockwise on screen.
  const cycle = [...points].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );
  let top = 0;
  let topY = Number.POSITIVE_INFINITY;
  for (let i = 0; i < 4; i++) {
    const midY = (cycle[i].y + cycle[(i + 1) % 4].y) / 2;
    if (midY < topY - 1e-9) {
      topY = midY;
      top = i;
    }
  }
  return [cycle[top], cycle[(top + 1) % 4], cycle[(top + 2) % 4], cycle[(top + 3) % 4]];
}

/**
 * Signed area of a polygon via the shoelace formula. Positive for clockwise order on screen
 * (y down), negative for counter-clockwise.
 */
export function signedArea(polygon: readonly Point[]): number {
  let sum = 0;
  for (let i = 0, n = polygon.length; i < n; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % n];
    sum += a.x * b.y - b.x * a.y;
  }
  return sum / 2;
}

/** Unsigned area of a quad (or any simple polygon). */
export function quadArea(quad: readonly Point[]): number {
  return Math.abs(signedArea(quad));
}

/**
 * Returns true if the quad is strictly convex. A strictly convex quad is necessarily simple, so
 * this also rejects self-intersecting ("bow-tie") quads: the cross products of consecutive edges
 * must be non-zero and share one sign. Corner order (clockwise or not) does not matter.
 */
export function isConvexQuad(quad: readonly Point[]): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const c = quad[(i + 2) % 4];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) < 1e-9) return false;
    const s = Math.sign(cross);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/**
 * Interior angles at each corner in degrees, in corner order. For a convex quad they sum to 360.
 */
export function quadAngles(quad: Quad): [number, number, number, number] {
  const angles: [number, number, number, number] = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    const p = quad[i];
    const prev = quad[(i + 3) % 4];
    const next = quad[(i + 1) % 4];
    const ux = prev.x - p.x;
    const uy = prev.y - p.y;
    const vx = next.x - p.x;
    const vy = next.y - p.y;
    angles[i] = (Math.atan2(Math.abs(ux * vy - uy * vx), ux * vx + uy * vy) * 180) / Math.PI;
  }
  return angles;
}

/** Side lengths in the order top, right, bottom, left. */
export function quadSideLengths(quad: Quad): [number, number, number, number] {
  return [
    distance(quad[0], quad[1]),
    distance(quad[1], quad[2]),
    distance(quad[2], quad[3]),
    distance(quad[3], quad[0]),
  ];
}

/**
 * Largest distance between corresponding corners of two quads. Both quads must use the same
 * corner order.
 */
export function maxCornerDistance(a: Quad, b: Quad): number {
  return Math.max(
    distance(a[0], b[0]),
    distance(a[1], b[1]),
    distance(a[2], b[2]),
    distance(a[3], b[3]),
  );
}

/** Multiplies all coordinates by `sx`, `sy` (e.g. to map between image resolutions). */
export function scaleQuad(quad: Quad, sx: number, sy: number = sx): Quad {
  return [
    { x: quad[0].x * sx, y: quad[0].y * sy },
    { x: quad[1].x * sx, y: quad[1].y * sy },
    { x: quad[2].x * sx, y: quad[2].y * sy },
    { x: quad[3].x * sx, y: quad[3].y * sy },
  ];
}

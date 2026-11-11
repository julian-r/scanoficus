import { type Point, signedArea } from './quad.ts';

/**
 * Intersection of two convex polygons using Sutherland–Hodgman clipping of `subject` against
 * every edge of `clip`. Both polygons may be in either orientation. Returns the (possibly empty)
 * intersection polygon.
 */
export function clipConvexPolygon(subject: readonly Point[], clip: readonly Point[]): Point[] {
  // Inside test is "left of edge" for counter-clockwise clip polygons; flip for clockwise ones.
  const orientation = Math.sign(signedArea(clip)) || 1;
  let output: Point[] = [...subject];
  for (let i = 0, n = clip.length; i < n && output.length > 0; i++) {
    const a = clip[i];
    const b = clip[(i + 1) % n];
    const side = (p: Point) =>
      orientation * ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
    const input = output;
    output = [];
    for (let j = 0, m = input.length; j < m; j++) {
      const cur = input[j];
      const prev = input[(j + m - 1) % m];
      const sCur = side(cur);
      const sPrev = side(prev);
      if (sCur >= 0) {
        if (sPrev < 0) output.push(intersect(prev, cur, sPrev, sCur));
        output.push(cur);
      } else if (sPrev >= 0) {
        output.push(intersect(prev, cur, sPrev, sCur));
      }
    }
  }
  return output;
}

/**
 * Intersection over union of two convex polygons (typically quads), in `[0, 1]`. Returns 0 if
 * either polygon has zero area.
 */
export function convexIoU(a: readonly Point[], b: readonly Point[]): number {
  const areaA = Math.abs(signedArea(a));
  const areaB = Math.abs(signedArea(b));
  if (areaA === 0 || areaB === 0) return 0;
  const inter = Math.abs(signedArea(clipConvexPolygon(a, b)));
  return inter / (areaA + areaB - inter);
}

/** Point on segment `p → q` where the signed side value crosses zero. */
function intersect(p: Point, q: Point, sp: number, sq: number): Point {
  const t = sp / (sp - sq);
  return { x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t };
}

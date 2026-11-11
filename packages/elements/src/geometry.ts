import { isConvexQuad, type Point, type Quad } from '@scanoficus/core/geometry';

/** Placement of a `sourceWidth × sourceHeight` image scaled to fit (contain) a box. */
export interface Fit {
  scale: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Fits an image into a box preserving aspect ratio, centered (CSS `object-fit: contain`). */
export function fitContain(
  sourceWidth: number,
  sourceHeight: number,
  boxWidth: number,
  boxHeight: number,
): Fit {
  const scale = Math.min(boxWidth / sourceWidth, boxHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { scale, x: (boxWidth - width) / 2, y: (boxHeight - height) / 2, width, height };
}

/** The quad covering a whole `width × height` image. */
export function fullQuad(width: number, height: number): Quad {
  return [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
}

/**
 * Moves corner `index` of `quad` to `point`, clamped to the image. Returns `null` if the result
 * would not be a convex quad (the editor then keeps the previous position).
 */
export function moveCorner(
  quad: Quad,
  index: number,
  point: Point,
  width: number,
  height: number,
): Quad | null {
  const clamped = {
    x: Math.max(0, Math.min(width, point.x)),
    y: Math.max(0, Math.min(height, point.y)),
  };
  const next = quad.map((p, i) => (i === index ? clamped : p)) as unknown as Quad;
  return isConvexQuad(next) ? next : null;
}

/** SVG `points` attribute for a polygon. */
export function svgPoints(points: readonly Point[]): string {
  return points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

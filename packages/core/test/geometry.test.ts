import { describe, expect, it } from 'vitest';
import {
  applyHomography,
  clipConvexPolygon,
  convexIoU,
  estimateOutputSize,
  homographyFromQuads,
  invertMatrix3,
  isConvexQuad,
  multiplyMatrix3,
  orderQuad,
  type Point,
  type Quad,
  quadAngles,
  quadArea,
  rectangleAspectRatio,
  rectToQuadHomography,
  scaleQuad,
} from '../src/geometry/index.ts';
import { Random } from './support/random.ts';

const square: Quad = [
  { x: 0, y: 0 },
  { x: 10, y: 0 },
  { x: 10, y: 10 },
  { x: 0, y: 10 },
];

function rotate(points: readonly Point[], degrees: number, cx = 0, cy = 0): Point[] {
  const a = (degrees * Math.PI) / 180;
  return points.map((p) => ({
    x: cx + (p.x - cx) * Math.cos(a) - (p.y - cy) * Math.sin(a),
    y: cy + (p.x - cx) * Math.sin(a) + (p.y - cy) * Math.cos(a),
  }));
}

function randomConvexQuad(rnd: Random): Quad {
  // Perturbed rectangle corners stay convex for perturbations below half the side length.
  const w = rnd.range(50, 300);
  const h = rnd.range(50, 300);
  const j = () => rnd.range(-0.2, 0.2);
  return [
    { x: j() * w, y: j() * h },
    { x: w + j() * w, y: j() * h },
    { x: w + j() * w, y: h + j() * h },
    { x: j() * w, y: h + j() * h },
  ];
}

describe('orderQuad', () => {
  it('orders shuffled corners as TL, TR, BR, BL', () => {
    const shuffled = [square[2], square[0], square[3], square[1]];
    expect(orderQuad(shuffled)).toEqual(square);
  });

  it('returns a valid clockwise permutation for rotated quads', () => {
    for (let deg = -40; deg <= 40; deg += 10) {
      const rotated = rotate(square, deg, 5, 5);
      const ordered = orderQuad([rotated[3], rotated[1], rotated[0], rotated[2]]);
      expect(ordered).toEqual(rotated);
    }
  });

  it('keeps all four points for a 45° diamond', () => {
    const diamond = [
      { x: 5, y: 0 },
      { x: 10, y: 5 },
      { x: 5, y: 10 },
      { x: 0, y: 5 },
    ];
    const ordered = orderQuad([diamond[2], diamond[0], diamond[3], diamond[1]]);
    expect(new Set(ordered)).toEqual(new Set(diamond));
    expect(isConvexQuad(ordered)).toBe(true);
  });

  it('rejects other point counts', () => {
    expect(() => orderQuad(square.slice(0, 3))).toThrow(RangeError);
  });
});

describe('quad properties', () => {
  it('detects convex, concave, self-intersecting and degenerate quads', () => {
    expect(isConvexQuad(square)).toBe(true);
    expect(isConvexQuad([...square].reverse())).toBe(true);
    const bowTie = [square[0], square[2], square[1], square[3]];
    expect(isConvexQuad(bowTie)).toBe(false);
    const concave = [square[0], { x: 5, y: 8 }, square[2], square[3]];
    expect(isConvexQuad(concave)).toBe(false);
    const degenerate = [square[0], { x: 5, y: 0 }, square[1], square[2]];
    expect(isConvexQuad(degenerate)).toBe(false);
  });

  it('computes area and interior angles', () => {
    expect(quadArea(square)).toBe(100);
    expect(quadAngles(square)).toEqual([90, 90, 90, 90]);
    const trapezoid: Quad = [
      { x: 2, y: 0 },
      { x: 8, y: 0 },
      { x: 10, y: 4 },
      { x: 0, y: 4 },
    ];
    expect(quadArea(trapezoid)).toBe(32);
    const angles = quadAngles(trapezoid);
    expect(angles.reduce((a, b) => a + b, 0)).toBeCloseTo(360, 9);
    expect(angles[3]).toBeCloseTo((Math.atan2(4, 2) * 180) / Math.PI, 9);
  });

  it('scales quads', () => {
    expect(scaleQuad(square, 2, 0.5)[2]).toEqual({ x: 20, y: 5 });
  });
});

describe('homography', () => {
  it('maps the four source points onto the destination points', () => {
    const rnd = new Random(1);
    for (let i = 0; i < 100; i++) {
      const src = randomConvexQuad(rnd);
      const dst = randomConvexQuad(rnd);
      const h = homographyFromQuads(src, dst);
      expect(h).not.toBeNull();
      for (let k = 0; k < 4; k++) {
        const p = applyHomography(h!, src[k]);
        expect(p.x).toBeCloseTo(dst[k].x, 6);
        expect(p.y).toBeCloseTo(dst[k].y, 6);
      }
    }
  });

  it('round-trips arbitrary points through the inverse', () => {
    const rnd = new Random(2);
    for (let i = 0; i < 50; i++) {
      const h = homographyFromQuads(randomConvexQuad(rnd), randomConvexQuad(rnd))!;
      const inv = invertMatrix3(h)!;
      const product = multiplyMatrix3(inv, h);
      for (let k = 0; k < 9; k++) expect(product[k] / product[8]).toBeCloseTo(k % 4 ? 0 : 1, 9);
      const p = { x: rnd.range(0, 200), y: rnd.range(0, 200) };
      const q = applyHomography(inv, applyHomography(h, p));
      expect(q.x).toBeCloseTo(p.x, 6);
      expect(q.y).toBeCloseTo(p.y, 6);
    }
  });

  it('maps a rectangle onto a quad', () => {
    const quad: Quad = [
      { x: 10, y: 20 },
      { x: 110, y: 25 },
      { x: 100, y: 160 },
      { x: 5, y: 150 },
    ];
    const h = rectToQuadHomography(50, 70, quad)!;
    expect(applyHomography(h, { x: 50, y: 70 }).x).toBeCloseTo(100, 9);
    expect(applyHomography(h, { x: 0, y: 70 }).y).toBeCloseTo(150, 9);
  });

  it('returns null for degenerate correspondences', () => {
    const collinear = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
      { x: 3, y: 3 },
    ];
    expect(homographyFromQuads(collinear, square)).toBeNull();
    expect(invertMatrix3([1, 2, 3, 2, 4, 6, 0, 0, 1])).toBeNull();
  });
});

describe('convexIoU', () => {
  it('handles identical, disjoint and partially overlapping quads', () => {
    expect(convexIoU(square, square)).toBeCloseTo(1, 12);
    const far = scaleQuad(square, 1).map((p) => ({ x: p.x + 100, y: p.y }));
    expect(convexIoU(square, far)).toBe(0);
    const shifted = square.map((p) => ({ x: p.x + 5, y: p.y }));
    expect(convexIoU(square, shifted)).toBeCloseTo(50 / 150, 12);
  });

  it('is independent of polygon orientation', () => {
    const rotated = rotate(square, 45, 5, 5);
    const a = convexIoU(square, rotated);
    expect(convexIoU([...square].reverse(), rotated)).toBeCloseTo(a, 12);
    expect(convexIoU(square, [...rotated].reverse())).toBeCloseTo(a, 12);
    // Intersection of a square and its 45° rotation is a regular octagon.
    const side = 10 * (Math.SQRT2 - 1);
    const octagon = 2 * (1 + Math.SQRT2) * side * side;
    expect(a).toBeCloseTo(octagon / (200 - octagon), 9);
  });

  it('clips to the inner polygon when one contains the other', () => {
    const inner: Quad = [
      { x: 2, y: 2 },
      { x: 4, y: 2 },
      { x: 4, y: 4 },
      { x: 2, y: 4 },
    ];
    expect(quadArea(clipConvexPolygon(square, inner))).toBeCloseTo(4, 12);
    expect(convexIoU(square, inner)).toBeCloseTo(0.04, 12);
  });
});

/** Projects a w×h rectangle with a pinhole camera (focal f, principal point c) after a 3D tilt. */
function projectRectangle(
  w: number,
  h: number,
  tiltX: number,
  tiltY: number,
  f: number,
  c: Point,
): Quad {
  const corners = [
    [-w / 2, -h / 2],
    [w / 2, -h / 2],
    [w / 2, h / 2],
    [-w / 2, h / 2],
  ].map(([X, Y]) => {
    const y1 = Y * Math.cos(tiltX);
    const z1 = Y * Math.sin(tiltX);
    const x2 = X * Math.cos(tiltY) + z1 * Math.sin(tiltY);
    const z2 = -X * Math.sin(tiltY) + z1 * Math.cos(tiltY);
    const z = 3 + z2;
    return { x: c.x + (f * x2) / z, y: c.y + (f * y1) / z };
  });
  return [corners[0], corners[1], corners[2], corners[3]];
}

describe('estimateOutputSize', () => {
  const square4k = scaleQuad(square, 400);

  it('uses the longest opposite edges', () => {
    const quad: Quad = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 90, y: 50 },
      { x: 10, y: 60 },
    ];
    const size = estimateOutputSize(quad);
    expect(size.width).toBe(100);
    expect(size.height).toBe(Math.round(Math.hypot(10, 60)));
  });

  it('clamps the pixel area preserving aspect', () => {
    const square = estimateOutputSize(scaleQuad(square4k, 1), { maxPixels: 12_000_000 });
    expect(square.width * square.height).toBeLessThanOrEqual(12_000_000);
    expect(square.width).toBe(square.height);
    const nearSquare = estimateOutputSize(scaleQuad(square4k, 1, 0.9), {
      maxLongSide: 4000,
      maxPixels: 12_000_000,
    });
    expect(nearSquare.width * nearSquare.height).toBeLessThanOrEqual(12_000_000);
    expect(nearSquare.width / nearSquare.height).toBeCloseTo(1 / 0.9, 2);
  });

  it('clamps to maxLongSide preserving aspect', () => {
    const size = estimateOutputSize(scaleQuad(square, 100), { maxLongSide: 250 });
    expect(size).toEqual({ width: 250, height: 250 });
  });

  it('recovers the true aspect ratio of a tilted rectangle', () => {
    const image = { width: 1000, height: 750 };
    const center = { x: 500, y: 375 };
    for (const [tx, ty] of [
      [0.5, 0.2],
      [-0.4, 0.35],
      [0.3, -0.5],
    ]) {
      const quad = projectRectangle(1, Math.SQRT2, tx, ty, 900, center);
      const ratio = rectangleAspectRatio(quad, center.x, center.y)!;
      expect(ratio).toBeCloseTo(1 / Math.SQRT2, 6);
      const size = estimateOutputSize(quad, { imageSize: image });
      expect(size.width / size.height).toBeCloseTo(1 / Math.SQRT2, 1);
    }
  });

  it('falls back to the affine ratio without perspective', () => {
    const rect = scaleQuad(square, 30, 20);
    expect(rectangleAspectRatio(rect, 500, 375)).toBeCloseTo(1.5, 9);
  });
});

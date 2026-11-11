import type { Point, Quad } from './quad.ts';

/**
 * A 3×3 matrix in row-major order: `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
 * A homography maps `(x, y)` to `((m00 x + m01 y + m02) / w, (m10 x + m11 y + m12) / w)` with
 * `w = m20 x + m21 y + m22`.
 */
export type Matrix3 = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/** The 3×3 identity matrix. */
export const IDENTITY3: Matrix3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Matrix product `a · b` (apply `b` first, then `a`). */
export function multiplyMatrix3(a: Matrix3, b: Matrix3): Matrix3 {
  const r = new Array<number>(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return r as unknown as Matrix3;
}

/** Inverse of a 3×3 matrix via the adjugate, or `null` if it is (numerically) singular. */
export function invertMatrix3(m: Matrix3): Matrix3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const scale = Math.max(...m.map(Math.abs));
  if (!Number.isFinite(det) || Math.abs(det) <= 1e-12 * scale * scale * scale) return null;
  const k = 1 / det;
  return [
    A * k,
    (c * h - b * i) * k,
    (b * f - c * e) * k,
    B * k,
    (a * i - c * g) * k,
    (c * d - a * f) * k,
    C * k,
    (b * g - a * h) * k,
    (a * e - b * d) * k,
  ];
}

/** Applies a homography to a point (including the perspective division). */
export function applyHomography(h: Matrix3, p: Point): Point {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  return {
    x: (h[0] * p.x + h[1] * p.y + h[2]) / w,
    y: (h[3] * p.x + h[4] * p.y + h[5]) / w,
  };
}

/**
 * Computes the homography `H` with `H(src[i]) = dst[i]` for four point correspondences.
 *
 * Both point sets are first normalized (Hartley: centroid at the origin, mean distance √2) to
 * keep the linear system well conditioned for pixel-sized coordinates. With `h22 = 1` the four
 * correspondences give an 8×8 linear system, solved by Gaussian elimination with partial
 * pivoting. Returns `null` for degenerate input (e.g. three collinear points).
 */
export function homographyFromQuads(src: readonly Point[], dst: readonly Point[]): Matrix3 | null {
  if (src.length !== 4 || dst.length !== 4) throw new RangeError('Expected 4 point pairs');
  const ns = normalization(src);
  const nd = normalization(dst);
  if (!ns || !nd) return null;
  const s = src.map((p) => applyHomography(ns, p));
  const d = dst.map((p) => applyHomography(nd, p));

  // Rows: [x, y, 1, 0, 0, 0, -u x, -u y | u] and [0, 0, 0, x, y, 1, -v x, -v y | v].
  const a = new Float64Array(8 * 9);
  for (let i = 0; i < 4; i++) {
    const { x, y } = s[i];
    const { x: u, y: v } = d[i];
    a.set([x, y, 1, 0, 0, 0, -u * x, -u * y, u], i * 18);
    a.set([0, 0, 0, x, y, 1, -v * x, -v * y, v], i * 18 + 9);
  }
  const h = solveAugmented(a, 8);
  if (!h) return null;
  const normalized: Matrix3 = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  const ndInv = invertMatrix3(nd);
  if (!ndInv) return null;
  const result = multiplyMatrix3(ndInv, multiplyMatrix3(normalized, ns));
  return normalizeScale(result);
}

/**
 * Homography mapping the axis-aligned rectangle `[0, width] × [0, height]` (corners in TL, TR,
 * BR, BL order) onto `quad`. This is the "inverse" mapping used for rectification: for every
 * output pixel it gives the source location to sample.
 */
export function rectToQuadHomography(width: number, height: number, quad: Quad): Matrix3 | null {
  return homographyFromQuads(
    [
      { x: 0, y: 0 },
      { x: width, y: 0 },
      { x: width, y: height },
      { x: 0, y: height },
    ],
    quad,
  );
}

/** Similarity transform moving the centroid to the origin with mean distance √2. */
function normalization(points: readonly Point[]): Matrix3 | null {
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.x;
    cy += p.y;
  }
  cx /= points.length;
  cy /= points.length;
  let meanDist = 0;
  for (const p of points) meanDist += Math.hypot(p.x - cx, p.y - cy);
  meanDist /= points.length;
  if (!(meanDist > 0) || !Number.isFinite(meanDist)) return null;
  const k = Math.SQRT2 / meanDist;
  return [k, 0, -k * cx, 0, k, -k * cy, 0, 0, 1];
}

/** Scales a homography so that `h22 = 1` when possible (otherwise unit Frobenius norm). */
function normalizeScale(m: Matrix3): Matrix3 {
  let k = m[8];
  if (Math.abs(k) < 1e-12) k = Math.hypot(...m);
  return m.map((v) => v / k) as unknown as Matrix3;
}

/**
 * Solves the `n × n` system stored as an augmented row-major `n × (n + 1)` matrix in place using
 * Gaussian elimination with partial pivoting. Returns `null` if the system is singular.
 */
function solveAugmented(a: Float64Array, n: number): Float64Array | null {
  const stride = n + 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(a[col * stride + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r * stride + col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-12) return null;
    if (pivot !== col) {
      for (let k = col; k < stride; k++) {
        const t = a[col * stride + k];
        a[col * stride + k] = a[pivot * stride + k];
        a[pivot * stride + k] = t;
      }
    }
    const inv = 1 / a[col * stride + col];
    for (let r = col + 1; r < n; r++) {
      const f = a[r * stride + col] * inv;
      if (f === 0) continue;
      for (let k = col; k < stride; k++) a[r * stride + k] -= f * a[col * stride + k];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let sum = a[r * stride + n];
    for (let k = r + 1; k < n; k++) sum -= a[r * stride + k] * x[k];
    x[r] = sum / a[r * stride + r];
  }
  return x;
}

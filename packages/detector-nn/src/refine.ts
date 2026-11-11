import type { Point, Quad } from '@scanoficus/core/geometry';
import { downscaleGray, type GrayImage } from '@scanoficus/core/image';

/** Options of {@link refineQuad} (defaults as tuned for DocQuadNet in scanoficus-ml). */
export interface RefineOptions {
  /** Refinement works on the gray frame area-downscaled to this long side. Default 1024. */
  longSide?: number;
  /** Search window across each side, as a fraction of the image diagonal. Default 0.008. */
  searchFraction?: number;
  /** Profiles per side. Default 48. */
  samples?: number;
  /** Minimum fraction of profiles with an edge for a side to be re-fitted. Default 0.4. */
  minFound?: number;
  /** Maximum angle (degrees) between the re-fitted and the predicted side. Default 4. */
  maxAngle?: number;
  /** Local maxima below this fraction of the profile maximum are ignored. Default 0.35. */
  peakRatio?: number;
}

const SIGMA = 1;
const RADIUS = 4; // OpenCV's 9-tap kernel for sigma 1 on float images
const KERNEL = (() => {
  const k = Array.from({ length: 2 * RADIUS + 1 }, (_, i) =>
    Math.exp(-((i - RADIUS) ** 2) / (2 * SIGMA * SIGMA)),
  );
  const sum = k.reduce((a, b) => a + b, 0);
  return Float32Array.from(k, (v) => v / sum);
})();
/** Huber constant of OpenCV's `fitLine(DIST_HUBER)` with `param = 0`. */
const HUBER = 1.345;

/**
 * Edge refinement of a predicted quad (a port of `scanoficus_ml/refine.py`): for each side, the
 * gradient across the side (Gaussian σ = 1, Sobel) is profiled at `samples` positions within
 * ± `searchFraction` of the diagonal; the strongest local maximum weighted by closeness to the
 * side (Gaussian, σ = window / 2) gives a sub-pixel edge point; a robust (Huber) line through the
 * points replaces the side if enough points were found and the line stays within the window and
 * `maxAngle` of the side. Corners are intersections of adjacent sides and fall back to the
 * predicted corner if they move more than twice the window.
 *
 * Gradients are computed lazily, only at the pixels the profiles touch, so the cost does not
 * grow with the frame area. Coordinates follow the core convention (pixel centers at +0.5).
 *
 * @param gray - The frame in gray (any size; downscaled to `longSide` first).
 * @param quad - Predicted quad in `gray` pixels, corners in order around the quad.
 * @returns The refined quad in `gray` pixels, same corner order.
 */
export function refineQuad(gray: GrayImage, quad: Quad, options: RefineOptions = {}): Quad {
  const longSide = options.longSide ?? 1024;
  const samples = options.samples ?? 48;
  const minFound = options.minFound ?? 0.4;
  const maxAngle = ((options.maxAngle ?? 4) * Math.PI) / 180;
  const peakRatio = options.peakRatio ?? 0.35;
  const small = downscaleGray(gray, longSide);
  const sx = small.width / gray.width;
  const sy = small.height / gray.height;
  const q = quad.map((p) => ({ x: p.x * sx, y: p.y * sy }));
  const field = new GradientField(small);
  const search = Math.max(
    4,
    (options.searchFraction ?? 0.008) * Math.hypot(small.width, small.height),
  );
  const offsets: number[] = [];
  for (let o = -search; o < search + 0.5; o += 0.5) offsets.push(o);
  const n = offsets.length;
  const row = new Float64Array(n);
  const lines: { p: Point; d: Point }[] = [];
  for (let i = 0; i < 4; i++) {
    const a = q[i];
    const b = q[(i + 1) % 4];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) return quad;
    const t = { x: dx / length, y: dy / length };
    const normal = { x: -t.y, y: t.x };
    const points: Point[] = [];
    for (let j = 0; j < samples; j++) {
      const s = 0.12 + (0.76 * j) / (samples - 1);
      const cx = a.x + s * dx;
      const cy = a.y + s * dy;
      let top = 0;
      for (let k = 0; k < n; k++) {
        const v = field.response(cx + offsets[k] * normal.x, cy + offsets[k] * normal.y, normal);
        row[k] = v;
        if (v > top) top = v;
      }
      if (top < 4) continue;
      let best = -1;
      let bestScore = -1;
      for (let k = 1; k < n - 1; k++) {
        const v = row[k];
        if (v < row[k - 1] || v <= row[k + 1] || v < peakRatio * top) continue;
        const score = v * Math.exp(-0.5 * (offsets[k] / (0.5 * search)) ** 2);
        if (score > bestScore) {
          bestScore = score;
          best = k;
        }
      }
      if (best < 0) continue;
      const denom = row[best - 1] - 2 * row[best] + row[best + 1];
      const frac = denom < 0 ? (0.5 * (row[best - 1] - row[best + 1])) / denom : 0;
      const off = offsets[best] + frac * 0.5;
      points.push({ x: cx + off * normal.x, y: cy + off * normal.y });
    }
    let line = { p: a, d: t };
    if (points.length >= minFound * samples) {
      const fit = fitLineHuber(points);
      const cos = Math.min(1, Math.abs(fit.d.x * t.x + fit.d.y * t.y));
      const mid = { x: a.x + 0.5 * dx, y: a.y + 0.5 * dy };
      const distance = Math.abs((mid.x - fit.p.x) * -fit.d.y + (mid.y - fit.p.y) * fit.d.x);
      if (Math.acos(cos) <= maxAngle && distance <= search) line = fit;
    }
    lines.push(line);
  }
  const out: Point[] = [];
  for (let i = 0; i < 4; i++) {
    const c = intersect(lines[(i + 3) % 4], lines[i]);
    const keep = c && Math.hypot(c.x - q[i].x, c.y - q[i].y) <= 2 * search;
    const p = keep ? c : q[i];
    out.push({ x: p.x / sx, y: p.y / sy });
  }
  return out as unknown as Quad;
}

function intersect(l1: { p: Point; d: Point }, l2: { p: Point; d: Point }): Point | null {
  const cross = l1.d.x * l2.d.y - l1.d.y * l2.d.x;
  if (Math.abs(cross) < 1e-9) return null;
  const t = ((l2.p.x - l1.p.x) * l2.d.y - (l2.p.y - l1.p.y) * l2.d.x) / cross;
  return { x: l1.p.x + t * l1.d.x, y: l1.p.y + t * l1.d.y };
}

/** Weighted total least squares line, iteratively reweighted with Huber weights. */
function fitLineHuber(points: Point[]): { p: Point; d: Point } {
  const weights = new Float64Array(points.length).fill(1);
  let line = { p: points[0], d: { x: 1, y: 0 } };
  for (let iter = 0; iter < 20; iter++) {
    let sw = 0;
    let mx = 0;
    let my = 0;
    points.forEach((p, i) => {
      sw += weights[i];
      mx += weights[i] * p.x;
      my += weights[i] * p.y;
    });
    mx /= sw;
    my /= sw;
    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    points.forEach((p, i) => {
      const u = p.x - mx;
      const v = p.y - my;
      sxx += weights[i] * u * u;
      syy += weights[i] * v * v;
      sxy += weights[i] * u * v;
    });
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const next = { p: { x: mx, y: my }, d: { x: Math.cos(angle), y: Math.sin(angle) } };
    const moved =
      Math.hypot(next.p.x - line.p.x, next.p.y - line.p.y) +
      Math.abs(next.d.x * line.d.y - next.d.y * line.d.x);
    line = next;
    if (iter > 0 && moved < 1e-4) break;
    points.forEach((p, i) => {
      const r = Math.abs((p.x - mx) * -line.d.y + (p.y - my) * line.d.x);
      weights[i] = r < HUBER ? 1 : HUBER / r;
    });
  }
  return line;
}

/**
 * Lazily evaluated Gaussian-blurred Sobel gradients (OpenCV defaults: reflect-101 borders),
 * cached per pixel.
 */
class GradientField {
  private readonly hblur: Float32Array;
  private readonly blur: Float32Array;
  private readonly gx: Float32Array;
  private readonly gy: Float32Array;
  /** Per pixel: bit 1 = hblur, 2 = blur, 4 = gradients computed. */
  private readonly done: Uint8Array;

  private readonly image: GrayImage;

  constructor(image: GrayImage) {
    this.image = image;
    const n = image.width * image.height;
    // Reuse the buffers of the previous call (same frame size in a video stream).
    if (pool?.done.length !== n) {
      pool = {
        hblur: new Float32Array(n),
        blur: new Float32Array(n),
        gx: new Float32Array(n),
        gy: new Float32Array(n),
        done: new Uint8Array(n),
      };
    } else pool.done.fill(0);
    ({ hblur: this.hblur, blur: this.blur, gx: this.gx, gy: this.gy, done: this.done } = pool);
  }

  /**
   * |∇I · n| bilinearly interpolated at continuous point (x, y), or 0 outside the region whose
   * pixel-index coordinates lie in [1, size − 2].
   */
  response(x: number, y: number, n: Point): number {
    const { width, height } = this.image;
    const u = x - 0.5;
    const v = y - 0.5;
    if (!(u >= 1 && v >= 1 && u <= width - 2 && v <= height - 2)) return 0;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const fx = u - x0;
    const fy = v - y0;
    const x1 = Math.min(x0 + 1, width - 1);
    const y1 = Math.min(y0 + 1, height - 1);
    const p00 = this.gradient(x0, y0);
    const p10 = this.gradient(x1, y0);
    const p01 = this.gradient(x0, y1);
    const p11 = this.gradient(x1, y1);
    const w00 = (1 - fx) * (1 - fy);
    const w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy;
    const w11 = fx * fy;
    const { gx, gy } = this;
    const ix = w00 * gx[p00] + w10 * gx[p10] + w01 * gx[p01] + w11 * gx[p11];
    const iy = w00 * gy[p00] + w10 * gy[p10] + w01 * gy[p01] + w11 * gy[p11];
    return Math.abs(ix * n.x + iy * n.y);
  }

  /** Computes the Sobel gradient at pixel (x, y) if needed; returns its index. */
  private gradient(x: number, y: number): number {
    const { width, height } = this.image;
    const i = y * width + x;
    if (this.done[i] & 4) return i;
    const xm = reflect(x - 1, width);
    const xp = reflect(x + 1, width);
    const ym = reflect(y - 1, height);
    const yp = reflect(y + 1, height);
    const b = (bx: number, by: number) => this.blurred(bx, by);
    this.gx[i] = b(xp, ym) - b(xm, ym) + 2 * (b(xp, y) - b(xm, y)) + b(xp, yp) - b(xm, yp);
    this.gy[i] = b(xm, yp) - b(xm, ym) + 2 * (b(x, yp) - b(x, ym)) + b(xp, yp) - b(xp, ym);
    this.done[i] |= 4;
    return i;
  }

  private blurred(x: number, y: number): number {
    const { width, height } = this.image;
    const i = y * width + x;
    if (!(this.done[i] & 2)) {
      let s = 0;
      for (let k = -RADIUS; k <= RADIUS; k++)
        s += KERNEL[k + RADIUS] * this.hblurred(x, reflect(y + k, height));
      this.blur[i] = s;
      this.done[i] |= 2;
    }
    return this.blur[i];
  }

  private hblurred(x: number, y: number): number {
    const { width, data } = this.image;
    const i = y * width + x;
    if (!(this.done[i] & 1)) {
      const row = y * width;
      let s = 0;
      for (let k = -RADIUS; k <= RADIUS; k++)
        s += KERNEL[k + RADIUS] * data[row + reflect(x + k, width)];
      this.hblur[i] = s;
      this.done[i] |= 1;
    }
    return this.hblur[i];
  }
}

let pool: {
  hblur: Float32Array;
  blur: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
  done: Uint8Array;
} | null = null;

/** OpenCV's BORDER_REFLECT_101 index mapping. */
function reflect(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

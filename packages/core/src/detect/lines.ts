import type { Point } from '../geometry/quad.ts';

/** Infinite line in Hesse normal form: points `p` with `nx * p.x + ny * p.y = rho`, |n| = 1. */
export interface Line {
  nx: number;
  ny: number;
  rho: number;
}

/** Intersection of two lines, or `null` if they are (nearly) parallel. */
export function intersectLines(a: Line, b: Line): Point | null {
  const det = a.nx * b.ny - a.ny * b.nx;
  if (Math.abs(det) < 1e-6) return null;
  return {
    x: (a.rho * b.ny - a.ny * b.rho) / det,
    y: (a.nx * b.rho - a.rho * b.nx) / det,
  };
}

/** Line through two distinct points. */
export function lineThrough(a: Point, b: Point): Line {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  const nx = -dy / len;
  const ny = dx / len;
  return { nx, ny, rho: nx * a.x + ny * a.y };
}

/**
 * Weighted point accumulator for robust total-least-squares line fitting. Points are added
 * once; {@link LineFitter.fit} runs a few iteratively reweighted passes (Tukey biweight on the
 * orthogonal residual) starting from an initial line, which suppresses outliers such as text
 * strokes or background edges that leak into the search band.
 */
export class LineFitter {
  private xs: Float32Array;
  private ys: Float32Array;
  private ws: Float32Array;
  /** Number of points added since the last {@link LineFitter.reset}. */
  count = 0;

  constructor(capacity: number) {
    this.xs = new Float32Array(capacity);
    this.ys = new Float32Array(capacity);
    this.ws = new Float32Array(capacity);
  }

  reset(): void {
    this.count = 0;
  }

  add(x: number, y: number, weight: number): void {
    if (this.count === this.xs.length) this.grow();
    this.xs[this.count] = x;
    this.ys[this.count] = y;
    this.ws[this.count] = weight;
    this.count++;
  }

  /**
   * Fits a line to the added points, starting from `initial` and down-weighting points with
   * orthogonal residual approaching `scale` (zero weight beyond it). Returns `null` if the
   * remaining weight is degenerate.
   */
  fit(initial: Line, scale: number, iterations = 3): Line | null {
    let line = initial;
    for (let it = 0; it < iterations; it++) {
      let sw = 0;
      let sx = 0;
      let sy = 0;
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      for (let i = 0; i < this.count; i++) {
        const x = this.xs[i];
        const y = this.ys[i];
        const r = (line.nx * x + line.ny * y - line.rho) / scale;
        if (r <= -1 || r >= 1) continue;
        const t = 1 - r * r;
        const w = this.ws[i] * t * t;
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
        syy += w * y * y;
      }
      if (sw <= 0) return null;
      const mx = sx / sw;
      const my = sy / sw;
      const cxx = sxx / sw - mx * mx;
      const cxy = sxy / sw - mx * my;
      const cyy = syy / sw - my * my;
      // The normal is the eigenvector of the covariance with the smallest eigenvalue.
      const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy); // direction of largest spread
      let nx = -Math.sin(angle);
      let ny = Math.cos(angle);
      if (nx * line.nx + ny * line.ny < 0) {
        nx = -nx;
        ny = -ny;
      }
      line = { nx, ny, rho: nx * mx + ny * my };
    }
    return line;
  }

  private grow(): void {
    const n = Math.max(64, this.xs.length * 2);
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const ws = new Float32Array(n);
    xs.set(this.xs);
    ys.set(this.ys);
    ws.set(this.ws);
    this.xs = xs;
    this.ys = ys;
    this.ws = ws;
  }
}

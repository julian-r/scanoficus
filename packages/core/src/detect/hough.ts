import type { Line } from './lines.ts';

/** Angular resolution of the accumulator: 1° bins over [0°, 180°). */
const THETA_BINS = 180;
/** Each edge pixel votes for orientations within ±this many bins of its gradient direction. */
const VOTE_SPREAD = 3;

/**
 * Orientation-constrained Hough transform for straight lines.
 *
 * Lines are parameterized as `x cos θ + y sin θ = ρ` with θ ∈ [0°, 180°) and coordinates relative
 * to the image center (which halves the ρ range). Every edge pixel only votes for θ within ±3° of
 * its gradient direction instead of all 180 bins, which makes voting ~25× cheaper and produces
 * much sharper peaks than the classic transform.
 */
export class HoughTransform {
  readonly width: number;
  readonly height: number;
  private readonly rhoMax: number;
  private readonly rhoBins: number;
  private readonly acc: Int32Array;
  /** Accumulator box-filtered over ±1 ρ bin (peaks of slanted lines straddle ρ bins). */
  private readonly smoothed: Int32Array;
  private readonly cos = new Float32Array(THETA_BINS);
  private readonly sin = new Float32Array(THETA_BINS);
  /** Scratch list of peak cells: packed `theta * rhoBins + rho` indices. */
  private peaks: Int32Array;
  /** Number of selected lines per 10° orientation sector. */
  private readonly sectorCounts = new Uint8Array(THETA_BINS / 10);

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.rhoMax = Math.ceil(Math.hypot(width, height) / 2) + 1;
    this.rhoBins = 2 * this.rhoMax + 1;
    this.acc = new Int32Array(THETA_BINS * this.rhoBins);
    this.smoothed = new Int32Array(THETA_BINS * this.rhoBins);
    this.peaks = new Int32Array(256);
    for (let t = 0; t < THETA_BINS; t++) {
      this.cos[t] = Math.cos((t * Math.PI) / THETA_BINS);
      this.sin[t] = Math.sin((t * Math.PI) / THETA_BINS);
    }
  }

  /**
   * Accumulates votes for the given edge pixels (flat indices) with their gradients and returns
   * up to `maxLines` peak lines with at least `minVotes` votes, strongest first, in image
   * coordinates (origin at the top-left corner). Near-duplicate peaks are suppressed and at
   * most `maxPerOrientation` lines are kept per 10° orientation sector so that strongly
   * textured backgrounds (stripes, tiles) cannot crowd out the document edges.
   */
  detect(
    edges: Int32Array,
    edgeCount: number,
    gx: Float32Array,
    gy: Float32Array,
    minVotes: number,
    maxLines: number,
    maxPerOrientation: number,
    out: Line[],
  ): number {
    const { width, rhoMax, rhoBins, acc, cos, sin } = this;
    const cx = this.width / 2;
    const cy = this.height / 2;
    acc.fill(0);
    const binsPerRadian = THETA_BINS / Math.PI;
    for (let e = 0; e < edgeCount; e++) {
      const p = edges[e];
      const px = (p % width) + 0.5 - cx;
      const py = ((p / width) | 0) + 0.5 - cy;
      let center = Math.round(Math.atan2(gy[p], gx[p]) * binsPerRadian);
      center = ((center % THETA_BINS) + THETA_BINS) % THETA_BINS;
      for (let d = -VOTE_SPREAD; d <= VOTE_SPREAD; d++) {
        let t = center + d;
        if (t < 0) t += THETA_BINS;
        else if (t >= THETA_BINS) t -= THETA_BINS;
        const r = Math.round(px * cos[t] + py * sin[t]) + rhoMax;
        acc[t * rhoBins + r]++;
      }
    }

    // A line whose angle falls between two θ bins spreads its votes over neighboring ρ bins
    // (by up to L·sin(0.5°) for length L), so peaks are detected on a ρ-smoothed accumulator.
    const smoothed = this.smoothed;
    for (let t = 0; t < THETA_BINS; t++) {
      const row = t * rhoBins;
      smoothed[row] = 0;
      smoothed[row + rhoBins - 1] = 0;
      for (let r = 1; r < rhoBins - 1; r++) {
        smoothed[row + r] = acc[row + r - 1] + acc[row + r] + acc[row + r + 1];
      }
    }

    // Collect 3×3 local maxima (θ wraps around with ρ mirrored).
    let peakCount = 0;
    for (let t = 0; t < THETA_BINS; t++) {
      const row = t * rhoBins;
      for (let r = 1; r < rhoBins - 1; r++) {
        const v = smoothed[row + r];
        if (v < minVotes || !this.isLocalMax(t, r, v)) continue;
        if (peakCount === this.peaks.length) {
          const grown = new Int32Array(peakCount * 2);
          grown.set(this.peaks);
          this.peaks = grown;
        }
        this.peaks[peakCount++] = row + r;
      }
    }
    const peaks = this.peaks.subarray(0, peakCount).sort((a, b) => smoothed[b] - smoothed[a]);

    // Greedy selection with duplicate suppression and per-orientation caps.
    const sectorCounts = this.sectorCounts.fill(0);
    let count = 0;
    for (let k = 0; k < peakCount && count < maxLines; k++) {
      const t = (peaks[k] / rhoBins) | 0;
      const rho = (peaks[k] % rhoBins) - rhoMax;
      const sector = (t / 10) | 0;
      if (sectorCounts[sector] >= maxPerOrientation) continue;
      const nx = cos[t];
      const ny = sin[t];
      // Shift ρ from center-relative to top-left-origin coordinates.
      const line: Line = { nx, ny, rho: rho + nx * cx + ny * cy };
      let duplicate = false;
      for (let j = 0; j < count && !duplicate; j++) duplicate = similarLines(out[j], line, cx, cy);
      if (duplicate) continue;
      sectorCounts[sector]++;
      out[count++] = line;
    }
    return count;
  }

  private isLocalMax(t: number, r: number, v: number): boolean {
    const { smoothed, rhoBins } = this;
    for (let dt = -1; dt <= 1; dt++) {
      let tt = t + dt;
      let rr = r;
      if (tt < 0 || tt >= THETA_BINS) {
        tt = (tt + THETA_BINS) % THETA_BINS;
        rr = rhoBins - 1 - r; // θ ± 180° flips the sign of ρ
      }
      for (let dr = -1; dr <= 1; dr++) {
        if (dt === 0 && dr === 0) continue;
        const ri = rr + dr;
        if (ri < 0 || ri >= rhoBins) continue;
        const u = smoothed[tt * rhoBins + ri];
        // Strict on one side so that plateaus produce a single peak.
        if (u > v || (u === v && (dt < 0 || (dt === 0 && dr < 0)))) return false;
      }
    }
    return true;
  }
}

/** Lines within 3° and 6 px (measured at the image center) count as duplicates. */
function similarLines(a: Line, b: Line, cx: number, cy: number): boolean {
  const dot = a.nx * b.nx + a.ny * b.ny;
  if (Math.abs(dot) < Math.cos((3 * Math.PI) / 180)) return false;
  const s = dot < 0 ? -1 : 1;
  const ra = a.rho - a.nx * cx - a.ny * cy;
  const rb = s * (b.rho - b.nx * cx - b.ny * cy);
  return Math.abs(ra - rb) <= 6;
}

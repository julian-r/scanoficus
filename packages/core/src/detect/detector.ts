import {
  isConvexQuad,
  maxCornerDistance,
  orderQuad,
  type Point,
  type Quad,
  quadArea,
  scaleQuad,
} from '../geometry/quad.ts';
import { downscaleGray, fitLongSide, type GrayImage } from '../image.ts';
import {
  convolveSeparable,
  gaussianKernel,
  gradientNeighborOffset,
  hysteresis,
  medianPositive,
  nonMaximumSuppression,
  sobel,
} from './edges.ts';
import { HoughTransform } from './hough.ts';
import { intersectLines, type Line, LineFitter, lineThrough } from './lines.ts';

/** Options for {@link detectDocument}. */
export interface DetectOptions {
  /**
   * Longer side, in pixels, of the resolution the detector works at. Larger inputs are
   * area-downscaled first (pass an already downscaled image to avoid that per-frame cost).
   * Default 384.
   */
  maxSide?: number;
  /** Minimum document area as a fraction of the image area. Default 0.04. */
  minArea?: number;
  /**
   * Minimum confidence in `[0, 1]` for a quad to be reported. Default {@link DEFAULT_MIN_SCORE}.
   * Pass 0 to always get the best candidate (e.g. to escalate unsure frames to another detector).
   */
  minScore?: number;
  /** Also return the ranked alternative candidates. Default false. */
  candidates?: boolean;
}

/**
 * Default `minScore` of {@link detectDocument}. Calibrated at the default processing size (384)
 * on real data (see docs/benchmarks.md): about 7 % of photos without any document (desks,
 * tables, screens) yield a quad of at least this score, and 98 % of the returned quads on the
 * SmartDoc tuning split are right (Jaccard index ≥ 0.9).
 */
export const DEFAULT_MIN_SCORE = 0.75;

/**
 * Score from which a detection is reliable: about 2 % of photos without any document reach it,
 * and 99.9 % of the quads at or above it on the SmartDoc tuning split are right. Below it (or
 * without a detection), a hybrid scanner can escalate the frame to a slower, learned detector.
 */
export const CONFIDENT_SCORE = 0.85;

/**
 * A quad with a confidence score in `[0, 1]`: the edge support along its sides (mostly the
 * length-weighted mean, partly the weakest side), lowered for sides on the frame border, quads
 * below 10 % of the frame and an inside darker than the surroundings. See {@link
 * DEFAULT_MIN_SCORE} and {@link CONFIDENT_SCORE} for calibrated thresholds.
 */
export interface ScoredQuad {
  quad: Quad;
  score: number;
}

/** Result of {@link detectDocument}. Coordinates are in the input image's pixel space. */
export interface Detection extends ScoredQuad {
  /**
   * Ranked candidates (best first, including the winner) if requested via options. Runner-ups
   * are ordered by score; the winner may score lower than a runner-up it encloses (a page
   * around a block of its content).
   */
  candidates?: ScoredQuad[];
}

/** Standard deviation of the pre-blur at the processing resolution. */
const BLUR_SIGMA = 1.2;
/** Canny thresholds: high = max(MIN_HIGH, HIGH_PER_MEDIAN · median gradient), low = high / 2. */
const MIN_HIGH_THRESHOLD = 1;
const HIGH_PER_MEDIAN = 2.5;
/** Number of Hough lines considered when assembling quads, and per 10° sector. */
const MAX_LINES = 30;
const MAX_LINES_PER_SECTOR = 10;
/** The four frame borders are appended to the Hough lines as "virtual" sides. */
const FRAME_LINES = 4;
/** Score multiplier per side lying on the frame border (the document continues beyond it). */
const FRAME_PENALTY = 0.85;
/** Quads smaller than this fraction of the frame are slightly penalized (∝ area^0.25). */
const FULL_SIZE_AREA = 0.1;
/** Band (px) around a line within which edge pixels count as support. */
const SUPPORT_BAND = 1.5;
/** Support profiles are binned along the line in steps of this many pixels. */
const SUPPORT_BIN = 2;
/** Edge orientation must be within this angle of a side's normal to support it. */
const COS_SUPPORT = Math.cos((15 * Math.PI) / 180);
/** Adjacent document sides meet at 35°–145°; opposite sides differ by at most 45°. */
const COS_MIN_CORNER = Math.cos((35 * Math.PI) / 180);
const SIN_MIN_ADJACENT = Math.sin((30 * Math.PI) / 180);
const COS_MAX_OPPOSITE = Math.cos((45 * Math.PI) / 180);
/** A side must be at least this much inside the frame to be judged. */
const MIN_VISIBLE = 0.5;
/** Exponent of the paper-brightness factor (inside vs. surrounding band, see `brightnessFactor`). */
const BRIGHTNESS_EXPONENT = 0.5;
/** Inner region and surrounding band of `brightnessFactor`, as scale factors about the centroid. */
const INNER_SCALE = 0.85;
const OUTER_SCALE = 1.15;
/** Minimum score of an enclosing candidate, relative to the best (see `preferEnclosing`). */
const ENCLOSING_SCORE = 0.85;
const ENCLOSING_BRIGHTNESS = 1;
/** Number of top candidates that are refined and re-scored. */
const REFINE_CANDIDATES = 6;

/**
 * Reusable working memory for {@link detectDocument}. Keep one per video stream (or worker):
 * buffers are allocated on first use and re-allocated only when the processing size changes.
 * The fields are implementation details, not part of the public API.
 */
export class DetectorContext {
  width = 0;
  height = 0;
  small: GrayImage | null = null;
  blurTmp: Float32Array = new Float32Array(0);
  blurred: Float32Array = new Float32Array(0);
  gx: Float32Array = new Float32Array(0);
  gy: Float32Array = new Float32Array(0);
  mag: Float32Array = new Float32Array(0);
  nms: Float32Array = new Float32Array(0);
  edges: Uint8Array = new Uint8Array(0);
  edgeList: Int32Array = new Int32Array(0);
  /** Per row: prefix sums of the gray values (`width + 1` entries per row). */
  rowPrefix: Float32Array = new Float32Array(0);
  histogram: Uint32Array = new Uint32Array(2048);
  kernel: Float32Array = gaussianKernel(BLUR_SIGMA);
  hough: HoughTransform | null = null;
  /** Hough lines followed by the four frame lines. */
  lines: Line[] = [];
  fitter: LineFitter = new LineFitter(1024);
  /** Per line: prefix sums over support bins of bins containing an edge pixel along the line. */
  supportPrefix: Int32Array = new Int32Array(0);
  binFlags: Uint8Array = new Uint8Array(0);
  /** Per line: visible parameter range `[t0, t1]` inside the frame. */
  visible: Float32Array = new Float32Array(MAX_LINES * 2);
  supportBins = 0;

  /** Ensures all buffers match a `width × height` processing size. */
  ensure(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    const n = width * height;
    this.blurTmp = new Float32Array(n);
    this.blurred = new Float32Array(n);
    this.gx = new Float32Array(n);
    this.gy = new Float32Array(n);
    this.mag = new Float32Array(n);
    this.nms = new Float32Array(n);
    this.edges = new Uint8Array(n);
    this.edgeList = new Int32Array(n);
    this.rowPrefix = new Float32Array((width + 1) * height);
    this.hough = new HoughTransform(width, height);
    this.supportBins = Math.ceil((2 * Math.hypot(width, height)) / SUPPORT_BIN) + 2;
    this.supportPrefix = new Int32Array(MAX_LINES * (this.supportBins + 1));
    this.binFlags = new Uint8Array(this.supportBins);
  }
}

/**
 * Detects the most likely document quadrilateral in a gray image.
 *
 * Pipeline (at ≤ `maxSide` px): Gaussian blur → Sobel → Canny (non-maximum suppression,
 * hysteresis with thresholds derived from the median gradient magnitude) → orientation-
 * constrained Hough transform → robust re-fit of each line → enumeration of all quads formed by
 * two pairs of roughly opposite lines (the frame borders are added as virtual lines) → scoring
 * → robust sub-pixel re-fit of the best candidates' sides → final re-scoring → a candidate
 * that encloses the best one (the page around a block of its content) wins if it scores almost
 * as well and the margin between them is at least as bright as the block.
 *
 * The score of a quad is the fraction of each side's visible length that is supported by edge
 * pixels whose gradient is perpendicular to the side, blended with the weakest side, weighted
 * by a mild preference for quads above 10 % of the frame, and lowered if the inside is darker
 * than the band around it (paper is almost always brighter than what it lies on; screens,
 * windows and dark furniture fronts are not). The contrast polarity may change along a side
 * (glare on a glossy table), so it is not required to be consistent.
 *
 * Working with lines rather than closed contours makes the detector tolerant to broken edges
 * (shadows, low contrast, occlusions), rounded or folded corners, and documents with corners
 * slightly outside the frame (corners are line intersections and may lie outside the image).
 * A document that continues beyond the frame is closed by the frame border, which earns only a
 * small fixed support; at most two adjacent frame sides are allowed so that the frame itself is
 * never reported. Sides that are mostly outside the frame are rejected.
 *
 * @param gray - Gray image (any size; downscaled internally to `maxSide`).
 * @param context - Reusable buffers; pass the same context every frame to avoid allocations.
 * @returns The best quad (TL, TR, BR, BL) in `gray`'s pixel coordinates, or `null` if no
 *   candidate scores at least `minScore`.
 */
export function detectDocument(
  gray: GrayImage,
  options: DetectOptions = {},
  context: DetectorContext = new DetectorContext(),
): Detection | null {
  const maxSide = options.maxSide ?? 384;
  const minArea = options.minArea ?? 0.04;
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const ctx = context;

  const target = fitLongSide(gray.width, gray.height, maxSide);
  let work = gray;
  if (target.scale !== 1) {
    const reusable = ctx.small?.width === target.width && ctx.small.height === target.height;
    work = downscaleGray(gray, maxSide, reusable ? (ctx.small ?? undefined) : undefined);
    ctx.small = work;
  }
  const { width, height } = work;
  if (width < 16 || height < 16) return null;
  ctx.ensure(width, height);

  convolveSeparable(work.data, width, height, ctx.kernel, ctx.blurTmp, ctx.blurred);
  sobel(ctx.blurred, width, height, ctx.gx, ctx.gy, ctx.mag);
  nonMaximumSuppression(ctx.gx, ctx.gy, ctx.mag, width, height, ctx.nms);
  const median = medianPositive(ctx.mag, ctx.histogram);
  const high = Math.max(MIN_HIGH_THRESHOLD, HIGH_PER_MEDIAN * median);
  const edgeCount = hysteresis(ctx.nms, width, height, high / 2, high, ctx.edges, ctx.edgeList);

  const minVotes = Math.max(12, 0.08 * Math.min(width, height));
  const lineCount = (ctx.hough as HoughTransform).detect(
    ctx.edgeList,
    edgeCount,
    ctx.gx,
    ctx.gy,
    minVotes,
    MAX_LINES,
    MAX_LINES_PER_SECTOR,
    ctx.lines,
  );
  if (lineCount < 2) return null;
  for (let i = 0; i < lineCount; i++) {
    ctx.lines[i] = refitLine(ctx, ctx.lines[i], edgeCount);
    buildSupportProfile(ctx, i, edgeCount);
  }
  ctx.lines[lineCount] = { nx: 1, ny: 0, rho: 0 };
  ctx.lines[lineCount + 1] = { nx: 0, ny: 1, rho: 0 };
  ctx.lines[lineCount + 2] = { nx: 1, ny: 0, rho: width };
  ctx.lines[lineCount + 3] = { nx: 0, ny: 1, rho: height };

  const shortlist = enumerateQuads(ctx, lineCount, minArea * width * height);
  buildRowPrefix(ctx, work.data);
  const refined: ScoredQuad[] = [];
  for (const candidate of shortlist) {
    const quad = refineQuad(ctx, candidate.quad, edgeCount);
    const score = scoreQuadBySampling(ctx, quad) * brightnessFactor(ctx, quad);
    if (score <= 0) continue;
    if (refined.some((r) => maxCornerDistance(r.quad, quad) < 2)) continue;
    refined.push({ quad, score });
  }
  refined.sort((a, b) => b.score - a.score);
  preferEnclosing(ctx, refined);
  const best = refined[0];
  if (!best || best.score < minScore) return null;

  // Width and height are rounded independently when downscaling, so map each axis separately.
  const backX = gray.width / width;
  const backY = gray.height / height;
  const toInput = (q: ScoredQuad): ScoredQuad => ({
    quad: orderQuad(scaleQuad(q.quad, backX, backY)),
    score: q.score,
  });
  const detection: Detection = toInput(best);
  if (options.candidates) detection.candidates = refined.map(toInput);
  return detection;
}

/** Re-fits a Hough line to nearby edge pixels with matching orientation (Tukey IRLS). */
function refitLine(ctx: DetectorContext, line: Line, edgeCount: number): Line {
  const { edgeList, gx, gy, mag, width, fitter } = ctx;
  const cosTol = Math.cos((10 * Math.PI) / 180);
  fitter.reset();
  for (let e = 0; e < edgeCount; e++) {
    const p = edgeList[e];
    const x = (p % width) + 0.5;
    const y = ((p / width) | 0) + 0.5;
    const d = line.nx * x + line.ny * y - line.rho;
    if (d < -2.5 || d > 2.5) continue;
    if (Math.abs(gx[p] * line.nx + gy[p] * line.ny) < cosTol * mag[p]) continue;
    fitter.add(x, y, mag[p]);
  }
  if (fitter.count < 8) return line;
  return fitter.fit(line, 2.5) ?? line;
}

/**
 * Builds prefix sums of edge support along line `index`, binned by the line
 * parameter `t` (position along the line), so that the support of any segment can be read in
 * O(1) while enumerating quads. Also records the parameter range visible inside the frame.
 */
function buildSupportProfile(ctx: DetectorContext, index: number, edgeCount: number): void {
  const { edgeList, gx, gy, mag, width, height, supportBins, binFlags } = ctx;
  const line = ctx.lines[index];
  const dx = -line.ny;
  const dy = line.nx;
  const offset = Math.hypot(width, height);
  binFlags.fill(0);
  for (let e = 0; e < edgeCount; e++) {
    const p = edgeList[e];
    const x = (p % width) + 0.5;
    const y = ((p / width) | 0) + 0.5;
    const d = line.nx * x + line.ny * y - line.rho;
    if (d < -SUPPORT_BAND || d > SUPPORT_BAND) continue;
    if (Math.abs(gx[p] * line.nx + gy[p] * line.ny) < COS_SUPPORT * mag[p]) continue;
    binFlags[((x * dx + y * dy + offset) / SUPPORT_BIN) | 0] = 1;
  }
  const base = index * (supportBins + 1);
  const prefix = ctx.supportPrefix;
  prefix[base] = 0;
  for (let b = 0; b < supportBins; b++) prefix[base + b + 1] = prefix[base + b] + binFlags[b];
  const range = visibleRange(line, width, height);
  ctx.visible[index * 2] = range[0];
  ctx.visible[index * 2 + 1] = range[1];
}

/**
 * Parameter range `[t0, t1]` (along direction `(-ny, nx)`) of the part of `line` inside the
 * region where gradients exist (1.5 px inset from the frame), via Liang–Barsky clipping.
 * Returns an empty range (`t0 > t1`) if the line misses the frame.
 */
function visibleRange(line: Line, width: number, height: number): [number, number] {
  const dx = -line.ny;
  const dy = line.nx;
  const ox = line.nx * line.rho; // point on the line with t = 0
  const oy = line.ny * line.rho;
  let t0 = Number.NEGATIVE_INFINITY;
  let t1 = Number.POSITIVE_INFINITY;
  const clip = (o: number, d: number, lo: number, hi: number) => {
    if (Math.abs(d) < 1e-9) {
      if (o < lo || o > hi) t0 = Number.POSITIVE_INFINITY;
      return;
    }
    let a = (lo - o) / d;
    let b = (hi - o) / d;
    if (a > b) [a, b] = [b, a];
    if (a > t0) t0 = a;
    if (b < t1) t1 = b;
  };
  clip(ox, dx, 1.5, width - 1.5);
  clip(oy, dy, 1.5, height - 1.5);
  return [t0, t1];
}

/** Evidence for one side of a candidate quad. */
interface SideEvidence {
  /** Full side length (px). */
  length: number;
  /** Length of the part inside the frame (px). */
  visible: number;
  /** True if the side lies on the frame border. */
  frame: boolean;
  /** Fraction of the visible part supported by edge pixels along the side. */
  support: number;
}

const sideScratch: SideEvidence[] = Array.from({ length: 4 }, () => ({
  length: 0,
  visible: 0,
  frame: false,
  support: 0,
}));

/**
 * Combines side evidence into a confidence in [0, 1]: the length-weighted supported fraction of
 * the visible real sides blended with the weakest real side (so that one strong side cannot
 * carry a quad), times a penalty per frame side and a mild preference for larger quads. Quads with a side mostly outside the frame, more than two frame sides or two opposite
 * frame sides score 0.
 */
function combineEvidence(sides: readonly SideEvidence[], areaRatio: number): number {
  let frameSides = 0;
  for (let i = 0; i < 4; i++) {
    const s = sides[i];
    if (s.frame) {
      frameSides++;
      if (sides[(i + 2) % 4].frame) return 0;
    } else if (s.length <= 0 || s.visible < MIN_VISIBLE * s.length) {
      return 0;
    }
  }
  if (frameSides > 2) return 0;
  let visible = 0;
  let supported = 0;
  let weakest = 1;
  for (const s of sides) {
    if (s.frame) continue;
    visible += s.visible;
    supported += s.support * s.visible;
    if (s.support < weakest) weakest = s.support;
  }
  const support = 0.6 * (supported / visible) + 0.4 * weakest;
  return support * FRAME_PENALTY ** frameSides * Math.min(1, areaRatio / FULL_SIZE_AREA) ** 0.25;
}

/**
 * Enumerates quads formed by two pairs of roughly opposite lines (Hough lines and frame lines)
 * and returns the best few by the O(1) profile-based score.
 */
function enumerateQuads(ctx: DetectorContext, lineCount: number, minArea: number): ScoredQuad[] {
  const { lines, width, height } = ctx;
  const n = lineCount + FRAME_LINES;
  // Pairwise intersections of sufficiently non-parallel lines.
  const corners: (Point | null)[] = new Array(n * n).fill(null);
  const opposite: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = lines[i];
      const b = lines[j];
      const sin = Math.abs(a.nx * b.ny - a.ny * b.nx);
      if (sin >= SIN_MIN_ADJACENT) {
        const p = intersectLines(a, b);
        corners[i * n + j] = p;
        corners[j * n + i] = p;
      }
      // Frame lines come last (i < j), so `i < lineCount` excludes pairs of two frame lines,
      // which are never opposite sides of a document.
      if (i < lineCount && Math.abs(a.nx * b.nx + a.ny * b.ny) >= COS_MAX_OPPOSITE) {
        opposite.push([i, j]);
      }
    }
  }
  const marginX = 0.15 * width;
  const marginY = 0.15 * height;
  const inBounds = (p: Point | null): p is Point =>
    p !== null &&
    p.x > -marginX &&
    p.x < width + marginX &&
    p.y > -marginY &&
    p.y < height + marginY;

  const best: ScoredQuad[] = [];
  const frameArea = width * height;
  for (let a = 0; a < opposite.length; a++) {
    const [i1, i2] = opposite[a];
    for (let b = a + 1; b < opposite.length; b++) {
      const [j1, j2] = opposite[b];
      if (j1 === i1 || j1 === i2 || j2 === i1 || j2 === i2) continue;
      const c0 = corners[i1 * n + j1];
      const c1 = corners[i1 * n + j2];
      const c2 = corners[i2 * n + j2];
      const c3 = corners[i2 * n + j1];
      if (!inBounds(c0) || !inBounds(c1) || !inBounds(c2) || !inBounds(c3)) continue;
      const quad: Quad = [c0, c1, c2, c3];
      if (!isConvexQuad(quad) || !hasPlausibleCorners(quad)) continue;
      const area = quadArea(quad);
      if (area < minArea) continue;
      profileEvidence(ctx, i1, lineCount, c0, c1, sideScratch[0]);
      profileEvidence(ctx, j2, lineCount, c1, c2, sideScratch[1]);
      profileEvidence(ctx, i2, lineCount, c2, c3, sideScratch[2]);
      profileEvidence(ctx, j1, lineCount, c3, c0, sideScratch[3]);
      const score = combineEvidence(sideScratch, area / frameArea);
      if (score <= 0) continue;
      insertTop(best, { quad, score }, REFINE_CANDIDATES);
    }
  }
  return best;
}

/** Inserts `item` into the descending list `list`, keeping at most `capacity` entries. */
function insertTop(list: ScoredQuad[], item: ScoredQuad, capacity: number): void {
  if (list.length === capacity && item.score <= list[capacity - 1].score) return;
  let i = list.length < capacity ? list.length : capacity - 1;
  list[i] = item;
  while (i > 0 && list[i - 1].score < item.score) {
    list[i] = list[i - 1];
    list[--i] = item;
  }
}

/** Rejects quads with corner angles outside 35°–145° (implausible even under perspective). */
function hasPlausibleCorners(quad: Quad): boolean {
  for (let i = 0; i < 4; i++) {
    const p = quad[i];
    const prev = quad[(i + 3) % 4];
    const next = quad[(i + 1) % 4];
    const ux = prev.x - p.x;
    const uy = prev.y - p.y;
    const vx = next.x - p.x;
    const vy = next.y - p.y;
    const cos = (ux * vx + uy * vy) / Math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy));
    if (!(Math.abs(cos) <= COS_MIN_CORNER)) return false;
  }
  return true;
}

/** Side evidence from the precomputed support profile of line `index` between `a` and `b`. */
function profileEvidence(
  ctx: DetectorContext,
  index: number,
  lineCount: number,
  a: Point,
  b: Point,
  out: SideEvidence,
): void {
  out.length = Math.hypot(b.x - a.x, b.y - a.y);
  out.support = 0;
  out.frame = index >= lineCount;
  if (out.frame) {
    out.visible = out.length;
    return;
  }
  const line = ctx.lines[index];
  const dx = -line.ny;
  const dy = line.nx;
  const ta = a.x * dx + a.y * dy;
  const tb = b.x * dx + b.y * dy;
  const lo = Math.max(Math.min(ta, tb), ctx.visible[index * 2]);
  const hi = Math.min(Math.max(ta, tb), ctx.visible[index * 2 + 1]);
  out.visible = Math.max(0, hi - lo);
  if (out.visible <= 0) return;
  const offset = Math.hypot(ctx.width, ctx.height);
  const b0 = ((lo + offset) / SUPPORT_BIN) | 0;
  const b1 = ((hi + offset) / SUPPORT_BIN) | 0;
  const base = index * (ctx.supportBins + 1);
  out.support = (ctx.supportPrefix[base + b1 + 1] - ctx.supportPrefix[base + b0]) / (b1 - b0 + 1);
}

/** True if segment `a → b` lies on one of the frame borders. */
function isFrameSide(a: Point, b: Point, width: number, height: number): boolean {
  const on = (u: number, v: number, at: number) => Math.abs(u - at) < 0.5 && Math.abs(v - at) < 0.5;
  return on(a.x, b.x, 0) || on(a.x, b.x, width) || on(a.y, b.y, 0) || on(a.y, b.y, height);
}

/**
 * Re-fits every non-frame side of a candidate to the edge pixels near it (two passes with a
 * shrinking band, sub-pixel edge positions, Tukey IRLS) and intersects consecutive sides for the
 * corners. Sides without enough evidence keep their current line. Returns the input if
 * refinement would produce an invalid quad.
 */
function refineQuad(ctx: DetectorContext, quad: Quad, edgeCount: number): Quad {
  let current = quad;
  for (const band of [3, 1.5]) {
    const lines: Line[] = [];
    for (let s = 0; s < 4; s++) {
      const a = current[s];
      const b = current[(s + 1) % 4];
      lines.push(
        isFrameSide(a, b, ctx.width, ctx.height)
          ? lineThrough(a, b)
          : refineSide(ctx, a, b, band, edgeCount),
      );
    }
    const next: Point[] = [];
    for (let s = 0; s < 4; s++) {
      const p = intersectLines(lines[(s + 3) % 4], lines[s]);
      if (!p) return current;
      next.push(p);
    }
    const candidate: Quad = [next[0], next[1], next[2], next[3]];
    if (!isConvexQuad(candidate) || !hasPlausibleCorners(candidate)) return current;
    current = candidate;
  }
  return current;
}

/** Robust sub-pixel line fit to the edge pixels supporting side `a → b`. */
function refineSide(
  ctx: DetectorContext,
  a: Point,
  b: Point,
  band: number,
  edgeCount: number,
): Line {
  const { edgeList, gx, gy, mag, width, fitter } = ctx;
  const line = lineThrough(a, b);
  const dx = -line.ny;
  const dy = line.nx;
  const ta = a.x * dx + a.y * dy;
  const tb = b.x * dx + b.y * dy;
  // Ignore the 5% next to each corner where the adjacent side's edges blur in.
  const margin = 0.05 * Math.abs(tb - ta);
  const lo = Math.min(ta, tb) + margin;
  const hi = Math.max(ta, tb) - margin;
  const cosTol = Math.cos((20 * Math.PI) / 180);
  fitter.reset();
  for (let e = 0; e < edgeCount; e++) {
    const p = edgeList[e];
    let x = (p % width) + 0.5;
    let y = ((p / width) | 0) + 0.5;
    const d = line.nx * x + line.ny * y - line.rho;
    if (d < -band || d > band) continue;
    const t = x * dx + y * dy;
    if (t < lo || t > hi) continue;
    const m = mag[p];
    if (Math.abs(gx[p] * line.nx + gy[p] * line.ny) < cosTol * m) continue;
    // Sub-pixel edge position: vertex of the parabola through the magnitudes along the
    // (quantized) gradient direction.
    const o = gradientNeighborOffset(gx[p], gy[p], width);
    const before = mag[p - o];
    const after = mag[p + o];
    const curvature = before - 2 * m + after;
    if (curvature < 0) {
      const shift = Math.max(-0.5, Math.min(0.5, (0.5 * (before - after)) / curvature));
      x += shift * (o === width ? 0 : o === width - 1 ? -1 : 1);
      y += shift * (o === 1 ? 0 : 1);
    }
    fitter.add(x, y, m);
  }
  if (fitter.count < 8 || fitter.count < 0.15 * (hi - lo)) return line;
  return fitter.fit(line, band) ?? line;
}

/**
 * Scores an arbitrary quad by sampling each visible side every pixel and looking for an edge
 * pixel with matching orientation within ±1 px across the side.
 */
function scoreQuadBySampling(ctx: DetectorContext, quad: Quad): number {
  const { edges, gx, gy, mag, width, height } = ctx;
  for (let s = 0; s < 4; s++) {
    const a = quad[s];
    const b = quad[(s + 1) % 4];
    const out = sideScratch[s];
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    out.length = length;
    out.frame = isFrameSide(a, b, width, height);
    out.visible = length;
    out.support = 0;
    if (out.frame) continue;
    const line = lineThrough(a, b);
    const steps = Math.max(1, Math.round(length));
    let visible = 0;
    let supported = 0;
    for (let k = 0; k <= steps; k++) {
      const u = k / steps;
      const x = a.x + (b.x - a.x) * u;
      const y = a.y + (b.y - a.y) * u;
      if (x < 1.5 || y < 1.5 || x > width - 1.5 || y > height - 1.5) continue;
      visible++;
      for (let off = -1; off <= 1; off++) {
        const px = Math.floor(x + off * line.nx);
        const py = Math.floor(y + off * line.ny);
        if (px < 0 || py < 0 || px >= width || py >= height) continue;
        const p = py * width + px;
        if (edges[p] && Math.abs(gx[p] * line.nx + gy[p] * line.ny) >= COS_SUPPORT * mag[p]) {
          supported++;
          break;
        }
      }
    }
    out.visible = (visible / (steps + 1)) * length;
    out.support = visible > 0 ? supported / visible : 0;
  }
  return combineEvidence(sideScratch, quadArea(quad) / (width * height));
}

/** Row-wise prefix sums of `data` into `ctx.rowPrefix`. */
function buildRowPrefix(ctx: DetectorContext, data: Uint8Array | Uint8ClampedArray): void {
  const { width, height, rowPrefix } = ctx;
  for (let y = 0; y < height; y++) {
    let sum = 0;
    const row = y * (width + 1);
    const src = y * width;
    rowPrefix[row] = 0;
    for (let x = 0; x < width; x++) {
      sum += data[src + x];
      rowPrefix[row + x + 1] = sum;
    }
  }
}

/** Sum and pixel count of the gray values inside a convex quad (pixel centers, clipped to the frame). */
function quadSum(ctx: DetectorContext, quad: Quad, out: { sum: number; count: number }): void {
  const { width, height, rowPrefix } = ctx;
  let top = Number.POSITIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  for (const p of quad) {
    top = Math.min(top, p.y);
    bottom = Math.max(bottom, p.y);
  }
  out.sum = 0;
  out.count = 0;
  const y0 = Math.max(0, Math.ceil(top - 0.5));
  const y1 = Math.min(height - 1, Math.floor(bottom - 0.5));
  for (let y = y0; y <= y1; y++) {
    const cy = y + 0.5;
    let left = Number.POSITIVE_INFINITY;
    let right = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < 4; i++) {
      const a = quad[i];
      const b = quad[(i + 1) % 4];
      if (a.y <= cy === b.y <= cy) continue;
      const x = a.x + ((cy - a.y) / (b.y - a.y)) * (b.x - a.x);
      if (x < left) left = x;
      if (x > right) right = x;
    }
    const x0 = Math.max(0, Math.ceil(left - 0.5));
    const x1 = Math.min(width - 1, Math.floor(right - 0.5));
    if (x1 < x0) continue;
    const row = y * (width + 1);
    out.sum += rowPrefix[row + x1 + 1] - rowPrefix[row + x0];
    out.count += x1 - x0 + 1;
  }
}

const sumScratch = { sum: 0, count: 0 };

/**
 * Paper-brightness factor in (0, 1]: 1 if the inside of the quad (shrunk to {@link INNER_SCALE}
 * about its centroid) is at least as bright on average as the band around it (up to
 * {@link OUTER_SCALE}), else (inside / band)^{@link BRIGHTNESS_EXPONENT}. Paper is almost always
 * brighter than what it lies on, while screens, windows, doors and dark furniture fronts are not.
 */
function scaleAboutCentroid(quad: Quad, f: number): Quad {
  const cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4;
  const cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4;
  return quad.map((p) => ({ x: cx + (p.x - cx) * f, y: cy + (p.y - cy) * f })) as unknown as Quad;
}

function brightnessFactor(ctx: DetectorContext, quad: Quad): number {
  const scaled = (f: number) => scaleAboutCentroid(quad, f);
  quadSum(ctx, scaled(INNER_SCALE), sumScratch);
  if (!sumScratch.count) return 1;
  const inner = sumScratch.sum / sumScratch.count;
  quadSum(ctx, quad, sumScratch);
  const quadTotal = sumScratch.sum;
  const quadCount = sumScratch.count;
  quadSum(ctx, scaled(OUTER_SCALE), sumScratch);
  const bandCount = sumScratch.count - quadCount;
  if (bandCount <= 0) return 1;
  const band = (sumScratch.sum - quadTotal) / bandCount;
  return inner >= band ? 1 : ((inner + 1) / (band + 1)) ** BRIGHTNESS_EXPONENT;
}

/** True if every corner of `inner` lies inside (or within 1 px of) the convex quad `outer`. */
function encloses(outer: Quad, inner: Quad): boolean {
  for (const p of inner) {
    for (let i = 0; i < 4; i++) {
      const a = outer[i];
      const b = outer[(i + 1) % 4];
      const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
      if (cross < -Math.hypot(b.x - a.x, b.y - a.y)) return false;
    }
  }
  return true;
}

/**
 * Moves a candidate that encloses the best one to the front if it scores at least
 * {@link ENCLOSING_SCORE} of the best and the region between them is at least as bright as the
 * best quad's inside: then the best quad is a block of content (text, a picture) on a page and
 * the enclosing quad is the page. A table top around a page fails the brightness test.
 */
function preferEnclosing(ctx: DetectorContext, ranked: ScoredQuad[]): void {
  const best = ranked[0];
  if (!best) return;
  quadSum(ctx, scaleAboutCentroid(best.quad, INNER_SCALE), sumScratch);
  if (!sumScratch.count) return;
  const inner = sumScratch.sum / sumScratch.count;
  quadSum(ctx, best.quad, sumScratch);
  const bestSum = sumScratch.sum;
  const bestCount = sumScratch.count;
  for (let i = 1; i < ranked.length; i++) {
    const c = ranked[i];
    if (c.score < ENCLOSING_SCORE * best.score) break;
    if (!encloses(c.quad, best.quad)) continue;
    quadSum(ctx, c.quad, sumScratch);
    const ringCount = sumScratch.count - bestCount;
    if (ringCount < 0.1 * bestCount) continue;
    if ((sumScratch.sum - bestSum) / ringCount < ENCLOSING_BRIGHTNESS * inner) continue;
    ranked.splice(i, 1);
    ranked.unshift(c);
    return;
  }
}

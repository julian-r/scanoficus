import type { Point, Quad } from '@scanoficus/core/geometry';
import type { RefinerSettings } from './bundle.ts';
import { areaInteger, resizeRgb, roundHalfEven } from './preprocess.ts';

/**
 * Corner refiner geometry and its CPU implementation, a port of `scanoficus_ml/refiner.py` and
 * the `net:` step of `scanoficus_ml/pipeline.py` (`OwnPipeline`):
 *
 * 1. The frame is area-downscaled (one `cv2.resize(INTER_AREA)`) to the work image with the
 *    long side `workSide` (1600) if it is larger; the quad is scaled by `workWidth / width`.
 * 2. The work image gets an area pyramid (2× box halvings while the short side is ≥ 64).
 * 3. Per pass and corner, a 64×64 crop in the canonical corner frame (centred on the corner,
 *    rotated so the bisector of its interior angle points to the bottom right, side
 *    `scale × √area`) is sampled bilinearly from the pyramid level whose pixel is at most 1.5
 *    crop pixels, with clamp-to-edge: exactly OpenCV 5's `warpAffine(INTER_LINEAR,
 *    BORDER_REPLICATE)` on the CPU (float32 coordinates and weights with fused multiply-adds,
 *    round half to even).
 * 4. The network's offset (half-crop units, crop frame) is rotated back and added.
 *
 * Quads are in the core convention (pixel centres at +0.5), the same numbers as the Python
 * `corners × (width, height)`; like refiner.py, the crop matrix treats them as OpenCV pixel
 * indices plus the half-pixel term of `Pyramid.crop`.
 *
 * @module
 */

/** A corner's crop frame: image vector = R · crop vector (crop vectors in units of side / crop). */
export interface CornerFrame {
  centre: Point;
  /** Rotation, row-major `[r00, r01, r10, r11]`. */
  r: [number, number, number, number];
  /** Crop side in image (work-image) pixels. */
  side: number;
}

/** Offsets of one refiner pass: `[dx, dy]` per corner, half-crop units in the crop frame. */
export interface RefinerPass {
  scale: number;
  offsets: [number, number][];
  /** Corner-confidence logits per corner, if the refiner has that output (round 3), else null. */
  cornerLogits: number[] | null;
}

/** Result of the corner refinement. */
export interface RefinerResult {
  /** Refined quad in frame pixels, same corner order as the input quad. */
  quad: Quad;
  passes: RefinerPass[];
}

const f32 = Math.fround;
/** float32 fused multiply-add `a · b + c` (exact product in float64, then one rounding). */
const fma32 = (a: number, b: number, c: number) => f32(a * b + c);

/** `cv2.contourArea` of the quad's corners cast to float32 (summed in float64). */
function contourArea(q: readonly Point[]): number {
  let a = 0;
  let prev = q[3];
  for (const p of q) {
    a += f32(prev.x) * f32(p.y) - f32(prev.y) * f32(p.x);
    prev = p;
  }
  return Math.abs(a * 0.5);
}

/** `refiner.corner_frame`: the crop frame of corner `i` at crop scale `scale`. */
export function cornerFrame(q: readonly Point[], i: number, scale: number): CornerFrame {
  const c = q[i];
  const unit = (p: Point) => {
    const x = p.x - c.x;
    const y = p.y - c.y;
    const n = Math.sqrt(x * x + y * y) + 1e-9;
    return { x: x / n, y: y / n };
  };
  const u = unit(q[(i + 3) % 4]);
  const v = unit(q[(i + 1) % 4]);
  let b = { x: u.x + v.x, y: u.y + v.y };
  if (Math.sqrt(b.x * b.x + b.y * b.y) < 1e-6) b = { x: -u.y, y: u.x };
  const a = Math.atan2(b.y, b.x) - Math.PI / 4;
  const side = scale * Math.sqrt(Math.max(contourArea(q), 1));
  return {
    centre: { x: c.x, y: c.y },
    r: [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)],
    side,
  };
}

/**
 * `Pyramid.crop`'s level choice and inverse affine map: crop pixel `(x, y)` (integer indices)
 * samples level `level` at `(m0·x + m1·y + m2, m3·x + m4·y + m5)` (pixel indices). The six
 * values are rounded to float32, as `cv::warpAffine` does.
 */
export function cropPlan(
  frame: CornerFrame,
  levels: number,
  crop: number,
): { level: number; m: Float32Array } {
  const step = frame.side / crop;
  let level = 0;
  while (level + 1 < levels && step / 2 ** (level + 1) >= 0.75) level++;
  const f = 2 ** level;
  const [r00, r01, r10, r11] = frame.r;
  const a00 = (r00 * step) / f;
  const a01 = (r01 * step) / f;
  const a10 = (r10 * step) / f;
  const a11 = (r11 * step) / f;
  const h = crop / 2 - 0.5;
  const o0 = frame.centre.x / f - (a00 * h + a01 * h) + (0.5 / f - 0.5);
  const o1 = frame.centre.y / f - (a10 * h + a11 * h) + (0.5 / f - 0.5);
  return { level, m: Float32Array.of(a00, a01, o0, a10, a11, o1) };
}

/** `refiner.apply_offset`: the corner moved by a network offset. */
export function applyOffset(frame: CornerFrame, offset: readonly [number, number]): Point {
  const t0 = offset[0] * (frame.side / 2);
  const t1 = offset[1] * (frame.side / 2);
  const [r00, r01, r10, r11] = frame.r;
  return { x: frame.centre.x + (r00 * t0 + r01 * t1), y: frame.centre.y + (r10 * t0 + r11 * t1) };
}

/** Size of the work image: `round(side × s)` with `s = min(1, workSide / long side)`. */
export function workSize(
  width: number,
  height: number,
  workSide: number,
): { width: number; height: number } {
  const s = Math.min(1, workSide / Math.max(width, height));
  if (s >= 1) return { width, height };
  return { width: roundHalfEven(width * s), height: roundHalfEven(height * s) };
}

/** Sizes of the pyramid levels of a `width × height` work image (`refiner.Pyramid`). */
export function pyramidSizes(width: number, height: number): [number, number][] {
  const sizes: [number, number][] = [[width, height]];
  for (let [w, h] = sizes[0]; Math.min(w, h) >= 64; ) {
    w >>= 1;
    h >>= 1;
    sizes.push([w, h]);
  }
  return sizes;
}

/** An interleaved RGB image. */
export interface RgbImage {
  width: number;
  height: number;
  data: Uint8Array;
}

/** The work image and its pyramid, from interleaved RGB frame bytes. */
export function buildPyramid(frame: RgbImage, workSide: number): RgbImage[] {
  const size = workSize(frame.width, frame.height, workSide);
  const work =
    size.width === frame.width && size.height === frame.height
      ? frame
      : {
          ...size,
          data: resizeRgb(frame.data, frame.width, frame.height, size.width, size.height),
        };
  const levels: RgbImage[] = [work];
  for (const [w, h] of pyramidSizes(work.width, work.height).slice(1)) {
    const p = levels[levels.length - 1];
    levels.push({ width: w, height: h, data: areaInteger(p.data, p.width, p.height, 2, 2) });
  }
  return levels;
}

/**
 * `cv2.warpAffine(img, M, (crop, crop), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE)` as
 * implemented in OpenCV 5 (float32 coordinates and bilinear weights with fused multiply-adds,
 * round half to even), into `out` (interleaved RGB, `crop × crop × 3`).
 */
export function warpCrop(img: RgbImage, m: Float32Array, crop: number, out: Uint8Array): void {
  const { width, height, data } = img;
  const px = (x: number, y: number, c: number) =>
    data[
      (Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))) * 3 + c
    ];
  for (let y = 0; y < crop; y++) {
    for (let x = 0; x < crop; x++) {
      const sx = fma32(m[0], x, fma32(m[1], y, m[2]));
      const sy = fma32(m[3], x, fma32(m[4], y, m[5]));
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const ax = f32(sx - x0);
      const ay = f32(sy - y0);
      for (let c = 0; c < 3; c++) {
        const p00 = px(x0, y0, c);
        const p01 = px(x0 + 1, y0, c);
        const p10 = px(x0, y0 + 1, c);
        const p11 = px(x0 + 1, y0 + 1, c);
        const v0 = fma32(ax, p01 - p00, p00);
        const v1 = fma32(ax, p11 - p10, p10);
        const v = fma32(ay, f32(v1 - v0), v0);
        out[(y * crop + x) * 3 + c] = Math.min(255, Math.max(0, roundHalfEven(v)));
      }
    }
  }
}

/**
 * The quad in work-image pixels. `coordinates: 'frame'` (round 2, `pipeline.py`): frame pixels
 * times `s = workWidth / width`. `'normalized'` (round 3, `verify.run_pipeline`): the network's
 * normalized corners times the work-image size.
 */
export function toWork(
  quad: readonly Point[],
  frame: { width: number; height: number },
  work: { width: number; height: number },
  settings: Pick<RefinerSettings, 'coordinates'>,
): Point[] {
  if (settings.coordinates === 'normalized') {
    return quad.map((p) => ({ x: p.x * work.width, y: p.y * work.height }));
  }
  return scalePoints(quad, work.width / frame.width);
}

/** The refined work-image quad in frame pixels (the inverse of {@link toWork}'s mapping). */
export function fromWork(
  q: readonly Point[],
  frame: { width: number; height: number },
  work: { width: number; height: number },
  settings: Pick<RefinerSettings, 'coordinates'>,
): Quad {
  if (settings.coordinates === 'normalized') {
    const sx = frame.width / work.width;
    const sy = frame.height / work.height;
    return q.map((p) => ({ x: p.x * sx, y: p.y * sy })) as unknown as Quad;
  }
  return scalePoints(q, work.width / frame.width, true) as unknown as Quad;
}

/** Offsets (and corner logits, with 3 outputs per crop) of one pass from 4 crops' outputs. */
export function readPass(out: Float32Array, scale: number): RefinerPass {
  const per = out.length / 4;
  const offsets = [0, 1, 2, 3].map((n) => [out[n * per], out[n * per + 1]] as [number, number]);
  const cornerLogits = per > 2 ? [0, 1, 2, 3].map((n) => out[n * per + 2]) : null;
  return { scale, offsets, cornerLogits };
}

/** Quad times `s` (frame → work image) or divided by it (back), like pipeline.py. */
export function scalePoints(q: readonly Point[], s: number, divide = false): Point[] {
  return q.map((p) => (divide ? { x: p.x / s, y: p.y / s } : { x: p.x * s, y: p.y * s }));
}

/**
 * The corner refinement on the CPU. `runRefiner` maps `n` crops (NCHW floats in [0, 1]) to `n`
 * outputs `[dx, dy(, logit)], …` (e.g. a `CpuRunner` of the refiner network). `quad` is in frame
 * pixels, or normalized to the frame with `settings.coordinates: 'normalized'`. Also records the crops
 * (`crops`, one `n × crop × crop × 3` uint8 array per pass) when given an array.
 */
export function refineCornersCpu(
  frame: RgbImage,
  quad: Quad,
  settings: RefinerSettings,
  runRefiner: (input: Float32Array, n: number) => Float32Array,
  crops?: Uint8Array[],
): RefinerResult {
  const levels = buildPyramid(frame, settings.workSide);
  const work = { width: levels[0].width, height: levels[0].height };
  let q = toWork(quad, frame, work, settings);
  const { crop } = settings;
  const plane = crop * crop;
  const passes: RefinerPass[] = [];
  const hwc = new Uint8Array(4 * plane * 3);
  for (const scale of settings.scales) {
    const frames = [0, 1, 2, 3].map((i) => cornerFrame(q, i, scale));
    const input = new Float32Array(4 * 3 * plane);
    frames.forEach((f, n) => {
      const { level, m } = cropPlan(f, levels.length, crop);
      const out = hwc.subarray(n * plane * 3, (n + 1) * plane * 3);
      warpCrop(levels[level], m, crop, out);
      for (let c = 0; c < 3; c++) {
        for (let i = 0; i < plane; i++) input[(n * 3 + c) * plane + i] = out[i * 3 + c] / 255;
      }
    });
    crops?.push(hwc.slice());
    const result = runRefiner(input, 4);
    const pass = readPass(result, scale);
    q = frames.map((f, n) => applyOffset(f, pass.offsets[n]));
    passes.push(pass);
  }
  return { quad: fromWork(q, frame, work, settings), passes };
}

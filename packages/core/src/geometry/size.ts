import { distance, type Quad } from './quad.ts';

/** Options for {@link estimateOutputSize}. */
export interface OutputSizeOptions {
  /**
   * Size of the image the quad lives in. When given, the true aspect ratio of the
   * (rectangular) document is recovered from the perspective distortion, assuming a pinhole
   * camera with square pixels and the principal point at the image center. Without it, the
   * size is the maximum of opposite edge lengths, which over-/under-estimates the aspect ratio
   * under strong perspective.
   */
  imageSize?: { width: number; height: number };
  /** Clamp the longer output side to this many pixels (aspect ratio is preserved). */
  maxLongSide?: number;
  /** Clamp the output area to this many pixels (aspect ratio is preserved). */
  maxPixels?: number;
}

/**
 * Estimates the pixel size of the rectified document for a quad.
 *
 * Base estimate: width = max(top, bottom) edge length, height = max(left, right) edge length.
 * With `imageSize`, the aspect ratio is corrected using the method of Zhang & He, "Whiteboard
 * scanning and image enhancement" (2007): the focal length is estimated from the vanishing
 * geometry of the quad and the rectangle's aspect ratio follows from the back-projected edge
 * vectors. If the quad has (nearly) parallel opposite sides the focal length is unobservable and
 * the affine aspect ratio is used instead. The longer of the base width/height is kept and the
 * other side is derived from the corrected ratio.
 */
export function estimateOutputSize(
  quad: Quad,
  options: OutputSizeOptions = {},
): { width: number; height: number } {
  const [tl, tr, br, bl] = quad;
  let width = Math.max(distance(tl, tr), distance(bl, br));
  let height = Math.max(distance(tl, bl), distance(tr, br));
  if (options.imageSize) {
    const ratio = rectangleAspectRatio(
      quad,
      options.imageSize.width / 2,
      options.imageSize.height / 2,
    );
    if (ratio !== null) {
      if (width >= height) height = width / ratio;
      else width = height * ratio;
    }
  }
  let k = 1;
  if (options.maxLongSide !== undefined)
    k = Math.min(k, options.maxLongSide / Math.max(width, height));
  if (options.maxPixels !== undefined)
    k = Math.min(k, Math.sqrt(options.maxPixels / (width * height)));
  // Floor when scaling down so that rounding never exceeds the caps.
  const round = k < 1 ? Math.floor : Math.round;
  return {
    width: Math.max(1, round(width * k)),
    height: Math.max(1, round(height * k)),
  };
}

/**
 * Width/height ratio of the real-world rectangle imaged as `quad` (Zhang & He 2007), with the
 * principal point at `(u0, v0)`. Returns `null` if the estimate is not physically plausible.
 */
export function rectangleAspectRatio(quad: Quad, u0: number, v0: number): number | null {
  // Paper notation: m1 = (0,0), m2 = (w,0), m3 = (0,h), m4 = (w,h) of the rectangle.
  const [tl, tr, br, bl] = quad;
  const m1: Vec3 = [tl.x, tl.y, 1];
  const m2: Vec3 = [tr.x, tr.y, 1];
  const m3: Vec3 = [bl.x, bl.y, 1];
  const m4: Vec3 = [br.x, br.y, 1];
  const k2 = dot(cross(m1, m4), m3) / dot(cross(m2, m4), m3);
  const k3 = dot(cross(m1, m4), m2) / dot(cross(m3, m4), m2);
  if (!Number.isFinite(k2) || !Number.isFinite(k3)) return null;
  const n2: Vec3 = [k2 * m2[0] - m1[0], k2 * m2[1] - m1[1], k2 * m2[2] - m1[2]];
  const n3: Vec3 = [k3 * m3[0] - m1[0], k3 * m3[1] - m1[1], k3 * m3[2] - m1[2]];

  // Relative tolerance below which the quad is treated as affine (no usable vanishing points).
  const affine = Math.abs(n2[2]) < 1e-3 * Math.abs(k2) || Math.abs(n3[2]) < 1e-3 * Math.abs(k3);
  let ratio2: number;
  if (affine) {
    ratio2 = (n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2);
  } else {
    const f2 =
      -(
        n2[0] * n3[0] -
        (n2[0] * n3[2] + n2[2] * n3[0]) * u0 +
        n2[2] * n3[2] * u0 * u0 +
        (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
      ) /
      (n2[2] * n3[2]);
    if (!(f2 > 0)) {
      ratio2 = (n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2);
    } else {
      // |A^-1 n|^2 with A = [[f, 0, u0], [0, f, v0], [0, 0, 1]].
      const norm = (n: Vec3) =>
        ((n[0] - u0 * n[2]) ** 2 + (n[1] - v0 * n[2]) ** 2) / f2 + n[2] ** 2;
      ratio2 = norm(n2) / norm(n3);
    }
  }
  const ratio = Math.sqrt(ratio2);
  return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
}

type Vec3 = [number, number, number];

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

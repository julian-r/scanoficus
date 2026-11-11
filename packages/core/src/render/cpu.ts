import { applyFilter, type ScanFilter } from '../filters/filters.ts';
import type { Matrix3 } from '../geometry/homography.ts';
import type { Quad } from '../geometry/quad.ts';
import { createRgba, type RgbaImage } from '../image.ts';
import { rectifyCpu } from '../warp/cpu.ts';

/** Clockwise page rotation in degrees. */
export type Rotation = 0 | 90 | 180 | 270;

/** How a page is cut out of its source image and processed. */
export interface PageRenderOptions {
  /** Document corners (TL, TR, BR, BL) in source pixels. */
  quad: Quad;
  filter: ScanFilter;
  rotation: Rotation;
  /** Size of the rectified page before rotation. */
  width: number;
  height: number;
}

/** Normalizes any angle (multiple of 90°) to a {@link Rotation}. */
export function normalizeRotation(degrees: number): Rotation {
  return ((((Math.round(degrees / 90) * 90) % 360) + 360) % 360) as Rotation;
}

/** Rotates an image clockwise by `rotation`. */
export function rotateRgba(image: RgbaImage, rotation: Rotation): RgbaImage {
  if (rotation === 0) return image;
  const { width: w, height: h, data } = image;
  const quarter = rotation !== 180;
  const out = createRgba(quarter ? h : w, quarter ? w : h);
  const m = rotatedToUnrotated(rotation, w, h);
  for (let y = 0; y < out.height; y++) {
    for (let x = 0; x < out.width; x++) {
      // Pixel centers map to pixel centers; floor recovers the source index.
      const sx = Math.floor(m[0] * (x + 0.5) + m[1] * (y + 0.5) + m[2]);
      const sy = Math.floor(m[3] * (x + 0.5) + m[4] * (y + 0.5) + m[5]);
      const s = (sy * w + sx) * 4;
      const o = (y * out.width + x) * 4;
      out.data[o] = data[s];
      out.data[o + 1] = data[s + 1];
      out.data[o + 2] = data[s + 2];
      out.data[o + 3] = data[s + 3];
    }
  }
  return out;
}

/**
 * Affine map from continuous coordinates of the rotated image to those of the unrotated
 * `width × height` image (clockwise rotation).
 */
export function rotatedToUnrotated(rotation: Rotation, width: number, height: number): Matrix3 {
  switch (rotation) {
    case 0:
      return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    case 90:
      return [0, 1, 0, -1, 0, height, 0, 0, 1];
    case 180:
      return [-1, 0, width, 0, -1, height, 0, 0, 1];
    case 270:
      return [0, -1, width, 1, 0, 0, 0, 0, 1];
  }
}

/** CPU reference page pipeline: rectify, filter, rotate. */
export function renderPageCpu(source: RgbaImage, options: PageRenderOptions): RgbaImage {
  const page = rectifyCpu(source, options.quad, options.width, options.height);
  return rotateRgba(applyFilter(page, options.filter), options.rotation);
}

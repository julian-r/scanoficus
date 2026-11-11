import { resampleArea } from './internal/resample.ts';

/**
 * Minimal, DOM-free image containers used throughout `@scanoficus/core`.
 *
 * Coordinate convention (used by every module): pixel `(i, j)` covers the continuous square
 * `[i, i + 1) × [j, j + 1)`, so its center is at `(i + 0.5, j + 0.5)`. Scaling an image by a
 * factor `s` scales continuous coordinates by exactly `s`.
 *
 * @module
 */

/** Single-channel 8-bit image, row-major, `data.length === width * height`. */
export interface GrayImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;
}

/** Single-channel floating point image, row-major, `data.length === width * height`. */
export interface FloatImage {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

/**
 * 8-bit RGBA image with the same memory layout as the DOM `ImageData`
 * (`data.length === width * height * 4`, non-premultiplied). An `ImageData` instance can be
 * passed wherever an `RgbaImage` is expected.
 */
export interface RgbaImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

/** Allocates a zero-filled {@link GrayImage}. */
export function createGray(width: number, height: number): GrayImage {
  return { width, height, data: new Uint8Array(width * height) };
}

/** Allocates a zero-filled (transparent black) {@link RgbaImage}. */
export function createRgba(width: number, height: number): RgbaImage {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) };
}

/**
 * Converts RGBA to luma using the Rec. 601 weights (`0.299 R + 0.587 G + 0.114 B`) in fixed point.
 * Alpha is ignored.
 *
 * @param out - Optional destination of the same size, reused to avoid allocation.
 */
export function rgbaToGray(rgba: RgbaImage, out?: GrayImage): GrayImage {
  const { width, height, data } = rgba;
  const dst = out ?? createGray(width, height);
  assertSameSize(dst, width, height);
  const g = dst.data;
  for (let i = 0, p = 0, n = width * height; i < n; i++, p += 4) {
    // 19595 + 38470 + 7471 = 65536; +32768 rounds to nearest.
    g[i] = (19595 * data[p] + 38470 * data[p + 1] + 7471 * data[p + 2] + 32768) >> 16;
  }
  return dst;
}

/**
 * Returns the size an image must be resized to so that its longer side equals `longSide`,
 * preserving aspect ratio. Images that are already small enough keep their size.
 */
export function fitLongSide(
  width: number,
  height: number,
  longSide: number,
): { width: number; height: number; scale: number } {
  const longest = Math.max(width, height);
  if (longest <= longSide) return { width, height, scale: 1 };
  const scale = longSide / longest;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
}

/**
 * Area-averaging downscale of a gray image so that its longer side is at most `longSide`.
 * Each output pixel is the exact area-weighted mean of the input pixels it covers (fractional
 * coverage included), which avoids the aliasing of nearest/bilinear sampling. If the image is
 * already small enough it is returned as is (not copied).
 *
 * @param out - Optional destination with the target size, reused to avoid allocation.
 */
export function downscaleGray(gray: GrayImage, longSide: number, out?: GrayImage): GrayImage {
  const size = fitLongSide(gray.width, gray.height, longSide);
  if (size.width === gray.width && size.height === gray.height) return gray;
  const dst = out ?? createGray(size.width, size.height);
  assertSameSize(dst, size.width, size.height);
  resampleArea(gray.data, gray.width, gray.height, dst.data, dst.width, dst.height);
  return dst;
}

function assertSameSize(img: { width: number; height: number }, width: number, height: number) {
  if (img.width !== width || img.height !== height) {
    throw new RangeError(
      `Expected a ${width}x${height} output image, got ${img.width}x${img.height}`,
    );
  }
}

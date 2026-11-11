import { type Matrix3, rectToQuadHomography } from '../geometry/homography.ts';
import type { Quad } from '../geometry/quad.ts';
import { createRgba, type RgbaImage } from '../image.ts';

/**
 * Perspective-rectifies the region `quad` of `rgba` into an `outWidth × outHeight` image
 * (CPU reference implementation / fallback for the WebGL2 path).
 *
 * For every output pixel center the source position is obtained from the homography mapping the
 * output rectangle onto the quad (i.e. the inverse warp) and sampled bilinearly. Samples outside
 * the source are clamped to the nearest edge pixel, so corners slightly outside the frame produce
 * smeared borders instead of transparent holes. Bilinear sampling does not low-pass filter, so
 * heavy minification (output much smaller than the quad) aliases; choose the output size with
 * `estimateOutputSize` to stay near 1:1.
 *
 * @param out - Optional destination of size `outWidth × outHeight`, reused to avoid allocation.
 */
export function rectifyCpu(
  rgba: RgbaImage,
  quad: Quad,
  outWidth: number,
  outHeight: number,
  out?: RgbaImage,
): RgbaImage {
  const h = rectToQuadHomography(outWidth, outHeight, quad);
  if (!h) throw new RangeError('Cannot rectify a degenerate quad');
  const dst = out ?? createRgba(outWidth, outHeight);
  if (dst.width !== outWidth || dst.height !== outHeight) {
    throw new RangeError(`Expected a ${outWidth}x${outHeight} output image`);
  }
  warpRows(rgba, h, dst);
  return dst;
}

/** Inverse-warps every output pixel through `h` (output → source) with bilinear sampling. */
function warpRows(src: RgbaImage, h: Matrix3, dst: RgbaImage): void {
  const { width: sw, height: sh, data: s } = src;
  const { width: dw, height: dh, data: d } = dst;
  const maxX = sw - 1;
  const maxY = sh - 1;
  for (let y = 0; y < dh; y++) {
    const oy = y + 0.5;
    // The homogeneous source coordinates are affine in x, so step them incrementally.
    let u = h[0] * 0.5 + h[1] * oy + h[2];
    let v = h[3] * 0.5 + h[4] * oy + h[5];
    let w = h[6] * 0.5 + h[7] * oy + h[8];
    let o = y * dw * 4;
    for (let x = 0; x < dw; x++, o += 4) {
      // Continuous source coordinates → index space (pixel centers at integers).
      let fx = u / w - 0.5;
      let fy = v / w - 0.5;
      u += h[0];
      v += h[3];
      w += h[6];
      fx = Math.max(0, Math.min(fx, maxX));
      fy = Math.max(0, Math.min(fy, maxY));
      const x0 = fx | 0;
      const y0 = fy | 0;
      const x1 = x0 < maxX ? x0 + 1 : x0;
      const y1 = y0 < maxY ? y0 + 1 : y0;
      const ax = fx - x0;
      const ay = fy - y0;
      const p00 = (y0 * sw + x0) * 4;
      const p10 = (y0 * sw + x1) * 4;
      const p01 = (y1 * sw + x0) * 4;
      const p11 = (y1 * sw + x1) * 4;
      for (let c = 0; c < 4; c++) {
        const top = s[p00 + c] + (s[p10 + c] - s[p00 + c]) * ax;
        const bottom = s[p01 + c] + (s[p11 + c] - s[p01 + c]) * ax;
        d[o + c] = top + (bottom - top) * ay;
      }
    }
  }
}

import type { GrayImage } from '../image.ts';

/** Axis-aligned pixel rectangle. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Focus measure: variance of the 4-neighbor Laplacian, over the whole image interior or over
 * `region` (clamped to the image; e.g. the bounding box of the detected document so that the
 * background does not dominate). Higher is sharper. The value depends strongly on content and
 * resolution (100–4000 was measured for the same scene on iPhone), so compare values of the
 * same scene over time, e.g. against the recent maximum, rather than using a fixed threshold.
 */
export function sharpness(gray: GrayImage, region?: Rect): number {
  const { width, height, data } = gray;
  const x0 = Math.max(1, Math.floor(region?.x ?? 0));
  const y0 = Math.max(1, Math.floor(region?.y ?? 0));
  const x1 = Math.min(width - 1, Math.ceil(region ? region.x + region.width : width));
  const y1 = Math.min(height - 1, Math.ceil(region ? region.y + region.height : height));
  if (x1 - x0 < 1 || y1 - y0 < 1) return 0;
  let sum = 0;
  let sumSq = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * width + x;
      const lap = data[i - 1] + data[i + 1] + data[i - width] + data[i + width] - 4 * data[i];
      sum += lap;
      sumSq += lap * lap;
    }
  }
  const n = (x1 - x0) * (y1 - y0);
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

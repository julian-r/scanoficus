import type { GrayImage } from '../image.ts';

/**
 * Focus measure: variance of the 4-neighbor Laplacian over the image interior. Higher is sharper;
 * the absolute value depends on content and resolution, so compare values of the same scene over
 * time (e.g. pick the sharpest of several frames, or require a fraction of the running maximum)
 * rather than using a universal threshold.
 */
export function sharpness(gray: GrayImage): number {
  const { width, height, data } = gray;
  if (width < 3 || height < 3) return 0;
  let sum = 0;
  let sumSq = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = data[i - 1] + data[i + 1] + data[i - width] + data[i + width] - 4 * data[i];
      sum += lap;
      sumSq += lap * lap;
    }
  }
  const n = (width - 2) * (height - 2);
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

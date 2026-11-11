import { convexIoU, distance, type Quad } from '../../src/geometry/index.ts';
import { createRgba, type RgbaImage } from '../../src/image.ts';

/**
 * Mean corner distance between two quads, minimized over the four cyclic relabelings (corner
 * labels are ambiguous for quads rotated by ~45°).
 */
export function meanCornerError(truth: Quad, detected: Quad): number {
  let best = Number.POSITIVE_INFINITY;
  for (let r = 0; r < 4; r++) {
    let sum = 0;
    for (let i = 0; i < 4; i++) sum += distance(truth[i], detected[(i + r) % 4]);
    best = Math.min(best, sum / 4);
  }
  return best;
}

export interface SceneOutcome {
  /** Whether the detector returned a quad. */
  found: boolean;
  iou: number;
  /** Mean corner error as a fraction of the image diagonal. */
  cornerError: number;
}

export function evaluateDetection(
  truth: Quad | null,
  detected: Quad | null,
  width: number,
  height: number,
): SceneOutcome {
  if (!truth || !detected) {
    return { found: detected !== null, iou: 0, cornerError: Number.NaN };
  }
  return {
    found: true,
    iou: convexIoU(truth, detected),
    cornerError: meanCornerError(truth, detected) / Math.hypot(width, height),
  };
}

/** Copy of `image` with quad outlines drawn on it (for debugging failures). */
export function drawQuads(
  image: RgbaImage,
  quads: { quad: Quad; color: [number, number, number] }[],
): RgbaImage {
  const out = createRgba(image.width, image.height);
  out.data.set(image.data);
  for (const { quad, color } of quads) {
    for (let s = 0; s < 4; s++) {
      const a = quad[s];
      const b = quad[(s + 1) % 4];
      const steps = Math.ceil(distance(a, b) * 2);
      for (let k = 0; k <= steps; k++) {
        const x = Math.floor(a.x + ((b.x - a.x) * k) / steps);
        const y = Math.floor(a.y + ((b.y - a.y) * k) / steps);
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const p = (y * image.width + x) * 4;
        out.data[p] = color[0];
        out.data[p + 1] = color[1];
        out.data[p + 2] = color[2];
      }
    }
  }
  return out;
}

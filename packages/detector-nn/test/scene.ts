/**
 * Synthetic test frames, bit-identical to `scene()` in `scripts/make_fixtures.py` (integer
 * arithmetic only), so the fixtures store only hashes and expected outputs, not pixels.
 */

export type Corners = [number, number][];

export interface SceneCase {
  name: string;
  width: number;
  height: number;
  page: Corners | 'noise' | null;
  text: Corners | null;
  seed: number;
}

function hash32(x: number, y: number, c: number, seed: number): number {
  let h =
    (Math.imul(x, 0x9e3779b1) >>> 0) ^
    (Math.imul(y, 0x85ebca77) >>> 0) ^
    (Math.imul(c, 0xc2b2ae3d) >>> 0) ^
    seed;
  h >>>= 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39) >>> 0;
  h ^= h >>> 15;
  return h >>> 0;
}

function inside(px: number, py: number, quad: Corners): boolean {
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % 4];
    if ((2 * bx - 2 * ax) * (py - 2 * ay) - (2 * by - 2 * ay) * (px - 2 * ax) < 0) return false;
  }
  return true;
}

/** Renders a scene as RGBA (`ImageData` layout, alpha 255). */
export function renderScene(scene: SceneCase): {
  width: number;
  height: number;
  data: Uint8ClampedArray;
} {
  const { width, height, page, text, seed } = scene;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      const px = 2 * x + 1;
      const py = 2 * y + 1;
      const onPage = Array.isArray(page) && inside(px, py, page);
      const ink =
        text !== null &&
        inside(px, py, text) &&
        Math.floor(y / 6) % 3 === 0 &&
        Math.floor(x / 23) % 5 !== 4;
      for (let c = 0; c < 3; c++) {
        const h = hash32(x, y, c, seed);
        let v: number;
        if (page === 'noise') v = h & 255;
        else if (ink) v = 60 + (h & 15);
        else if (onPage) v = 225 - 5 * c - (h & 15);
        else v = 50 + 15 * c + Math.floor(((x + 2 * y) * 60) / (width + 2 * height)) + (h & 15);
        data[p + c] = v;
      }
      data[p + 3] = 255;
    }
  }
  return { width, height, data };
}

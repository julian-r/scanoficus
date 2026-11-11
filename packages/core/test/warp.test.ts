import { describe, expect, it } from 'vitest';
import {
  applyHomography,
  invertMatrix3,
  type Quad,
  rectToQuadHomography,
} from '../src/geometry/index.ts';
import { createRgba, type RgbaImage } from '../src/image.ts';
import { planGlRectify, rectifyCpu, rectifyShaderMatrix } from '../src/warp/index.ts';

const quad: Quad = [
  { x: 60, y: 40 },
  { x: 330, y: 70 },
  { x: 300, y: 270 },
  { x: 40, y: 230 },
];
const OUT_W = 200;
const OUT_H = 150;

/**
 * Renders a document-plane pattern under the perspective `quad`: red = u, green = v ramps and a
 * black 10×10 % marker square at u, v ∈ [0.6, 0.7]. Outside the quad the scene is blue.
 */
function renderScene(width: number, height: number): RgbaImage {
  const toDoc = invertMatrix3(rectToQuadHomography(1, 1, quad)!)!;
  const img = createRgba(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const { x: u, y: v } = applyHomography(toDoc, { x: x + 0.5, y: y + 0.5 });
      const o = (y * width + x) * 4;
      const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
      const marker = u >= 0.6 && u < 0.7 && v >= 0.6 && v < 0.7;
      img.data[o] = inside && !marker ? u * 255 : 0;
      img.data[o + 1] = inside && !marker ? v * 255 : 0;
      img.data[o + 2] = inside ? 0 : 255;
      img.data[o + 3] = 255;
    }
  }
  return img;
}

describe('rectifyCpu', () => {
  const scene = renderScene(400, 300);
  const out = rectifyCpu(scene, quad, OUT_W, OUT_H);

  it('maps the document plane onto the output rectangle', () => {
    let maxError = 0;
    for (let y = 2; y < OUT_H - 2; y += 7) {
      for (let x = 2; x < OUT_W - 2; x += 7) {
        const u = (x + 0.5) / OUT_W;
        const v = (y + 0.5) / OUT_H;
        if (u > 0.57 && u < 0.73 && v > 0.57 && v < 0.73) continue; // marker and its border
        const o = (y * OUT_W + x) * 4;
        maxError = Math.max(
          maxError,
          Math.abs(out.data[o] - u * 255),
          Math.abs(out.data[o + 1] - v * 255),
        );
        expect(out.data[o + 2]).toBeLessThan(40); // no background inside the document
      }
    }
    // Ramps are linear in the document plane, so bilinear sampling is accurate to ~1 level.
    expect(maxError).toBeLessThan(3);
  });

  it('places the marker square where expected', () => {
    const at = (u: number, v: number) => {
      const o = (Math.floor(v * OUT_H) * OUT_W + Math.floor(u * OUT_W)) * 4;
      return out.data[o] + out.data[o + 1];
    };
    expect(at(0.65, 0.65)).toBe(0);
    expect(at(0.62, 0.68)).toBe(0);
    expect(at(0.55, 0.65)).toBeGreaterThan(100);
    expect(at(0.75, 0.65)).toBeGreaterThan(100);
    expect(at(0.65, 0.55)).toBeGreaterThan(100);
  });

  it('reuses an output buffer and validates its size', () => {
    const reused = createRgba(OUT_W, OUT_H);
    expect(rectifyCpu(scene, quad, OUT_W, OUT_H, reused)).toBe(reused);
    expect(() => rectifyCpu(scene, quad, 10, 10, reused)).toThrow(RangeError);
  });
});

describe('rectifyShaderMatrix', () => {
  it('maps fragment coordinates to the same source locations as the CPU warp', () => {
    const m = rectifyShaderMatrix(quad, 400, 300, OUT_W, OUT_H);
    const h = rectToQuadHomography(OUT_W, OUT_H, quad)!;
    for (const [x, y] of [
      [0.5, 0.5],
      [199.5, 0.5],
      [120.5, 77.5],
      [0.5, 149.5],
    ]) {
      // gl_FragCoord has its origin at the bottom-left; output row y is fragment row H - y.
      const fx = x;
      const fy = OUT_H - y;
      // Column-major: column k is m[3k..3k+2].
      const px = m[0] * fx + m[3] * fy + m[6];
      const py = m[1] * fx + m[4] * fy + m[7];
      const pz = m[2] * fx + m[5] * fy + m[8];
      const expected = applyHomography(h, { x, y });
      expect((px / pz) * 400).toBeCloseTo(expected.x, 2);
      expect((py / pz) * 300).toBeCloseTo(expected.y, 2);
    }
  });
});

describe('planGlRectify', () => {
  it('keeps sizes within the limits unchanged', () => {
    expect(planGlRectify(4032, 3024, 2480, 3508, 8192)).toEqual({
      sourceScale: 1,
      textureWidth: 4032,
      textureHeight: 3024,
      outWidth: 2480,
      outHeight: 3508,
    });
  });

  it('downscales oversized sources and outputs uniformly', () => {
    const plan = planGlRectify(8064, 6048, 5000, 7000, 4096);
    expect(plan.textureWidth).toBe(4096);
    expect(plan.textureHeight).toBe(3072);
    expect(plan.outHeight).toBe(4096);
    expect(plan.outWidth).toBe(Math.floor(5000 * (4096 / 7000)));
  });

  it('honors a smaller render limit', () => {
    const plan = planGlRectify(1000, 1000, 3000, 2000, 4096, 2048);
    expect([plan.outWidth, plan.outHeight]).toEqual([2048, 1365]);
  });
});

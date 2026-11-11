import { describe, expect, it } from 'vitest';
import { applyFilter, SCAN_FILTERS, type ScanFilter } from '../src/filters/index.ts';
import { applyHomography, type Quad, rectToQuadHomography } from '../src/geometry/index.ts';
import { createRgba, type RgbaImage } from '../src/image.ts';
import {
  computeFilterPlan,
  type FilterPlan,
  normalizeRotation,
  pageShaderMatrices,
  type Rotation,
  renderPageCpu,
  rotateRgba,
} from '../src/render/index.ts';
import { rectifyCpu } from '../src/warp/index.ts';

function numbered(width: number, height: number): RgbaImage {
  const img = createRgba(width, height);
  for (let i = 0; i < width * height; i++) img.data.set([i, 0, 0, 255], i * 4);
  return img;
}

describe('rotateRgba', () => {
  it('rotates clockwise', () => {
    // 3×2 image: row 0 = 0 1 2, row 1 = 3 4 5.
    const red = (img: RgbaImage) =>
      Array.from({ length: img.width * img.height }, (_, i) => img.data[i * 4]);
    expect(red(rotateRgba(numbered(3, 2), 90))).toEqual([3, 0, 4, 1, 5, 2]);
    expect(red(rotateRgba(numbered(3, 2), 180))).toEqual([5, 4, 3, 2, 1, 0]);
    expect(red(rotateRgba(numbered(3, 2), 270))).toEqual([2, 5, 1, 4, 0, 3]);
    let img = numbered(3, 2);
    for (let i = 0; i < 4; i++) img = rotateRgba(img, 90);
    expect(red(img)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('normalizes angles', () => {
    expect([-90, 0, 450, 720, 180].map(normalizeRotation)).toEqual([270, 0, 90, 0, 180]);
  });
});

const quad: Quad = [
  { x: 30, y: 20 },
  { x: 200, y: 35 },
  { x: 190, y: 260 },
  { x: 15, y: 240 },
];

describe('pageShaderMatrices', () => {
  const W = 120;
  const H = 160;
  const apply = (m: Float32Array, x: number, y: number) => {
    const z = m[2] * x + m[5] * y + m[8];
    return { x: (m[0] * x + m[3] * y + m[6]) / z, y: (m[1] * x + m[4] * y + m[7]) / z };
  };
  for (const rotation of [0, 90, 180, 270] as Rotation[]) {
    it(`maps rotated fragments consistently for ${rotation}°`, () => {
      const m = pageShaderMatrices(quad, 400, 300, W, H, rotation);
      const h = rectToQuadHomography(W, H, quad)!;
      const quarter = rotation % 180 !== 0;
      const canvasW = quarter ? H : W;
      const canvasH = quarter ? W : H;
      // Pixel (x, y) of the unrotated page lands at these rotated canvas pixels.
      const rotatedPixel = (x: number, y: number) =>
        rotation === 0
          ? [x, y]
          : rotation === 90
            ? [H - 1 - y, x]
            : rotation === 180
              ? [W - 1 - x, H - 1 - y]
              : [y, W - 1 - x];
      for (const [x, y] of [
        [0, 0],
        [W - 1, 0],
        [37, 101],
        [W - 1, H - 1],
      ]) {
        const [rx, ry] = rotatedPixel(x, y);
        const frag = { x: rx + 0.5, y: canvasH - (ry + 0.5) };
        expect(rx).toBeLessThan(canvasW);
        const src = apply(m.source, frag.x, frag.y);
        const expected = applyHomography(h, { x: x + 0.5, y: y + 0.5 });
        expect(src.x * 400).toBeCloseTo(expected.x, 2);
        expect(src.y * 300).toBeCloseTo(expected.y, 2);
        const page = apply(m.page, frag.x, frag.y);
        expect(page.x).toBeCloseTo((x + 0.5) / W, 5);
        expect(page.y).toBeCloseTo((y + 0.5) / H, 5);
      }
    });
  }
});

/** CPU emulation of the page fragment shader for a rectified page. */
function emulateShader(page: RgbaImage, plan: FilterPlan): RgbaImage {
  const out = createRgba(page.width, page.height);
  const sample = (tex: RgbaImage, u: number, v: number, c: number) => {
    const x = Math.max(0, Math.min(tex.width - 1, u * tex.width - 0.5));
    const y = Math.max(0, Math.min(tex.height - 1, v * tex.height - 0.5));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(tex.width - 1, x0 + 1);
    const y1 = Math.min(tex.height - 1, y0 + 1);
    const at = (xx: number, yy: number) => tex.data[(yy * tex.width + xx) * 4 + c] / 255;
    const top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * (x - x0);
    const bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * (x - x0);
    return top + (bottom - top) * (y - y0);
  };
  const luma = (v: number[]) => 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2];
  for (let y = 0; y < page.height; y++) {
    for (let x = 0; x < page.width; x++) {
      const o = (y * page.width + x) * 4;
      const c = [0, 1, 2].map((k) => page.data[o + k] / 255);
      let v = c;
      if (plan.mode !== 0) {
        const u = (x + 0.5) / page.width;
        const w = (y + 0.5) / page.height;
        const flat = c.map((value, k) =>
          Math.min(1, value / Math.max(sample(plan.background!, u, w, k), 1 / 255)),
        );
        if (plan.mode === 3) {
          const t =
            sample(plan.statistics!, u, w, 0) *
            (1 + plan.k * (sample(plan.statistics!, u, w, 1) * (255 / 128) - 1));
          v = [0, 0, 0].map(() => (luma(flat) < t ? 0 : 1));
        } else {
          v = flat.map((value, k) => {
            const range = Math.max((plan.white[k] - plan.black[k]) / 255, 1 / 255);
            return Math.max(0, Math.min(1, (value - plan.black[k] / 255) / range));
          });
          const l = luma(v);
          v =
            plan.mode === 2
              ? [l, l, l]
              : v.map((value) => Math.max(0, Math.min(1, l + (value - l) * plan.saturation)));
        }
      }
      out.data.set([v[0] * 255, v[1] * 255, v[2] * 255, 255], o);
    }
  }
  return out;
}

/** A photographed page: text bars, illumination gradient and yellow cast, inside a scene. */
function scene(): RgbaImage {
  const toPage = (x: number, y: number) => ({ u: (x - 20) / 360, v: (y - 15) / 480 });
  const img = createRgba(400, 510);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const { u, v } = toPage(x + 0.5, y + 0.5);
      const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
      const ink = inside && u > 0.1 && u < 0.9 && (v * 40) % 1 > 0.6 && (u * 12) % 1 < 0.8;
      const light = 0.5 + 0.5 * u;
      const base = !inside ? 60 : ink ? 35 : 235;
      img.data.set(
        [base * light, base * light * 0.95, base * light * 0.8, 255],
        (y * img.width + x) * 4,
      );
    }
  }
  return img;
}

describe('computeFilterPlan', () => {
  const pageQuad: Quad = [
    { x: 20, y: 15 },
    { x: 380, y: 15 },
    { x: 380, y: 495 },
    { x: 20, y: 495 },
  ];
  const page = rectifyCpu(scene(), pageQuad, 360, 480);
  const preview = rectifyCpu(scene(), pageQuad, 135, 180);

  for (const filter of SCAN_FILTERS as ScanFilter[]) {
    it(`GPU plan for '${filter}' matches the CPU filter`, () => {
      const plan = computeFilterPlan(preview, filter, page.width, page.height);
      const gpu = emulateShader(page, plan);
      const cpu = applyFilter(page, filter);
      let diff = 0;
      let mismatched = 0;
      for (let p = 0; p < cpu.data.length; p += 4) {
        const d =
          Math.abs(gpu.data[p] - cpu.data[p]) +
          Math.abs(gpu.data[p + 1] - cpu.data[p + 1]) +
          Math.abs(gpu.data[p + 2] - cpu.data[p + 2]);
        diff += d / 3;
        if (d / 3 > 64) mismatched++;
      }
      const pixels = cpu.data.length / 4;
      if (filter === 'bw') expect(mismatched / pixels).toBeLessThan(0.05);
      else expect(diff / pixels).toBeLessThan(filter === 'photo' ? 0.5 : 12);
    });
  }
});

describe('renderPageCpu', () => {
  it('rectifies, filters and rotates', () => {
    const source = scene();
    const out = renderPageCpu(source, {
      quad,
      filter: 'grayscale',
      rotation: 90,
      width: 60,
      height: 80,
    });
    expect([out.width, out.height]).toEqual([80, 60]);
    const expected = rotateRgba(applyFilter(rectifyCpu(source, quad, 60, 80), 'grayscale'), 90);
    expect(out.data).toEqual(expected.data);
  });
});

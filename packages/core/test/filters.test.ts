import { describe, expect, it } from 'vitest';
import {
  adaptiveThreshold,
  applyFilter,
  enhanceColor,
  removeShadows,
  SCAN_FILTERS,
  toGrayscale,
} from '../src/filters/index.ts';
import { createRgba, type RgbaImage, rgbaToGray } from '../src/image.ts';

const W = 240;
const H = 320;

/** True where the synthetic page has ink: horizontal "text" bars every 16 px. */
const isInk = (x: number, y: number) =>
  x >= 20 && x < W - 20 && y % 16 >= 6 && y % 16 < 9 && (x >> 4) % 3 !== 2;

/**
 * A page with dark text bars under a strong illumination gradient (left 45%, right 100%) and a
 * slight yellow cast, like a sheet photographed next to a window.
 */
function shadedPage(): RgbaImage {
  const img = createRgba(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const light = 0.45 + (0.55 * x) / (W - 1);
      const base = isInk(x, y) ? 40 : 235;
      const o = (y * W + x) * 4;
      img.data[o] = base * light;
      img.data[o + 1] = base * light * 0.97;
      img.data[o + 2] = base * light * 0.85;
      img.data[o + 3] = 255;
    }
  }
  return img;
}

/** Range (max − min) of the luma of paper pixels. */
function paperRange(img: RgbaImage): number {
  const gray = rgbaToGray(img);
  let min = 255;
  let max = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // Skip ink and its immediate neighborhood (resampling halos).
      if (isInk(x, y) || isInk(x, y - 1) || isInk(x, y + 1) || isInk(x - 1, y) || isInk(x + 1, y))
        continue;
      const v = gray.data[y * W + x];
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  return max - min;
}

describe('removeShadows', () => {
  it('flattens an illumination gradient and whitens the paper', () => {
    const page = shadedPage();
    expect(paperRange(page)).toBeGreaterThan(100);
    const flat = removeShadows(page);
    expect(paperRange(flat)).toBeLessThan(25);
    const paper = (10 * W + 120) * 4; // y = 10 is a gap between text lines
    expect(flat.data[paper]).toBeGreaterThan(235);
    expect(flat.data[paper + 2]).toBeGreaterThan(235); // yellow cast removed
  });

  it('keeps ink dark', () => {
    const flat = rgbaToGray(removeShadows(shadedPage()));
    expect(flat.data[7 * W + 30]).toBeLessThan(80);
    expect(flat.data[7 * W + 200]).toBeLessThan(80);
  });
});

describe('adaptiveThreshold', () => {
  for (const method of ['sauvola', 'bradley'] as const) {
    it(`${method}: outputs only 0/255 and separates ink from shaded paper`, () => {
      const binary = adaptiveThreshold(rgbaToGray(shadedPage()), { method });
      let errors = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = binary.data[y * W + x];
          expect(v === 0 || v === 255).toBe(true);
          if ((v === 0) !== isInk(x, y)) errors++;
        }
      }
      expect(errors / (W * H)).toBeLessThan(0.02);
    });
  }
});

describe('enhanceColor', () => {
  it('stretches each channel to the full range', () => {
    const img = createRgba(64, 1);
    for (let x = 0; x < 64; x++) img.data.set([60 + x, 80 + x, 100 + x, 255], x * 4);
    const out = enhanceColor(img, { lowPercentile: 0, highPercentile: 0, saturation: 1 });
    expect(Array.from(out.data.subarray(0, 3))).toEqual([0, 0, 0]);
    expect(Array.from(out.data.subarray(63 * 4, 63 * 4 + 3))).toEqual([255, 255, 255]);
    for (let x = 1; x < 64; x++)
      expect(out.data[x * 4]).toBeGreaterThanOrEqual(out.data[(x - 1) * 4]);
  });

  it('boosts saturation around luma', () => {
    // Black and white anchors keep the levels stretch an identity.
    const img = createRgba(3, 1);
    img.data.set([0, 0, 0, 255, 255, 255, 255, 255, 200, 100, 100, 255]);
    const plain = enhanceColor(img, { lowPercentile: 0, highPercentile: 0, saturation: 1 });
    const boosted = enhanceColor(img, { lowPercentile: 0, highPercentile: 0, saturation: 1.5 });
    const chroma = (d: Uint8ClampedArray) => d[8] - d[9];
    expect(chroma(boosted.data)).toBeGreaterThan(chroma(plain.data));
  });
});

describe('applyFilter', () => {
  const page = shadedPage();

  it('produces images of the same size for every filter', () => {
    for (const filter of SCAN_FILTERS) {
      const out = applyFilter(page, filter);
      expect([out.width, out.height]).toEqual([W, H]);
      expect(out.data).not.toBe(page.data);
    }
  });

  it("'photo' copies the pixels unchanged", () => {
    expect(applyFilter(page, 'photo').data).toEqual(page.data);
  });

  it("'grayscale' has equal channels and 'bw' only black and white", () => {
    const gray = applyFilter(page, 'grayscale');
    for (let p = 0; p < gray.data.length; p += 4) {
      expect(gray.data[p]).toBe(gray.data[p + 1]);
      expect(gray.data[p]).toBe(gray.data[p + 2]);
    }
    const bw = applyFilter(page, 'bw');
    expect(new Set(bw.data)).toEqual(new Set([0, 255]));
  });

  it('toGrayscale preserves alpha', () => {
    const img = createRgba(1, 1);
    img.data.set([10, 20, 30, 77]);
    expect(Array.from(toGrayscale(img).data)).toEqual([18, 18, 18, 77]);
  });
});

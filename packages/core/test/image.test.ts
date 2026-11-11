import { describe, expect, it } from 'vitest';
import { createGray, createRgba, downscaleGray, fitLongSide, rgbaToGray } from '../src/image.ts';

describe('rgbaToGray', () => {
  it('uses Rec. 601 luma and ignores alpha', () => {
    const rgba = createRgba(4, 1);
    rgba.data.set([255, 255, 255, 0, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    expect(Array.from(rgbaToGray(rgba).data)).toEqual([255, 76, 150, 29]);
  });

  it('writes into a provided output and validates its size', () => {
    const out = createGray(4, 1);
    expect(rgbaToGray(createRgba(4, 1), out)).toBe(out);
    expect(() => rgbaToGray(createRgba(4, 1), createGray(2, 2))).toThrow(RangeError);
  });
});

describe('fitLongSide', () => {
  it('scales the longer side and keeps small images', () => {
    expect(fitLongSide(1920, 1080, 384)).toEqual({ width: 384, height: 216, scale: 0.2 });
    expect(fitLongSide(300, 200, 384)).toEqual({ width: 300, height: 200, scale: 1 });
  });
});

describe('downscaleGray', () => {
  it('averages exact 2×2 blocks', () => {
    const gray = createGray(4, 2);
    gray.data.set([0, 255, 10, 20, 255, 0, 30, 40]);
    const small = downscaleGray(gray, 2);
    expect([small.width, small.height]).toEqual([2, 1]);
    expect(Array.from(small.data)).toEqual([128, 25]);
  });

  it('preserves the mean for fractional factors', () => {
    const gray = createGray(97, 61);
    for (let i = 0; i < gray.data.length; i++) gray.data[i] = (i * 37) % 256;
    const mean = (d: Uint8Array) => d.reduce((a, b) => a + b, 0) / d.length;
    const small = downscaleGray(gray, 40);
    expect(small.width).toBe(40);
    // Edge pixels carry equal total weight, so the mean is preserved up to rounding.
    expect(Math.abs(mean(small.data) - mean(gray.data))).toBeLessThan(1);
  });

  it('returns the input when no downscaling is needed', () => {
    const gray = createGray(10, 10);
    expect(downscaleGray(gray, 10)).toBe(gray);
  });
});

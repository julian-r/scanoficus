import { describe, expect, it } from 'vitest';
import { DetectorContext, detectDocument, sharpness } from '../src/detect/index.ts';
import { createGray, createRgba, rgbaToGray } from '../src/image.ts';
import { evaluateDetection } from './support/metrics.ts';
import { generateScene, type SceneClass } from './support/synthetic.ts';

/**
 * Regression thresholds per synthetic class on 40 seeds (the full `pnpm eval` uses 300 seeds;
 * see the README for those numbers). They sit a few points below the measured rates so that
 * the suite is stable but still catches real regressions.
 */
const THRESHOLDS: Record<Exclude<SceneClass, 'none'>, { success: number; cornerError: number }> = {
  easy: { success: 1, cornerError: 0.002 },
  perspective: { success: 1, cornerError: 0.002 },
  noisy: { success: 0.97, cornerError: 0.003 },
  shadow: { success: 0.97, cornerError: 0.003 },
  border: { success: 0.9, cornerError: 0.01 },
  textured: { success: 0.92, cornerError: 0.005 },
  'low-contrast': { success: 0.65, cornerError: 0.02 },
};
const SEEDS = 40;

describe('detectDocument on synthetic scenes', () => {
  const context = new DetectorContext();

  for (const [kind, threshold] of Object.entries(THRESHOLDS) as [
    Exclude<SceneClass, 'none'>,
    { success: number; cornerError: number },
  ][]) {
    it(`detects documents in '${kind}' scenes`, () => {
      let success = 0;
      const errors: number[] = [];
      for (let seed = 1000; seed < 1000 + SEEDS; seed++) {
        const scene = generateScene(kind, seed);
        const detection = detectDocument(rgbaToGray(scene.image), {}, context);
        const outcome = evaluateDetection(scene.quad, detection?.quad ?? null, 384, 288);
        if (outcome.iou >= 0.9) {
          success++;
          errors.push(outcome.cornerError);
        }
      }
      expect(success / SEEDS).toBeGreaterThanOrEqual(threshold.success);
      const meanError = errors.reduce((a, b) => a + b, 0) / errors.length;
      expect(meanError).toBeLessThan(threshold.cornerError);
    });
  }

  // Document-free scenes include uniform rectangular objects (books, boxes), which an edge-based
  // detector cannot tell from paper: 12.5% on these 40 seeds, 4.7% on the 300-seed eval with
  // gray input (6.3% with RGBA input).
  it('rarely reports documents in document-free scenes', () => {
    let falsePositives = 0;
    for (let seed = 1000; seed < 1000 + SEEDS; seed++) {
      const scene = generateScene('none', seed);
      if (detectDocument(rgbaToGray(scene.image), {}, context)) falsePositives++;
    }
    expect(falsePositives / SEEDS).toBeLessThanOrEqual(0.2);
  });
});

describe('detectDocument API', () => {
  it('returns quads in input coordinates for inputs larger than maxSide', () => {
    const scene = generateScene('easy', 7, 768, 576);
    const detection = detectDocument(rgbaToGray(scene.image), { maxSide: 384 });
    expect(detection).not.toBeNull();
    const outcome = evaluateDetection(scene.quad, detection!.quad, 768, 576);
    expect(outcome.iou).toBeGreaterThan(0.97);
  });

  it('maps X and Y back to the input separately for odd sizes', () => {
    // 1000×699 → 384×268 at processing size: the Y factor (699/268) differs from X (1000/384).
    const gray = createGray(1000, 699);
    gray.data.fill(40);
    for (let y = 120; y < 680; y++) gray.data.fill(220, y * 1000 + 150, y * 1000 + 850);
    const detection = detectDocument(gray)!;
    const ys = detection.quad.map((p) => p.y).sort((a, b) => a - b);
    // Old single-factor mapping put the bottom edge at ≈678.9 (1000/384 · 680 · 268/699).
    expect(ys[3]).toBeCloseTo(680, 0);
    expect(Math.abs(ys[3] - 680)).toBeLessThan(0.5);
    expect(Math.abs(ys[0] - 120)).toBeLessThan(0.5);
  });

  it('orders corners TL, TR, BR, BL and returns ranked candidates on request', () => {
    const scene = generateScene('easy', 3);
    const detection = detectDocument(rgbaToGray(scene.image), { candidates: true })!;
    const [tl, tr, br, bl] = detection.quad;
    expect(tl.x).toBeLessThan(tr.x);
    expect(bl.x).toBeLessThan(br.x);
    expect(tl.y).toBeLessThan(bl.y);
    expect(tr.y).toBeLessThan(br.y);
    expect(detection.candidates?.[0]).toEqual({ quad: detection.quad, score: detection.score });
    // Runner-ups follow by score (only the winner may be promoted over a higher score).
    const scores = detection.candidates!.slice(1).map((c) => c.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });

  /** 384×288 gray image filled with `background` and the given rectangles drawn in order. */
  function drawRects(background: number, rects: [number, number, number, number, number][]) {
    const gray = createGray(384, 288);
    gray.data.fill(background);
    for (const [x0, y0, x1, y1, value] of rects) {
      for (let y = y0; y < y1; y++) gray.data.fill(value, y * 384 + x0, y * 384 + x1);
    }
    return gray;
  }

  it('scores a quad lower if its inside is darker than its surroundings', () => {
    const rect: [number, number, number, number] = [80, 60, 300, 230];
    const paper = detectDocument(drawRects(60, [[...rect, 200]]), { minScore: 0 })!;
    const screen = detectDocument(drawRects(200, [[...rect, 60]]), { minScore: 0 })!;
    expect(paper.score).toBeGreaterThan(0.95);
    expect(screen.score).toBeLessThan(0.7);
  });

  /** `gray` mirrored in X and/or Y. */
  function mirror(gray: ReturnType<typeof createGray>, mx: boolean, my: boolean) {
    const out = createGray(gray.width, gray.height);
    for (let y = 0; y < gray.height; y++) {
      for (let x = 0; x < gray.width; x++) {
        const sx = mx ? gray.width - 1 - x : x;
        const sy = my ? gray.height - 1 - y : y;
        out.data[y * gray.width + x] = gray.data[sy * gray.width + sx];
      }
    }
    return out;
  }

  // A page (200) on a table (120) whose top-left corner fades into the table, with a block of
  // content (170) framed by dark lines on it. The block's edges are stronger and it scores
  // higher, but the page encloses it and the margin between them is brighter than the block.
  // The page is symmetric in X (40–344), so the same checks hold for every mirrored copy; the
  // mirrored copies give the candidates both corner orientations.
  for (const [mx, my] of [
    [false, false],
    [true, false],
    [false, true],
    [true, true],
  ] as const) {
    it(`prefers the page over a framed block of content on it (mirrored x ${mx}, y ${my})`, () => {
      const page = drawRects(120, [
        [40, 20, 344, 270, 200],
        [100, 70, 290, 230, 40],
        [102, 72, 288, 228, 170],
      ]);
      for (let y = 20; y < 60; y++) {
        for (let x = 40; x < 100; x++) {
          const fade = Math.min(1, ((x - 40) / 60) * ((y - 20) / 40) * 2);
          page.data[y * 384 + x] = Math.round(120 + 80 * fade);
        }
      }
      const gray = mirror(page, mx, my);
      const detection = detectDocument(gray, { candidates: true, minScore: 0 })!;
      const xs = detection.quad.map((p) => p.x).sort((a, b) => a - b);
      expect(Math.abs(xs[0] - 40)).toBeLessThan(2);
      expect(Math.abs(xs[3] - 344)).toBeLessThan(2);
      const block = detection.candidates!.find((c) => c.quad.every((p) => p.x > 90));
      expect(block!.score).toBeGreaterThan(detection.score);
    });
  }

  it('returns a document with two sides on the frame border at the default threshold', () => {
    // A page (210) on a dark table (50) that leaves the frame at the top and on the left.
    const gray = drawRects(50, [[0, 0, 250, 190, 210]]);
    const detection = detectDocument(gray);
    expect(detection).not.toBeNull();
    const xs = detection!.quad.map((p) => p.x).sort((a, b) => a - b);
    const ys = detection!.quad.map((p) => p.y).sort((a, b) => a - b);
    expect(xs[0]).toBeLessThan(1);
    expect(Math.abs(xs[3] - 250)).toBeLessThan(1.5);
    expect(ys[0]).toBeLessThan(1);
    expect(Math.abs(ys[3] - 190)).toBeLessThan(1.5);
  });

  it('uses colour edges for RGBA input: warm paper on a cool table of similar brightness', () => {
    const image = createRgba(384, 288);
    let seed = 7;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let y = 0; y < 288; y++) {
      for (let x = 0; x < 384; x++) {
        const paper = x >= 80 && x < 300 && y >= 50 && y < 240;
        const color = paper ? [232, 220, 186] : [204, 219, 238];
        const noise = (random() - 0.5) * 16;
        const p = 4 * (y * 384 + x);
        for (let c = 0; c < 3; c++) image.data[p + c] = color[c] + noise;
        image.data[p + 3] = 255;
      }
    }
    const gray = detectDocument(rgbaToGray(image), { minScore: 0 });
    expect(gray === null || gray.score < 0.5).toBe(true);
    const color = detectDocument(image)!;
    expect(color.score).toBeGreaterThan(0.9);
    const truth = [
      { x: 80, y: 50 },
      { x: 300, y: 50 },
      { x: 300, y: 240 },
      { x: 80, y: 240 },
    ];
    color.quad.forEach((p, i) => {
      expect(Math.hypot(p.x - truth[i].x, p.y - truth[i].y)).toBeLessThan(1);
    });
  });

  it('gives the same result for RGBA input without chroma as for its luma', () => {
    const scene = generateScene('perspective', 5);
    const gray = detectDocument(rgbaToGray(scene.image))!;
    const neutral = createRgba(scene.image.width, scene.image.height);
    const luma = rgbaToGray(scene.image).data;
    for (let i = 0; i < luma.length; i++) {
      neutral.data.fill(luma[i], 4 * i, 4 * i + 3);
      neutral.data[4 * i + 3] = 255;
    }
    expect(detectDocument(neutral)).toEqual(gray);
  });

  it('returns null for a blank frame and for tiny images', () => {
    const blank = createGray(320, 240);
    blank.data.fill(128);
    expect(detectDocument(blank)).toBeNull();
    expect(detectDocument(createGray(8, 8))).toBeNull();
  });

  it('respects minArea', () => {
    const scene = generateScene('easy', 11);
    const gray = rgbaToGray(scene.image);
    expect(detectDocument(gray, { minArea: 0.95 })).toBeNull();
  });
});

describe('sharpness', () => {
  it('is zero for flat images and decreases with blur', () => {
    const flat = createGray(50, 50);
    flat.data.fill(90);
    expect(sharpness(flat)).toBe(0);
    const sharpScene = rgbaToGray(generateScene('easy', 5).image);
    const blurred = createGray(sharpScene.width, sharpScene.height);
    const w = sharpScene.width;
    for (let i = 0; i < blurred.data.length; i++) {
      const a = sharpScene.data;
      blurred.data[i] =
        (a[i] * 4 +
          (a[i - 1] ?? a[i]) +
          (a[i + 1] ?? a[i]) +
          (a[i - w] ?? a[i]) +
          (a[i + w] ?? a[i])) /
        8;
    }
    expect(sharpness(blurred)).toBeLessThan(sharpness(sharpScene) * 0.8);
  });
});

import { describe, expect, it } from 'vitest';
import {
  isSelfIntersectingQuad,
  photoIoU,
  photoReference,
  smartDocJaccard,
  visibleFraction,
  visibleIoU,
} from '../bench/metrics.ts';
import type { Quad } from '../src/geometry/index.ts';

/** Quad from SmartDoc corner order (TL, BL, BR, TR), as in eval_seg.py, to TL, TR, BR, BL. */
function fromSmartDoc(points: [number, number][]): Quad {
  const [tl, bl, br, tr] = points.map(([x, y]) => ({ x, y }));
  return [tl, tr, br, bl];
}

describe('smartDocJaccard', () => {
  // Reference values from the official algorithm (cv2.getPerspectiveTransform +
  // cv2.perspectiveTransform + polygon intersection, as in smartdoc15-ch1-eval/eval_seg.py),
  // computed with OpenCV 4 and shapely. The first three cases are the examples in eval_seg.py.
  const gt = fromSmartDoc([
    [200, 200],
    [100, 800],
    [600, 800],
    [600, 300],
  ]);
  const frameGt = fromSmartDoc([
    [698.087, 200.476],
    [692.141, 891.077],
    [1253.18, 869.656],
    [1178.15, 191.515],
  ]);
  const a4 = { width: 2100, height: 2970 };
  const cases: [string, Quad, Quad, { width: number; height: number }, number][] = [
    [
      'partial overlap',
      gt,
      fromSmartDoc([
        [300, 200],
        [100, 600],
        [500, 700],
        [600, 200],
      ]),
      a4,
      0.550136818113834,
    ],
    ['perfect match', gt, gt, a4, 1],
    [
      'no overlap',
      gt,
      fromSmartDoc([
        [300, 100],
        [300, 200],
        [600, 200],
        [600, 100],
      ]),
      a4,
      0,
    ],
    [
      'close detection',
      frameGt,
      fromSmartDoc([
        [700, 210],
        [690, 880],
        [1250, 875],
        [1180, 195],
      ]),
      a4,
      0.9785073435618998,
    ],
    [
      'whole frame',
      frameGt,
      fromSmartDoc([
        [0, 0],
        [0, 1080],
        [1920, 1080],
        [1920, 0],
      ]),
      a4,
      0.16783944619616203,
    ],
    [
      'letter-size reference',
      fromSmartDoc([
        [500, 300],
        [450, 900],
        [1500, 1000],
        [1400, 250],
      ]),
      fromSmartDoc([
        [520, 320],
        [470, 880],
        [1000, 950],
        [1380, 270],
      ]),
      { width: 2159, height: 2794 },
      0.7176376329709555,
    ],
  ];
  for (const [name, truth, detected, reference, expected] of cases) {
    it(`matches the official evaluation: ${name}`, () => {
      expect(smartDocJaccard(truth, detected, reference)).toBeCloseTo(expected, 5);
    });
  }

  it('scores missing detections 0', () => {
    expect(smartDocJaccard(gt, null, a4)).toBe(0);
  });

  it('scores self-intersecting detections 0', () => {
    const [tl, tr, br, bl] = gt;
    expect(smartDocJaccard(gt, [tl, tr, bl, br], a4)).toBe(0);
    expect(isSelfIntersectingQuad([tl, tr, bl, br])).toBe(true);
    expect(isSelfIntersectingQuad(gt)).toBe(false);
  });

  it('does not depend on the corner labels of the detection', () => {
    const [tl, tr, br, bl] = frameGt;
    const close = fromSmartDoc([
      [700, 210],
      [690, 880],
      [1250, 875],
      [1180, 195],
    ]);
    const rotated: Quad = [close[1], close[2], close[3], close[0]];
    expect(smartDocJaccard([tl, tr, br, bl], rotated, a4)).toBeCloseTo(
      smartDocJaccard(frameGt, close, a4),
      12,
    );
  });
});

describe('visibleIoU and visibleFraction', () => {
  const square = (x: number, y: number, s: number): Quad => [
    { x, y },
    { x: x + s, y },
    { x: x + s, y: y + s },
    { x, y: y + s },
  ];

  it('compares only the parts inside the frame', () => {
    // Ground truth half outside the frame; the detection covers exactly the visible half.
    const truth = square(-50, 0, 100);
    const visibleHalf: Quad = [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 50, y: 100 },
      { x: 0, y: 100 },
    ];
    expect(visibleIoU(truth, visibleHalf, 200, 200)).toBeCloseTo(1, 12);
    expect(visibleFraction(truth, 200, 200)).toBeCloseTo(0.5, 12);
    expect(visibleIoU(truth, null, 200, 200)).toBe(0);
  });
});

describe('photo metrics', () => {
  const truth: Quad = [
    { x: 10, y: 10 },
    { x: 110, y: 20 },
    { x: 100, y: 220 },
    { x: 0, y: 200 },
  ];

  it('derives the JI reference from the ground-truth side lengths; the JI does not depend on it', () => {
    const reference = photoReference(truth);
    expect(reference.width).toBeCloseTo(0.5 * (Math.hypot(100, 10) + Math.hypot(100, 20)), 9);
    expect(reference.height).toBeCloseTo(0.5 * (Math.hypot(10, 200) + Math.hypot(10, 190)), 9);
    const detected: Quad = [
      { x: 14, y: 6 },
      { x: 112, y: 24 },
      { x: 96, y: 224 },
      { x: 2, y: 196 },
    ];
    expect(smartDocJaccard(truth, detected, reference)).toBeCloseTo(
      smartDocJaccard(truth, detected, { width: 2100, height: 2970 }),
      9,
    );
  });

  it('takes the IoU of the full quads, also outside the image, and 0 for bow ties', () => {
    expect(photoIoU(truth, truth)).toBeCloseTo(1, 12);
    expect(photoIoU(truth, null)).toBe(0);
    expect(photoIoU(truth, [truth[0], truth[2], truth[1], truth[3]])).toBe(0);
  });
});

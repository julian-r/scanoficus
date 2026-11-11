/**
 * Procedural synthetic document scenes with exact ground-truth quads, used by the detector tests
 * and the evaluation script. A scene is a "photo" of a paper sheet (text-like strokes, optional
 * picture block) under a pinhole camera with random pose, composited on a background, lit with
 * optional shadows, then blurred and corrupted with noise. Rendering is 2×2 supersampled.
 */

import type { Matrix3, Point, Quad } from '../../src/geometry/index.ts';
import { applyHomography, homographyFromQuads, invertMatrix3 } from '../../src/geometry/index.ts';
import { createRgba, type RgbaImage } from '../../src/image.ts';
import { Random } from './random.ts';

/** Difficulty classes of the synthetic benchmark. */
export const SCENE_CLASSES = [
  'easy',
  'perspective',
  'noisy',
  'shadow',
  'low-contrast',
  'border',
  'textured',
  'none',
] as const;
export type SceneClass = (typeof SCENE_CLASSES)[number];

export interface Scene {
  kind: SceneClass;
  seed: number;
  image: RgbaImage;
  /** Ground-truth document corners (TL, TR, BR, BL of the paper), or null for 'none'. */
  quad: Quad | null;
}

interface ClassParams {
  areaRange: [number, number];
  maxTilt: number;
  maxRoll: number;
  background: 'plain' | 'light' | 'textured';
  paperContrast?: [number, number];
  noise: [number, number];
  blur: [number, number];
  shadow: number;
  softShadow: number;
  border: boolean;
}

const PARAMS: Record<SceneClass, ClassParams> = {
  easy: {
    areaRange: [0.25, 0.6],
    maxTilt: 15,
    maxRoll: 10,
    background: 'plain',
    noise: [1, 3],
    blur: [0.3, 0.7],
    shadow: 0.1,
    softShadow: 0,
    border: false,
  },
  perspective: {
    areaRange: [0.15, 0.6],
    maxTilt: 40,
    maxRoll: 45,
    background: 'plain',
    noise: [2, 4],
    blur: [0.3, 0.8],
    shadow: 0.15,
    softShadow: 0,
    border: false,
  },
  noisy: {
    areaRange: [0.2, 0.6],
    maxTilt: 25,
    maxRoll: 20,
    background: 'plain',
    noise: [6, 14],
    blur: [0.8, 2],
    shadow: 0.15,
    softShadow: 0,
    border: false,
  },
  shadow: {
    areaRange: [0.2, 0.6],
    maxTilt: 25,
    maxRoll: 20,
    background: 'plain',
    noise: [2, 5],
    blur: [0.3, 1],
    shadow: 0.55,
    softShadow: 0.7,
    border: false,
  },
  'low-contrast': {
    areaRange: [0.2, 0.6],
    maxTilt: 25,
    maxRoll: 20,
    background: 'light',
    paperContrast: [12, 35],
    noise: [1.5, 4],
    blur: [0.3, 1],
    shadow: 0.15,
    softShadow: 0,
    border: false,
  },
  border: {
    areaRange: [0.45, 0.8],
    maxTilt: 20,
    maxRoll: 15,
    background: 'plain',
    noise: [2, 5],
    blur: [0.3, 1],
    shadow: 0.15,
    softShadow: 0,
    border: true,
  },
  textured: {
    areaRange: [0.2, 0.6],
    maxTilt: 25,
    maxRoll: 20,
    background: 'textured',
    noise: [2, 6],
    blur: [0.3, 1],
    shadow: 0.2,
    softShadow: 0,
    border: false,
  },
  none: {
    areaRange: [0, 0],
    maxTilt: 0,
    maxRoll: 0,
    background: 'textured',
    noise: [2, 6],
    blur: [0.3, 1],
    shadow: 0.2,
    softShadow: 0,
    border: false,
  },
};

const SUPERSAMPLE = 2;

/** Renders a reproducible scene for the given class and seed. */
export function generateScene(kind: SceneClass, seed: number, width = 384, height = 288): Scene {
  const rnd = new Random(seed * 7919 + SCENE_CLASSES.indexOf(kind) * 104729 + 1);
  const p = PARAMS[kind];

  const background = makeBackground(rnd, p, kind === 'none');
  const bgLuma = background.meanLuma;
  let quad: Quad | null = null;
  let toDoc: Matrix3 | null = null;
  let paper: Texture | null = null;
  if (kind !== 'none') {
    quad = placeDocument(rnd, p, width, height);
    const unit: Quad = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    toDoc = invertMatrix3(homographyFromQuads(unit, quad) as Matrix3);
    const paperLuma = p.paperContrast
      ? Math.min(250, bgLuma + rnd.range(p.paperContrast[0], p.paperContrast[1]))
      : rnd.range(215, 248);
    paper = makePaper(rnd, paperLuma, quad);
  }
  const light = makeLighting(rnd, p, width, height, quad);

  // Supersampled render, box-filtered down to the output resolution.
  const rgb = new Float32Array(width * height * 3);
  const step = 1 / SUPERSAMPLE;
  const sampleWeight = 1 / (SUPERSAMPLE * SUPERSAMPLE);
  const color: [number, number, number] = [0, 0, 0];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let sy = 0; sy < SUPERSAMPLE; sy++) {
        for (let sx = 0; sx < SUPERSAMPLE; sx++) {
          const px = x + (sx + 0.5) * step;
          const py = y + (sy + 0.5) * step;
          let inside = false;
          if (toDoc && paper) {
            const d = applyHomography(toDoc, { x: px, y: py });
            if (d.x >= 0 && d.x <= 1 && d.y >= 0 && d.y <= 1) {
              paper.sample(d.x, d.y, color);
              inside = true;
            }
          }
          if (!inside) background.sample(px, py, color);
          const l = light(px, py);
          r += color[0] * l;
          g += color[1] * l;
          b += color[2] * l;
        }
      }
      const o = (y * width + x) * 3;
      rgb[o] = r * sampleWeight;
      rgb[o + 1] = g * sampleWeight;
      rgb[o + 2] = b * sampleWeight;
    }
  }

  blurRgb(rgb, width, height, rnd.range(p.blur[0], p.blur[1]));
  const sigma = rnd.range(p.noise[0], p.noise[1]);
  const image = createRgba(width, height);
  for (let i = 0, n = width * height; i < n; i++) {
    // Mostly luminance noise plus a little chroma noise, like a phone sensor after denoising.
    const common = rnd.gaussian() * sigma;
    for (let c = 0; c < 3; c++) {
      image.data[i * 4 + c] = rgb[i * 3 + c] + common + rnd.gaussian() * sigma * 0.3;
    }
    image.data[i * 4 + 3] = 255;
  }
  return { kind, seed, image, quad };
}

interface Texture {
  sample(x: number, y: number, out: [number, number, number]): void;
}

interface Background extends Texture {
  meanLuma: number;
}

function makeBackground(rnd: Random, p: ClassParams, noDocument: boolean): Background {
  const tint = (luma: number): [number, number, number] => [
    luma * rnd.range(0.85, 1.15),
    luma * rnd.range(0.9, 1.1),
    luma * rnd.range(0.8, 1.15),
  ];
  const base = tint(p.background === 'light' ? rnd.range(165, 205) : rnd.range(35, 130));
  const waves = Array.from({ length: 3 }, () => ({
    fx: rnd.range(-1, 1) / rnd.range(60, 300),
    fy: rnd.range(-1, 1) / rnd.range(60, 300),
    phase: rnd.range(0, 2 * Math.PI),
    amp: rnd.range(2, 8),
  }));
  const lowFrequency = (x: number, y: number) => {
    let v = 0;
    for (const w of waves) v += w.amp * Math.sin(2 * Math.PI * (w.fx * x + w.fy * y) + w.phase);
    return v;
  };
  const plain: Background = {
    meanLuma: (base[0] + base[1] + base[2]) / 3,
    sample(x, y, out) {
      const v = lowFrequency(x, y);
      out[0] = base[0] + v;
      out[1] = base[1] + v;
      out[2] = base[2] + v;
    },
  };
  if (p.background !== 'textured') return plain;

  const style = rnd.pick(noDocument ? ['wood', 'fabric'] : ['wood', 'fabric', 'clutter']);
  if (style === 'wood') {
    const angle = rnd.range(0, Math.PI);
    const period = rnd.range(6, 22);
    const amp = rnd.range(12, 35);
    const warp = rnd.range(1, 4);
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);
    return {
      meanLuma: plain.meanLuma,
      sample(x, y, out) {
        const along = x * ca + y * sa;
        const across = -x * sa + y * ca;
        const v = amp * Math.sin((2 * Math.PI * (along + warp * Math.sin(across / 23))) / period);
        plain.sample(x, y, out);
        out[0] += v;
        out[1] += v * 0.8;
        out[2] += v * 0.6;
      },
    };
  }
  if (style === 'fabric') {
    const period = rnd.range(3, 7);
    const amp = rnd.range(10, 25);
    return {
      meanLuma: plain.meanLuma,
      sample(x, y, out) {
        const v = amp * Math.sin(x * (6.283 / period)) * Math.sin(y * (6.283 / period));
        plain.sample(x, y, out);
        out[0] += v;
        out[1] += v;
        out[2] += v;
      },
    };
  }
  // Clutter: random rotated rectangles (other objects on the desk) with strong straight edges.
  const objects = Array.from({ length: rnd.int(6, 14) }, () => {
    const cx = rnd.range(-40, 420);
    const cy = rnd.range(-40, 330);
    const angle = rnd.range(0, Math.PI);
    return {
      cx,
      cy,
      ca: Math.cos(angle),
      sa: Math.sin(angle),
      hw: rnd.range(5, 90),
      hh: rnd.range(2, 50),
      color: tint(rnd.range(20, 200)),
    };
  });
  return {
    meanLuma: plain.meanLuma,
    sample(x, y, out) {
      plain.sample(x, y, out);
      for (const o of objects) {
        const dx = x - o.cx;
        const dy = y - o.cy;
        if (Math.abs(dx * o.ca + dy * o.sa) < o.hw && Math.abs(-dx * o.sa + dy * o.ca) < o.hh) {
          out[0] = o.color[0];
          out[1] = o.color[1];
          out[2] = o.color[2];
        }
      }
    },
  };
}

/**
 * Projects a paper rectangle with a random camera pose and returns its image quad. The doc is
 * scaled (equivalent to changing the focal length) to hit a random target area and translated
 * into the frame; for the 'border' class one or two sides are pushed onto or slightly over the
 * frame edge.
 */
function placeDocument(rnd: Random, p: ClassParams, width: number, height: number): Quad {
  const aspect = rnd.pick([Math.SQRT2, 1.294, 1.586, 1, 2.2]);
  const landscape = rnd.chance(0.25);
  const hw = landscape ? aspect / 2 : 0.5;
  const hh = landscape ? 0.5 : aspect / 2;
  const tiltX = (rnd.range(-p.maxTilt, p.maxTilt) * Math.PI) / 180;
  const tiltY = (rnd.range(-p.maxTilt, p.maxTilt) * Math.PI) / 180;
  const roll = (rnd.range(-p.maxRoll, p.maxRoll) * Math.PI) / 180;
  const focal = width * rnd.range(0.8, 1.1);
  const depth = 2.2;
  const corners = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ].map(([X, Y]) => {
    // Rotate about the x axis, then the y axis, then the optical axis.
    const y1 = Y * Math.cos(tiltX);
    const z1 = Y * Math.sin(tiltX);
    const x2 = X * Math.cos(tiltY) + z1 * Math.sin(tiltY);
    const z2 = -X * Math.sin(tiltY) + z1 * Math.cos(tiltY);
    const x3 = x2 * Math.cos(roll) - y1 * Math.sin(roll);
    const y3 = x2 * Math.sin(roll) + y1 * Math.cos(roll);
    const z = depth + z2;
    return { x: (focal * x3) / z, y: (focal * y3) / z };
  });
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    area += a.x * b.y - b.x * a.y;
  }
  area = Math.abs(area) / 2;
  const targetArea = rnd.range(p.areaRange[0], p.areaRange[1]) * width * height;
  let scale = Math.sqrt(targetArea / area);
  const bounds = (k: number) => {
    const xs = corners.map((c) => c.x * k);
    const ys = corners.map((c) => c.y * k);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  };
  let b = bounds(scale);
  // Shrink until the document fits with a small margin.
  const margin = 0.03 * width;
  const fit = Math.min(
    (width - 2 * margin) / (b.x1 - b.x0),
    (height - 2 * margin) / (b.y1 - b.y0),
    1,
  );
  scale *= fit;
  b = bounds(scale);
  let tx: number;
  let ty: number;
  if (p.border) {
    // Push against one vertical frame side and, half of the time, one horizontal side too,
    // touching it or overflowing by up to 3% of the width. Thanks to the fitting margin the
    // document never overflows two opposite sides.
    const over = () => rnd.range(-0.005, 0.03) * width;
    tx = rnd.chance(0.5) ? -b.x0 - over() : width - b.x1 + over();
    if (rnd.chance(0.5)) ty = rnd.chance(0.5) ? -b.y0 - over() : height - b.y1 + over();
    else ty = rnd.range(margin - b.y0, height - margin - b.y1);
  } else {
    tx = rnd.range(margin - b.x0, width - margin - b.x1);
    ty = rnd.range(margin - b.y0, height - margin - b.y1);
  }
  const q = corners.map((c) => ({ x: c.x * scale + tx, y: c.y * scale + ty }));
  return [q[0], q[1], q[2], q[3]];
}

/** Paper texture: tinted paper with margins, text-like word strokes and an optional picture. */
function makePaper(rnd: Random, luma: number, quad: Quad): Texture {
  const tw = 240;
  const aspect =
    (dist(quad[0], quad[3]) + dist(quad[1], quad[2])) /
    (dist(quad[0], quad[1]) + dist(quad[3], quad[2]));
  const th = Math.max(60, Math.round(tw * aspect));
  const tex = new Float32Array(tw * th * 3);
  const paper: [number, number, number] = [
    luma * rnd.range(0.97, 1.02),
    luma * rnd.range(0.97, 1.01),
    luma * rnd.range(0.93, 1),
  ];
  for (let i = 0; i < tw * th; i++) tex.set(paper, i * 3);
  const fill = (x0: number, y0: number, x1: number, y1: number, c: number[]) => {
    for (let y = Math.max(0, Math.round(y0)); y < Math.min(th, Math.round(y1)); y++) {
      for (let x = Math.max(0, Math.round(x0)); x < Math.min(tw, Math.round(x1)); x++) {
        tex.set(c, (y * tw + x) * 3);
      }
    }
  };
  const margin = tw * rnd.range(0.07, 0.13);
  const ink = rnd.range(20, 70);
  const lineHeight = rnd.range(7, 11);
  let y = margin;
  if (rnd.chance(0.5)) {
    const pictureH = th * rnd.range(0.15, 0.3);
    if (rnd.chance(0.5)) {
      fill(margin, y, tw - margin, y + pictureH, [
        rnd.range(40, 200),
        rnd.range(40, 200),
        rnd.range(40, 200),
      ]);
      y += pictureH + lineHeight;
    }
  }
  while (y < th - margin - lineHeight) {
    if (rnd.chance(0.12)) {
      y += lineHeight; // paragraph break
      continue;
    }
    let x = margin;
    const end = tw - margin - (rnd.chance(0.2) ? rnd.range(0, tw * 0.4) : 0);
    while (x < end) {
      const w = Math.min(end - x, rnd.range(6, 28));
      fill(x, y, x + w, y + lineHeight * 0.55, [ink, ink, ink]);
      x += w + rnd.range(3, 5);
    }
    y += lineHeight;
  }
  return {
    sample(u, v, out) {
      const x = Math.min(tw - 1, Math.max(0, u * tw - 0.5));
      const yy = Math.min(th - 1, Math.max(0, v * th - 0.5));
      const x0 = Math.floor(x);
      const y0 = Math.floor(yy);
      const x1 = Math.min(tw - 1, x0 + 1);
      const y1 = Math.min(th - 1, y0 + 1);
      const fx = x - x0;
      const fy = yy - y0;
      for (let c = 0; c < 3; c++) {
        const a = tex[(y0 * tw + x0) * 3 + c] * (1 - fx) + tex[(y0 * tw + x1) * 3 + c] * fx;
        const b = tex[(y1 * tw + x0) * 3 + c] * (1 - fx) + tex[(y1 * tw + x1) * 3 + c] * fx;
        out[c] = a * (1 - fy) + b * fy;
      }
    },
  };
}

/** Multiplicative illumination: exposure, linear shadow gradient, optional soft cast shadow. */
function makeLighting(
  rnd: Random,
  p: ClassParams,
  width: number,
  height: number,
  quad: Quad | null,
): (x: number, y: number) => number {
  const exposure = rnd.range(0.92, 1.05);
  const angle = rnd.range(0, 2 * Math.PI);
  const gx = Math.cos(angle);
  const gy = Math.sin(angle);
  const span = Math.abs(gx) * width + Math.abs(gy) * height;
  const offset = Math.min(0, gx * width) + Math.min(0, gy * height);
  const strength = rnd.range(p.shadow * 0.5, p.shadow);
  let cast: ((x: number, y: number) => number) | null = null;
  if (quad && rnd.chance(p.softShadow)) {
    // Shadow of a hand/phone: half-plane through a point inside the document, soft edge.
    const t = rnd.range(0.3, 0.7);
    const c: Point = {
      x: quad[0].x + (quad[2].x - quad[0].x) * t,
      y: quad[0].y + (quad[2].y - quad[0].y) * t,
    };
    const a = rnd.range(0, 2 * Math.PI);
    const nx = Math.cos(a);
    const ny = Math.sin(a);
    const softness = rnd.range(4, 18);
    const depthOf = rnd.range(0.3, 0.45);
    cast = (x, y) => {
      const d = ((x - c.x) * nx + (y - c.y) * ny) / softness;
      return 1 - depthOf / (1 + Math.exp(-d));
    };
  }
  return (x, y) => {
    const t = (x * gx + y * gy - offset) / span;
    const l = exposure * (1 - strength * t);
    return cast ? l * cast(x, y) : l;
  };
}

function blurRgb(rgb: Float32Array, width: number, height: number, sigma: number): void {
  const radius = Math.ceil(sigma * 3);
  const kernel = Array.from({ length: 2 * radius + 1 }, (_, i) =>
    Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)),
  );
  const sum = kernel.reduce((a, b) => a + b, 0);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const tmp = new Float32Array(rgb.length);
  const pass = (src: Float32Array, dst: Float32Array, horizontal: boolean) => {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 3; c++) {
          let acc = 0;
          for (let k = -radius; k <= radius; k++) {
            const xx = horizontal ? Math.min(width - 1, Math.max(0, x + k)) : x;
            const yy = horizontal ? y : Math.min(height - 1, Math.max(0, y + k));
            acc += src[(yy * width + xx) * 3 + c] * kernel[k + radius];
          }
          dst[(y * width + x) * 3 + c] = acc;
        }
      }
    }
  };
  pass(rgb, tmp, true);
  pass(tmp, rgb, false);
}

function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

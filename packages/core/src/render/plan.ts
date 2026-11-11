import {
  COLOR_SATURATION,
  estimateBackground,
  levelPoints,
  removeShadows,
  type ScanFilter,
  sauvolaWindow,
} from '../filters/filters.ts';
import { createRgba, type RgbaImage, rgbaToGray } from '../image.ts';

/** Shader modes, one per {@link ScanFilter}. */
export const FILTER_MODES: Record<ScanFilter, number> = { photo: 0, color: 1, grayscale: 2, bw: 3 };

/** Sauvola sensitivity used by the `'bw'` filter (same default as `adaptiveThreshold`). */
const SAUVOLA_K = 0.2;

/**
 * Everything the per-pixel GPU filter needs, derived on the CPU from a low-resolution preview of
 * the rectified page. Expensive, non-local steps of the CPU filters (background estimation,
 * histograms, local statistics) run at preview resolution; the full-resolution pass only does
 * per-pixel arithmetic and texture lookups.
 */
export interface FilterPlan {
  /** Shader mode (see {@link FILTER_MODES}). */
  mode: number;
  /** Paper background per channel (RGB, 0–255), sampled bilinearly over the page. */
  background: RgbaImage | null;
  /** Levels black/white points per channel, in 0–255, applied after shadow removal. */
  black: [number, number, number];
  white: [number, number, number];
  saturation: number;
  /** For `'bw'`: local mean (R) and standard deviation (G) of the shadow-free luma, 0–255. */
  statistics: RgbaImage | null;
  /** Sauvola sensitivity. */
  k: number;
}

/** Plan of the `'photo'` filter: rectification only. */
export const PHOTO_PLAN: FilterPlan = {
  mode: 0,
  background: null,
  black: [0, 0, 0],
  white: [255, 255, 255],
  saturation: 1,
  statistics: null,
  k: SAUVOLA_K,
};

/**
 * Computes the {@link FilterPlan} for `filter` from `preview`, a downscaled rectified page whose
 * full-resolution size is `outWidth × outHeight` (unrotated). Mirrors `applyFilter`:
 * - `'color'`: shadow removal, per-channel 1 % levels, saturation boost;
 * - `'grayscale'`: shadow removal, luma, levels from the luma histogram;
 * - `'bw'`: Sauvola on the shadow-free luma, with the local statistics computed at preview
 *   resolution over the window `adaptiveThreshold` would use at full resolution (an
 *   approximation: thin strokes lower the local variance less at low resolution).
 */
export function computeFilterPlan(
  preview: RgbaImage,
  filter: ScanFilter,
  outWidth: number,
  outHeight: number,
): FilterPlan {
  if (filter === 'photo') return PHOTO_PLAN;
  const plan: FilterPlan = {
    ...PHOTO_PLAN,
    mode: FILTER_MODES[filter],
    black: [0, 0, 0],
    white: [255, 255, 255],
  };
  plan.background = backgroundTexture(preview);
  const flat = removeShadows(preview);
  const n = flat.width * flat.height;
  if (filter === 'color') {
    plan.saturation = COLOR_SATURATION;
    for (let c = 0; c < 3; c++) {
      const histogram = new Uint32Array(256);
      for (let p = c; p < flat.data.length; p += 4) histogram[flat.data[p]]++;
      [plan.black[c], plan.white[c]] = levelPoints(histogram, n, 0.01, 0.01);
    }
    return plan;
  }
  const luma = rgbaToGray(flat);
  if (filter === 'grayscale') {
    const histogram = new Uint32Array(256);
    for (let i = 0; i < n; i++) histogram[luma.data[i]]++;
    const [black, white] = levelPoints(histogram, n, 0.01, 0.01);
    plan.black = [black, black, black];
    plan.white = [white, white, white];
    return plan;
  }
  const scale = preview.width / outWidth;
  const radius = Math.max(1, Math.round((sauvolaWindow(outWidth, outHeight) * scale) / 2));
  plan.statistics = localStatistics(luma.data, luma.width, luma.height, radius);
  return plan;
}

/** Background estimate of the preview packed as an RGBA texture. */
function backgroundTexture(preview: RgbaImage): RgbaImage {
  const bg = estimateBackground(preview);
  const out = createRgba(bg.width, bg.height);
  for (let i = 0; i < bg.width * bg.height; i++) {
    out.data[i * 4] = bg.channels[0][i];
    out.data[i * 4 + 1] = bg.channels[1][i];
    out.data[i * 4 + 2] = bg.channels[2][i];
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

/** Box-window mean (R) and standard deviation (G) via integral images. */
function localStatistics(data: Uint8Array, w: number, h: number, r: number): RgbaImage {
  const stride = w + 1;
  const sum = new Float64Array(stride * (h + 1));
  const sq = new Float64Array(stride * (h + 1));
  for (let y = 0; y < h; y++) {
    let rs = 0;
    let rq = 0;
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x];
      rs += v;
      rq += v * v;
      const i = (y + 1) * stride + x + 1;
      sum[i] = sum[i - stride] + rs;
      sq[i] = sq[i - stride] + rq;
    }
  }
  const out = createRgba(w, h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w, x + r + 1);
      const count = (x1 - x0) * (y1 - y0);
      const box = (t: Float64Array) =>
        t[y1 * stride + x1] - t[y0 * stride + x1] - t[y1 * stride + x0] + t[y0 * stride + x0];
      const mean = box(sum) / count;
      const variance = Math.max(0, box(sq) / count - mean * mean);
      const o = (y * w + x) * 4;
      out.data[o] = mean;
      out.data[o + 1] = Math.sqrt(variance);
      out.data[o + 3] = 255;
    }
  }
  return out;
}

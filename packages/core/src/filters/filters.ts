import { createGray, createRgba, type GrayImage, type RgbaImage, rgbaToGray } from '../image.ts';
import { resampleArea } from '../internal/resample.ts';

/**
 * Output styles of a scanned page, mirroring VisionKit's document camera:
 * - `'color'`: shadow-free, white-balanced, contrast-stretched color (VisionKit "Color").
 * - `'grayscale'`: the same cleanup in gray (VisionKit "Grayscale").
 * - `'bw'`: binarized black text on white paper (VisionKit "Black & White").
 * - `'photo'`: the rectified pixels unchanged (VisionKit "Photo", the "original" of PLAN.md).
 */
export type ScanFilter = 'color' | 'grayscale' | 'bw' | 'photo';

/** All filters, in the order a filter picker would present them. */
export const SCAN_FILTERS: readonly ScanFilter[] = ['color', 'grayscale', 'bw', 'photo'];

/** Copy of `rgba` with every pixel replaced by its luma (alpha preserved). */
export function toGrayscale(rgba: RgbaImage): RgbaImage {
  const gray = rgbaToGray(rgba);
  const out = createRgba(rgba.width, rgba.height);
  for (let i = 0, p = 0; i < gray.data.length; i++, p += 4) {
    const v = gray.data[i];
    out.data[p] = v;
    out.data[p + 1] = v;
    out.data[p + 2] = v;
    out.data[p + 3] = rgba.data[p + 3];
  }
  return out;
}

/** Options for {@link enhanceColor}. */
export interface EnhanceOptions {
  /** Fraction of darkest pixels per channel mapped to 0. Default 0.01. */
  lowPercentile?: number;
  /** Fraction of brightest pixels per channel mapped to 255. Default 0.01. */
  highPercentile?: number;
  /** Saturation multiplier applied around the luma after stretching. Default 1.15. */
  saturation?: number;
}

/**
 * Auto-levels plus a mild saturation boost: each channel is stretched linearly so that its
 * `lowPercentile` / `1 − highPercentile` quantiles map to 0 / 255 (stretching channels
 * independently also neutralizes a color cast of the paper), then chroma is scaled by
 * `saturation`.
 */
export function enhanceColor(rgba: RgbaImage, options: EnhanceOptions = {}): RgbaImage {
  const low = options.lowPercentile ?? 0.01;
  const high = options.highPercentile ?? 0.01;
  const saturation = options.saturation ?? 1.15;
  const { data } = rgba;
  const n = rgba.width * rgba.height;
  const luts = [0, 1, 2].map((c) => {
    const histogram = new Uint32Array(256);
    for (let p = c; p < data.length; p += 4) histogram[data[p]]++;
    return levelsLut(histogram, n, low, high);
  });
  const out = createRgba(rgba.width, rgba.height);
  const o = out.data;
  const [lr, lg, lb] = luts;
  for (let p = 0; p < data.length; p += 4) {
    const r = lr[data[p]];
    const g = lg[data[p + 1]];
    const b = lb[data[p + 2]];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    o[p] = y + (r - y) * saturation;
    o[p + 1] = y + (g - y) * saturation;
    o[p + 2] = y + (b - y) * saturation;
    o[p + 3] = data[p + 3];
  }
  return out;
}

/** Linear levels lookup table from a histogram and clipping fractions at both ends. */
function levelsLut(
  histogram: Uint32Array,
  total: number,
  low: number,
  high: number,
): Uint8ClampedArray {
  let black = 0;
  for (let acc = 0; black < 255 && acc + histogram[black] <= low * total; black++) {
    acc += histogram[black];
  }
  let white = 255;
  for (let acc = 0; white > 0 && acc + histogram[white] <= high * total; white--) {
    acc += histogram[white];
  }
  const lut = new Uint8ClampedArray(256);
  if (white <= black) {
    for (let v = 0; v < 256; v++) lut[v] = v;
    return lut;
  }
  const k = 255 / (white - black);
  for (let v = 0; v < 256; v++) lut[v] = (v - black) * k;
  return lut;
}

/** Options for {@link removeShadows}. */
export interface ShadowRemovalOptions {
  /** Long side (px) of the background estimate. Default 96. */
  backgroundSize?: number;
  /**
   * Radius (px at background resolution) of the max filter that erases dark content (text,
   * lines) from the background estimate. Default 3.
   */
  dilateRadius?: number;
}

/**
 * Flattens uneven illumination (shadows, vignetting, colored light) of a rectified page.
 *
 * The paper background of each channel is estimated at low resolution: the channel is
 * area-downscaled to `backgroundSize`, a max filter removes dark foreground strokes and a box blur
 * smooths the result. Every pixel is then divided by its (bilinearly upsampled) background and
 * rescaled so that paper becomes white, which removes shading while preserving ink contrast.
 * Dividing per channel also white-balances the paper.
 */
export function removeShadows(rgba: RgbaImage, options: ShadowRemovalOptions = {}): RgbaImage {
  const { width, height, data } = rgba;
  const size = options.backgroundSize ?? 96;
  const radius = options.dilateRadius ?? 3;
  const scale = Math.min(1, size / Math.max(width, height));
  const bw = Math.max(1, Math.round(width * scale));
  const bh = Math.max(1, Math.round(height * scale));
  const channel = new Uint8Array(width * height);
  const backgrounds = [0, 1, 2].map((c) => {
    for (let i = 0, p = c; i < channel.length; i++, p += 4) channel[i] = data[p];
    const small = new Float32Array(bw * bh);
    if (bw === width && bh === height) small.set(channel);
    else resampleArea(channel, width, height, small, bw, bh);
    const dilated = maxFilter(small, bw, bh, radius);
    return boxBlur(dilated, bw, bh, radius);
  });

  const out = createRgba(width, height);
  const o = out.data;
  // Background sample positions of output pixel centers (bilinear, edge-clamped).
  const fx = (x: number) => Math.max(0, Math.min(bw - 1, ((x + 0.5) * bw) / width - 0.5));
  const fy = (y: number) => Math.max(0, Math.min(bh - 1, ((y + 0.5) * bh) / height - 0.5));
  for (let y = 0; y < height; y++) {
    const sy = fy(y);
    const y0 = sy | 0;
    const y1 = Math.min(bh - 1, y0 + 1);
    const ay = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = fx(x);
      const x0 = sx | 0;
      const x1 = Math.min(bw - 1, x0 + 1);
      const ax = sx - x0;
      const p = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        const bg = backgrounds[c];
        const top = bg[y0 * bw + x0] + (bg[y0 * bw + x1] - bg[y0 * bw + x0]) * ax;
        const bottom = bg[y1 * bw + x0] + (bg[y1 * bw + x1] - bg[y1 * bw + x0]) * ax;
        const b = top + (bottom - top) * ay;
        o[p + c] = (data[p + c] * 255) / Math.max(b, 1);
      }
      o[p + 3] = data[p + 3];
    }
  }
  return out;
}

/** Separable square max filter (morphological dilation of bright regions). */
function maxFilter(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = 0;
      for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++)
        m = Math.max(m, src[y * w + k]);
      tmp[y * w + x] = m;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = 0;
      for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++)
        m = Math.max(m, tmp[k * w + x]);
      out[y * w + x] = m;
    }
  }
  return out;
}

/** Separable box blur with edge-clamped windows. */
function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += src[y * w + Math.max(0, Math.min(w - 1, x + k))];
      tmp[y * w + x] = s / (2 * r + 1);
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += tmp[Math.max(0, Math.min(h - 1, y + k)) * w + x];
      out[y * w + x] = s / (2 * r + 1);
    }
  }
  return out;
}

/** Options for {@link adaptiveThreshold}. */
export interface ThresholdOptions {
  /**
   * `'sauvola'` (default): `T = m · (1 + k · (s / 128 − 1))`, robust for faint strokes on paper.
   * `'bradley'`: `T = m · (1 − k)`, cheaper (no variance) but more sensitive to noise.
   */
  method?: 'sauvola' | 'bradley';
  /** Side of the square local window in px (odd). Default: ~1/24 of the shorter side, ≥ 15. */
  window?: number;
  /** Sensitivity `k`. Default 0.2 for Sauvola, 0.15 for Bradley. */
  k?: number;
}

/**
 * Local adaptive binarization with integral images (O(1) per pixel for any window size). Pixels
 * darker than their local threshold become 0 (ink), all others 255 (paper).
 */
export function adaptiveThreshold(gray: GrayImage, options: ThresholdOptions = {}): GrayImage {
  const { width: w, height: h, data } = gray;
  const method = options.method ?? 'sauvola';
  const k = options.k ?? (method === 'sauvola' ? 0.2 : 0.15);
  const windowSize = options.window ?? Math.max(15, Math.round(Math.min(w, h) / 24) | 1);
  const r = windowSize >> 1;
  const stride = w + 1;
  // Float64 keeps sums exact for any realistic image (sum of squares exceeds 2^32 at ~66 K px).
  const sum = new Float64Array(stride * (h + 1));
  const sumSq = method === 'sauvola' ? new Float64Array(stride * (h + 1)) : null;
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    let rowSq = 0;
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x];
      rowSum += v;
      rowSq += v * v;
      const i = (y + 1) * stride + x + 1;
      sum[i] = sum[i - stride] + rowSum;
      if (sumSq) sumSq[i] = sumSq[i - stride] + rowSq;
    }
  }
  const out = createGray(w, h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w, x + r + 1);
      const a = y0 * stride + x0;
      const b = y0 * stride + x1;
      const c = y1 * stride + x0;
      const d = y1 * stride + x1;
      const count = (x1 - x0) * (y1 - y0);
      const mean = (sum[d] - sum[b] - sum[c] + sum[a]) / count;
      let threshold: number;
      if (sumSq) {
        const variance = (sumSq[d] - sumSq[b] - sumSq[c] + sumSq[a]) / count - mean * mean;
        threshold = mean * (1 + k * (Math.sqrt(Math.max(0, variance)) / 128 - 1));
      } else {
        threshold = mean * (1 - k);
      }
      out.data[y * w + x] = data[y * w + x] < threshold ? 0 : 255;
    }
  }
  return out;
}

/**
 * Applies a {@link ScanFilter} to a rectified page and returns a new image:
 * `'color'` = {@link removeShadows} + {@link enhanceColor}; `'grayscale'` = shadow removal, luma,
 * levels stretch; `'bw'` = shadow removal, luma, Sauvola {@link adaptiveThreshold};
 * `'photo'` = an unmodified copy.
 */
export function applyFilter(rgba: RgbaImage, filter: ScanFilter): RgbaImage {
  switch (filter) {
    case 'photo': {
      const copy = createRgba(rgba.width, rgba.height);
      copy.data.set(rgba.data);
      return copy;
    }
    case 'color':
      return enhanceColor(removeShadows(rgba), { saturation: 1.2 });
    case 'grayscale':
      return enhanceColor(toGrayscale(removeShadows(rgba)), { saturation: 1 });
    case 'bw': {
      const binary = adaptiveThreshold(rgbaToGray(removeShadows(rgba)));
      const out = createRgba(rgba.width, rgba.height);
      for (let i = 0, p = 0; i < binary.data.length; i++, p += 4) {
        const v = binary.data[i];
        out.data[p] = v;
        out.data[p + 1] = v;
        out.data[p + 2] = v;
        out.data[p + 3] = 255;
      }
      return out;
    }
  }
}

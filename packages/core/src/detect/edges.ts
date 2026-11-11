/**
 * Low-level edge extraction on single-channel buffers: separable Gaussian blur, Sobel gradients
 * and Canny (non-maximum suppression + hysteresis). All functions write into caller-provided
 * buffers so that the per-frame detection path does not allocate.
 */

/** Normalized 1D Gaussian kernel with radius `ceil(3σ)`. */
export function gaussianKernel(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return kernel;
}

/**
 * Separable convolution with a symmetric 1D kernel, clamping coordinates at the borders.
 * `tmp` and `out` must hold `width * height` floats; `src` and `out` may not alias.
 */
export function convolveSeparable(
  src: ArrayLike<number>,
  width: number,
  height: number,
  kernel: Float32Array,
  tmp: Float32Array,
  out: Float32Array,
): void {
  const r = (kernel.length - 1) >> 1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      if (x >= r && x < width - r) {
        for (let k = -r; k <= r; k++) acc += src[row + x + k] * kernel[k + r];
      } else {
        for (let k = -r; k <= r; k++) {
          const xx = x + k < 0 ? 0 : x + k >= width ? width - 1 : x + k;
          acc += src[row + xx] * kernel[k + r];
        }
      }
      tmp[row + x] = acc;
    }
  }
  for (let y = 0; y < height; y++) {
    const interior = y >= r && y < height - r;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      if (interior) {
        for (let k = -r; k <= r; k++) acc += tmp[(y + k) * width + x] * kernel[k + r];
      } else {
        for (let k = -r; k <= r; k++) {
          const yy = y + k < 0 ? 0 : y + k >= height ? height - 1 : y + k;
          acc += tmp[yy * width + x] * kernel[k + r];
        }
      }
      out[y * width + x] = acc;
    }
  }
}

/**
 * Sobel gradients normalized to intensity units per pixel (kernel sum of 8 divided out).
 * The one-pixel image border is set to zero.
 */
export function sobel(
  src: Float32Array,
  width: number,
  height: number,
  gx: Float32Array,
  gy: Float32Array,
  mag: Float32Array,
): void {
  gx.fill(0);
  gy.fill(0);
  mag.fill(0);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const a = src[i - width - 1];
      const b = src[i - width];
      const c = src[i - width + 1];
      const d = src[i - 1];
      const f = src[i + 1];
      const g = src[i + width - 1];
      const h = src[i + width];
      const k = src[i + width + 1];
      const dx = (c + 2 * f + k - a - 2 * d - g) * 0.125;
      const dy = (g + 2 * h + k - a - 2 * b - c) * 0.125;
      gx[i] = dx;
      gy[i] = dy;
      mag[i] = Math.sqrt(dx * dx + dy * dy);
    }
  }
}

/** tan(22.5°): boundary between horizontal/vertical and diagonal gradient directions. */
const TAN_22_5 = 0.41421356;

/**
 * Offset (in the flat buffer) of the neighbor along the gradient direction, quantized to one of
 * four directions as in Canny's non-maximum suppression. The opposite neighbor is `-offset`.
 */
export function gradientNeighborOffset(dx: number, dy: number, width: number): number {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (ay <= TAN_22_5 * ax) return 1;
  if (ax <= TAN_22_5 * ay) return width;
  return dx > 0 === dy > 0 ? width + 1 : width - 1;
}

/**
 * Canny non-maximum suppression: `out[i] = mag[i]` where the magnitude is a local maximum along
 * the (quantized) gradient direction, else 0. Ties are broken towards the lower index so that a
 * plateau of two equal pixels yields exactly one edge pixel.
 */
export function nonMaximumSuppression(
  gx: Float32Array,
  gy: Float32Array,
  mag: Float32Array,
  width: number,
  height: number,
  out: Float32Array,
): void {
  out.fill(0);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const m = mag[i];
      if (m === 0) continue;
      const o = gradientNeighborOffset(gx[i], gy[i], width);
      if (m > mag[i - o] && m >= mag[i + o]) out[i] = m;
    }
  }
}

/**
 * Median of the non-zero values of `values` (approximated with a 1/8-unit histogram over
 * `[0, 256)`), used as a robust noise/texture level for automatic Canny thresholds.
 */
export function medianPositive(values: Float32Array, histogram: Uint32Array): number {
  const bins = histogram.length;
  const scale = bins / 256;
  histogram.fill(0);
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v <= 0) continue;
    const b = Math.min(bins - 1, (v * scale) | 0);
    histogram[b]++;
    count++;
  }
  if (count === 0) return 0;
  const half = count / 2;
  let acc = 0;
  for (let b = 0; b < bins; b++) {
    acc += histogram[b];
    if (acc >= half) return (b + 0.5) / scale;
  }
  return 256;
}

/**
 * Canny hysteresis: pixels of `nms` above `high` seed edges which grow 8-connected through
 * pixels above `low`. Writes 1 into `edges` for edge pixels (0 elsewhere) and their flat indices
 * into `list`; returns the number of edge pixels. `list` must hold `width * height` entries and
 * doubles as the flood-fill stack.
 */
export function hysteresis(
  nms: Float32Array,
  width: number,
  height: number,
  low: number,
  high: number,
  edges: Uint8Array,
  list: Int32Array,
): number {
  edges.fill(0);
  let count = 0;
  for (let i = 0; i < nms.length; i++) {
    if (nms[i] < high || edges[i]) continue;
    edges[i] = 1;
    // Edges found so far occupy list[0..count); the region grown from this seed is appended
    // and processed in place, so the list doubles as the BFS queue.
    let head = count;
    list[count++] = i;
    while (head < count) {
      const p = list[head++];
      const x = p % width;
      const y = (p - x) / width;
      const x0 = x > 0 ? -1 : 0;
      const x1 = x < width - 1 ? 1 : 0;
      const y0 = y > 0 ? -1 : 0;
      const y1 = y < height - 1 ? 1 : 0;
      for (let dy = y0; dy <= y1; dy++) {
        for (let dx = x0; dx <= x1; dx++) {
          const q = p + dy * width + dx;
          if (!edges[q] && nms[q] >= low) {
            edges[q] = 1;
            list[count++] = q;
          }
        }
      }
    }
  }
  return count;
}

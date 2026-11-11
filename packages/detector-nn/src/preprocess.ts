import type { RgbaImage } from '@scanoficus/core/image';

/**
 * CPU preprocessing that reproduces the Python reference (`scanoficus_ml.data.preprocess`)
 * bit-exactly on the test frames: the frame is halved with exact 2×2 box means while both sides
 * are even and at least twice the target, then stretched to `size × size` with OpenCV's
 * `INTER_AREA` (fractional coverage weights in float32, round half to even). Returns the RGB
 * planes (CHW) as bytes; the network input is `byte / 255`.
 */
export function preprocessRgba(frame: RgbaImage, size: number): Uint8Array {
  let width = frame.width;
  let height = frame.height;
  if (frame.data.length !== width * height * 4) throw new RangeError('Expected RGBA pixels');
  // Interleaved RGB copy (alpha dropped).
  let rgb: Uint8Array = new Uint8Array(width * height * 3);
  for (let i = 0, p = 0; i < width * height; i++, p += 4) {
    rgb[3 * i] = frame.data[p];
    rgb[3 * i + 1] = frame.data[p + 1];
    rgb[3 * i + 2] = frame.data[p + 2];
  }
  while (width >= 2 * size && height >= 2 * size && width % 2 === 0 && height % 2 === 0) {
    rgb = areaInteger(rgb, width, height, 2, 2);
    width /= 2;
    height /= 2;
  }
  const out = resizeArea(rgb, width, height, size, size);
  // Interleaved → planar.
  const planar = new Uint8Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    planar[i] = out[3 * i];
    planar[plane + i] = out[3 * i + 1];
    planar[2 * plane + i] = out[3 * i + 2];
  }
  return planar;
}

/** Converts CHW bytes to the float network input in [0, 1]. */
export function bytesToInput(
  bytes: Uint8Array,
  out: Float32Array = new Float32Array(bytes.length),
): Float32Array {
  for (let i = 0; i < bytes.length; i++) out[i] = bytes[i] / 255;
  return out;
}

/** OpenCV `cvRound` for floats: round half to even. */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function saturate(v: number): number {
  const r = roundHalfEven(v);
  return Math.min(255, Math.max(0, r));
}

/** `cv::resize(INTER_AREA)` of interleaved RGB bytes (downscaling only). */
function resizeArea(src: Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8Array {
  if (sw === dw && sh === dh) return src.slice();
  const scaleX = 1 / (dw / sw);
  const scaleY = 1 / (dh / sh);
  if (scaleX < 1 || scaleY < 1) throw new RangeError('Upscaling is not supported');
  const ix = Math.round(scaleX);
  const iy = Math.round(scaleY);
  if (Math.abs(scaleX - ix) < Number.EPSILON && Math.abs(scaleY - iy) < Number.EPSILON) {
    return areaInteger(src, sw, sh, ix, iy);
  }
  const xtab = areaTab(sw, dw, scaleX);
  const ytab = areaTab(sh, dh, scaleY);
  const f = Math.fround;
  const rowLen = dw * 3;
  const dst = new Uint8Array(dh * rowLen);
  const buf = new Float32Array(rowLen);
  const sum = new Float32Array(rowLen);
  let prev = ytab.di[0];
  for (let j = 0; j < ytab.di.length; j++) {
    const beta = ytab.alpha[j];
    const dy = ytab.di[j];
    const row = ytab.si[j] * sw * 3;
    buf.fill(0);
    for (let k = 0; k < xtab.di.length; k++) {
      const s = row + xtab.si[k] * 3;
      const d = xtab.di[k] * 3;
      const alpha = xtab.alpha[k];
      // Horizontal pass: fused float32 multiply-add (the product is exact in float64); the
      // vertical pass below rounds the product first. This matches OpenCV 4 on arm64 exactly.
      buf[d] = f(buf[d] + src[s] * alpha);
      buf[d + 1] = f(buf[d + 1] + src[s + 1] * alpha);
      buf[d + 2] = f(buf[d + 2] + src[s + 2] * alpha);
    }
    if (dy !== prev) {
      const o = prev * rowLen;
      for (let x = 0; x < rowLen; x++) {
        dst[o + x] = saturate(sum[x]);
        sum[x] = beta * buf[x];
      }
      prev = dy;
    } else {
      for (let x = 0; x < rowLen; x++) sum[x] = f(sum[x] + f(beta * buf[x]));
    }
  }
  const o = prev * rowLen;
  for (let x = 0; x < rowLen; x++) dst[o + x] = saturate(sum[x]);
  return dst;
}

/** OpenCV's `computeResizeAreaTab`: source index, destination index and weight per tap. */
function areaTab(
  ssize: number,
  dsize: number,
  scale: number,
): { si: Int32Array; di: Int32Array; alpha: Float32Array } {
  const si: number[] = [];
  const di: number[] = [];
  const alpha: number[] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cell = Math.min(scale, ssize - fsx1);
    let sx2 = Math.floor(fsx2);
    let sx1 = Math.ceil(fsx1);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      si.push(sx1 - 1);
      di.push(dx);
      alpha.push((sx1 - fsx1) / cell);
    }
    for (let sx = sx1; sx < sx2; sx++) {
      si.push(sx);
      di.push(dx);
      alpha.push(1 / cell);
    }
    if (fsx2 - sx2 > 1e-3) {
      si.push(sx2);
      di.push(dx);
      alpha.push(Math.min(Math.min(fsx2 - sx2, 1), cell) / cell);
    }
  }
  return { si: Int32Array.from(si), di: Int32Array.from(di), alpha: Float32Array.from(alpha) };
}

/** Integer-factor area resize (`resizeAreaFast`); 2×2 rounds half up like OpenCV's SIMD path. */
function areaInteger(src: Uint8Array, sw: number, sh: number, fx: number, fy: number): Uint8Array {
  const dw = Math.floor(sw / fx);
  const dh = Math.floor(sh / fy);
  const dst = new Uint8Array(dw * dh * 3);
  const area = fx * fy;
  const scale = Math.fround(1 / area);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let v = 0; v < fy; v++) {
          const row = (y * fy + v) * sw;
          for (let u = 0; u < fx; u++) s += src[(row + x * fx + u) * 3 + c];
        }
        dst[(y * dw + x) * 3 + c] = area === 4 ? (s + 2) >> 2 : saturate(Math.fround(s * scale));
      }
    }
  }
  return dst;
}

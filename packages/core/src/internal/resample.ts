/**
 * Separable area-average resampling of a single-channel buffer (downscaling only): every output
 * sample is the exact area-weighted mean of the input samples it covers. Integer destinations
 * (`Uint8Array`) are rounded to nearest; `Uint8ClampedArray` rounds by itself.
 */
export function resampleArea(
  src: ArrayLike<number>,
  srcW: number,
  srcH: number,
  dst: Uint8Array | Uint8ClampedArray | Float32Array,
  dstW: number,
  dstH: number,
): void {
  const bias = dst instanceof Uint8Array ? 0.5 : 0;
  const xTaps = areaTaps(srcW, dstW);
  const yTaps = areaTaps(srcH, dstH);
  // Horizontal pass into a float buffer of dstW x srcH.
  const tmp = new Float32Array(dstW * srcH);
  for (let y = 0; y < srcH; y++) {
    const row = y * srcW;
    for (let x = 0; x < dstW; x++) {
      const start = xTaps.start[x];
      const end = xTaps.start[x + 1];
      let sum = 0;
      for (let t = start; t < end; t++) sum += src[row + xTaps.index[t]] * xTaps.weight[t];
      tmp[y * dstW + x] = sum;
    }
  }
  // Vertical pass.
  for (let y = 0; y < dstH; y++) {
    const start = yTaps.start[y];
    const end = yTaps.start[y + 1];
    for (let x = 0; x < dstW; x++) {
      let sum = 0;
      for (let t = start; t < end; t++) sum += tmp[yTaps.index[t] * dstW + x] * yTaps.weight[t];
      dst[y * dstW + x] = sum + bias;
    }
  }
}

interface AreaTaps {
  /** `start[i]..start[i + 1]` indexes the taps of output sample `i`. */
  start: Int32Array;
  index: Int32Array;
  weight: Float32Array;
}

/** Computes, for each output sample, the covered input samples and their normalized weights. */
function areaTaps(srcLen: number, dstLen: number): AreaTaps {
  const ratio = srcLen / dstLen;
  const maxTaps = Math.ceil(ratio) + 1;
  const start = new Int32Array(dstLen + 1);
  const index = new Int32Array(dstLen * maxTaps);
  const weight = new Float32Array(dstLen * maxTaps);
  let n = 0;
  for (let i = 0; i < dstLen; i++) {
    start[i] = n;
    const a = i * ratio;
    const b = Math.min(srcLen, (i + 1) * ratio);
    for (let s = Math.floor(a); s < b; s++) {
      const w = Math.min(b, s + 1) - Math.max(a, s);
      if (w <= 1e-9) continue;
      index[n] = s;
      weight[n] = w / ratio;
      n++;
    }
  }
  start[dstLen] = n;
  return { start, index, weight };
}

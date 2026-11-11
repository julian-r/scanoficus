import { orderQuad, type Point, type Quad } from '@scanoficus/core/geometry';

/** Decoded network output for one frame. */
export interface NnOutput {
  /**
   * The predicted quad (TL, TR, BR, BL) in frame pixels, regardless of presence. Corners may lie
   * outside the frame.
   */
  quad: Quad;
  /** Document-presence probability in [0, 1] (sigmoid of the presence logit). */
  presence: number;
  /** The 9 raw outputs: x0, y0, …, x3, y3 normalized to the frame, then the presence logit. */
  heads: Float32Array;
}

/** Maps the 9 raw outputs to a quad in a `width × height` frame and a presence probability. */
export function decodeHeads(heads: Float32Array, width: number, height: number): NnOutput {
  const points: Point[] = [];
  for (let i = 0; i < 4; i++)
    points.push({ x: heads[2 * i] * width, y: heads[2 * i + 1] * height });
  return { quad: orderQuad(points), presence: 1 / (1 + Math.exp(-heads[8])), heads };
}

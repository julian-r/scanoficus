/**
 * Planar geometry for document scanning: points and quads, corner ordering and validation,
 * homographies, convex polygon intersection and output size estimation.
 *
 * @module
 */
export {
  applyHomography,
  homographyFromQuads,
  IDENTITY3,
  invertMatrix3,
  type Matrix3,
  multiplyMatrix3,
  rectToQuadHomography,
} from './homography.ts';
export { clipConvexPolygon, convexIoU } from './polygon.ts';
export {
  distance,
  isConvexQuad,
  maxCornerDistance,
  orderQuad,
  type Point,
  type Quad,
  quadAngles,
  quadArea,
  quadSideLengths,
  scaleQuad,
  signedArea,
} from './quad.ts';
export { estimateOutputSize, type OutputSizeOptions, rectangleAspectRatio } from './size.ts';

/**
 * Perspective rectification: a CPU reference implementation (works anywhere, including Node)
 * and a WebGL2 implementation for full-resolution captures in the browser.
 *
 * @module
 */
export { rectifyCpu } from './cpu.ts';
export {
  createGlRectifier,
  type GlRectifier,
  type GlRectifyPlan,
  planGlRectify,
  rectifyShaderMatrix,
} from './gl.ts';

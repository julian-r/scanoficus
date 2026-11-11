/**
 * Page rendering: rectification, scan filter and rotation in one step, on the GPU (WebGL2) with
 * a CPU fallback, plus image encoding.
 *
 * @module
 */
export {
  normalizeRotation,
  type PageRenderOptions,
  type Rotation,
  renderPageCpu,
  rotateRgba,
} from './cpu.ts';
export {
  createGlPageRenderer,
  type GlPageRenderer,
  type GlPageSource,
  pageShaderMatrices,
} from './gl.ts';
export { computeFilterPlan, type FilterPlan } from './plan.ts';
export {
  createPageRenderer,
  type EncodeOptions,
  encodeCanvas,
  type PageRenderer,
  type PageRendererOptions,
  type PageSource,
} from './renderer.ts';

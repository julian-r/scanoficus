import { type Matrix3, multiplyMatrix3, rectToQuadHomography } from '../geometry/homography.ts';
import type { Quad } from '../geometry/quad.ts';
import type { RgbaImage } from '../image.ts';
import { FULLSCREEN_VERTEX_SHADER, linkProgram, readFramebuffer } from '../internal/gl.ts';
import { planGlRectify } from '../warp/gl.ts';
import { type PageRenderOptions, type Rotation, rotatedToUnrotated } from './cpu.ts';
import { computeFilterPlan, type FilterPlan, PHOTO_PLAN } from './plan.ts';

/** Long side of the preview from which filter statistics are computed (larger for B&W). */
const PREVIEW_SIZE = 384;
const PREVIEW_SIZE_BW = 768;

/** Matrices (column-major) used by the page shader for one output canvas. */
export interface PageShaderMatrices {
  /** `gl_FragCoord` → homogeneous normalized source texture coordinates. */
  source: Float32Array;
  /** `gl_FragCoord` → normalized coordinates of the unrotated page (for filter textures). */
  page: Float32Array;
}

/**
 * Builds the shader matrices for rendering the rectified, rotated page into a canvas of the
 * rotated size. With `F` = flip from bottom-left fragment coordinates to the rotated top-left
 * pixel frame and `R` = rotated → unrotated page coordinates:
 * `source = Nsrc · H · R · F` and `page = Npage · R · F`, where `H` maps the page rectangle onto
 * `quad` and `N*` normalize by the respective size.
 */
export function pageShaderMatrices(
  quad: Quad,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  rotation: Rotation,
): PageShaderMatrices {
  const h = rectToQuadHomography(width, height, quad);
  if (!h) throw new RangeError('Cannot render a degenerate quad');
  const canvasHeight = rotation === 90 || rotation === 270 ? width : height;
  const flip: Matrix3 = [1, 0, 0, 0, -1, canvasHeight, 0, 0, 1];
  const toPage = multiplyMatrix3(rotatedToUnrotated(rotation, width, height), flip);
  const source = multiplyMatrix3(
    [1 / sourceWidth, 0, 0, 0, 1 / sourceHeight, 0, 0, 0, 1],
    multiplyMatrix3(h, toPage),
  );
  const page = multiplyMatrix3([1 / width, 0, 0, 0, 1 / height, 0, 0, 0, 1], toPage);
  return { source: columnMajor(source), page: columnMajor(page) };
}

function columnMajor(m: Matrix3): Float32Array {
  return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
}

/** Image sources the GPU renderer accepts (RGBA pixels are uploaded directly). */
export type GlPageSource =
  | ImageBitmap
  | HTMLCanvasElement
  | OffscreenCanvas
  | HTMLImageElement
  | RgbaImage;

/** WebGL2 page renderer bound to one canvas. */
export interface GlPageRenderer {
  /**
   * Renders the page into the canvas (resized to the rotated output size, possibly reduced to
   * respect GPU limits as described for `planGlRectify`).
   */
  render(source: GlPageSource, options: PageRenderOptions): void;
  dispose(): void;
}

// Mirrors applyFilter: photo, color (shadow removal + levels + saturation), grayscale, bw.
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform mat3 u_source_map;
uniform mat3 u_page_map;
uniform sampler2D u_source;
uniform sampler2D u_background;
uniform sampler2D u_statistics;
uniform int u_mode;
uniform vec3 u_black;
uniform vec3 u_white;
uniform float u_saturation;
uniform float u_k;
out vec4 outColor;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
void main() {
  vec3 s = u_source_map * vec3(gl_FragCoord.xy, 1.0);
  vec3 c = texture(u_source, s.xy / s.z).rgb;
  if (u_mode == 0) { outColor = vec4(c, 1.0); return; }
  vec2 p = (u_page_map * vec3(gl_FragCoord.xy, 1.0)).xy;
  vec3 flat_ = min(c / max(texture(u_background, p).rgb, vec3(1.0 / 255.0)), vec3(1.0));
  if (u_mode == 3) {
    vec2 st = texture(u_statistics, p).rg;
    float t = st.r * (1.0 + u_k * (st.g * (255.0 / 128.0) - 1.0));
    outColor = vec4(vec3(dot(flat_, LUMA) < t ? 0.0 : 1.0), 1.0);
    return;
  }
  vec3 v = clamp((flat_ - u_black) / max(u_white - u_black, vec3(1.0 / 255.0)), 0.0, 1.0);
  float y = dot(v, LUMA);
  v = u_mode == 2 ? vec3(y) : clamp(vec3(y) + (v - vec3(y)) * u_saturation, 0.0, 1.0);
  outColor = vec4(v, 1.0);
}`;

/**
 * Creates a WebGL2 renderer that rectifies, filters and rotates a page in one full-resolution
 * pass. Non-local filter statistics are computed on the CPU from a small preview rendered with
 * the same shader (see {@link computeFilterPlan}). Returns `null` without WebGL2.
 */
export function createGlPageRenderer(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): GlPageRenderer | null {
  const gl = canvas.getContext('webgl2', {
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
    antialias: false,
  }) as WebGL2RenderingContext | null;
  if (!gl) return null;
  const program = linkProgram(gl, FULLSCREEN_VERTEX_SHADER, FRAGMENT_SHADER);
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const textures = [gl.createTexture(), gl.createTexture(), gl.createTexture()];
  textures.forEach((texture, unit) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  });
  gl.useProgram(program);
  const uniform = (name: string) => gl.getUniformLocation(program, name);
  gl.uniform1i(uniform('u_source'), 0);
  gl.uniform1i(uniform('u_background'), 1);
  gl.uniform1i(uniform('u_statistics'), 2);
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
  const maxRenderSize = Math.min(
    maxTextureSize,
    gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number,
    viewport[0],
    viewport[1],
  );
  let scratch: HTMLCanvasElement | OffscreenCanvas | null = null;

  const upload = (unit: number, image: TexImageSource | RgbaImage) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, textures[unit]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    if ('data' in image) {
      const { width, height, data } = image;
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    }
  };

  const draw = (
    plan: FilterPlan,
    matrices: PageShaderMatrices,
    canvasWidth: number,
    canvasHeight: number,
  ) => {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    gl.viewport(0, 0, canvasWidth, canvasHeight);
    gl.uniformMatrix3fv(uniform('u_source_map'), false, matrices.source);
    gl.uniformMatrix3fv(uniform('u_page_map'), false, matrices.page);
    gl.uniform1i(uniform('u_mode'), plan.mode);
    gl.uniform3fv(
      uniform('u_black'),
      plan.black.map((v) => v / 255),
    );
    gl.uniform3fv(
      uniform('u_white'),
      plan.white.map((v) => v / 255),
    );
    gl.uniform1f(uniform('u_saturation'), plan.saturation);
    gl.uniform1f(uniform('u_k'), plan.k);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  return {
    render(source, options) {
      const sourceWidth = 'naturalWidth' in source ? source.naturalWidth : source.width;
      const sourceHeight = 'naturalHeight' in source ? source.naturalHeight : source.height;
      const sizes = planGlRectify(
        sourceWidth,
        sourceHeight,
        options.width,
        options.height,
        maxTextureSize,
        maxRenderSize,
      );
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      if (sizes.sourceScale < 1) {
        if ('data' in source) throw new RangeError('RGBA source exceeds MAX_TEXTURE_SIZE');
        scratch ??=
          typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(1, 1)
            : document.createElement('canvas');
        scratch.width = sizes.textureWidth;
        scratch.height = sizes.textureHeight;
        const ctx = scratch.getContext('2d') as CanvasRenderingContext2D | null;
        if (!ctx) throw new Error('2D canvas unavailable for downscaling');
        ctx.drawImage(source, 0, 0, sizes.textureWidth, sizes.textureHeight);
        upload(0, scratch);
      } else {
        upload(0, source);
      }
      const { quad, filter, rotation } = options;
      const { outWidth: width, outHeight: height } = sizes;
      let plan = PHOTO_PLAN;
      if (filter !== 'photo') {
        const previewSide = filter === 'bw' ? PREVIEW_SIZE_BW : PREVIEW_SIZE;
        const k = Math.min(1, previewSide / Math.max(width, height));
        const pw = Math.max(1, Math.round(width * k));
        const ph = Math.max(1, Math.round(height * k));
        draw(plan, pageShaderMatrices(quad, sourceWidth, sourceHeight, pw, ph, 0), pw, ph);
        plan = computeFilterPlan(readFramebuffer(gl, pw, ph), filter, width, height);
        if (plan.background) upload(1, plan.background);
        if (plan.statistics) upload(2, plan.statistics);
      }
      const quarter = rotation === 90 || rotation === 270;
      draw(
        plan,
        pageShaderMatrices(quad, sourceWidth, sourceHeight, width, height, rotation),
        quarter ? height : width,
        quarter ? width : height,
      );
    },
    dispose() {
      for (const texture of textures) gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      if (scratch) scratch.width = scratch.height = 0;
    },
  };
}

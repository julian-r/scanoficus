import { multiplyMatrix3, rectToQuadHomography } from '../geometry/homography.ts';
import type { Quad } from '../geometry/quad.ts';
import type { RgbaImage } from '../image.ts';
import { FULLSCREEN_VERTEX_SHADER, linkProgram, readFramebuffer } from '../internal/gl.ts';

/** Sizes chosen by {@link planGlRectify} for one rectification under GPU limits. */
export interface GlRectifyPlan {
  /** Factor applied to the source before upload (1 = full resolution). */
  sourceScale: number;
  /** Size of the uploaded source texture. */
  textureWidth: number;
  textureHeight: number;
  /** Size of the rendered output (the requested size, scaled down if it exceeds the limits). */
  outWidth: number;
  outHeight: number;
}

/**
 * Chooses texture and output sizes that respect the GPU limits.
 *
 * Strategy (downscaling, not tiling): if the source's longer side exceeds `maxTextureSize`, the
 * source is uniformly downscaled to fit before upload; if the requested output exceeds
 * `maxRenderSize` (the smaller of `MAX_TEXTURE_SIZE`, `MAX_RENDERBUFFER_SIZE` and the viewport
 * limit), the output is uniformly scaled down too. Older iPhones report 4096, recent ones up to
 * 16384 (query, never assume), so a 12 MP photo (4032 px) normally fits unchanged; only 48 MP
 * frames on 4096-limited GPUs lose resolution. Tiling would preserve those pixels but multiplies
 * code and memory traffic for a case where the output is capped anyway.
 */
export function planGlRectify(
  sourceWidth: number,
  sourceHeight: number,
  outWidth: number,
  outHeight: number,
  maxTextureSize: number,
  maxRenderSize: number = maxTextureSize,
): GlRectifyPlan {
  const sourceScale = Math.min(1, maxTextureSize / Math.max(sourceWidth, sourceHeight));
  const outScale = Math.min(1, maxRenderSize / Math.max(outWidth, outHeight));
  return {
    sourceScale,
    textureWidth: Math.max(1, Math.floor(sourceWidth * sourceScale)),
    textureHeight: Math.max(1, Math.floor(sourceHeight * sourceScale)),
    outWidth: Math.max(1, Math.floor(outWidth * outScale)),
    outHeight: Math.max(1, Math.floor(outHeight * outScale)),
  };
}

/**
 * The 3×3 matrix (column-major, ready for `uniformMatrix3fv`) that maps a fragment's
 * `gl_FragCoord.xy` in an `outWidth × outHeight` framebuffer (origin bottom-left) to normalized
 * texture coordinates of the source (origin at the first uploaded row, i.e. the image top), as a
 * homogeneous vector to be divided by its `z`.
 *
 * It is `N · H · F`: `F` flips y to a top-left origin, `H` maps the output rectangle onto `quad`
 * (pixel coordinates of a `sourceWidth × sourceHeight` source) and `N` normalizes by the source
 * size. Because of `N`, a uniformly downscaled texture (see {@link planGlRectify}) can use the
 * same matrix as the original source.
 */
export function rectifyShaderMatrix(
  quad: Quad,
  sourceWidth: number,
  sourceHeight: number,
  outWidth: number,
  outHeight: number,
): Float32Array {
  const h = rectToQuadHomography(outWidth, outHeight, quad);
  if (!h) throw new RangeError('Cannot rectify a degenerate quad');
  const flip = [1, 0, 0, 0, -1, outHeight, 0, 0, 1] as const;
  const normalize = [1 / sourceWidth, 0, 0, 0, 1 / sourceHeight, 0, 0, 0, 1] as const;
  const m = multiplyMatrix3(normalize, multiplyMatrix3(h, flip));
  // Row-major → column-major.
  return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
}

/** WebGL2 perspective rectifier bound to one canvas. */
export interface GlRectifier {
  /** `MAX_TEXTURE_SIZE` of the context. */
  readonly maxTextureSize: number;
  /**
   * Renders the rectified `quad` region of `source` into the canvas, which is resized to the
   * planned output size (see {@link planGlRectify}). Returns the plan actually used.
   *
   * @param source - Image, bitmap, video frame, canvas or `ImageData`.
   * @param quad - Corners (TL, TR, BR, BL) in `source` pixel coordinates.
   */
  rectify(
    source: TexImageSource,
    sourceWidth: number,
    sourceHeight: number,
    quad: Quad,
    outWidth: number,
    outHeight: number,
  ): GlRectifyPlan;
  /** Reads the current canvas content back as an {@link RgbaImage} (top row first). */
  readPixels(): RgbaImage;
  /** Releases all GPU resources. The rectifier must not be used afterwards. */
  dispose(): void;
}

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform mat3 u_map;
uniform sampler2D u_source;
out vec4 outColor;
void main() {
  vec3 p = u_map * vec3(gl_FragCoord.xy, 1.0);
  outColor = texture(u_source, p.xy / p.z);
}`;

/**
 * Creates a WebGL2 rectifier rendering into `canvas` (an `HTMLCanvasElement` or an
 * `OffscreenCanvas`, e.g. inside a worker). The warp is a single full-screen triangle whose
 * fragment shader applies {@link rectifyShaderMatrix} and samples the source bilinearly with
 * edge clamping, matching {@link rectifyCpu}. Returns `null` if WebGL2 is unavailable, in which
 * case callers should fall back to `rectifyCpu`.
 */
export function createGlRectifier(canvas: HTMLCanvasElement | OffscreenCanvas): GlRectifier | null {
  const gl = canvas.getContext('webgl2', {
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
    antialias: false,
  }) as WebGL2RenderingContext | null;
  if (!gl) return null;
  const program = linkProgram(gl, FULLSCREEN_VERTEX_SHADER, FRAGMENT_SHADER);
  const buffer = gl.createBuffer();
  const texture = gl.createTexture();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  for (const [k, v] of [
    [gl.TEXTURE_MIN_FILTER, gl.LINEAR],
    [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
    [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
    [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
  ]) {
    gl.texParameteri(gl.TEXTURE_2D, k, v);
  }
  const mapLocation = gl.getUniformLocation(program, 'u_map');
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
  const maxRenderSize = Math.min(
    maxTextureSize,
    gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number,
    viewport[0],
    viewport[1],
  );
  let scratch: HTMLCanvasElement | OffscreenCanvas | null = null;

  /** Uniformly downscaled copy of an oversized source via a 2D canvas. */
  const downscale = (source: TexImageSource, w: number, h: number): TexImageSource => {
    if (source instanceof ImageData) {
      throw new RangeError('ImageData sources larger than MAX_TEXTURE_SIZE are not supported');
    }
    if (!scratch) {
      scratch =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(w, h)
          : document.createElement('canvas');
    }
    scratch.width = w;
    scratch.height = h;
    const ctx = scratch.getContext('2d') as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error('2D canvas unavailable for downscaling');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, w, h);
    return scratch;
  };

  return {
    maxTextureSize,
    rectify(source, sourceWidth, sourceHeight, quad, outWidth, outHeight) {
      const plan = planGlRectify(
        sourceWidth,
        sourceHeight,
        outWidth,
        outHeight,
        maxTextureSize,
        maxRenderSize,
      );
      const upload =
        plan.sourceScale < 1 ? downscale(source, plan.textureWidth, plan.textureHeight) : source;
      canvas.width = plan.outWidth;
      canvas.height = plan.outHeight;
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, upload);
      gl.uniformMatrix3fv(
        mapLocation,
        false,
        rectifyShaderMatrix(quad, sourceWidth, sourceHeight, plan.outWidth, plan.outHeight),
      );
      gl.viewport(0, 0, plan.outWidth, plan.outHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return plan;
    },
    readPixels() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return readFramebuffer(gl, canvas.width, canvas.height);
    },
    dispose() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      if (scratch) scratch.width = scratch.height = 0;
    },
  };
}

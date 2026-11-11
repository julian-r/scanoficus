import { createRgba, type RgbaImage } from '../image.ts';

/** Shared WebGL2 helpers. */

/** Vertex shader drawing one full-screen triangle from `a_position`. */
export const FULLSCREEN_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); }`;

/** Reads the framebuffer as an RgbaImage, top row first. */
export function readFramebuffer(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
): RgbaImage {
  const raw = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
  const out = createRgba(width, height);
  const row = width * 4;
  for (let y = 0; y < height; y++) {
    out.data.set(raw.subarray((height - 1 - y) * row, (height - y) * row), y * row);
  }
  return out;
}

/** Compiles and links a vertex/fragment shader pair, throwing with the info log on failure. */
export function linkProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string,
): WebGLProgram {
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, vertexSource],
    [gl.FRAGMENT_SHADER, fragmentSource],
  ] as const) {
    const shader = gl.createShader(type);
    if (!shader) throw new Error('Failed to create shader');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`Shader compile error: ${gl.getShaderInfoLog(shader)}`);
    }
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

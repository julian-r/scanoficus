import type { RgbaImage } from '../image.ts';
import { type PageRenderOptions, renderPageCpu } from './cpu.ts';
import { createGlPageRenderer, type GlPageRenderer, type GlPageSource } from './gl.ts';

/** Anything a page can be rendered from: decoded images, canvases or RGBA pixels. */
export type PageSource = GlPageSource;

/** Output encoding of {@link PageRenderer.render}. */
export interface EncodeOptions {
  /** MIME type. Default `'image/jpeg'` (what the PDF writer embeds). */
  type?: string;
  /** Encoder quality in [0, 1] for lossy types. Default 0.9. */
  quality?: number;
}

/** Renders pages (rectify + filter + rotate) and encodes them. */
export interface PageRenderer {
  /** Backend in use; switches from `'webgl2'` to `'cpu'` if the GPU path fails. */
  readonly backend: 'webgl2' | 'cpu';
  render(source: PageSource, options: PageRenderOptions, encode?: EncodeOptions): Promise<Blob>;
  /** Releases GPU resources and canvases. */
  dispose(): void;
}

/** Options for {@link createPageRenderer}. */
export interface PageRendererOptions {
  /** Force a backend. Default `'auto'`: WebGL2 when available, else CPU. */
  backend?: 'auto' | 'webgl2' | 'cpu';
}

type AnyCanvas = HTMLCanvasElement | OffscreenCanvas;

function createCanvas(): AnyCanvas {
  return typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(1, 1)
    : document.createElement('canvas');
}

/** Encodes a canvas with `convertToBlob` (OffscreenCanvas) or `toBlob`. */
export function encodeCanvas(canvas: AnyCanvas, options: EncodeOptions = {}): Promise<Blob> {
  const type = options.type ?? 'image/jpeg';
  const quality = options.quality ?? 0.9;
  if ('convertToBlob' in canvas) return canvas.convertToBlob({ type, quality });
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Canvas encoding failed'))),
      type,
      quality,
    ),
  );
}

/**
 * Creates a {@link PageRenderer}. The WebGL2 backend renders full resolution in one shader pass
 * (see `createGlPageRenderer`); if WebGL2 is unavailable, the context is lost or rendering throws,
 * the renderer permanently falls back to the CPU reference pipeline (`renderPageCpu`), which is
 * slower and needs the source pixels in memory.
 */
export function createPageRenderer(options: PageRendererOptions = {}): PageRenderer {
  const backend = options.backend ?? 'auto';
  let glCanvas: AnyCanvas | null = null;
  let gl: GlPageRenderer | null = null;
  if (backend !== 'cpu') {
    glCanvas = createCanvas();
    gl = createGlPageRenderer(glCanvas);
    glCanvas.addEventListener('webglcontextlost', () => {
      gl = null;
    });
    if (!gl && backend === 'webgl2') throw new Error('WebGL2 is not available');
  }
  let cpuCanvas: AnyCanvas | null = null;

  const toRgba = (source: PageSource): RgbaImage => {
    if ('data' in source) return source;
    cpuCanvas ??= createCanvas();
    const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
    const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
    cpuCanvas.width = width;
    cpuCanvas.height = height;
    const ctx = cpuCanvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.drawImage(source, 0, 0);
    return ctx.getImageData(0, 0, width, height);
  };

  const renderer: PageRenderer = {
    get backend() {
      return gl ? 'webgl2' : 'cpu';
    },
    async render(source, renderOptions, encode) {
      if (gl && glCanvas) {
        try {
          gl.render(source, renderOptions);
          return await encodeCanvas(glCanvas, encode);
        } catch (error) {
          if (backend === 'webgl2') throw error;
          gl.dispose();
          gl = null;
        }
      }
      const page = renderPageCpu(toRgba(source), renderOptions);
      cpuCanvas ??= createCanvas();
      cpuCanvas.width = page.width;
      cpuCanvas.height = page.height;
      const ctx = cpuCanvas.getContext('2d') as CanvasRenderingContext2D;
      const pixels = page.data as Uint8ClampedArray<ArrayBuffer>;
      ctx.putImageData(new ImageData(pixels, page.width, page.height), 0, 0);
      return encodeCanvas(cpuCanvas, encode);
    },
    dispose() {
      gl?.dispose();
      gl = null;
      for (const canvas of [glCanvas, cpuCanvas]) if (canvas) canvas.width = canvas.height = 0;
    },
  };
  return renderer;
}

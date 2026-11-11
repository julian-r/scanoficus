import { fitLongSide } from '../image.ts';

/** Options for {@link startFrameLoop}. */
export interface FrameLoopOptions {
  /** Upper bound on callbacks per second. Default 30. */
  maxFps?: number;
}

/**
 * Calls `onFrame` for newly presented video frames, at most `maxFps` times per second. Uses
 * `requestVideoFrameCallback` (Safari 15.4+, Chrome 83+), which fires only for new frames; falls
 * back to `requestAnimationFrame`, skipping ticks where `currentTime` did not advance. Returns a
 * function that stops the loop.
 */
export function startFrameLoop(
  video: HTMLVideoElement,
  onFrame: (timeMs: number) => void,
  options: FrameLoopOptions = {},
): () => void {
  const minInterval = 1000 / (options.maxFps ?? 30) - 1;
  let last = Number.NEGATIVE_INFINITY;
  let stopped = false;
  let handle = 0;
  const tick = (now: number) => {
    if (stopped) return;
    if (now - last >= minInterval) {
      last = now;
      onFrame(now);
    }
    schedule();
  };
  const hasRvfc = typeof video.requestVideoFrameCallback === 'function';
  let lastMediaTime = -1;
  const schedule = () => {
    if (hasRvfc) {
      handle = video.requestVideoFrameCallback(tick);
      return;
    }
    handle = requestAnimationFrame((now) => {
      if (video.currentTime === lastMediaTime) {
        schedule();
        return;
      }
      lastMediaTime = video.currentTime;
      tick(now);
    });
  };
  schedule();
  return () => {
    stopped = true;
    if (hasRvfc) video.cancelVideoFrameCallback(handle);
    else cancelAnimationFrame(handle);
  };
}

/** Sources {@link FrameGrabber} can read: a playing video or a decoded image. */
export type GrabSource = HTMLVideoElement | ImageBitmap | HTMLCanvasElement | OffscreenCanvas;

/**
 * Extracts downscaled RGBA frames from a video (or a decoded still) for detection, reusing one
 * 2D canvas. Drawing with `drawImage` lets the browser downscale on the GPU; only the small
 * result is read back.
 */
export class FrameGrabber {
  private canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

  /**
   * Returns the current image of `source` with its longer side at most `maxSide`, or `null` if
   * a video has no frame yet.
   */
  grab(source: GrabSource, maxSide: number): ImageData | null {
    const isVideo = 'videoWidth' in source;
    const sourceWidth = isVideo ? source.videoWidth : source.width;
    const sourceHeight = isVideo ? source.videoHeight : source.height;
    if (sourceWidth === 0 || sourceHeight === 0) return null;
    const { width, height } = fitLongSide(sourceWidth, sourceHeight, maxSide);
    if (!this.canvas) {
      this.canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(width, height)
          : document.createElement('canvas');
      this.context = this.canvas.getContext('2d', { willReadFrequently: true }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
    }
    const ctx = this.context;
    if (!ctx) return null;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    ctx.drawImage(source, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  }

  /** Releases the canvas memory. */
  dispose(): void {
    if (this.canvas) this.canvas.width = this.canvas.height = 0;
    this.canvas = null;
    this.context = null;
  }
}

/**
 * Calls `onHidden` when the page is hidden or being unloaded/frozen (`visibilitychange` to
 * hidden, `pagehide`) and `onVisible` when it becomes visible again (`visibilitychange`,
 * `pageshow` from the back/forward cache). Returns an unsubscribe function.
 */
export function watchPageVisibility(onHidden: () => void, onVisible: () => void): () => void {
  const visibility = () => (document.visibilityState === 'hidden' ? onHidden() : onVisible());
  const show = (event: PageTransitionEvent) => {
    if (event.persisted && document.visibilityState === 'visible') onVisible();
  };
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('pagehide', onHidden);
  window.addEventListener('pageshow', show);
  return () => {
    document.removeEventListener('visibilitychange', visibility);
    window.removeEventListener('pagehide', onHidden);
    window.removeEventListener('pageshow', show);
  };
}

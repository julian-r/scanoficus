import {
  type Detection,
  type DetectOptions,
  DetectorContext,
  detectDocument,
} from '../detect/detector.ts';
import { scaleQuad } from '../geometry/quad.ts';
import { fitLongSide, type RgbaImage } from '../image.ts';

/**
 * A camera frame handed to a {@link DocumentDetector}: an `ImageBitmap` (e.g. from
 * `createImageBitmap(video)`) or RGBA pixels such as `ImageData`.
 */
export type DetectorFrame = ImageBitmap | RgbaImage;

/**
 * Asynchronous document detector. This is the plugin point for alternative detectors (e.g. a
 * future ML model): the scanner only depends on this interface.
 *
 * Ownership: `ImageBitmap` frames are consumed — the detector closes them once processed (in a
 * worker after transfer), so callers must not use them afterwards. RGBA frames are not modified
 * unless an implementation documents otherwise (see `transferPixels` of `createDetectorWorker`).
 */
export interface DocumentDetector {
  /**
   * Detects the document in `frame`. Resolves with the quad in the frame's pixel coordinates,
   * or `null` if none was found.
   */
  detect(frame: DetectorFrame, options?: DetectOptions): Promise<Detection | null>;
  /**
   * Optional: detection on a full-resolution still (a captured photo or an image given to
   * `addImage`). When present, the `ScannerController` calls it for the still re-detection
   * instead of `detect()` on a frame downscaled to `maxSide`, so a detector can use the full
   * resolution (e.g. for corner refinement). The frame is a copy owned by the detector
   * (`ImageBitmap`, consumed) or the caller's RGBA pixels (not modified). If it rejects, the
   * controller falls back to `detect()` on the downscaled frame.
   */
  detectStill?(still: DetectorFrame, options?: DetectOptions): Promise<Detection | null>;
  /** Releases resources (terminates workers). Pending detections reject. */
  dispose(): void;
}

/** Request message sent to the detector worker. */
export interface DetectRequest {
  id: number;
  frame: DetectorFrame;
  options?: DetectOptions;
}

/** Response message posted by the detector worker. */
export type DetectResponse =
  | { id: number; detection: Detection | null }
  | { id: number; error: string };

/** Any message posted by the detector worker: `{ ready: true }` once loaded, then responses. */
export type DetectorWorkerMessage = DetectResponse | { ready: true };

type Canvas2D = OffscreenCanvas | HTMLCanvasElement;

/**
 * Reads frames at detection resolution and runs {@link detectDocument} on the RGBA pixels (luma
 * plus colour edges), reusing all buffers between frames. Shared by the worker entry and the main-thread detector.
 */
export class FrameProcessor {
  private readonly context = new DetectorContext();
  private canvas: Canvas2D | null = null;

  /** Detects a document in `frame`; closes `ImageBitmap` frames. */
  process(frame: DetectorFrame, options: DetectOptions = {}): Detection | null {
    const maxSide = options.maxSide ?? 384;
    if (isImageBitmap(frame)) {
      try {
        const pixels = this.bitmapToPixels(frame, maxSide);
        const detection = detectDocument(pixels, options, this.context);
        return (
          detection &&
          scaleDetection(detection, frame.width / pixels.width, frame.height / pixels.height)
        );
      } finally {
        frame.close();
      }
    }
    if (
      !(frame.data instanceof Uint8ClampedArray) ||
      frame.data.length !== frame.width * frame.height * 4
    ) {
      throw new TypeError('Expected an ImageBitmap or RGBA pixels with ImageData layout');
    }
    // detectDocument area-downscales internally, uses colour edges and returns frame coordinates.
    return detectDocument(frame, options, this.context);
  }

  /** Draws the bitmap downscaled by the browser (cheap, GPU-assisted) and reads it back. */
  private bitmapToPixels(bitmap: ImageBitmap, maxSide: number): RgbaImage {
    const size = fitLongSide(bitmap.width, bitmap.height, maxSide);
    if (!this.canvas) {
      this.canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(size.width, size.height)
          : document.createElement('canvas');
    }
    const canvas = this.canvas;
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
      | OffscreenCanvasRenderingContext2D
      | CanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error('2D canvas context unavailable');
    ctx.drawImage(bitmap, 0, 0, size.width, size.height);
    return ctx.getImageData(0, 0, size.width, size.height);
  }
}

/** Runs one request through `processor`, converting failures into error responses. */
export function handleDetectRequest(
  processor: FrameProcessor,
  request: DetectRequest,
): DetectResponse {
  try {
    return { id: request.id, detection: processor.process(request.frame, request.options) };
  } catch (error) {
    return { id: request.id, error: error instanceof Error ? error.message : String(error) };
  }
}

function isImageBitmap(frame: DetectorFrame): frame is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && frame instanceof ImageBitmap;
}

function scaleDetection(detection: Detection, sx: number, sy: number): Detection {
  const scaled: Detection = { quad: scaleQuad(detection.quad, sx, sy), score: detection.score };
  if (detection.candidates) {
    scaled.candidates = detection.candidates.map((c) => ({
      quad: scaleQuad(c.quad, sx, sy),
      score: c.score,
    }));
  }
  return scaled;
}

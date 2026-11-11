import {
  type Detection,
  type DetectOptions,
  DetectorContext,
  detectDocument,
} from '../detect/detector.ts';
import { scaleQuad } from '../geometry/quad.ts';
import { fitLongSide, type GrayImage, type RgbaImage, rgbaToGray } from '../image.ts';

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

type Canvas2D = OffscreenCanvas | HTMLCanvasElement;

/**
 * Converts frames to gray at detection resolution and runs {@link detectDocument}, reusing all
 * buffers between frames. Shared by the worker entry and the main-thread detector.
 */
export class FrameProcessor {
  private readonly context = new DetectorContext();
  private canvas: Canvas2D | null = null;

  /** Detects a document in `frame`; closes `ImageBitmap` frames. */
  process(frame: DetectorFrame, options: DetectOptions = {}): Detection | null {
    const maxSide = options.maxSide ?? 384;
    if (isImageBitmap(frame)) {
      try {
        const gray = this.bitmapToGray(frame, maxSide);
        const detection = detectDocument(gray, options, this.context);
        return (
          detection &&
          scaleDetection(detection, frame.width / gray.width, frame.height / gray.height)
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
    // detectDocument area-downscales internally and returns frame coordinates.
    return detectDocument(rgbaToGray(frame), options, this.context);
  }

  /** Draws the bitmap downscaled by the browser (cheap, GPU-assisted) and reads it back. */
  private bitmapToGray(bitmap: ImageBitmap, maxSide: number): GrayImage {
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
    const pixels = ctx.getImageData(0, 0, size.width, size.height);
    return rgbaToGray(pixels);
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

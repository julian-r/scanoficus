import {
  type CameraFeatures,
  type CameraOptions,
  openCamera,
  type StillCapture,
} from '../camera/camera.ts';
import { FrameGrabber, startFrameLoop } from '../camera/frames.ts';
import type { RgbaImage } from '../image.ts';
import type { PageSource } from '../render/renderer.ts';

/** A still from {@link ScannerCamera.captureStill}. */
export interface ScannerStill {
  image: PageSource;
  width: number;
  height: number;
  /**
   * True if the still shows exactly the preview's field of view (e.g. a video frame), so that
   * preview coordinates can be scaled onto it. False for stills that may be cropped or
   * framed differently (Image Capture photos).
   */
  matchesPreview: boolean;
}

/**
 * What the {@link ScannerController} needs from a live camera. The default implementation
 * ({@link openScannerCamera}) wraps `openCamera`, `startFrameLoop` and `FrameGrabber`; tests and
 * custom capture pipelines can provide their own.
 */
export interface ScannerCamera {
  /** Current video frame size; detections are reported in these coordinates. */
  readonly width: number;
  readonly height: number;
  readonly features: CameraFeatures;
  /** Starts calling `callback` for new frames; returns a stop function. */
  onFrame(callback: (timeMs: number) => void): () => void;
  /** Current frame downscaled to at most `maxSide`, or `null` if no frame is available. */
  grabFrame(maxSide: number): RgbaImage | null;
  /** Captures a still at the best available resolution. */
  captureStill(): Promise<ScannerStill>;
  setTorch(on: boolean): Promise<boolean>;
  /** Registers a callback for the camera ending unexpectedly (track ended, device lost). */
  onEnded(callback: () => void): void;
  stop(): void;
}

/** Opens the device camera into `video` as a {@link ScannerCamera}. */
export async function openScannerCamera(
  video: HTMLVideoElement,
  options: CameraOptions = {},
): Promise<ScannerCamera> {
  const session = await openCamera(video, options);
  const grabber = new FrameGrabber();
  return {
    get width() {
      return session.width;
    },
    get height() {
      return session.height;
    },
    features: session.features,
    onFrame: (callback) => startFrameLoop(video, callback),
    grabFrame: (maxSide) => grabber.grab(video, maxSide),
    async captureStill() {
      const still: StillCapture = await session.captureStill();
      return { ...still, matchesPreview: still.method === 'video-frame' };
    },
    setTorch: (on) => session.setTorch(on),
    onEnded(callback) {
      session.track.addEventListener('ended', callback);
    },
    stop() {
      session.stop();
      grabber.dispose();
    },
  };
}

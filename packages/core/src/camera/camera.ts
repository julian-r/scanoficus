/**
 * Camera acquisition for document scanning, following docs/platform-notes.md and the iPhone
 * lab measurements summarized there:
 * - rear camera via `facingMode: 'environment'` (the system-preferred, often logical
 *   multi-lens camera; no physical lens is guaranteed) or an explicit `deviceId`;
 * - staged, deadline-guarded startup (acquiring → playing → waiting-for-frames), because on iOS
 *   `getUserMedia` can succeed while `video.play()` / the first frame never arrives;
 * - stills from the video track at full stream resolution by default (`takePhoto()` returned
 *   only preview-sized images, slower, and failed with maximum photo settings on iPhone);
 * - capability-gated torch and serialized camera reconfiguration.
 */

/** Why the camera could not be opened. */
export type CameraErrorCode =
  | 'insecure-context'
  | 'unsupported'
  | 'permission-denied'
  | 'not-found'
  | 'not-readable'
  | 'overconstrained'
  /** Startup was interrupted (e.g. `play()` rejected with AbortError) or cancelled. */
  | 'aborted'
  /** The preview did not start playing or deliver a frame before the deadline. */
  | 'start-timeout'
  | 'unknown';

/** Codes for which simply trying again can succeed. */
const RECOVERABLE: ReadonlySet<CameraErrorCode> = new Set([
  'aborted',
  'start-timeout',
  'not-readable',
  'unknown',
]);

/** Error thrown by {@link openCamera}, with a machine-readable {@link CameraErrorCode}. */
export class CameraError extends Error {
  readonly code: CameraErrorCode;

  constructor(code: CameraErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'CameraError';
    this.code = code;
  }

  /**
   * True if retrying can help (interrupted or stalled startup, busy camera). Note that an
   * AbortError is not a permission denial.
   */
  get recoverable(): boolean {
    return RECOVERABLE.has(this.code);
  }
}

/** Startup stages reported through {@link CameraOptions.onStage}. */
export type CameraStartStage = 'acquiring' | 'playing' | 'waiting-for-frames';

/** Options for {@link openCamera}. */
export interface CameraOptions {
  /**
   * Exact camera to use (from {@link listCameras}). Default: the system-preferred rear camera
   * (`facingMode: 'environment'`). Device order and labels are not reliable for choosing a lens
   * automatically; offer a picker instead.
   */
  deviceId?: string;
  /**
   * Stream resolution preset. `'hd'` (default) requests ideal 1920×1080, which iPhones deliver
   * as 1080×1920 portrait at 30 fps. `'max'` requests ideal 4096×4096 (3024×4032 at 30 fps was
   * measured on iPhone), giving higher-resolution stills from the video track at a higher
   * processing and memory cost. Always use the actual size the session reports.
   */
  resolution?: 'hd' | 'max';
  /**
   * How {@link CameraSession.captureStill} captures. `'frame'` (default): the current video frame
   * at stream resolution. `'photo'`: Image Capture `takePhoto()`, kept only if it decodes to at
   * least the stream resolution; otherwise (or on failure) the frame is used.
   */
  stillCapture?: 'frame' | 'photo';
  /** Deadline (ms) for `play()` and for the first frame, each. Default 4000. */
  startTimeoutMs?: number;
  /** Called when startup enters a new stage. */
  onStage?: (stage: CameraStartStage) => void;
  /** Cancels a pending start; the stream is released and `aborted` is thrown. */
  signal?: AbortSignal;
}

/** Optional controls the active camera advertises. */
export interface CameraFeatures {
  /** `torch` constraint advertised by `getCapabilities()`. */
  torch: boolean;
  /** Image Capture `takePhoto()` is available for this track. */
  photo: boolean;
}

/** A captured still image. */
export interface StillCapture {
  image: ImageBitmap;
  width: number;
  height: number;
  /** `'photo'`: Image Capture `takePhoto()`; `'video-frame'`: current video frame. */
  method: 'photo' | 'video-frame';
}

/** An open camera stream attached to a `<video>` element. */
export interface CameraSession {
  readonly video: HTMLVideoElement;
  readonly stream: MediaStream;
  readonly track: MediaStreamTrack;
  readonly features: CameraFeatures;
  /** Actual video frame size (`videoWidth × videoHeight`, e.g. portrait on phones). */
  readonly width: number;
  readonly height: number;
  /** Switches the torch; resolves to the applied state (false if unsupported). */
  setTorch(on: boolean): Promise<boolean>;
  /**
   * Captures a still according to {@link CameraOptions.stillCapture}. Calls are serialized
   * with other camera reconfigurations. The result reports its actual size; re-detect the
   * document on it rather than reusing preview coordinates.
   */
  captureStill(): Promise<StillCapture>;
  /** Turns the torch off, stops all tracks and detaches the stream. */
  stop(): void;
}

/** Minimal typing of the Image Capture API (not in TypeScript's DOM lib). */
interface ImageCaptureLike {
  takePhoto(): Promise<Blob>;
}
type ImageCaptureConstructor = new (track: MediaStreamTrack) => ImageCaptureLike;

const PHOTO_TIMEOUT_MS = 5000;

/** Video input devices. Labels and ids are only meaningful after camera permission is granted. */
export async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === 'videoinput');
}

/**
 * Opens the camera and starts playing it in `video` (configured `muted`, `playsinline` and
 * `autoplay`, as iOS requires for inline preview). Requests video only.
 *
 * Startup runs in stages, each reported via `onStage`: `acquiring` (`getUserMedia`, which may
 * wait for the permission prompt and therefore has no deadline), `playing` (`video.play()`,
 * deadline `startTimeoutMs`) and `waiting-for-frames` (first presented frame, same deadline).
 * On any failure everything acquired so far is released before throwing.
 *
 * @throws {CameraError} with a code describing the failure; check `recoverable` to decide
 *   whether to retry.
 */
export async function openCamera(
  video: HTMLVideoElement,
  options: CameraOptions = {},
): Promise<CameraSession> {
  if (typeof isSecureContext !== 'undefined' && !isSecureContext) {
    throw new CameraError('insecure-context', 'Camera access requires HTTPS');
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraError('unsupported', 'getUserMedia is not available in this browser');
  }
  const { signal, onStage } = options;
  const deadline = options.startTimeoutMs ?? 4000;
  const side = options.resolution === 'max' ? 4096 : null;
  const constraints: MediaTrackConstraints = {
    width: { ideal: side ?? 1920 },
    height: { ideal: side ?? 1080 },
  };
  if (options.deviceId) constraints.deviceId = { exact: options.deviceId };
  else constraints.facingMode = { ideal: 'environment' };

  throwIfAborted(signal);
  onStage?.('acquiring');
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
  } catch (error) {
    throw toCameraError(error);
  }
  const release = () => {
    for (const t of stream.getTracks()) t.stop();
    if (video.srcObject === stream) video.srcObject = null;
  };
  const track = stream.getVideoTracks()[0];
  if (!track) {
    release();
    throw new CameraError('not-found', 'The stream has no video track');
  }

  try {
    throwIfAborted(signal);
    onStage?.('playing');
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.setAttribute('playsinline', '');
    video.srcObject = stream;
    await guarded(video.play(), deadline, signal, 'The video preview did not start playing');
    onStage?.('waiting-for-frames');
    await guarded(firstFrame(video), deadline, signal, 'The camera delivered no frames');
  } catch (error) {
    release();
    throw error instanceof CameraError ? error : toCameraError(error);
  }

  const capabilities = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & {
    torch?: boolean;
  };
  const ImageCaptureCtor = (globalThis as { ImageCapture?: ImageCaptureConstructor }).ImageCapture;
  let imageCapture: ImageCaptureLike | null =
    options.stillCapture === 'photo' && ImageCaptureCtor ? new ImageCaptureCtor(track) : null;
  const features: CameraFeatures = {
    torch: capabilities.torch === true,
    photo: !!ImageCaptureCtor,
  };
  let torchOn = false;
  // Serializes takePhoto() and applyConstraints(): overlapping calls reconfigure the capture
  // session concurrently, which older Safari versions handle badly.
  let queue: Promise<unknown> = Promise.resolve();
  const serialized = <T>(task: () => Promise<T>): Promise<T> => {
    const run = queue.then(task, task);
    queue = run.catch(() => undefined);
    return run;
  };

  const frameStill = async (): Promise<StillCapture> => {
    const image = await captureVideoFrame(video);
    return { image, width: image.width, height: image.height, method: 'video-frame' };
  };

  return {
    video,
    stream,
    track,
    features,
    get width() {
      return video.videoWidth;
    },
    get height() {
      return video.videoHeight;
    },
    setTorch(on) {
      if (!features.torch) return Promise.resolve(false);
      return serialized(async () => {
        await track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
        torchOn = on;
        return on;
      });
    },
    captureStill() {
      return serialized(async () => {
        if (imageCapture && track.readyState === 'live') {
          try {
            const blob = await withTimeout(imageCapture.takePhoto(), PHOTO_TIMEOUT_MS);
            const image = await createImageBitmap(blob, { imageOrientation: 'from-image' });
            // Keep the photo only if it really is at least the stream resolution.
            if (image.width * image.height >= video.videoWidth * video.videoHeight) {
              return { image, width: image.width, height: image.height, method: 'photo' };
            }
            image.close();
          } catch {
            // takePhoto() failed on this camera; stop trying it for this session.
            imageCapture = null;
          }
        }
        return frameStill();
      });
    },
    stop() {
      if (torchOn) {
        track
          .applyConstraints({ advanced: [{ torch: false } as MediaTrackConstraintSet] })
          .catch(() => {});
      }
      release();
    },
  };
}

/** The current video frame at stream resolution as an ImageBitmap (canvas fallback). */
async function captureVideoFrame(video: HTMLVideoElement): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(video);
  } catch {
    const canvas =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(video.videoWidth, video.videoHeight)
        : Object.assign(document.createElement('canvas'), {
            width: video.videoWidth,
            height: video.videoHeight,
          });
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | null;
    if (!ctx) throw new Error('2D canvas unavailable for frame capture');
    ctx.drawImage(video, 0, 0);
    const bitmap = await createImageBitmap(canvas);
    canvas.width = canvas.height = 0;
    return bitmap;
  }
}

/** Resolves when the video presents its first frame (rVFC, else first `currentTime` advance). */
function firstFrame(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    if (typeof video.requestVideoFrameCallback === 'function') {
      video.requestVideoFrameCallback(() => resolve());
      return;
    }
    const check = () => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) resolve();
      else requestAnimationFrame(check);
    };
    check();
  });
}

/** Races `promise` against a deadline and an abort signal, mapping both to CameraErrors. */
function guarded<T>(
  promise: Promise<T>,
  ms: number,
  signal: AbortSignal | undefined,
  timeoutMessage: string,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    };
    const onAbort = () => {
      cleanup();
      reject(new CameraError('aborted', 'Camera start was cancelled'));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new CameraError('start-timeout', timeoutMessage));
    }, ms);
    signal?.addEventListener('abort', onAbort);
    promise.then(
      (v) => {
        cleanup();
        resolve(v);
      },
      (e) => {
        cleanup();
        reject(e);
      },
    );
  });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new CameraError('aborted', 'Camera start was cancelled');
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Timed out')), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

/** Maps `getUserMedia` / `play()` DOMExceptions to {@link CameraError}s. */
export function toCameraError(error: unknown): CameraError {
  const name = (error as { name?: string })?.name;
  const message = (error as { message?: string })?.message ?? String(error);
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new CameraError('permission-denied', 'Camera permission was denied', { cause: error });
    case 'NotFoundError':
      return new CameraError('not-found', 'No camera found', { cause: error });
    case 'NotReadableError':
      return new CameraError('not-readable', 'The camera is in use or unavailable', {
        cause: error,
      });
    case 'AbortError':
      // Seen on iOS when play() is interrupted during startup; retrying usually works.
      return new CameraError('aborted', 'Camera startup was interrupted', { cause: error });
    case 'OverconstrainedError':
      return new CameraError('overconstrained', 'The selected camera is not available', {
        cause: error,
      });
    default:
      return new CameraError('unknown', message, { cause: error });
  }
}

/**
 * Camera acquisition for document scanning, following docs/platform-notes.md:
 * rear camera via `facingMode: 'environment'` (the system-preferred rear camera; no lens is
 * guaranteed) with optional explicit `deviceId`, 1080p preview, capability-checked torch, still
 * capture via Image Capture `takePhoto()` with a video-frame fallback, and serialized camera
 * reconfiguration.
 */

/** Why the camera could not be opened. */
export type CameraErrorCode =
  | 'insecure-context'
  | 'unsupported'
  | 'permission-denied'
  | 'not-found'
  | 'not-readable'
  | 'overconstrained'
  | 'unknown';

/** Error thrown by {@link openCamera}, with a machine-readable {@link CameraErrorCode}. */
export class CameraError extends Error {
  readonly code: CameraErrorCode;

  constructor(code: CameraErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'CameraError';
    this.code = code;
  }
}

/** Options for {@link openCamera}. */
export interface CameraOptions {
  /**
   * Exact camera to use (from {@link listCameras}). Default: the system-preferred rear camera
   * (`facingMode: 'environment'`). Let users pick another rear camera when focus or field of view
   * is poor (e.g. close-focus limits of some Pro iPhones); device order and labels are not
   * reliable for choosing automatically.
   */
  deviceId?: string;
  /** Ideal preview size (may resolve lower). Default 1920 × 1080. */
  width?: number;
  height?: number;
  /** Ideal preview frame rate. Default 30. */
  frameRate?: number;
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
  /** `'photo'`: Image Capture `takePhoto()`; `'video-frame'`: current preview frame. */
  method: 'photo' | 'video-frame';
}

/** An open camera stream attached to a `<video>` element. */
export interface CameraSession {
  readonly video: HTMLVideoElement;
  readonly stream: MediaStream;
  readonly track: MediaStreamTrack;
  readonly features: CameraFeatures;
  /** Current video frame size (`videoWidth × videoHeight`). */
  readonly width: number;
  readonly height: number;
  /** Switches the torch; resolves to the applied state (false if unsupported). */
  setTorch(on: boolean): Promise<boolean>;
  /**
   * Captures a still at the best available resolution: `takePhoto()` when available (falls back
   * to the preview frame on error or after 5 s, and stops trying `takePhoto()` after a failure),
   * else the current video frame at stream resolution. Calls are serialized with other camera
   * reconfigurations. The still may have a different size or crop than the preview, so the
   * document should be re-detected on it.
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
 * Opens the camera and starts playing it in `video` (which is configured `muted`, `playsinline`
 * and `autoplay`, as iOS requires for inline camera preview). Requests video only.
 *
 * @throws {CameraError} with a code describing the failure.
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
  const constraints: MediaTrackConstraints = {
    width: { ideal: options.width ?? 1920 },
    height: { ideal: options.height ?? 1080 },
    frameRate: { ideal: options.frameRate ?? 30 },
  };
  if (options.deviceId) constraints.deviceId = { exact: options.deviceId };
  else constraints.facingMode = { ideal: 'environment' };

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
  } catch (error) {
    throw toCameraError(error);
  }
  const track = stream.getVideoTracks()[0];
  if (!track) {
    for (const t of stream.getTracks()) t.stop();
    throw new CameraError('not-found', 'The stream has no video track');
  }

  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.setAttribute('playsinline', '');
  video.srcObject = stream;
  try {
    await waitForMetadata(video);
    await video.play();
  } catch (error) {
    for (const t of stream.getTracks()) t.stop();
    video.srcObject = null;
    throw new CameraError('not-readable', 'Could not start the video preview', { cause: error });
  }

  const capabilities = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & {
    torch?: boolean;
  };
  const ImageCaptureCtor = (globalThis as { ImageCapture?: ImageCaptureConstructor }).ImageCapture;
  let imageCapture: ImageCaptureLike | null = ImageCaptureCtor ? new ImageCaptureCtor(track) : null;
  const features: CameraFeatures = { torch: capabilities.torch === true, photo: !!imageCapture };
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
    const image = await createImageBitmap(video);
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
            return { image, width: image.width, height: image.height, method: 'photo' };
          } catch {
            // Remember the failure; the preview frame is the dependable fallback.
            imageCapture = null;
            features.photo = false;
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
      for (const t of stream.getTracks()) t.stop();
      if (video.srcObject === stream) video.srcObject = null;
    },
  };
}

function waitForMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener('loadedmetadata', done);
      video.removeEventListener('error', fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener('loadedmetadata', done);
      video.removeEventListener('error', fail);
      reject(new Error('Video element error'));
    };
    video.addEventListener('loadedmetadata', done);
    video.addEventListener('error', fail);
  });
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

/** Maps `getUserMedia` DOMExceptions to {@link CameraError}s. */
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
    case 'AbortError':
      return new CameraError('not-readable', 'The camera is in use or unavailable', {
        cause: error,
      });
    case 'OverconstrainedError':
      return new CameraError('overconstrained', 'The selected camera is not available', {
        cause: error,
      });
    default:
      return new CameraError('unknown', message, { cause: error });
  }
}

import {
  CameraError,
  type CameraFeatures,
  type CameraOptions,
  type CameraStartStage,
} from '../camera/camera.ts';
import { FrameGrabber, watchPageVisibility } from '../camera/frames.ts';
import { onDeviceMotionGranted } from '../camera/motion.ts';
import type { Detection, DetectOptions } from '../detect/detector.ts';
import { sharpness } from '../detect/sharpness.ts';
import { QuadTracker, type QuadTrackerOptions } from '../detect/tracker.ts';
import type { ScanFilter } from '../filters/filters.ts';
import { convexIoU } from '../geometry/polygon.ts';
import { type Quad, scaleQuad } from '../geometry/quad.ts';
import { estimateOutputSize } from '../geometry/size.ts';
import { type RgbaImage, rgbaToGray } from '../image.ts';
import type { PdfOptions } from '../pdf/pdf.ts';
import { normalizeRotation, type Rotation } from '../render/cpu.ts';
import { createPageRenderer, type PageRenderer, type PageSource } from '../render/renderer.ts';
import { createDetectorWorker } from '../worker/client.ts';
import type { DocumentDetector } from '../worker/processor.ts';
import { openScannerCamera, type ScannerCamera, type ScannerStill } from './camera-source.ts';
import { Emitter } from './emitter.ts';
import {
  type CaptureBlocker,
  CaptureTrigger,
  type CaptureTriggerOptions,
  type CaptureTriggerStatus,
} from './trigger.ts';

/**
 * Scanner lifecycle. `idle` → `starting` (opening the camera) → `scanning` (live detection) →
 * `capturing` (taking and processing a still) → back to `scanning`, or `reviewing` once the user
 * is done (camera released). `error` is entered when the camera fails; `start()` retries.
 */
export type ScannerState = 'idle' | 'starting' | 'scanning' | 'capturing' | 'reviewing' | 'error';

/** One scanned page. Immutable: edits produce a new object with the same `id`. */
export interface ScanPage {
  readonly id: string;
  /** The captured still or imported image the page is cut from. */
  readonly source: PageSource;
  readonly sourceWidth: number;
  readonly sourceHeight: number;
  /** Document corners in source pixels (TL, TR, BR, BL). */
  readonly quad: Quad;
  /** Whether `quad` was detected (false: whole image used). */
  readonly detected: boolean;
  readonly filter: ScanFilter;
  readonly rotation: Rotation;
  /** Rectified size before rotation. */
  readonly width: number;
  readonly height: number;
  /** Rendered page (JPEG by default) and an object URL for it. */
  readonly image: Blob;
  readonly url: string;
}

/** Changes accepted by {@link ScannerController.updatePage}. */
export interface PageChanges {
  quad?: Quad;
  filter?: ScanFilter;
  /** Absolute rotation in degrees (multiples of 90). */
  rotation?: number;
}

/** Live detection state, in video pixel coordinates. */
export interface LiveDetection {
  /** Smoothed document quad, or `null`. */
  quad: Quad | null;
  /** Size of the video frame the quad refers to. */
  frameWidth: number;
  frameHeight: number;
  /** The tracker reports the quad as stable. */
  stable: boolean;
  /**
   * Auto-capture progress in [0, 1]: fills while the document is ready (stable and sharp) for
   * `stableMs`, decays while it is not.
   */
  progress: number;
  /**
   * Laplacian-variance sharpness of the document region (whole frame without a detection) and
   * its ratio to the recent reference (a percentile of the last 1.5 s); only the ratio is used
   * for auto-capture, because absolute values depend on scene and scale.
   */
  sharpness: number;
  relativeSharpness: number;
  /** Duration of the last detection and detection rate. */
  detectMs: number;
  fps: number;
  /** Why auto-capture is (not) ready, with the measured values and their limits. */
  autoCapture?: AutoCaptureInfo;
}

/** Auto-capture criteria of the latest frame: measured values and limits (for debug overlays). */
export interface AutoCaptureInfo {
  /** The first criterion that fails, or `null` if the frame is ready. */
  blocker: CaptureBlocker | null;
  ready: boolean;
  /** Fastest corner of the tracked motion (diagonals/s; `null`: too few frames) and the limit. */
  speed: number | null;
  maxSpeed: number;
  /** Scatter of the detections around that motion (diagonals) and the limit. */
  jitter: number | null;
  maxJitter: number;
  /**
   * Detections in the stability window and the minimum; time they span and the minimum (half
   * the window).
   */
  windowFrames: number;
  minFrames: number;
  windowSpanMs: number;
  minWindowSpanMs: number;
  /** Sharpness reference and the minimum ratio to it. */
  sharpnessReference: number;
  minRelativeSharpness: number;
  /** Ready time accumulated so far (progress × `stableMs`) and the required time. */
  readyMs: number;
  stableMs: number;
  /**
   * Device rotation rate (°/s, opt-in motion sensor; `null` without recent samples) and the
   * limit.
   */
  rotationRate: number | null;
  maxRotationRate: number;
  /** A sharper recent frame is kept as the still for the next auto-capture. */
  bestFrame: boolean;
}

/** Immutable view of the controller, replaced on every change (for `useSyncExternalStore`). */
export interface ScannerSnapshot {
  readonly state: ScannerState;
  readonly pages: readonly ScanPage[];
  readonly detection: LiveDetection | null;
  readonly autoCapture: boolean;
  readonly torch: boolean;
  readonly features: CameraFeatures | null;
  readonly error: Error | null;
  /** Camera startup stage while `starting` (acquiring → playing → waiting-for-frames). */
  readonly cameraStage: CameraStartStage | null;
}

/** Events emitted by {@link ScannerController}. */
export interface ScannerEvents extends Record<string, unknown> {
  state: { state: ScannerState; previous: ScannerState };
  detection: LiveDetection;
  /** Debug: the downscaled frame given to the detector and its raw result. */
  frame: { image: RgbaImage; detection: Detection | null };
  pages: { pages: readonly ScanPage[] };
  capture: { page: ScanPage };
  error: { error: Error };
  change: ScannerSnapshot;
}

/** Options for {@link ScannerController}. */
export interface ScannerOptions {
  /**
   * Detector to use, owned by the caller (not disposed by the controller, no fallback on
   * failure). Prefer `createDetector` unless the detector is shared.
   */
  detector?: DocumentDetector;
  /**
   * Factory for a detector owned by the controller: created on first use, disposed with the
   * controller, and replaced by the main-thread detector if it fails (e.g. a worker that cannot
   * load). Default: `createDetectorWorker()` (main thread where workers are unavailable).
   */
  createDetector?: () => DocumentDetector;
  /** Page renderer. Default: `createPageRenderer()` (WebGL2 with CPU fallback). */
  renderer?: PageRenderer;
  /** Camera constraints for the default camera. */
  camera?: CameraOptions;
  /** Camera factory (for custom pipelines and tests). Default: `openScannerCamera`. */
  openCamera?: (video: HTMLVideoElement, options: CameraOptions) => Promise<ScannerCamera>;
  /** Detector options (processing size, minimum area, minimum score). */
  detection?: DetectOptions;
  /** Quad smoothing/stability options. */
  tracker?: QuadTrackerOptions;
  /** Capture automatically when the document is stable and sharp. Default true. */
  autoCapture?: boolean;
  /**
   * How long the document must be stable and sharp before auto-capture. Default 600 ms (was
   * 1000 ms with the earlier consecutive-frame rule).
   */
  stableMs?: number;
  /**
   * Minimum sharpness relative to the recent reference (75th percentile of the last 1.5 s) for
   * auto-capture. Default 0.6.
   */
  minRelativeSharpness?: number;
  /**
   * Further auto-capture rule options (progress decay, sharpness reference); `stableMs` and
   * `minRelativeSharpness` above take precedence.
   */
  trigger?: CaptureTriggerOptions;
  /**
   * Best-frame window (ms) of auto-capture: during the last part of the progress, each analyzed
   * video frame is also grabbed at full resolution and the sharpest ready one of this period is
   * kept (at most one, plus one being compared); auto-capture uses it as the still instead of
   * the frame after the decision. Only with video-frame stills (not `stillCapture: 'photo'`) and
   * cameras that implement `grabStill`. 0 disables it. Default 300.
   */
  bestFrameMs?: number;
  /**
   * Opt-in motion sensor: listen to `devicemotion` while scanning and hold auto-capture while
   * the device rotates faster than `maxRotationRate` (the last 300 ms). On iOS, call
   * `requestDeviceMotionPermission()` from a user gesture, before or during scanning (a grant
   * re-arms the listener). The listener is attached only in the `scanning` state. Default false.
   */
  deviceMotion?: boolean;
  /** Rotation-rate limit (°/s) of the motion sensor. Default 10. */
  maxRotationRate?: number;
  /** Filter of new pages. Default `'color'`. */
  defaultFilter?: ScanFilter;
  /**
   * Longer side cap of rendered pages. Default 4000 px (≈ 11 MP for A4), in line with the
   * ≤ 12 MP output budget of docs/platform-notes.md.
   */
  maxOutputSide?: number;
  /** Pixel-area cap of rendered pages. Default 12 000 000 (12 MP). */
  maxOutputPixels?: number;
  /** JPEG quality of rendered pages. Default 0.9. */
  jpegQuality?: number;
  /** Switch to `reviewing` after every capture (single-shot flows). Default false. */
  reviewAfterCapture?: boolean;
  /** Report the camera as stalled when no frame arrives for this long. Default 4000 ms. */
  stallTimeoutMs?: number;
  /** Clock (for tests). Default `performance.now`. */
  now?: () => number;
}

const NOT_STARTED = new Error('The scanner has no video element; call start(video) first');

/**
 * Headless document scanner: owns the camera, the live detection loop, auto-capture and the
 * page list, and exposes everything as state + events so that any UI (custom elements, React,
 * or your own) can drive it.
 */
export class ScannerController {
  private readonly emitter = new Emitter<ScannerEvents>();
  private readonly options: ScannerOptions;
  private readonly now: () => number;
  private readonly ownsDetector: boolean;
  private readonly ownsRenderer: boolean;
  /** Created lazily so that constructing a controller has no side effects (SSR, StrictMode). */
  private detector: DocumentDetector | null;
  private renderer: PageRenderer | null;
  private readonly tracker: QuadTracker;
  private readonly grabber = new FrameGrabber();
  private camera: ScannerCamera | null = null;
  private video: HTMLVideoElement | null = null;
  private stopFrames: (() => void) | null = null;
  private startGeneration = 0;
  private cancelStart: (() => void) | null = null;
  private unwatchVisibility: (() => void) | null = null;
  private stallTimer: ReturnType<typeof setInterval> | null = null;
  private resumeWhenVisible = false;
  private busy = false;
  private lastFrameAt = 0;
  /** Consecutive failures of `detectStill` (reported once, at the second). */
  private stillFailures = 0;
  private readonly trigger: CaptureTrigger;
  /** Progress of the last analyzed frame (decides whether to grab best-frame stills). */
  private lastProgress = 0;
  /** Best-frame candidate: a full-resolution video frame with its sharpness and preview quad. */
  private bestStill: {
    still: ScannerStill;
    sharpness: number;
    time: number;
    quad: Quad | null;
    width: number;
    height: number;
  } | null = null;
  private stopMotion: (() => void) | null = null;
  private rotationRate: number | null = null;
  private lastMotionAt = Number.NEGATIVE_INFINITY;
  private lastFastMotionAt = Number.NEGATIVE_INFINITY;
  /** Quad of the last capture; auto-capture waits until the document changes. */
  private capturedQuad: Quad | null = null;
  private replaceIndex: number | null = null;
  /** Incremented by stop() and dispose(): pending captures of an older epoch are discarded. */
  private captureEpoch = 0;
  private captureSequence = 0;
  /** Token of the capture that owns the live session (0: none). Cleared by releaseCamera(). */
  private activeCapture = 0;
  private nextPageId = 1;
  private detectTimes: number[] = [];
  private snapshot: ScannerSnapshot;
  private disposed = false;

  constructor(options: ScannerOptions = {}) {
    this.options = options;
    this.now = options.now ?? (() => performance.now());
    this.ownsDetector = !options.detector;
    this.detector = options.detector ?? null;
    this.ownsRenderer = !options.renderer;
    this.renderer = options.renderer ?? null;
    this.tracker = new QuadTracker(options.tracker);
    this.trigger = new CaptureTrigger({
      ...options.trigger,
      ...(options.stableMs !== undefined && { stableMs: options.stableMs }),
      ...(options.minRelativeSharpness !== undefined && {
        minRelativeSharpness: options.minRelativeSharpness,
      }),
    });
    this.snapshot = {
      state: 'idle',
      pages: [],
      detection: null,
      autoCapture: options.autoCapture ?? true,
      torch: false,
      features: null,
      error: null,
      cameraStage: null,
    };
  }

  /** Current immutable snapshot. */
  getSnapshot = (): ScannerSnapshot => this.snapshot;

  /** Subscribes to any change of the snapshot (for `useSyncExternalStore`). */
  subscribe = (listener: () => void): (() => void) => this.emitter.on('change', listener);

  /** Subscribes to an event; returns an unsubscribe function. */
  on<K extends keyof ScannerEvents>(
    type: K,
    listener: (payload: ScannerEvents[K]) => void,
  ): () => void {
    return this.emitter.on(type, listener);
  }

  get state(): ScannerState {
    return this.snapshot.state;
  }

  get pages(): readonly ScanPage[] {
    return this.snapshot.pages;
  }

  /** True after {@link ScannerController.dispose}; a disposed controller cannot be restarted. */
  get isDisposed(): boolean {
    return this.disposed;
  }

  /**
   * Opens the camera into `video` and starts live detection. Also used to retry after an error
   * or to resume from `reviewing` (the last video element is reused when omitted).
   */
  async start(video?: HTMLVideoElement): Promise<void> {
    this.assertAlive();
    const target = video ?? this.video;
    if (!target) throw NOT_STARTED;
    if (this.camera && target === this.video) return;
    if (this.state === 'starting' && target === this.video) return;
    this.releaseCamera();
    this.video = target;
    this.setState('starting');
    // Every start gets a generation; stop(), page hide or a newer start cancel older ones.
    this.watchLifecycle();
    const generation = ++this.startGeneration;
    const abort = new AbortController();
    this.cancelStart = () => abort.abort();
    const open = this.options.openCamera ?? openScannerCamera;
    const attempt = async (): Promise<ScannerCamera> => {
      try {
        return await open(target, {
          ...this.options.camera,
          signal: abort.signal,
          onStage: (stage) => {
            if (generation === this.startGeneration) this.update({ cameraStage: stage });
          },
        });
      } catch (error) {
        // iOS: getUserMedia can succeed while play()/the first frame never completes, and a
        // plain retry then works. Retry once automatically while the page is visible.
        const retryable =
          error instanceof CameraError &&
          error.recoverable &&
          !abort.signal.aborted &&
          (typeof document === 'undefined' || document.visibilityState === 'visible');
        if (!retryable) throw error;
        return open(target, { ...this.options.camera, signal: abort.signal });
      }
    };
    try {
      const camera = await attempt();
      if (this.disposed || generation !== this.startGeneration) {
        camera.stop();
        return;
      }
      this.cancelStart = null;
      this.camera = camera;
      camera.onEnded(() => {
        if (this.camera === camera) this.fail(new Error('The camera stopped unexpectedly'));
      });
      this.update({ features: camera.features, torch: false, error: null, cameraStage: null });
      this.tracker.reset();
      this.trigger.reset();
      this.lastProgress = 0;
      this.lastFrameAt = this.now();
      this.stopFrames = camera.onFrame(() => this.onFrame());
      this.startStallWatchdog();
      this.setState('scanning');
    } catch (error) {
      if (generation !== this.startGeneration || this.disposed) return; // cancelled
      this.cancelStart = null;
      this.update({ cameraStage: null });
      this.fail(asError(error));
    }
  }

  /** Retries opening the camera after an error (same as `start()` with the last video). */
  retry(): Promise<void> {
    return this.start();
  }

  /** Releases the camera and returns to `idle`. Pages are kept. */
  stop(): void {
    this.captureEpoch++;
    this.releaseCamera();
    this.unwatchVisibility?.();
    this.unwatchVisibility = null;
    this.setState('idle');
  }

  /** Ends scanning and shows the pages for review (releases the camera). */
  finish(): void {
    this.releaseCamera();
    this.setState('reviewing');
  }

  /** Enables or disables auto-capture. */
  setAutoCapture(enabled: boolean): void {
    this.trigger.restart();
    this.lastProgress = 0;
    this.dropBestStill();
    this.update({ autoCapture: enabled });
  }

  /** Switches the torch if the camera supports it. */
  async setTorch(on: boolean): Promise<void> {
    if (!this.camera) return;
    try {
      const applied = await this.camera.setTorch(on);
      this.update({ torch: applied });
    } catch (error) {
      this.emitter.emit('error', { error: asError(error) });
    }
  }

  /**
   * Captures a page now (manual shutter, also used by auto-capture). The document is
   * re-detected on the still (which may differ in size and crop from the preview); if that
   * fails the tracked preview quad is scaled to the still, and without any quad the whole
   * image is used. Resolves with the new page, or `null` if not scanning.
   */
  capture(): Promise<ScanPage | null> {
    // A best-frame still younger than bestFrameMs is as good as a fresh one, and sharp.
    const kept = this.bestStill;
    this.bestStill = null;
    if (kept && this.now() - kept.time <= (this.options.bestFrameMs ?? 300)) {
      return this.captureWith(kept);
    }
    if (kept) closeSource(kept.still.image);
    return this.captureWith(null);
  }

  /** Captures with `kept` (a best-frame still, owned by this call) or a fresh still. */
  private async captureWith(kept: ScannerController['bestStill']): Promise<ScanPage | null> {
    const camera = this.camera;
    if (!camera || this.state !== 'scanning') {
      if (kept) closeSource(kept.still.image);
      return null;
    }
    const previewQuad = kept ? kept.quad : (this.snapshot.detection?.quad ?? null);
    const previewWidth = kept ? kept.width : camera.width;
    const previewHeight = kept ? kept.height : camera.height;
    // `epoch` decides whether the result is still wanted (stop/dispose discard it); `token`
    // decides whether this capture still owns the live session (hide, finish or a newer
    // session take that away): only the owner may change state or the duplicate guard.
    const epoch = this.captureEpoch;
    const token = ++this.captureSequence;
    this.activeCapture = token;
    const valid = () => epoch === this.captureEpoch && !this.disposed;
    const owns = () => this.activeCapture === token;
    this.setState('capturing');
    let still: ScannerStill | null = null;
    try {
      still = kept ? kept.still : await camera.captureStill();
      // Preview corners map to the still only when both show the same framing (video frames);
      // a photo may be cropped differently, so it relies on its own detection.
      const preview =
        still.matchesPreview && previewQuad
          ? scaleQuad(previewQuad, still.width / previewWidth, still.height / previewHeight)
          : null;
      if (!valid()) throw new PageCancelled();
      const page = await this.createPage(still.image, still.width, still.height, preview, valid);
      if (!valid()) {
        URL.revokeObjectURL(page.url);
        throw new PageCancelled();
      }
      const replacing = this.replaceIndex;
      this.insertPage(page);
      this.emitter.emit('capture', { page });
      if (owns()) {
        this.activeCapture = 0;
        this.capturedQuad = this.snapshot.detection?.quad ?? previewQuad;
        this.trigger.restart();
        this.lastProgress = 0;
        if (this.options.reviewAfterCapture || replacing !== null) this.finish();
        else this.setState('scanning');
      }
      return page;
    } catch (error) {
      // The still is ours until a page is committed.
      if (still) closeSource(still.image);
      if (!(error instanceof PageCancelled)) this.emitter.emit('error', { error: asError(error) });
      if (owns()) {
        this.activeCapture = 0;
        if (this.isCapturing()) this.setState('scanning');
      }
      return null;
    }
  }

  /**
   * Adds a page from an existing image (e.g. a photo picked with
   * `<input type="file" accept="image/*" capture="environment">`). The document is detected
   * on the image; without a detection the whole image is used. Works in any state.
   */
  async addImage(image: Blob | PageSource): Promise<ScanPage> {
    this.assertAlive();
    const decoded = image instanceof Blob;
    const source = decoded
      ? await createImageBitmap(image, { imageOrientation: 'from-image' })
      : image;
    const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
    const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
    const valid = () => !this.disposed;
    try {
      if (!valid()) throw new PageCancelled();
      const page = await this.createPage(source, width, height, null, valid);
      if (!valid()) {
        URL.revokeObjectURL(page.url);
        throw new PageCancelled();
      }
      this.insertPage(page);
      return page;
    } catch (error) {
      // A bitmap decoded here is ours until the page is committed; caller sources are not.
      if (decoded) closeSource(source);
      if (error instanceof PageCancelled) {
        throw new Error('The scanner controller was disposed while adding an image');
      }
      throw error;
    }
  }

  /** Re-renders a page after editing its quad, filter or rotation. */
  async updatePage(id: string, changes: PageChanges): Promise<ScanPage> {
    const page = this.findPage(id);
    const quad = changes.quad ?? page.quad;
    const size =
      changes.quad !== undefined
        ? this.outputSize(quad, page.sourceWidth, page.sourceHeight)
        : { width: page.width, height: page.height };
    const filter = changes.filter ?? page.filter;
    const rotation =
      changes.rotation !== undefined ? normalizeRotation(changes.rotation) : page.rotation;
    const image = await this.render(page.source, { quad, filter, rotation, ...size });
    const updated: ScanPage = {
      ...page,
      quad,
      filter,
      rotation,
      ...size,
      image,
      url: URL.createObjectURL(image),
    };
    // The page may have been removed while rendering.
    const index = this.pages.findIndex((p) => p.id === id);
    if (index < 0) {
      URL.revokeObjectURL(updated.url);
      return updated;
    }
    URL.revokeObjectURL(this.pages[index].url);
    const pages = [...this.pages];
    pages[index] = updated;
    this.setPages(pages);
    return updated;
  }

  /** Removes a page and releases its image. */
  removePage(id: string): void {
    const page = this.findPage(id);
    const remaining = this.pages.filter((p) => p.id !== id);
    releasePage(page, remaining);
    this.setPages(remaining);
  }

  /** Moves a page to another position. */
  movePage(id: string, toIndex: number): void {
    const pages = this.pages.filter((p) => p.id !== id);
    pages.splice(Math.max(0, Math.min(pages.length, toIndex)), 0, this.findPage(id));
    this.setPages(pages);
  }

  /** Replaces a page with a new capture: resumes scanning; the next capture takes its place. */
  async retake(id: string): Promise<void> {
    this.replaceIndex = this.pages.indexOf(this.findPage(id));
    await this.start();
  }

  /** Removes all pages. */
  clear(): void {
    for (const page of this.pages) releasePage(page, []);
    this.setPages([]);
  }

  /** Builds a PDF with one page per scanned page (JPEGs embedded without re-encoding). */
  async exportPdf(options: PdfOptions = {}): Promise<Blob> {
    if (this.pages.length === 0) throw new Error('No pages to export');
    const jpegs = await Promise.all(
      this.pages.map(async (p) => new Uint8Array(await p.image.arrayBuffer())),
    );
    // The PDF writer is loaded on first export.
    const { createPdf } = await import('../pdf/pdf.ts');
    return new Blob([createPdf(jpegs, options) as Uint8Array<ArrayBuffer>], {
      type: 'application/pdf',
    });
  }

  /** Releases everything: camera, detector and renderer (if created by the controller), pages. */
  dispose(): void {
    if (this.disposed) return;
    this.captureEpoch++;
    this.releaseCamera();
    this.unwatchVisibility?.();
    this.clear();
    if (this.ownsDetector) this.detector?.dispose();
    if (this.ownsRenderer) this.renderer?.dispose();
    this.grabber.dispose();
    this.setState('idle');
    this.disposed = true;
    this.emitter.clear();
  }

  // ---- live loop -------------------------------------------------------------------------

  private onFrame(): void {
    const camera = this.camera;
    this.lastFrameAt = this.now();
    if (!camera || this.busy || this.state !== 'scanning') return;
    const maxSide = this.options.detection?.maxSide ?? 384;
    const frame = camera.grabFrame(maxSide);
    if (!frame) return;
    const gray = rgbaToGray(frame);
    this.busy = true;
    const started = this.now();
    // Best frame: near the end of the progress, grab the full-resolution frame as well.
    const bestFrameMs = this.options.bestFrameMs ?? 300;
    const grabStill =
      bestFrameMs > 0 &&
      this.snapshot.autoCapture &&
      this.capturedQuad === null &&
      this.lastProgress > 0 &&
      this.lastProgress >= Math.max(0.01, 1 - bestFrameMs / this.stableMs()) &&
      camera.grabStill
        ? camera.grabStill().catch(() => null)
        : null;
    const size = { width: camera.width, height: camera.height };
    let handedOver = false;
    this.detectWithFallback(frame)
      .then(async (detection) => {
        const still = grabStill ? await grabStill : null;
        this.busy = false;
        if (this.camera !== camera || this.state !== 'scanning') {
          if (still) closeSource(still.image);
          return;
        }
        if (this.emitter.has('frame')) this.emitter.emit('frame', { image: frame, detection });
        // Focus is judged on the document region, so the background does not dominate.
        const frameSharpness = sharpness(gray, detection ? boundingBox(detection.quad) : undefined);
        handedOver = true;
        this.handleDetection(camera, frame, detection, frameSharpness, started, still, size);
      })
      .catch((error) => {
        this.busy = false;
        // A best-frame still grabbed for a failed detection is nobody's: close it.
        if (!handedOver) void grabStill?.then((still) => still && closeSource(still.image));
        this.emitter.emit('error', { error: asError(error) });
      });
  }

  private stableMs(): number {
    return this.options.stableMs ?? this.options.trigger?.stableMs ?? 600;
  }

  private async detectWithFallback(frame: RgbaImage): Promise<Detection | null> {
    let detector = this.detector;
    try {
      // Creating the detector is inside the fallback boundary: a factory can throw (e.g.
      // `new Worker()` rejected by a content security policy).
      if (!detector) {
        detector = (this.options.createDetector ?? createDefaultDetector)();
        this.detector = detector;
      }
      return await detector.detect(frame, this.options.detection);
    } catch (error) {
      // A worker that cannot be created or loaded (CSP, bundling, 404) or crashed is replaced
      // by the main-thread detector; the reason is reported once as a (non-fatal) error event.
      if (!this.ownsDetector || this.disposed) throw error;
      if (this.detector === detector) {
        detector?.dispose();
        this.detector = createLazyMainThreadDetector();
        this.emitter.emit('error', {
          error: new Error(`Falling back to main-thread detection: ${asError(error).message}`),
        });
      }
      return (this.detector as DocumentDetector).detect(frame, this.options.detection);
    }
  }

  private handleDetection(
    camera: ScannerCamera,
    frame: RgbaImage,
    detection: Detection | null,
    frameSharpness: number,
    frameTime: number,
    still: ScannerStill | null,
    size: { width: number; height: number },
  ): void {
    const now = this.now();
    const observed =
      detection &&
      scaleQuad(detection.quad, camera.width / frame.width, camera.height / frame.height);
    const tracked = this.tracker.update(observed, frameTime);
    this.detectTimes.push(now);
    while (this.detectTimes.length > 1 && now - this.detectTimes[0] > 1000)
      this.detectTimes.shift();

    if (this.capturedQuad && (!tracked.quad || convexIoU(tracked.quad, this.capturedQuad) < 0.8)) {
      this.capturedQuad = null; // a different document (or none) is in view now
    }
    const maxRotationRate = this.options.maxRotationRate ?? 10;
    const motionFresh = frameTime - this.lastMotionAt <= 1000;
    let outside: CaptureBlocker | null = null;
    if (!this.snapshot.autoCapture) outside = 'disabled';
    else if (this.capturedQuad) outside = 'captured';
    else if (motionFresh && frameTime - this.lastFastMotionAt < 300) outside = 'device-motion';
    const status = this.trigger.update(tracked, frameSharpness, frameTime, outside);
    if (outside === 'disabled' || outside === 'captured') this.trigger.restart();
    const progress = outside === 'disabled' || outside === 'captured' ? 0 : status.progress;
    this.lastProgress = progress;
    this.keepBestStill(still, status, frameSharpness, frameTime, tracked.quad, size);

    const live: LiveDetection = {
      quad: tracked.quad,
      frameWidth: camera.width,
      frameHeight: camera.height,
      stable: tracked.stable,
      progress,
      sharpness: frameSharpness,
      relativeSharpness: status.relativeSharpness,
      detectMs: now - frameTime,
      fps: this.detectTimes.length,
      autoCapture: {
        blocker: status.blocker,
        ready: status.ready,
        speed: tracked.speed ?? null,
        maxSpeed: this.tracker.maxSpeed,
        jitter: tracked.jitter ?? null,
        maxJitter: this.tracker.stabilityThreshold,
        windowFrames: tracked.windowFrames ?? 0,
        minFrames: this.tracker.minWindowFrames,
        windowSpanMs: tracked.windowSpanMs ?? 0,
        minWindowSpanMs: this.tracker.windowMs / 2,
        sharpnessReference: status.sharpnessReference,
        minRelativeSharpness: this.trigger.minRelativeSharpness,
        readyMs: progress * this.trigger.stableMs,
        stableMs: this.trigger.stableMs,
        rotationRate: motionFresh ? this.rotationRate : null,
        maxRotationRate,
        bestFrame: this.bestStill !== null,
      },
    };
    this.update({ detection: live });
    this.emitter.emit('detection', live);
    if (status.fire && outside === null) {
      const kept = this.bestStill;
      this.bestStill = null;
      void this.captureWith(kept);
    }
  }

  /**
   * Best-frame buffer: keeps the sharpest ready full-resolution frame of the last
   * `bestFrameMs` (one slot); everything else is closed at once.
   */
  private keepBestStill(
    still: ScannerStill | null,
    status: CaptureTriggerStatus,
    frameSharpness: number,
    frameTime: number,
    quad: Quad | null,
    size: { width: number; height: number },
  ): void {
    const bestFrameMs = this.options.bestFrameMs ?? 300;
    const kept = this.bestStill;
    if (kept && (kept.time < frameTime - bestFrameMs || (!status.ready && status.progress === 0))) {
      this.dropBestStill();
    }
    if (!still) return;
    const current = this.bestStill;
    if (status.ready && (!current || frameSharpness >= current.sharpness)) {
      this.dropBestStill();
      this.bestStill = { still, sharpness: frameSharpness, time: frameTime, quad, ...size };
    } else {
      closeSource(still.image);
    }
  }

  private dropBestStill(): void {
    if (this.bestStill) closeSource(this.bestStill.still.image);
    this.bestStill = null;
  }

  /** Opt-in `devicemotion` listener (see {@link ScannerOptions.deviceMotion}). */
  private watchMotion(): void {
    this.unwatchMotion();
    if (!this.options.deviceMotion || typeof window === 'undefined') return;
    const limit = this.options.maxRotationRate ?? 10;
    const listener = (event: Event) => {
      const rate = (event as DeviceMotionEvent).rotationRate;
      if (!rate) return;
      const value = Math.hypot(rate.alpha ?? 0, rate.beta ?? 0, rate.gamma ?? 0);
      const now = this.now();
      this.rotationRate = value;
      this.lastMotionAt = now;
      if (value > limit) this.lastFastMotionAt = now;
    };
    const target = window;
    target.addEventListener('devicemotion', listener);
    // iOS: a permission granted later (requestDeviceMotionPermission) re-arms the listener.
    const offGranted = onDeviceMotionGranted(() => {
      if (this.state === 'scanning') this.watchMotion();
    });
    this.stopMotion = () => {
      target.removeEventListener('devicemotion', listener);
      offGranted();
    };
  }

  private unwatchMotion(): void {
    this.stopMotion?.();
    this.stopMotion = null;
    this.rotationRate = null;
    this.lastMotionAt = Number.NEGATIVE_INFINITY;
    this.lastFastMotionAt = Number.NEGATIVE_INFINITY;
  }

  // ---- pages -----------------------------------------------------------------------------

  /**
   * Detects, sizes and renders a page. `isValid` is checked after detection (before any
   * rendering); a cancelled operation rejects with {@link PageCancelled}. The caller owns
   * `source` until the returned page is committed.
   */
  private async createPage(
    source: PageSource,
    width: number,
    height: number,
    fallbackQuad: Quad | null,
    isValid: () => boolean,
  ): Promise<ScanPage> {
    const detected = await this.detectStill(source, width, height);
    if (!isValid()) throw new PageCancelled();
    const usable =
      detected && (!fallbackQuad || convexIoU(detected, fallbackQuad) > 0.5) ? detected : null;
    const quad: Quad = usable ??
      fallbackQuad ?? [
        { x: 0, y: 0 },
        { x: width, y: 0 },
        { x: width, y: height },
        { x: 0, y: height },
      ];
    const size = this.outputSize(quad, width, height);
    const filter = this.options.defaultFilter ?? 'color';
    const image = await this.render(source, { quad, filter, rotation: 0, ...size });
    return {
      id: `page-${this.nextPageId++}`,
      source,
      sourceWidth: width,
      sourceHeight: height,
      quad,
      detected: usable !== null || fallbackQuad !== null,
      filter,
      rotation: 0,
      ...size,
      image,
      url: URL.createObjectURL(image),
    };
  }

  private async detectStill(
    source: PageSource,
    width: number,
    height: number,
  ): Promise<Quad | null> {
    // A detector with `detectStill` gets the full-resolution still (a bitmap copy it owns).
    let copy: ImageBitmap | null = null;
    try {
      if (!this.detector) {
        this.detector = (this.options.createDetector ?? createDefaultDetector)();
      }
      const detector = this.detector;
      if (detector.detectStill) {
        if (!('data' in source)) copy = await createImageBitmap(source);
        const still = copy ?? (source as RgbaImage);
        // Read the size first: a bitmap transferred to a worker is detached (0 × 0) afterwards.
        const { width: stillWidth, height: stillHeight } = still;
        const detection = await detector.detectStill(still, this.options.detection);
        this.stillFailures = 0;
        return detection && scaleQuad(detection.quad, width / stillWidth, height / stillHeight);
      }
    } catch (error) {
      // Fall back to detect() on the downscaled frame (with its own fallback handling); report
      // once if the still path keeps failing (two in a row).
      if (++this.stillFailures === 2) {
        this.emitter.emit('error', {
          error: new Error(
            `Still detection failed, using the downscaled frame: ${asError(error).message}`,
          ),
        });
      }
    } finally {
      // The detector normally consumes the copy; closing a consumed (detached) bitmap is a no-op.
      copy?.close();
    }
    const maxSide = this.options.detection?.maxSide ?? 384;
    const frame = 'data' in source ? source : this.grabber.grab(source as ImageBitmap, maxSide);
    if (!frame) return null;
    try {
      const detection = await this.detectWithFallback(frame);
      return detection && scaleQuad(detection.quad, width / frame.width, height / frame.height);
    } catch {
      return null;
    }
  }

  private outputSize(quad: Quad, width: number, height: number): { width: number; height: number } {
    return estimateOutputSize(quad, {
      imageSize: { width, height },
      maxLongSide: this.options.maxOutputSide ?? 4000,
      maxPixels: this.options.maxOutputPixels ?? 12_000_000,
    });
  }

  private render(
    source: PageSource,
    options: { quad: Quad; filter: ScanFilter; rotation: Rotation; width: number; height: number },
  ): Promise<Blob> {
    // Never create (and leak) an owned renderer after dispose().
    if (this.disposed) return Promise.reject(new PageCancelled());
    this.renderer ??= createPageRenderer();
    return this.renderer.render(source, options, {
      type: 'image/jpeg',
      quality: this.options.jpegQuality ?? 0.9,
    });
  }

  private insertPage(page: ScanPage): void {
    const pages = [...this.pages];
    const index = this.replaceIndex;
    this.replaceIndex = null;
    if (index !== null && index < pages.length) {
      releasePage(pages[index], [...pages.filter((_, i) => i !== index), page]);
      pages[index] = page;
    } else {
      pages.push(page);
    }
    this.setPages(pages);
  }

  private findPage(id: string): ScanPage {
    const page = this.pages.find((p) => p.id === id);
    if (!page) throw new RangeError(`Unknown page ${id}`);
    return page;
  }

  private setPages(pages: ScanPage[]): void {
    this.update({ pages });
    this.emitter.emit('pages', { pages });
  }

  // ---- lifecycle -------------------------------------------------------------------------

  private watchLifecycle(): void {
    if (this.unwatchVisibility || typeof document === 'undefined') return;
    // Release the camera while hidden (best effort; see platform notes) and resume on return.
    this.unwatchVisibility = watchPageVisibility(
      () => {
        // A capture in progress keeps its still and finishes rendering, but it no longer
        // returns to scanning (the state is no longer 'capturing'); scanning resumes on return.
        const live = this.state;
        if (live === 'scanning' || live === 'starting' || live === 'capturing') {
          this.resumeWhenVisible = true;
          this.releaseCamera();
          this.setState('idle');
        }
      },
      () => {
        if (this.resumeWhenVisible && this.state === 'idle') {
          this.resumeWhenVisible = false;
          void this.start();
        }
      },
    );
  }

  /** Fails into the recoverable `error` state if a running camera stops delivering frames. */
  private startStallWatchdog(): void {
    if (this.stallTimer !== null) clearInterval(this.stallTimer);
    const timeout = this.options.stallTimeoutMs ?? 4000;
    this.stallTimer = setInterval(() => {
      if (this.state === 'scanning' && this.now() - this.lastFrameAt > timeout) {
        this.fail(new CameraError('start-timeout', 'The camera stopped delivering frames'));
      }
    }, 1000);
  }

  private releaseCamera(): void {
    // Invalidate (and abort) any pending start; a pending capture no longer owns the session.
    this.startGeneration++;
    this.activeCapture = 0;
    this.cancelStart?.();
    this.cancelStart = null;
    this.stopFrames?.();
    this.stopFrames = null;
    if (this.stallTimer !== null) clearInterval(this.stallTimer);
    this.stallTimer = null;
    this.camera?.stop();
    this.camera = null;
    this.busy = false;
    this.capturedQuad = null;
    this.tracker.reset();
    this.trigger.reset();
    this.lastProgress = 0;
    this.dropBestStill();
    this.unwatchMotion();
    if (this.snapshot.detection || this.snapshot.torch || this.snapshot.cameraStage) {
      this.update({ detection: null, torch: false, cameraStage: null });
    }
  }

  private fail(error: Error): void {
    this.releaseCamera();
    this.update({ error });
    this.emitter.emit('error', { error });
    this.setState('error');
  }

  private setState(state: ScannerState): void {
    const previous = this.snapshot.state;
    if (previous === state) return;
    // The motion sensor is only listened to while scanning.
    if (state === 'scanning') this.watchMotion();
    else this.unwatchMotion();
    this.update({ state });
    this.emitter.emit('state', { state, previous });
  }

  private update(changes: Partial<ScannerSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...changes };
    this.emitter.emit('change', this.snapshot);
  }

  private isCapturing(): boolean {
    return this.snapshot.state === 'capturing';
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('The scanner controller has been disposed');
  }
}

function createDefaultDetector(): DocumentDetector {
  if (typeof Worker === 'undefined') return createLazyMainThreadDetector();
  try {
    return createDetectorWorker();
  } catch {
    return createLazyMainThreadDetector();
  }
}

/**
 * The main-thread detector (the whole classic detector), loaded with `import()` on first use:
 * most sessions use the worker and never need it on the main thread.
 */
function createLazyMainThreadDetector(): DocumentDetector {
  let loading: Promise<DocumentDetector> | null = null;
  let disposed = false;
  const load = () =>
    (loading ??= import('../worker/main-thread.ts').then((m) => m.createMainThreadDetector()));
  return {
    async detect(frame, options) {
      if (disposed) throw new Error('Detector disposed');
      return (await load()).detect(frame, options);
    },
    dispose() {
      disposed = true;
      void loading?.then((detector) => detector.dispose());
    },
  };
}

/** Revokes the page URL and closes its source unless another page still uses it. */
function releasePage(page: ScanPage, others: readonly ScanPage[]): void {
  URL.revokeObjectURL(page.url);
  if (others.some((p) => p.source === page.source)) return;
  if ('close' in page.source) page.source.close();
}

function boundingBox(quad: Quad): { x: number; y: number; width: number; height: number } {
  const xs = quad.map((p) => p.x);
  const ys = quad.map((p) => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

/** Internal signal: a page operation was cancelled (stop/dispose) and must not commit. */
class PageCancelled extends Error {
  constructor() {
    super('Page operation cancelled');
  }
}

function closeSource(source: PageSource): void {
  if ('close' in source) source.close();
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

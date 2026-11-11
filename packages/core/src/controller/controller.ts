import type { CameraFeatures, CameraOptions } from '../camera/camera.ts';
import { FrameGrabber, watchPageVisibility } from '../camera/frames.ts';
import type { Detection, DetectOptions } from '../detect/detector.ts';
import { sharpness } from '../detect/sharpness.ts';
import { QuadTracker, type QuadTrackerOptions } from '../detect/tracker.ts';
import type { ScanFilter } from '../filters/filters.ts';
import { convexIoU } from '../geometry/polygon.ts';
import { type Quad, scaleQuad } from '../geometry/quad.ts';
import { estimateOutputSize } from '../geometry/size.ts';
import { type RgbaImage, rgbaToGray } from '../image.ts';
import { createPdf, type PdfOptions } from '../pdf/pdf.ts';
import { normalizeRotation, type Rotation } from '../render/cpu.ts';
import { createPageRenderer, type PageRenderer, type PageSource } from '../render/renderer.ts';
import { createDetectorWorker, createMainThreadDetector } from '../worker/client.ts';
import type { DocumentDetector } from '../worker/processor.ts';
import { openScannerCamera, type ScannerCamera } from './camera-source.ts';
import { Emitter } from './emitter.ts';

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
  /** Auto-capture progress in [0, 1] (stable and sharp time / required time). */
  progress: number;
  /** Laplacian-variance sharpness of the frame and its ratio to the recent maximum. */
  sharpness: number;
  relativeSharpness: number;
  /** Duration of the last detection and detection rate. */
  detectMs: number;
  fps: number;
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
  /** Detector to use. Default: a Web Worker detector, falling back to the main thread. */
  detector?: DocumentDetector;
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
  /** How long the document must be stable and sharp before auto-capture. Default 1000 ms. */
  stableMs?: number;
  /** Minimum sharpness relative to the recent maximum for auto-capture. Default 0.6. */
  minRelativeSharpness?: number;
  /** Filter of new pages. Default `'color'`. */
  defaultFilter?: ScanFilter;
  /**
   * Longer side cap of rendered pages. Default 4000 px (≈ 11 MP for A4), in line with the
   * ≤ 12 MP output budget of docs/platform-notes.md.
   */
  maxOutputSide?: number;
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
  private unwatchVisibility: (() => void) | null = null;
  private stallTimer: ReturnType<typeof setInterval> | null = null;
  private resumeWhenVisible = false;
  private busy = false;
  private lastFrameAt = 0;
  private peakSharpness = 0;
  private stableSince: number | null = null;
  /** Quad of the last capture; auto-capture waits until the document changes. */
  private capturedQuad: Quad | null = null;
  private replaceIndex: number | null = null;
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
    this.snapshot = {
      state: 'idle',
      pages: [],
      detection: null,
      autoCapture: options.autoCapture ?? true,
      torch: false,
      features: null,
      error: null,
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
    this.releaseCamera();
    this.video = target;
    this.setState('starting');
    try {
      const open = this.options.openCamera ?? openScannerCamera;
      const camera = await open(target, this.options.camera ?? {});
      if (this.disposed || this.state !== 'starting') {
        camera.stop();
        return;
      }
      this.camera = camera;
      camera.onEnded(() => {
        if (this.camera === camera) this.fail(new Error('The camera stopped unexpectedly'));
      });
      this.update({ features: camera.features, torch: false, error: null });
      this.tracker.reset();
      this.stableSince = null;
      this.lastFrameAt = this.now();
      this.stopFrames = camera.onFrame(() => this.onFrame());
      this.watchLifecycle();
      this.setState('scanning');
    } catch (error) {
      this.fail(asError(error));
    }
  }

  /** Releases the camera and returns to `idle`. Pages are kept. */
  stop(): void {
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
    this.stableSince = null;
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
  async capture(): Promise<ScanPage | null> {
    const camera = this.camera;
    if (!camera || this.state !== 'scanning') return null;
    const previewQuad = this.snapshot.detection?.quad ?? null;
    this.setState('capturing');
    try {
      const still = await camera.captureStill();
      const preview =
        previewQuad &&
        scaleQuad(previewQuad, still.width / camera.width, still.height / camera.height);
      const page = await this.createPage(still.image, still.width, still.height, preview);
      this.capturedQuad = previewQuad;
      this.stableSince = null;
      const replacing = this.replaceIndex;
      this.insertPage(page);
      this.emitter.emit('capture', { page });
      // The state may have changed while awaiting (stop, finish, page hidden).
      if (this.isCapturing()) {
        if (this.options.reviewAfterCapture || replacing !== null) this.finish();
        else this.setState('scanning');
      }
      return page;
    } catch (error) {
      this.emitter.emit('error', { error: asError(error) });
      if (this.isCapturing()) this.setState('scanning');
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
    const source =
      image instanceof Blob
        ? await createImageBitmap(image, { imageOrientation: 'from-image' })
        : image;
    const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
    const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
    const page = await this.createPage(source, width, height, null);
    this.insertPage(page);
    return page;
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
    return new Blob([createPdf(jpegs, options) as Uint8Array<ArrayBuffer>], {
      type: 'application/pdf',
    });
  }

  /** Releases everything: camera, detector and renderer (if created by the controller), pages. */
  dispose(): void {
    if (this.disposed) return;
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
    const frameSharpness = sharpness(rgbaToGray(frame));
    this.busy = true;
    const started = this.now();
    this.detectWithFallback(frame)
      .then((detection) => {
        this.busy = false;
        if (this.camera !== camera || this.state !== 'scanning') return;
        if (this.emitter.has('frame')) this.emitter.emit('frame', { image: frame, detection });
        this.handleDetection(camera, frame, detection, frameSharpness, this.now() - started);
      })
      .catch((error) => {
        this.busy = false;
        this.emitter.emit('error', { error: asError(error) });
      });
  }

  private async detectWithFallback(frame: RgbaImage): Promise<Detection | null> {
    this.detector ??= createDefaultDetector();
    try {
      return await this.detector.detect(frame, this.options.detection);
    } catch (error) {
      // A worker that cannot load (CSP, bundling) is replaced by the main-thread detector.
      if (!this.ownsDetector || this.disposed) throw error;
      this.detector.dispose();
      this.detector = createMainThreadDetector();
      return this.detector.detect(frame, this.options.detection);
    }
  }

  private handleDetection(
    camera: ScannerCamera,
    frame: RgbaImage,
    detection: Detection | null,
    frameSharpness: number,
    detectMs: number,
  ): void {
    const now = this.now();
    const observed =
      detection &&
      scaleQuad(detection.quad, camera.width / frame.width, camera.height / frame.height);
    const tracked = this.tracker.update(observed, now);
    this.peakSharpness = Math.max(this.peakSharpness * 0.98, frameSharpness);
    const relativeSharpness = this.peakSharpness > 0 ? frameSharpness / this.peakSharpness : 0;
    this.detectTimes.push(now);
    while (this.detectTimes.length > 1 && now - this.detectTimes[0] > 1000)
      this.detectTimes.shift();

    if (this.capturedQuad && (!tracked.quad || convexIoU(tracked.quad, this.capturedQuad) < 0.8)) {
      this.capturedQuad = null; // a different document (or none) is in view now
    }
    const ready =
      this.snapshot.autoCapture &&
      tracked.stable &&
      !tracked.holding &&
      this.capturedQuad === null &&
      relativeSharpness >= (this.options.minRelativeSharpness ?? 0.6);
    if (!ready) this.stableSince = null;
    else this.stableSince ??= now;
    const stableMs = this.options.stableMs ?? 1000;
    const progress =
      this.stableSince === null ? 0 : Math.min(1, (now - this.stableSince) / stableMs);

    const live: LiveDetection = {
      quad: tracked.quad,
      frameWidth: camera.width,
      frameHeight: camera.height,
      stable: tracked.stable,
      progress,
      sharpness: frameSharpness,
      relativeSharpness,
      detectMs,
      fps: this.detectTimes.length,
    };
    this.update({ detection: live });
    this.emitter.emit('detection', live);
    if (progress >= 1) void this.capture();
  }

  // ---- pages -----------------------------------------------------------------------------

  private async createPage(
    source: PageSource,
    width: number,
    height: number,
    fallbackQuad: Quad | null,
  ): Promise<ScanPage> {
    const detected = await this.detectStill(source, width, height);
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
    });
  }

  private render(
    source: PageSource,
    options: { quad: Quad; filter: ScanFilter; rotation: Rotation; width: number; height: number },
  ): Promise<Blob> {
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
        if (this.state === 'scanning' || this.state === 'starting') {
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
    const timeout = this.options.stallTimeoutMs ?? 4000;
    this.stallTimer = setInterval(() => {
      if (this.state === 'scanning' && this.now() - this.lastFrameAt > timeout) {
        this.fail(new Error('The camera stopped delivering frames'));
      }
    }, 1000);
  }

  private releaseCamera(): void {
    this.stopFrames?.();
    this.stopFrames = null;
    if (this.stallTimer !== null) clearInterval(this.stallTimer);
    this.stallTimer = null;
    this.camera?.stop();
    this.camera = null;
    this.busy = false;
    this.stableSince = null;
    this.capturedQuad = null;
    this.tracker.reset();
    if (this.snapshot.detection || this.snapshot.torch)
      this.update({ detection: null, torch: false });
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
  if (typeof Worker === 'undefined') return createMainThreadDetector();
  try {
    return createDetectorWorker();
  } catch {
    return createMainThreadDetector();
  }
}

/** Revokes the page URL and closes its source unless another page still uses it. */
function releasePage(page: ScanPage, others: readonly ScanPage[]): void {
  URL.revokeObjectURL(page.url);
  if (others.some((p) => p.source === page.source)) return;
  if ('close' in page.source) page.source.close();
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

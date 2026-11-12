import { PDFDocument } from 'pdf-lib';
import { describe, expect, it, vi } from 'vitest';
import { CameraError, type CameraOptions } from '../src/camera/index.ts';
import {
  type ScannerCamera,
  ScannerController,
  type ScannerStill,
} from '../src/controller/index.ts';
import { convexIoU, scaleQuad } from '../src/geometry/index.ts';
import {
  createDetectorWorker,
  createMainThreadDetector,
  type DocumentDetector,
} from '../src/worker/index.ts';
import { FakeWorker } from './support/fake-worker.ts';
import { FakeCamera, flush, jpeg, setup } from './support/scanner-fakes.ts';
import { generateScene } from './support/synthetic.ts';

const video = {} as HTMLVideoElement;

/** Fake camera with `grabStill` (best-frame buffer) that records grabs and closes. */
class BufferingCamera extends FakeCamera {
  grabbed: { id: number; closed: boolean }[] = [];
  captureStills = 0;
  async grabStill(): Promise<ScannerStill> {
    const entry = { id: this.grabbed.length, closed: false };
    this.grabbed.push(entry);
    const image = { ...this.still.image, id: entry.id, close: () => (entry.closed = true) };
    return { image, width: 768, height: 576, matchesPreview: true };
  }
  override async captureStill(): Promise<ScannerStill> {
    this.captureStills++;
    return super.captureStill();
  }
}

describe('ScannerController', () => {
  it('starts scanning and reports smoothed detections in video coordinates', async () => {
    const { controller, camera, states, frames } = setup({ autoCapture: false });
    await controller.start(video);
    expect(states).toEqual(['starting', 'scanning']);
    expect(controller.getSnapshot().features?.torch).toBe(true);
    await frames(3);
    const live = controller.getSnapshot().detection;
    expect(live?.frameWidth).toBe(768);
    expect(convexIoU(live!.quad!, camera.quad)).toBeGreaterThan(0.95);
    expect(live!.sharpness).toBeGreaterThan(0);
  });

  it('auto-captures once the document is stable and sharp for stableMs', async () => {
    const { controller, camera, frames, calls } = setup({ stableMs: 500 });
    const captures: string[] = [];
    controller.on('capture', ({ page }) => captures.push(page.id));
    await controller.start(video);
    await frames(8); // tracker needs 8 stable frames
    expect(captures).toHaveLength(0);
    await frames(8);
    expect(captures).toHaveLength(1);
    const page = controller.pages[0];
    expect(convexIoU(page.quad, camera.quad)).toBeGreaterThan(0.95);
    expect(page.detected).toBe(true);
    expect([page.sourceWidth, page.sourceHeight]).toEqual([768, 576]);
    expect(calls[0]).toMatchObject({ filter: 'color', rotation: 0 });
    expect(controller.state).toBe('scanning');
    // The same document does not trigger another capture.
    await frames(20);
    expect(captures).toHaveLength(1);
  });

  it('reports which auto-capture criterion fails', async () => {
    const { controller, frames } = setup({ stableMs: 500 });
    const blockers: (string | null | undefined)[] = [];
    controller.on('detection', (d) => blockers.push(d.autoCapture?.blocker));
    await controller.start(video);
    await frames(6);
    const info = controller.getSnapshot().detection?.autoCapture;
    expect(blockers[0]).toBe('settling');
    expect(blockers.at(-1)).toBeNull();
    expect(info).toMatchObject({
      ready: true,
      maxSpeed: 0.1,
      maxJitter: 0.015,
      minFrames: 4,
      minRelativeSharpness: 0.6,
      stableMs: 500,
      rotationRate: null,
      maxRotationRate: 10,
    });
    expect(info?.speed).toBeLessThan(0.01);
    expect(info?.readyMs).toBeGreaterThan(0);
  });

  it('auto-captures the sharpest recent full-resolution frame and closes the others', async () => {
    const camera = new BufferingCamera();
    const { controller, frames } = setup({ stableMs: 500 }, camera);
    const captures: unknown[] = [];
    controller.on('capture', ({ page }) => captures.push(page.source));
    await controller.start(video);
    await frames(16);
    expect(captures).toHaveLength(1);
    expect(camera.captureStills).toBe(0);
    expect(camera.grabbed.length).toBeGreaterThanOrEqual(2);
    // The page owns the kept frame; every other grabbed frame was closed.
    const kept = (captures[0] as { id: number }).id;
    expect(camera.grabbed.filter((g) => !g.closed).map((g) => g.id)).toEqual([kept]);
  });

  it('grabs best-frame stills only once the progress has started, even if stableMs < bestFrameMs', async () => {
    const camera = new BufferingCamera();
    const { controller, frames } = setup({ stableMs: 100, bestFrameMs: 300 }, camera);
    await controller.start(video);
    await frames(3); // settling: progress 0
    expect(controller.getSnapshot().detection?.progress).toBe(0);
    expect(camera.grabbed).toHaveLength(0);
    controller.setAutoCapture(false);
    await frames(5);
    expect(camera.grabbed).toHaveLength(0);
  });

  it('a manual capture uses a recent best frame', async () => {
    const camera = new BufferingCamera();
    const { controller, frames } = setup({ stableMs: 1000, bestFrameMs: 900 }, camera);
    await controller.start(video);
    await frames(7); // ready for a while, not yet captured
    expect(controller.pages).toHaveLength(0);
    expect(camera.grabbed.length).toBeGreaterThan(0);
    const page = await controller.capture();
    expect(camera.captureStills).toBe(0);
    expect((page?.source as unknown as { id?: number } | undefined)?.id).toBeTypeOf('number');
    expect(camera.grabbed.filter((g) => !g.closed)).toHaveLength(1);
  });

  it('listens to the motion sensor only while scanning and re-arms after a permission grant', async () => {
    class CountingTarget extends EventTarget {
      active = 0;
      added = 0;
      override addEventListener(...args: Parameters<EventTarget['addEventListener']>) {
        if (args[0] === 'devicemotion') {
          this.active++;
          this.added++;
        }
        super.addEventListener(...args);
      }
      override removeEventListener(...args: Parameters<EventTarget['removeEventListener']>) {
        if (args[0] === 'devicemotion') this.active--;
        super.removeEventListener(...args);
      }
    }
    const target = new CountingTarget();
    vi.stubGlobal('window', target);
    vi.stubGlobal('DeviceMotionEvent', { requestPermission: async () => 'granted' });
    const { requestDeviceMotionPermission } = await import('../src/camera/index.ts');
    const { controller, frames } = setup({ autoCapture: false, deviceMotion: true });
    expect(target.active).toBe(0);
    await controller.start(video);
    expect([target.active, target.added]).toEqual([1, 1]);
    expect(await requestDeviceMotionPermission()).toBe(true);
    expect([target.active, target.added]).toEqual([1, 2]);
    await frames(1);
    controller.finish();
    expect(target.active).toBe(0);
    expect(await requestDeviceMotionPermission()).toBe(true);
    expect(target.active).toBe(0); // not scanning: not re-armed
    controller.dispose();
    vi.unstubAllGlobals();
  });

  it('holds auto-capture while the opt-in motion sensor reports fast rotation', async () => {
    const target = new EventTarget();
    vi.stubGlobal('window', target);
    const { controller, frames } = setup({ stableMs: 300, deviceMotion: true });
    const captures: string[] = [];
    controller.on('capture', ({ page }) => captures.push(page.id));
    await controller.start(video);
    const shake = () => {
      const event = new Event('devicemotion');
      Object.defineProperty(event, 'rotationRate', { value: { alpha: 20, beta: 5, gamma: 0 } });
      target.dispatchEvent(event);
    };
    for (let i = 0; i < 12; i++) {
      shake();
      await frames(1);
    }
    const info = controller.getSnapshot().detection?.autoCapture;
    expect(info?.blocker).toBe('device-motion');
    expect(info?.rotationRate).toBeCloseTo(Math.hypot(20, 5), 6);
    expect(captures).toHaveLength(0);
    await frames(12); // the device is still now
    expect(captures).toHaveLength(1);
    controller.dispose();
    vi.unstubAllGlobals();
  });

  it('captures manually and can switch to review after capture', async () => {
    const { controller, frames, states } = setup({ autoCapture: false, reviewAfterCapture: true });
    await controller.start(video);
    await frames(2);
    const page = await controller.capture();
    expect(page).not.toBeNull();
    expect(states).toEqual(['starting', 'scanning', 'capturing', 'reviewing']);
    expect(controller.state).toBe('reviewing');
    expect(await controller.capture()).toBeNull(); // not scanning anymore
  });

  it('edits, reorders, retakes and removes pages', async () => {
    const { controller, frames, calls } = setup({ autoCapture: false });
    await controller.start(video);
    await frames(2);
    const first = (await controller.capture())!;
    const second = (await controller.capture())!;
    const updated = await controller.updatePage(first.id, { filter: 'bw', rotation: -90 });
    expect(updated).toMatchObject({ id: first.id, filter: 'bw', rotation: 270 });
    expect(updated.url).not.toBe(first.url);
    expect(calls.at(-1)).toMatchObject({ filter: 'bw', rotation: 270 });

    const smaller = scaleQuad(updated.quad, 0.5);
    const cropped = await controller.updatePage(first.id, { quad: smaller });
    expect(cropped.width).toBeLessThan(updated.width);

    controller.movePage(second.id, 0);
    expect(controller.pages.map((p) => p.id)).toEqual([second.id, first.id]);

    controller.finish();
    await controller.retake(second.id);
    expect(controller.state).toBe('scanning');
    await frames(2);
    const replacement = (await controller.capture())!;
    expect(controller.pages.map((p) => p.id)).toEqual([replacement.id, first.id]);
    expect(controller.state).toBe('reviewing');

    controller.removePage(first.id);
    expect(controller.pages).toHaveLength(1);
  });

  it('exports a PDF with one page per scan', async () => {
    const { controller, frames } = setup({ autoCapture: false });
    await controller.start(video);
    await frames(2);
    await controller.capture();
    await controller.capture();
    const pdf = await controller.exportPdf({ pageSize: 'a4' });
    expect(pdf.type).toBe('application/pdf');
    const doc = await PDFDocument.load(new Uint8Array(await pdf.arrayBuffer()));
    expect(doc.getPageCount()).toBe(2);
  });

  it('adds pages from images without a camera', async () => {
    const { controller } = setup();
    const scene = generateScene('perspective', 4, 640, 480);
    const page = await controller.addImage(scene.image);
    expect(controller.state).toBe('idle');
    expect(convexIoU(page.quad, scene.quad!)).toBeGreaterThan(0.95);
    const blank = generateScene('none', 4, 320, 240);
    const whole = await controller.addImage(blank.image);
    expect(whole.detected).toBe(false);
    expect(whole.quad[2]).toEqual({ x: 320, y: 240 });
  });

  it('gives detectors with detectStill the full-resolution still', async () => {
    const scene = generateScene('perspective', 4, 640, 480);
    const inner = createMainThreadDetector();
    const stills: { width: number; height: number }[] = [];
    const detector = {
      detect: vi.fn(inner.detect),
      detectStill: vi.fn(async (still: { width: number; height: number }) => {
        stills.push({ width: still.width, height: still.height });
        return { quad: scene.quad!, score: 0.9 };
      }),
      dispose: () => inner.dispose(),
    };
    const { controller } = setup({ detector });
    const page = await controller.addImage(scene.image);
    expect(stills).toEqual([{ width: 640, height: 480 }]);
    expect(detector.detect).not.toHaveBeenCalled();
    expect(convexIoU(page.quad, scene.quad!)).toBeGreaterThan(0.999);
    // A failing detectStill falls back to detect() on the downscaled frame.
    detector.detectStill.mockRejectedValueOnce(new Error('no GPU'));
    const fallback = await controller.addImage(scene.image);
    expect(detector.detect).toHaveBeenCalledTimes(1);
    expect(convexIoU(fallback.quad, scene.quad!)).toBeGreaterThan(0.95);
  });

  it('passes bitmap sources to detectStill as a copy and keeps their size', async () => {
    const scene = generateScene('perspective', 4, 640, 480);
    // A bitmap-like source: the controller copies it with createImageBitmap; the detector
    // consumes the copy like a worker transfer would (the size becomes 0 × 0).
    const copies: { width: number; height: number; close: () => void }[] = [];
    vi.stubGlobal('createImageBitmap', async (src: { width: number; height: number }) => {
      const copy = { width: src.width, height: src.height, close: vi.fn() };
      copies.push(copy);
      return copy;
    });
    try {
      const detector = {
        detect: async () => null,
        detectStill: async (still: { width: number; height: number }) => {
          Object.assign(still, { width: 0, height: 0 });
          return { quad: scaleQuad(scene.quad!, 0.5, 0.5), score: 0.9 };
        },
        dispose: () => {},
      };
      const { controller } = setup({ detector });
      // A 320×240 bitmap-like source (no pixels: the fake renderer does not read them).
      const bitmapLike = { width: 320, height: 240, close() {} };
      const page = await controller.addImage(bitmapLike as unknown as ImageBitmap);
      expect(copies).toHaveLength(1);
      expect(page.quad[0].x).toBeCloseTo(scene.quad![0].x * 0.5, 6);
      expect(Number.isFinite(page.quad[2].y)).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('closes the still copy the detector did not consume', async () => {
    const copies: { width: number; height: number; close: ReturnType<typeof vi.fn> }[] = [];
    vi.stubGlobal('createImageBitmap', async (src: { width: number; height: number }) => {
      const copy = { width: src.width, height: src.height, close: vi.fn() };
      copies.push(copy);
      return copy;
    });
    try {
      // A detector that answers without consuming (closing or transferring) the still.
      const detector = { detect: async () => null, detectStill: async () => null, dispose() {} };
      const { controller } = setup({ detector });
      const bitmapLike = { width: 320, height: 240, close() {} };
      await controller.addImage(bitmapLike as unknown as ImageBitmap);
      expect(copies).toHaveLength(1);
      expect(copies[0].close).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('reports a still path that keeps failing once and falls back to detect()', async () => {
    const scene = generateScene('perspective', 4, 640, 480);
    const inner = createMainThreadDetector();
    const detector = {
      detect: vi.fn(inner.detect),
      detectStill: vi.fn(async () => {
        throw new Error('GPU gone');
      }),
      dispose: () => inner.dispose(),
    };
    const { controller } = setup({ detector });
    const errors: string[] = [];
    controller.on('error', ({ error }) => errors.push(error.message));
    for (let i = 0; i < 3; i++) {
      const page = await controller.addImage(scene.image);
      expect(convexIoU(page.quad, scene.quad!)).toBeGreaterThan(0.95);
    }
    expect(detector.detect).toHaveBeenCalledTimes(3);
    expect(errors).toEqual(['Still detection failed, using the downscaled frame: GPU gone']);
  });

  it('toggles torch and auto-capture', async () => {
    const { controller, camera } = setup();
    await controller.start(video);
    await controller.setTorch(true);
    expect(camera.torch).toBe(true);
    expect(controller.getSnapshot().torch).toBe(true);
    controller.setAutoCapture(false);
    expect(controller.getSnapshot().autoCapture).toBe(false);
  });

  it('enters the error state when the camera fails or ends', async () => {
    const failing = setup({
      openCamera: async () => {
        throw new Error('denied');
      },
    });
    const errors: string[] = [];
    failing.controller.on('error', ({ error }) => errors.push(error.message));
    await failing.controller.start(video);
    expect(failing.controller.state).toBe('error');
    expect(errors).toEqual(['denied']);

    const { controller, camera } = setup();
    await controller.start(video);
    camera.end();
    expect(controller.state).toBe('error');
    expect(camera.stopped).toBe(true);
    await controller.start(); // retry with the same video element
    expect(controller.state).toBe('scanning');
  });

  it('notifies subscribers with new snapshots and cleans up on dispose', async () => {
    const { controller, camera } = setup();
    let notifications = 0;
    const before = controller.getSnapshot();
    controller.subscribe(() => notifications++);
    await controller.start(video);
    expect(notifications).toBeGreaterThan(0);
    expect(controller.getSnapshot()).not.toBe(before);
    controller.dispose();
    expect(camera.stopped).toBe(true);
    expect(controller.state).toBe('idle');
    await expect(controller.start(video)).rejects.toThrow('disposed');
  });

  it('retries a recoverable startup failure once, then succeeds', async () => {
    const camera = new FakeCamera();
    const opens: CameraOptions[] = [];
    const { controller } = setup({
      openCamera: async (_video, options) => {
        opens.push(options);
        if (opens.length === 1) {
          options.onStage?.('playing');
          throw new CameraError('aborted', 'Camera startup was interrupted');
        }
        return camera;
      },
    });
    await controller.start(video);
    expect(opens).toHaveLength(2);
    expect(controller.state).toBe('scanning');
    expect(controller.getSnapshot().cameraStage).toBeNull();
  });

  it('surfaces repeated recoverable failures and recovers via retry()', async () => {
    let fail = true;
    let opens = 0;
    const { controller } = setup({
      openCamera: async () => {
        opens++;
        if (fail) throw new CameraError('start-timeout', 'The camera delivered no frames');
        return new FakeCamera();
      },
    });
    await controller.start(video);
    expect(opens).toBe(2);
    expect(controller.state).toBe('error');
    expect((controller.getSnapshot().error as CameraError).recoverable).toBe(true);
    fail = false;
    await controller.retry();
    expect(controller.state).toBe('scanning');
  });

  it('does not retry a permission denial', async () => {
    let opens = 0;
    const { controller } = setup({
      openCamera: async () => {
        opens++;
        throw new CameraError('permission-denied', 'Camera permission was denied');
      },
    });
    await controller.start(video);
    expect(opens).toBe(1);
    expect(controller.state).toBe('error');
  });

  it('cancels a pending start when stopped', async () => {
    const camera = new FakeCamera();
    let resolveOpen: (c: ScannerCamera) => void = () => {};
    let signal: AbortSignal | undefined;
    const { controller } = setup({
      openCamera: (_video, options) => {
        signal = options.signal;
        options.onStage?.('acquiring');
        return new Promise((resolve) => {
          resolveOpen = resolve;
        });
      },
    });
    const pending = controller.start(video);
    expect(controller.getSnapshot().cameraStage).toBe('acquiring');
    controller.stop();
    expect(signal?.aborted).toBe(true);
    resolveOpen(camera);
    await pending;
    expect(camera.stopped).toBe(true);
    expect(controller.state).toBe('idle');
    expect(controller.getSnapshot()).toMatchObject({ error: null, cameraStage: null });
  });

  it('replaces a failed worker detector by the main-thread detector and reports why', async () => {
    // A worker whose script fails to load: it never answers, then reports a bare error event.
    const worker = new FakeWorker({ ready: false });
    worker.responsive = false;
    let created = 0;
    const { controller, camera, frames } = setup({
      detector: undefined,
      createDetector: () => {
        created++;
        const detector = createDetectorWorker({ worker });
        setTimeout(() => worker.error());
        return detector;
      },
      autoCapture: false,
    });
    const errors: string[] = [];
    controller.on('error', ({ error }) => errors.push(error.message));
    await controller.start(video);
    await frames(3);
    expect(created).toBe(1);
    expect(worker.terminated).toBe(true);
    expect(errors).toEqual([
      expect.stringContaining(
        'Falling back to main-thread detection: Detector worker failed to load',
      ),
    ]);
    expect(convexIoU(controller.getSnapshot().detection!.quad!, camera.quad)).toBeGreaterThan(0.95);
  });

  it('disposes detectors it created, but not injected ones', async () => {
    const disposed: string[] = [];
    const tracked = (name: string): DocumentDetector => {
      const inner = createMainThreadDetector();
      return {
        detect: (frame, options) => inner.detect(frame, options),
        dispose: () => disposed.push(name),
      };
    };
    const owned = setup({ detector: undefined, createDetector: () => tracked('owned') });
    await owned.controller.start(video);
    await owned.frames(1);
    owned.controller.dispose();
    const injected = setup({ detector: tracked('injected') });
    await injected.controller.start(video);
    await injected.frames(1);
    injected.controller.dispose();
    expect(disposed).toEqual(['owned']);
  });

  it('discards a capture that completes after stop() and releases its still', async () => {
    const { controller, camera, frames } = setup({ autoCapture: false });
    await controller.start(video);
    await frames(2);
    let release: (still: ScannerStill) => void = () => {};
    let closed = false;
    const image = {
      width: 768,
      height: 576,
      close: () => (closed = true),
    } as unknown as ImageBitmap;
    camera.nextStill = () => new Promise((resolve) => (release = resolve));
    const pending = controller.capture();
    controller.stop();
    release({ image, width: 768, height: 576, matchesPreview: true });
    expect(await pending).toBeNull();
    expect(closed).toBe(true);
    expect(controller.pages).toHaveLength(0);
    expect(controller.state).toBe('idle');
  });

  it('does not add pages whose rendering finishes after dispose()', async () => {
    let finish: (blob: Blob) => void = () => {};
    const revoked: string[] = [];
    const originalRevoke = URL.revokeObjectURL;
    URL.revokeObjectURL = (url: string) => {
      revoked.push(url);
      originalRevoke(url);
    };
    try {
      const controller = new ScannerController({
        detector: createMainThreadDetector(),
        renderer: {
          backend: 'cpu',
          render: () => new Promise<Blob>((resolve) => (finish = resolve)),
          dispose() {},
        },
      });
      const pending = controller.addImage(generateScene('easy', 8, 320, 240).image);
      await flush();
      controller.dispose();
      finish(new Blob([jpeg], { type: 'image/jpeg' }));
      await expect(pending).rejects.toThrow('disposed');
      expect(controller.pages).toHaveLength(0);
      expect(revoked).toHaveLength(1);
    } finally {
      URL.revokeObjectURL = originalRevoke;
    }
  });

  it('trusts the still detection of a differently framed photo', async () => {
    const { controller, camera, frames } = setup({ autoCapture: false });
    await controller.start(video);
    await frames(3); // preview quad tracked for scene 2
    // The photo shows a different crop (here: a different scene) than the preview.
    const photo = generateScene('easy', 15, 1024, 768);
    // Precondition: the preview quad, naively scaled to the photo, disagrees with the photo's
    // document (IoU ≤ 0.5), so the old overlap rule would have rejected the photo detection.
    const naive = scaleQuad(controller.getSnapshot().detection!.quad!, 1024 / 768, 768 / 576);
    expect(convexIoU(naive, photo.quad!)).toBeLessThanOrEqual(0.5);
    camera.nextStill = async () => ({
      image: photo.image,
      width: 1024,
      height: 768,
      matchesPreview: false,
    });
    const page = (await controller.capture())!;
    expect(convexIoU(page.quad, photo.quad!)).toBeGreaterThan(0.95);
    expect(page.detected).toBe(true);
  });

  it('caps rendered pages at 12 MP by default', async () => {
    const { controller, calls } = setup();
    // A blank 4000×4000 image: no document, so the whole (square) image is the page.
    const blank = { width: 4000, height: 4000, data: new Uint8ClampedArray(4000 * 4000 * 4) };
    const page = await controller.addImage(blank);
    expect(page.detected).toBe(false);
    expect(page.width * page.height).toBeLessThanOrEqual(12_000_000);
    expect(page.width).toBe(page.height);
    expect(calls.at(-1)).toMatchObject({ width: page.width, height: page.height });
  });

  it('stops page creation when disposed during still detection', async () => {
    let finishDetection: () => void = () => {};
    const deferred: DocumentDetector = {
      detect: () => new Promise((resolve) => (finishDetection = () => resolve(null))),
      dispose() {},
    };
    // No renderer option: the controller would create its own (see controller-renderer.test.ts).
    const { controller, camera } = setup({
      detector: deferred,
      renderer: undefined,
      autoCapture: false,
    });
    const errors: string[] = [];
    controller.on('error', ({ error }) => errors.push(error.message));
    await controller.start(video);
    let closed = false;
    const image = { ...camera.still.image, close: () => (closed = true) };
    camera.nextStill = async () => ({ image, width: 768, height: 576, matchesPreview: true });
    const pending = controller.capture();
    await flush();
    controller.dispose();
    finishDetection();
    expect(await pending).toBeNull();
    expect(closed).toBe(true);
    expect(errors).toEqual([]);

    const importing = setup({ detector: deferred, renderer: undefined });
    const added = importing.controller.addImage(generateScene('easy', 5, 320, 240).image);
    await flush();
    importing.controller.dispose();
    finishDetection();
    await expect(added).rejects.toThrow('disposed while adding an image');
  });

  it('closes an uncommitted still or decoded bitmap when rendering fails', async () => {
    const failing = {
      backend: 'cpu' as const,
      render: () => Promise.reject(new Error('render failed')),
      dispose() {},
    };
    const { controller, camera } = setup({ renderer: failing, autoCapture: false });
    const errors: string[] = [];
    controller.on('error', ({ error }) => errors.push(error.message));
    await controller.start(video);
    let stillClosed = false;
    const image = { ...camera.still.image, close: () => (stillClosed = true) };
    camera.nextStill = async () => ({ image, width: 768, height: 576, matchesPreview: true });
    expect(await controller.capture()).toBeNull();
    expect(stillClosed).toBe(true);
    expect(errors).toEqual(['render failed']);
    expect(controller.state).toBe('scanning');

    // A Blob is decoded by the controller: the bitmap is closed when the page fails.
    let bitmapClosed = false;
    const scene = generateScene('easy', 6, 320, 240).image;
    vi.stubGlobal('createImageBitmap', async () => ({
      ...scene,
      close: () => (bitmapClosed = true),
    }));
    try {
      await expect(controller.addImage(new Blob(['jpeg']))).rejects.toThrow('render failed');
      expect(bitmapClosed).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('falls back once when the detector factory throws synchronously', async () => {
    let calls = 0;
    const { controller, camera, frames } = setup({
      detector: undefined,
      createDetector: () => {
        calls++;
        throw new DOMException('Worker creation blocked', 'SecurityError');
      },
      autoCapture: false,
    });
    const errors: string[] = [];
    controller.on('error', ({ error }) => errors.push(error.message));
    await controller.start(video);
    await frames(4);
    expect(calls).toBe(1);
    expect(errors).toEqual(['Falling back to main-thread detection: Worker creation blocked']);
    expect(convexIoU(controller.getSnapshot().detection!.quad!, camera.quad)).toBeGreaterThan(0.95);
  });
});

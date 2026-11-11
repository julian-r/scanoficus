import { readFileSync } from 'node:fs';
import { PDFDocument } from 'pdf-lib';
import { describe, expect, it } from 'vitest';
import {
  type ScannerCamera,
  ScannerController,
  type ScannerOptions,
  type ScannerState,
} from '../src/controller/index.ts';
import { convexIoU, type Quad, scaleQuad } from '../src/geometry/index.ts';
import type { PageRenderer } from '../src/render/index.ts';
import { createMainThreadDetector } from '../src/worker/index.ts';
import { generateScene } from './support/synthetic.ts';

const jpeg = new Uint8Array(readFileSync(new URL('./fixtures/rgb.jpg', import.meta.url)));

/** Camera serving a synthetic scene: 384×288 preview frames, 768×576 stills. */
class FakeCamera implements ScannerCamera {
  readonly width = 768;
  readonly height = 576;
  readonly features = { torch: true, photo: false };
  torch = false;
  stopped = false;
  private callback: ((t: number) => void) | null = null;
  private ended: (() => void) | null = null;
  readonly frame;
  readonly still;
  readonly quad: Quad;

  constructor(seed = 2) {
    this.frame = generateScene('easy', seed, 384, 288);
    this.still = generateScene('easy', seed, 768, 576);
    this.quad = this.still.quad as Quad;
  }

  onFrame(callback: (t: number) => void) {
    this.callback = callback;
    return () => {
      this.callback = null;
    };
  }
  tick() {
    this.callback?.(0);
  }
  grabFrame() {
    return this.frame.image;
  }
  async captureStill() {
    return { image: this.still.image, width: 768, height: 576 };
  }
  async setTorch(on: boolean) {
    this.torch = on;
    return on;
  }
  onEnded(callback: () => void) {
    this.ended = callback;
  }
  end() {
    this.ended?.();
  }
  stop() {
    this.stopped = true;
  }
}

/** Renderer recording its calls and returning a real (tiny) JPEG. */
function fakeRenderer() {
  const calls: Parameters<PageRenderer['render']>[1][] = [];
  const renderer: PageRenderer = {
    backend: 'cpu',
    async render(_source, options) {
      calls.push(options);
      return new Blob([jpeg], { type: 'image/jpeg' });
    },
    dispose() {},
  };
  return { renderer, calls };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup(options: Partial<ScannerOptions> = {}, camera = new FakeCamera()) {
  let time = 0;
  const { renderer, calls } = fakeRenderer();
  const controller = new ScannerController({
    detector: createMainThreadDetector(),
    renderer,
    openCamera: async () => camera,
    now: () => time,
    ...options,
  });
  const states: ScannerState[] = [];
  controller.on('state', ({ state }) => states.push(state));
  /** Delivers `n` frames, advancing the clock by `ms` per frame. */
  const frames = async (n: number, ms = 100) => {
    for (let i = 0; i < n; i++) {
      time += ms;
      camera.tick();
      await flush();
    }
  };
  return { controller, camera, calls, states, frames };
}

const video = {} as HTMLVideoElement;

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
});

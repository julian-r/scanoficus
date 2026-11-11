/** Fakes for ScannerController tests: synthetic camera, recording renderer, setup helper. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  type ScannerCamera,
  ScannerController,
  type ScannerOptions,
  type ScannerState,
  type ScannerStill,
} from '../../src/controller/index.ts';
import type { Quad } from '../../src/geometry/index.ts';
import type { PageRenderer } from '../../src/render/index.ts';
import { createMainThreadDetector } from '../../src/worker/index.ts';
import { generateScene } from './synthetic.ts';

export const jpeg = new Uint8Array(readFileSync(join(process.cwd(), 'test/fixtures/rgb.jpg')));

/** Camera serving a synthetic scene: 384×288 preview frames, 768×576 stills. */
export class FakeCamera implements ScannerCamera {
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
  /** Overrides the next stills (e.g. a differently framed photo, or a deferred capture). */
  nextStill: (() => Promise<ScannerStill>) | null = null;
  async captureStill(): Promise<ScannerStill> {
    if (this.nextStill) return this.nextStill();
    return { image: this.still.image, width: 768, height: 576, matchesPreview: true };
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
export function fakeRenderer() {
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

export const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

export function setup(options: Partial<ScannerOptions> = {}, camera = new FakeCamera()) {
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

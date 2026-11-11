// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import type { ScannerStill } from '../src/controller/index.ts';
import { FakeCamera, setup } from './support/scanner-fakes.ts';

function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
  document.dispatchEvent(new Event('visibilitychange'));
}

afterEach(() => setVisibility('visible'));

describe('ScannerController page lifecycle', () => {
  it('releases the camera when hidden during a capture and resumes only when visible', async () => {
    const cameras: FakeCamera[] = [];
    const { controller, frames } = setup({
      autoCapture: false,
      openCamera: async () => {
        const camera = new FakeCamera();
        cameras.push(camera);
        return camera;
      },
    });
    await controller.start(document.createElement('video'));
    await frames(2);
    const first = cameras[0];
    let release: (still: ScannerStill) => void = () => {};
    first.nextStill = () => new Promise((resolve) => (release = resolve));
    const pending = controller.capture();
    expect(controller.state).toBe('capturing');

    setVisibility('hidden');
    expect(first.stopped).toBe(true);
    expect(controller.state).toBe('idle');

    // The still taken before hiding is still processed, but scanning does not restart.
    release({ image: first.still.image, width: 768, height: 576, matchesPreview: true });
    expect(await pending).not.toBeNull();
    expect(controller.pages).toHaveLength(1);
    expect(controller.state).toBe('idle');
    expect(cameras).toHaveLength(1);

    setVisibility('visible');
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(cameras).toHaveLength(2);
    expect(controller.state).toBe('scanning');
    controller.dispose();
  });
});

describe('ScannerController capture ownership', () => {
  for (const outcome of ['resolves', 'rejects'] as const) {
    it(`an older capture that ${outcome} after hide/show does not touch the newer session`, async () => {
      const cameras: FakeCamera[] = [];
      const { controller, frames } = setup({
        autoCapture: false,
        openCamera: async () => {
          const camera = new FakeCamera();
          cameras.push(camera);
          return camera;
        },
      });
      await controller.start(document.createElement('video'));
      await frames(2);
      let settleA: (still: ScannerStill) => void = () => {};
      let failA: (error: Error) => void = () => {};
      cameras[0].nextStill = () =>
        new Promise((resolve, reject) => {
          settleA = resolve;
          failA = reject;
        });
      const captureA = controller.capture();

      setVisibility('hidden');
      setVisibility('visible');
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(controller.state).toBe('scanning'); // session B
      cameras[1].nextStill = () => new Promise(() => {}); // capture B stays in progress
      const captureB = controller.capture();
      expect(controller.state).toBe('capturing');

      if (outcome === 'resolves') {
        settleA({ image: cameras[0].still.image, width: 768, height: 576, matchesPreview: true });
        expect(await captureA).not.toBeNull(); // A's page is kept…
        expect(controller.pages).toHaveLength(1);
      } else {
        failA(new Error('capture A failed'));
        expect(await captureA).toBeNull();
      }
      expect(controller.state).toBe('capturing'); // …but B's state is untouched
      expect(cameras[1].stopped).toBe(false);
      void captureB;
      controller.dispose();
    });
  }
});

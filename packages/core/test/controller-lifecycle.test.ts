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

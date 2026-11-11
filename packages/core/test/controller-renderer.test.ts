import { describe, expect, it, vi } from 'vitest';
import { createPageRenderer } from '../src/render/renderer.ts';
import type { DocumentDetector } from '../src/worker/index.ts';
import { flush, jpeg, setup } from './support/scanner-fakes.ts';
import { generateScene } from './support/synthetic.ts';

// Replace the default renderer factory by a spy returning a fake renderer, so that tests can
// assert directly whether the controller created its own renderer.
vi.mock('../src/render/renderer.ts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/render/renderer.ts')>();
  return {
    ...actual,
    createPageRenderer: vi.fn(() => ({
      backend: 'cpu',
      render: async () => new Blob([jpeg], { type: 'image/jpeg' }),
      dispose() {},
    })),
  };
});

describe('ScannerController default renderer', () => {
  it('creates its own renderer when needed (control)', async () => {
    vi.mocked(createPageRenderer).mockClear();
    const { controller } = setup({ renderer: undefined });
    await controller.addImage(generateScene('easy', 5, 320, 240).image);
    expect(createPageRenderer).toHaveBeenCalledTimes(1);
    controller.dispose();
  });

  it('never creates a renderer after dispose() while still detection is pending', async () => {
    vi.mocked(createPageRenderer).mockClear();
    let finishDetection: () => void = () => {};
    const deferred: DocumentDetector = {
      detect: () => new Promise((resolve) => (finishDetection = () => resolve(null))),
      dispose() {},
    };
    const { controller } = setup({ detector: deferred, renderer: undefined });
    const added = controller.addImage(generateScene('easy', 5, 320, 240).image);
    await flush();
    controller.dispose();
    finishDetection();
    await expect(added).rejects.toThrow('disposed while adding an image');
    expect(createPageRenderer).not.toHaveBeenCalled();
  });
});

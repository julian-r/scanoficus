import type { DetectOptions } from '../detect/detector.ts';
import { type DetectorFrame, type DocumentDetector, FrameProcessor } from './processor.ts';

/**
 * {@link DocumentDetector} running on the calling thread, for environments without module
 * workers or `OffscreenCanvas` (and for Node). Same results as the worker, but detection
 * (≈5 ms at 384 px on a laptop, more on phones) blocks the thread; run it at a reduced rate.
 */
export function createMainThreadDetector(): DocumentDetector {
  let processor: FrameProcessor | null = new FrameProcessor();
  return {
    detect(frame: DetectorFrame, detectOptions?: DetectOptions) {
      if (!processor) return Promise.reject(new Error('Detector disposed'));
      try {
        return Promise.resolve(processor.process(frame, detectOptions));
      } catch (error) {
        return Promise.reject(error);
      }
    },
    dispose() {
      processor = null;
    },
  };
}

import { CONFIDENT_SCORE, DEFAULT_MIN_SCORE } from '@scanoficus/core/detect';
import { createMainThreadDetector, type DocumentDetector } from '@scanoficus/core/worker';
import { isBitmap } from './detector.ts';

/** Options for {@link createHybridDetector}. */
export interface HybridDetectorOptions {
  /** The network detector (e.g. `createNnDetector` or `createNnDetectorWorker`). */
  nn: DocumentDetector;
  /** The classic detector. Default: `createMainThreadDetector()` from `@scanoficus/core/worker`. */
  classic?: DocumentDetector;
  /**
   * Classic detections scoring at least this are returned as is. Default `CONFIDENT_SCORE` from
   * `@scanoficus/core/detect`.
   */
  confidentScore?: number;
  /**
   * Called when the network fails on a frame. The frame is then answered by the classic
   * detector if its score reaches `DEFAULT_MIN_SCORE` (its usual threshold), else with `null`.
   */
  onError?: (error: unknown) => void;
}

/**
 * Classic first, network on doubt: every frame goes to the classic detector (with
 * `minScore: 0`); a quad scoring at least `confidentScore` is returned directly (see
 * `CONFIDENT_SCORE` for what that score means; it costs only the classic detector). Other
 * frames go to the network, which then decides alone (its presence threshold vetoes unsure
 * classic quads). The `score` of the result is therefore either the classic edge score or the
 * network's presence probability. If the network fails on a frame, the classic result answers
 * it when it reaches `DEFAULT_MIN_SCORE` (else `null`) and `onError` is called; a failing
 * classic detector rejects the frame. Both detectors are disposed with the hybrid.
 *
 * The classic detector gets the frame as it is: RGBA frames (e.g. `ImageData`) unchanged, so it
 * uses its colour edges; `ImageBitmap` frames as a clone (the core reads them back as RGBA), so
 * both detectors can consume their copy. Do not give the classic detector `transferPixels`:
 * the network still needs the RGBA buffer.
 */
export function createHybridDetector(options: HybridDetectorOptions): DocumentDetector {
  const { nn } = options;
  const classic = options.classic ?? createMainThreadDetector();
  const confident = options.confidentScore ?? CONFIDENT_SCORE;
  return {
    async detect(frame, detectOptions) {
      const copy = isBitmap(frame) ? await createImageBitmap(frame) : frame;
      let first: Awaited<ReturnType<DocumentDetector['detect']>>;
      try {
        first = await classic.detect(copy, { ...detectOptions, minScore: 0 });
      } catch (error) {
        if (isBitmap(frame)) frame.close();
        throw error;
      }
      if (first && first.score >= confident) {
        if (isBitmap(frame)) frame.close();
        return first;
      }
      try {
        return await nn.detect(frame, detectOptions);
      } catch (error) {
        options.onError?.(error);
        const minScore = detectOptions?.minScore ?? DEFAULT_MIN_SCORE;
        return first && first.score >= minScore ? first : null;
      }
    },
    dispose() {
      classic.dispose();
      nn.dispose();
    },
  };
}

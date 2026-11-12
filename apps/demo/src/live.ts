import { requestDeviceMotionPermission } from '@scanoficus/core/camera';
import type { ScanDocumentElement } from '@scanoficus/elements';
import { attachDebugOverlay } from './debug.ts';
import { createDemoDetector, type DetectorChoice, detectionMaxSide } from './detector.ts';
import { button, renderResults } from './results.ts';

/** Live scanning with the full `<scan-document>` flow in a full-screen overlay. */
export function mountLive(
  main: HTMLElement,
  debug: () => boolean,
  choice: () => DetectorChoice,
): () => void {
  const intro = document.createElement('section');
  intro.innerHTML = `<h1>Live scan</h1>
<p>Point the camera at a document. It is captured automatically when the outline is steady
(switch to manual mode in the top bar). Scan as many pages as you like, then tap Done to review,
change filters, crop, rotate and save a PDF.</p>`;
  const results = document.createElement('section');
  let close: (() => void) | null = null;
  let clearResults = () => {};

  const open = () => {
    // iOS only grants motion-sensor access from a user gesture: ask in this tap handler.
    if (motion.checked) void requestDeviceMotionPermission();
    const overlay = document.createElement('div');
    overlay.className = 'scanner-overlay';
    const scanner = document.createElement('scan-document') as ScanDocumentElement;
    scanner.setAttribute('page-size', 'a4');
    // The controller creates (and disposes) the chosen detector; it is used for live frames
    // and for the re-detection on the captured still. Its workers come from the libraries'
    // default worker URLs, which Vite bundles from `new Worker(new URL(…))`.
    const detector = choice();
    scanner.options = {
      camera: { resolution: highRes.checked ? 'max' : 'hd' },
      createDetector: () => createDemoDetector(detector),
      detection: { maxSide: detectionMaxSide(detector) },
      deviceMotion: motion.checked,
    };
    overlay.append(scanner);
    document.body.append(overlay);
    document.body.classList.add('scanning');
    const detachDebug =
      debug() && scanner.controller ? attachDebugOverlay(scanner.controller, overlay) : null;
    close = () => {
      detachDebug?.();
      overlay.remove();
      document.body.classList.remove('scanning');
      close = null;
    };
    scanner.addEventListener('scan-complete', (e) => {
      clearResults();
      clearResults = renderResults(results, e.detail.pages, e.detail.pdf);
      close?.();
    });
    scanner.addEventListener('scan-cancel', () => close?.());
    scanner.addEventListener('scan-error', (e) => console.error(e.detail.error));
  };

  const highResLabel = document.createElement('label');
  highResLabel.className = 'option';
  highResLabel.innerHTML =
    '<input type="checkbox"> High-resolution stream (sharper pages, more battery and memory)';
  const highRes = highResLabel.querySelector('input') as HTMLInputElement;
  const motionLabel = document.createElement('label');
  motionLabel.className = 'option';
  motionLabel.innerHTML =
    '<input type="checkbox"> Motion sensor: also wait while the phone rotates (asks for permission on iOS)';
  const motion = motionLabel.querySelector('input') as HTMLInputElement;
  motion.checked = localStorage.getItem('scanoficus-motion') === '1';
  motion.addEventListener('change', () => {
    localStorage.setItem('scanoficus-motion', motion.checked ? '1' : '0');
  });
  intro.append(highResLabel, motionLabel, button('Start scanning', open, 'primary'));
  main.replaceChildren(intro, results);
  return () => {
    close?.();
    clearResults();
  };
}

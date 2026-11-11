import type { ScanDocumentElement } from '@scanoficus/elements';
import { attachDebugOverlay } from './debug.ts';
import { createDemoDetector } from './detector.ts';
import { button, renderResults } from './results.ts';

/** Live scanning with the full `<scan-document>` flow in a full-screen overlay. */
export function mountLive(main: HTMLElement, debug: () => boolean): () => void {
  const intro = document.createElement('section');
  intro.innerHTML = `<h1>Live scan</h1>
<p>Point the camera at a document. It is captured automatically when the outline is steady
(switch to manual mode in the top bar). Scan as many pages as you like, then tap Done to review,
change filters, crop, rotate and save a PDF.</p>`;
  const results = document.createElement('section');
  let close: (() => void) | null = null;

  const open = () => {
    const overlay = document.createElement('div');
    overlay.className = 'scanner-overlay';
    const scanner = document.createElement('scan-document') as ScanDocumentElement;
    scanner.setAttribute('page-size', 'a4');
    scanner.options = { detector: createDemoDetector() };
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
      renderResults(results, e.detail.pages, e.detail.pdf);
      close?.();
    });
    scanner.addEventListener('scan-cancel', () => close?.());
    scanner.addEventListener('scan-error', (e) => console.error(e.detail.error));
  };

  intro.append(button('Start scanning', open, 'primary'));
  main.replaceChildren(intro, results);
  return () => close?.();
}

import { ScannerController } from '@scanoficus/core/controller';
import type { ScanCropEditorElement, ScanReviewElement } from '@scanoficus/elements';
import { createDemoDetector } from './detector.ts';
import { button, renderResults } from './results.ts';

/**
 * Scan from a photo: pick or take a picture (`capture="environment"` opens the native camera
 * on phones, often with better photographic processing than a video stream), detect the
 * document, adjust the corners, then review and export. No live camera access needed.
 */
export function mountPhoto(main: HTMLElement): () => void {
  const controller = new ScannerController({ detector: createDemoDetector() });
  const intro = document.createElement('section');
  intro.innerHTML = `<h1>Scan from a photo</h1>
<p>Take a picture with your camera app or choose one from your library. The document is
detected automatically; adjust the corners if needed.</p>`;
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.setAttribute('capture', 'environment');
  input.hidden = true;
  const stage = document.createElement('div');
  stage.className = 'stage';
  const results = document.createElement('section');

  const showReview = () => {
    const review = document.createElement('scan-review') as ScanReviewElement;
    review.setAttribute('no-camera', '');
    review.setAttribute('page-size', 'a4');
    review.controller = controller;
    review.addEventListener('scan-crop', (e) => showCrop(e.detail.page.id));
    review.addEventListener('scan-complete', (e) => {
      renderResults(results, e.detail.pages, e.detail.pdf);
      stage.replaceChildren();
    });
    stage.replaceChildren(review);
  };

  const showCrop = (pageId: string) => {
    const page = controller.pages.find((p) => p.id === pageId);
    if (!page) return;
    const editor = document.createElement('scan-crop-editor') as ScanCropEditorElement;
    editor.source = page.source;
    editor.quad = page.quad;
    editor.addEventListener('crop-cancel', showReview);
    editor.addEventListener('crop-confirm', async (e) => {
      await controller.updatePage(pageId, { quad: e.detail.quad });
      showReview();
    });
    stage.replaceChildren(editor);
  };

  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    stage.textContent = 'Detecting document…';
    try {
      const page = await controller.addImage(file);
      showCrop(page.id);
    } catch (error) {
      stage.textContent = `Could not read the image: ${(error as Error).message}`;
    }
  });

  intro.append(
    input,
    button('Take or choose a photo', () => input.click(), 'primary'),
  );
  main.replaceChildren(intro, stage, results);
  return () => controller.dispose();
}

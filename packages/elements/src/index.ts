/**
 * `@scanoficus/elements`: framework-free custom elements for a VisionKit-style document scanner.
 * Import `@scanoficus/elements/define` (or call {@link defineScanElements}) to register
 * `<scan-document>`, `<scan-camera>`, `<scan-crop-editor>` and `<scan-review>`.
 *
 * @module
 */
import { ScanCameraElement } from './scan-camera.ts';
import { ScanCropEditorElement } from './scan-crop-editor.ts';
import { ScanDocumentElement } from './scan-document.ts';
import { ScanReviewElement } from './scan-review.ts';

export type {
  QuadDetail,
  ScanCompleteDetail,
  ScanCropDetail,
  ScanElementEventMap,
  ScanErrorDetail,
  ScanExportImagesDetail,
} from './events.ts';
export { fitContain, fullQuad, moveCorner } from './geometry.ts';
export { ScanCameraElement, type ScanCameraMessages } from './scan-camera.ts';
export { ScanCropEditorElement } from './scan-crop-editor.ts';
export { ScanDocumentElement } from './scan-document.ts';
export { ScanReviewElement, type ScanReviewMessages } from './scan-review.ts';

/** Tag names and their classes. */
export const SCAN_ELEMENTS: {
  readonly 'scan-camera': typeof ScanCameraElement;
  readonly 'scan-crop-editor': typeof ScanCropEditorElement;
  readonly 'scan-review': typeof ScanReviewElement;
  readonly 'scan-document': typeof ScanDocumentElement;
} = {
  'scan-camera': ScanCameraElement,
  'scan-crop-editor': ScanCropEditorElement,
  'scan-review': ScanReviewElement,
  'scan-document': ScanDocumentElement,
};

/**
 * Registers all elements (idempotent; does nothing outside the browser, e.g. during SSR).
 * Sub-elements are registered before `<scan-document>`, which uses them.
 */
export function defineScanElements(registry?: CustomElementRegistry): void {
  const target = registry ?? (typeof customElements === 'undefined' ? null : customElements);
  if (!target) return;
  for (const [name, element] of Object.entries(SCAN_ELEMENTS)) {
    if (!target.get(name)) target.define(name, element);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scan-camera': ScanCameraElement;
    'scan-crop-editor': ScanCropEditorElement;
    'scan-review': ScanReviewElement;
    'scan-document': ScanDocumentElement;
  }
}

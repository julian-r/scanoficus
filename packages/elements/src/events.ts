import type { ScanPage } from '@scanoficus/core/controller';
import type { Quad } from '@scanoficus/core/geometry';

/** Detail of `scan-complete` (from `<scan-document>` and `<scan-review>`). */
export interface ScanCompleteDetail {
  pages: readonly ScanPage[];
  /** The PDF of all pages (built with the element's `page-size` setting). */
  pdf: Blob;
}

/** Detail of `scan-export-images` (from `<scan-review>`). */
export interface ScanExportImagesDetail {
  pages: readonly ScanPage[];
  images: Blob[];
}

/** Detail of `scan-crop` (from `<scan-review>`): the page the user wants to crop. */
export interface ScanCropDetail {
  page: ScanPage;
}

/** Detail of `scan-error`. */
export interface ScanErrorDetail {
  error: Error;
}

/** Detail of `quad-change` / `crop-confirm` (from `<scan-crop-editor>`). */
export interface QuadDetail {
  quad: Quad;
}

/** Typed `CustomEvent`s dispatched by the scanoficus elements (all bubble and are composed). */
export interface ScanElementEventMap {
  /** The user finished: pages and PDF are ready. */
  'scan-complete': CustomEvent<ScanCompleteDetail>;
  /** The user asked to export the pages as images. */
  'scan-export-images': CustomEvent<ScanExportImagesDetail>;
  /** The user closed the scanner without finishing. */
  'scan-cancel': CustomEvent<undefined>;
  /** `<scan-camera>`: the user tapped "Done". */
  'scan-done': CustomEvent<undefined>;
  /** `<scan-review>`: the user wants to adjust the corners of a page. */
  'scan-crop': CustomEvent<ScanCropDetail>;
  /** Camera or processing error. */
  'scan-error': CustomEvent<ScanErrorDetail>;
  /** `<scan-crop-editor>`: a corner was moved (fires on release / key press). */
  'quad-change': CustomEvent<QuadDetail>;
  /** `<scan-crop-editor>`: the user confirmed the crop. */
  'crop-confirm': CustomEvent<QuadDetail>;
  /** `<scan-crop-editor>`: the user discarded the edit. */
  'crop-cancel': CustomEvent<undefined>;
}

declare global {
  interface HTMLElementEventMap extends ScanElementEventMap {}
}

/**
 * Minimal dependency-free PDF writer that embeds JPEG pages without re-encoding.
 *
 * @module
 */
export { createPdf, type JpegInfo, PAGE_SIZES, type PdfOptions, readJpegInfo } from './pdf.ts';

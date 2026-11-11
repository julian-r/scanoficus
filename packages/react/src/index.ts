/**
 * `@scanoficus/react`: `useDocumentScanner` for custom UIs, and thin components wrapping the
 * scanoficus custom elements. SSR-safe: nothing touches browser APIs until mounted.
 *
 * @module
 */
export {
  CropEditor,
  type CropEditorProps,
  DocumentScanner,
  type DocumentScannerProps,
  ScanCamera,
  type ScanCameraProps,
  ScanReview,
  type ScanReviewProps,
} from './components.tsx';
export {
  type DocumentScannerApi,
  type UseDocumentScannerOptions,
  useDocumentScanner,
} from './use-document-scanner.ts';

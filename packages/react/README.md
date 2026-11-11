# @scanoficus/react

React bindings for the scanoficus document scanner. They work with React 18 and 19, React is a
peer dependency, and they are SSR-safe.

```tsx
import { DocumentScanner, useDocumentScanner } from '@scanoficus/react';

// Complete flow (wraps <scan-document>)
<DocumentScanner pageSize="a4" onComplete={({ pages, pdf }) => upload(pdf)} onCancel={close} />;

// Custom UI on the headless controller
const scanner = useDocumentScanner({ autoCapture: true });
// scanner.state, scanner.detection, scanner.pages, scanner.start(video), scanner.capture(), …
```

Also included: `<ScanCamera>`, `<ScanReview>` and `<CropEditor>`, which take a `controller`
(for example `useDocumentScanner().controller`). The components create their custom element
imperatively on mount, so props and events behave the same in React 18 and 19. On the server
they render an empty `<div>`.

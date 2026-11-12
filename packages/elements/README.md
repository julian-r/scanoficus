# @scanoficus/elements

Framework-free custom elements for the scanoficus document scanner, built on
`ScannerController` from `@scanoficus/core`. They use Shadow DOM, can be themed with CSS custom
properties and `::part()`, have accessible labels and keyboard support, and respect safe areas
in portrait and landscape.

```js
import '@scanoficus/elements/define'; // or: import { defineScanElements } from '@scanoficus/elements'
```

| Element              | Purpose                                                                                                         | Events                                                                              |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `<scan-document>`    | Complete flow: camera → review → crop, with an error view (retry or import a photo)                             | `scan-complete` `{ pages, pdf }`, `scan-export-images`, `scan-cancel`, `scan-error` |
| `<scan-camera>`      | Live preview, animated quad, shutter, auto/manual, torch, page counter                                          | `scan-done`, `scan-cancel`                                                          |
| `<scan-crop-editor>` | Drag the four corners (pointer and touch, with a loupe) or move them with the arrow keys; the quad stays convex | `quad-change`, `crop-confirm`, `crop-cancel`                                        |
| `<scan-review>`      | Page carousel, filter picker, rotate, crop, delete, retake, export                                              | `scan-complete`, `scan-export-images`, `scan-crop`, `scan-error`                    |

`<scan-document>` takes the attributes `auto-capture="false"`, `filter`,
`page-size="image|a4|letter"` and `device-motion` (opt-in: auto-capture also waits while the
motion sensor reports fast rotation; iOS asks for permission on the first tap inside the
element). The other elements take a `controller` property.

Themeable custom properties: `--scan-accent`, `--scan-background`, `--scan-foreground`,
`--scan-toolbar-background`, `--scan-quad-fill`, `--scan-quad-stroke`, `--scan-font-family` and
`--scan-radius`. To localize the labels, call `setMessages()` on `<scan-camera>` and
`<scan-review>`.

Importing the package during server-side rendering is safe: the elements are only defined in
the browser.

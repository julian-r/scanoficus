/**
 * `@scanoficus/core`: framework-agnostic building blocks of a VisionKit-style document scanner.
 * Everything is also available from tree-shakeable subpath exports (`@scanoficus/core/geometry`,
 * `/detect`, `/warp`, `/filters`, `/render`, `/pdf`, `/camera`, `/controller`, `/worker`,
 * `/image`).
 *
 * @module
 */
export * from './camera/index.ts';
export * from './controller/index.ts';
export * from './detect/index.ts';
export * from './filters/index.ts';
export * from './geometry/index.ts';
export * from './image.ts';
export * from './pdf/index.ts';
export * from './render/index.ts';
export * from './warp/index.ts';
export * from './worker/index.ts';

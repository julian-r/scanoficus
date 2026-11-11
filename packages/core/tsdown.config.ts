import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    image: 'src/image.ts',
    geometry: 'src/geometry/index.ts',
    detect: 'src/detect/index.ts',
    warp: 'src/warp/index.ts',
    filters: 'src/filters/index.ts',
    worker: 'src/worker/index.ts',
    'worker-entry': 'src/worker/entry.ts',
  },
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
  dts: true,
  sourcemap: true,
  clean: true,
});

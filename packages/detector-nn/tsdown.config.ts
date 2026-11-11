import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    docquadnet2: 'src/bundle-docquadnet2.ts',
    docquadnet3: 'src/bundle-docquadnet3.ts',
    'worker-entry': 'src/worker-entry.ts',
  },
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
  dts: true,
  sourcemap: true,
  clean: true,
});

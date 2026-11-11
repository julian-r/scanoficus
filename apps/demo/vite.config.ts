import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * `DEMO_BASE` sets the public base path, e.g. `/scanoficus/` for GitHub Pages; the default `./`
 * makes the build relocatable (it can be served from any directory).
 * Workspace packages are consumed from source via the `@scanoficus/source` export condition,
 * so the demo needs no prior library build and the detector worker is bundled by Vite.
 */
export default defineConfig({
  base: process.env.DEMO_BASE ?? './',
  plugins: [react()],
  resolve: { conditions: ['@scanoficus/source'] },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});

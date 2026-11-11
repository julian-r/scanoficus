import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Resolve workspace packages to their TypeScript source so tests never depend on a prior
  // build. The `ssr` block covers tests that run in the Node environment (SSR tests).
  resolve: { conditions: ['@scanoficus/source'] },
  ssr: {
    resolve: { conditions: ['@scanoficus/source'], externalConditions: ['@scanoficus/source'] },
  },
  test: {
    include: ['test/**/*.test.{ts,tsx}'],
    environment: 'happy-dom',
  },
});

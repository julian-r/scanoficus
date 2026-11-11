import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Resolve workspace packages to their TypeScript source so tests never depend on a prior build.
  resolve: { conditions: ['@scanoficus/source'] },
  ssr: {
    resolve: { conditions: ['@scanoficus/source'], externalConditions: ['@scanoficus/source'] },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
    testTimeout: 60_000,
  },
});

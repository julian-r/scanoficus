import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: { conditions: ['@scanoficus/source'] },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'happy-dom',
  },
});

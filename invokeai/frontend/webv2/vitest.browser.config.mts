import { playwright } from '@vitest/browser-playwright';
import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';

import viteConfig from './vite.config.mts';

export default mergeConfig(
  viteConfig,
  defineConfig({
    // Hook-bearing dependencies must be prebundled into the initial browser-test
    // graph. An optimization reload can otherwise leave the running suite with
    // two React instances and produce invalid-hook failures.
    optimizeDeps: { include: ['@dnd-kit/core', '@tanstack/react-query'] },
    test: {
      browser: {
        enabled: true,
        headless: true,
        instances: [{ browser: 'chromium' }],
        provider: playwright(),
      },
      include: ['src/**/*.browser.test.{ts,tsx}'],
    },
  })
);

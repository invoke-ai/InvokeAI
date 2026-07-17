import { playwright } from '@vitest/browser-playwright';
import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';

import viteConfig from './vite.config.mts';

export default mergeConfig(
  viteConfig,
  defineConfig({
    // dnd-kit must share the app's React instance in the browser-test module
    // graph, or its hooks dispatch against a second React copy and crash.
    optimizeDeps: { include: ['@dnd-kit/core'] },
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

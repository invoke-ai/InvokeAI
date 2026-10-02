import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';

import viteConfig from './vite.config.mts';

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      exclude: ['src/**/*.browser.test.{ts,tsx}'],
      include: ['src/**/*.test.{ts,tsx}'],
    },
  })
);

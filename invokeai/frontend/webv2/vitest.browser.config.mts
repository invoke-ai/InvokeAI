import type { BrowserCommand } from 'vitest/node';

import { playwright } from '@vitest/browser-playwright';
import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';

import viteConfig from './vite.config.mts';

/** Drives Chromium's own IME over CDP: `compose` replaces the in-progress text, `commit` inserts the final text. */
const imeCompose: BrowserCommand<[steps: readonly { kind: 'commit' | 'compose'; text: string }[]]> = async (
  { context, page },
  steps
) => {
  const session = await context.newCDPSession(page);

  try {
    for (const step of steps) {
      if (step.kind === 'compose') {
        await session.send('Input.imeSetComposition', {
          selectionEnd: step.text.length,
          selectionStart: step.text.length,
          text: step.text,
        });
      } else {
        await session.send('Input.insertText', { text: step.text });
      }
    }
  } finally {
    await session.detach();
  }
};

export default mergeConfig(
  viteConfig,
  defineConfig({
    define: {
      __CANVAS_GOLDEN_UPDATE__: process.env.CANVAS_GOLDEN_UPDATE === '1',
    },
    // Prebundle browser-test dependencies to prevent optimizer reloads and duplicate React instances.
    optimizeDeps: {
      include: [
        '@chakra-ui/react',
        '@chakra-ui/react/theme',
        '@dnd-kit/core',
        '@tanstack/react-query',
        '@tanstack/react-virtual',
        'idb',
        'i18next-http-backend',
        'react-hook-tanstack-virtual',
        'tinykeys',
      ],
    },
    test: {
      browser: {
        commands: { imeCompose },
        enabled: true,
        headless: true,
        instances: [{ browser: 'chromium' }],
        provider: playwright(),
      },
      include: ['src/**/*.browser.test.{ts,tsx}'],
      setupFiles: ['./scripts/browser-test-console.ts', './scripts/browser-test-viewport.ts'],
    },
  })
);

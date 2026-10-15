import type { BrowserContext, Route } from 'playwright';
import type { BrowserCommand } from 'vitest/node';

import { playwright } from '@vitest/browser-playwright';
import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';

import viteConfig from './vite.config.mts';

/** A previous test's pointer must not hover controls mounted by the next test. */
const resetBrowserPointer: BrowserCommand<[]> = async ({ page }) => {
  await page.mouse.move(-1, -1);
};

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

type ThumbnailRouteState = {
  handler: (route: Route) => Promise<void>;
  pattern: string;
  statuses: number[];
  urls: string[];
};

const thumbnailRoutes = new Map<string, ThumbnailRouteState>();

const stopThumbnailRoute = async (context: BrowserContext, sessionId: string) => {
  const state = thumbnailRoutes.get(sessionId);
  if (state) {
    await context.unroute(state.pattern, state.handler);
    thumbnailRoutes.delete(sessionId);
  }
};

const startThumbnailRoute: BrowserCommand<[pathname: string]> = async ({ context, sessionId }, pathname) => {
  await stopThumbnailRoute(context, sessionId);
  const path = new URL(pathname, 'http://localhost').pathname;
  const pattern = `**${path}*`;
  const state: ThumbnailRouteState = {
    handler: async (route) => {
      if (new URL(route.request().url()).pathname !== path) {
        await route.continue();
        return;
      }

      state.urls.push(route.request().url());
      const status = state.urls.length === 1 ? 404 : 200;
      state.statuses.push(status);
      if (status === 404) {
        await route.fulfill({ body: 'thumbnail missing', contentType: 'text/plain', status });
      } else {
        await route.fulfill({
          body: '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#4ade80"/></svg>',
          contentType: 'image/svg+xml',
          status,
        });
      }
    },
    pattern,
    statuses: [],
    urls: [],
  };
  thumbnailRoutes.set(sessionId, state);
  await context.route(pattern, state.handler);
};

const getThumbnailRequests: BrowserCommand<[], { statuses: number[]; urls: string[] }> = ({ sessionId }) => {
  const state = thumbnailRoutes.get(sessionId);
  return { statuses: state?.statuses ?? [], urls: state?.urls ?? [] };
};

const stopThumbnailRouteCommand: BrowserCommand<[]> = async ({ context, sessionId }) => {
  await stopThumbnailRoute(context, sessionId);
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
        commands: {
          getThumbnailRequests,
          imeCompose,
          resetBrowserPointer,
          startThumbnailRoute,
          stopThumbnailRoute: stopThumbnailRouteCommand,
        },
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

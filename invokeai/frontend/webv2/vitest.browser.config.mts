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
  /** Requests still to be answered with `failureStatus`. */
  failuresLeft: number;
  failureStatus: number;
  handler: (route: Route) => Promise<void>;
  matches: (url: URL) => boolean;
  statuses: number[];
  /** Playwright resource types: `image` for elements, `fetch` for scripted reads. */
  types: string[];
  urls: string[];
};

const thumbnailRoutes = new Map<string, ThumbnailRouteState>();

const stopThumbnailRoute = async (context: BrowserContext, sessionId: string) => {
  const state = thumbnailRoutes.get(sessionId);
  if (state) {
    await context.unroute(state.matches, state.handler);
    thumbnailRoutes.delete(sessionId);
  }
};

/**
 * Serves the images at `pathnames`, answering the first `failures` requests with `status` (default: one 404).
 * `endThumbnailRouteFailures` serves every later request.
 */
const startThumbnailRoute: BrowserCommand<
  [pathnames: string | string[], options?: { failures?: number; status?: number }]
> = async ({ context, sessionId }, pathnames, { failures = 1, status: failureStatus = 404 } = {}) => {
  await stopThumbnailRoute(context, sessionId);
  const paths = new Set(
    (Array.isArray(pathnames) ? pathnames : [pathnames]).map(
      (pathname) => new URL(pathname, 'http://localhost').pathname
    )
  );
  const state: ThumbnailRouteState = {
    failuresLeft: failures,
    failureStatus,
    handler: async (route) => {
      const request = route.request();
      state.urls.push(request.url());
      state.types.push(request.resourceType());
      const status = state.failuresLeft > 0 ? state.failureStatus : 200;
      state.failuresLeft = Math.max(0, state.failuresLeft - 1);
      state.statuses.push(status);
      if (status !== 200) {
        await route.fulfill({ body: 'thumbnail missing', contentType: 'text/plain', status });
      } else {
        await route.fulfill({
          body: '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#4ade80"/></svg>',
          contentType: 'image/svg+xml',
          status,
        });
      }
    },
    matches: (url) => paths.has(url.pathname),
    statuses: [],
    types: [],
    urls: [],
  };
  thumbnailRoutes.set(sessionId, state);
  await context.route(state.matches, state.handler);
};

const endThumbnailRouteFailures: BrowserCommand<[]> = ({ sessionId }) => {
  const state = thumbnailRoutes.get(sessionId);
  if (state) {
    state.failuresLeft = 0;
  }
};

const getThumbnailRequests: BrowserCommand<[], { statuses: number[]; types: string[]; urls: string[] }> = ({
  sessionId,
}) => {
  const state = thumbnailRoutes.get(sessionId);
  return { statuses: state?.statuses ?? [], types: state?.types ?? [], urls: state?.urls ?? [] };
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
          endThumbnailRouteFailures,
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

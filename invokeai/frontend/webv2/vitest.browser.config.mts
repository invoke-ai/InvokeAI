import type { BrowserContext, CDPSession, Route } from 'playwright';
import type { BrowserCommand } from 'vitest/node';

import { playwright } from '@vitest/browser-playwright';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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

/**
 * Drags with the real mouse through `points` (the test frame's viewport coordinates), so the browser hit-tests every
 * move and the release the way a user's drag is, which synthetic events dispatched at an element skip.
 */
const mouseDragPath: BrowserCommand<[points: readonly { x: number; y: number }[], frameWidth: number]> = async (
  { frame, page },
  points,
  frameWidth
) => {
  const box = await (await (await frame()).frameElement()).boundingBox();

  if (!box || points.length === 0) {
    throw new Error('mouseDragPath needs a visible test frame and at least one point');
  }

  const scale = box.width / frameWidth;
  const [start, ...rest] = points.map((point) => ({ x: box.x + point.x * scale, y: box.y + point.y * scale }));

  await page.mouse.move(start!.x, start!.y);
  await page.mouse.down();
  try {
    for (const point of rest) {
      await page.mouse.move(point.x, point.y, { steps: 6 });
    }
  } finally {
    // Never leave the button pressed for the next test.
    await page.mouse.up();
  }
};

/** A file for `nativeDrag`, written to disk so Chromium attaches it as the operating system would. */
type NativeDragFile = { base64: string; name: string };

/**
 * Drags files, a link (`uriList`) or plain text (`text`), alone or together, in from outside the page through
 * Chromium's own drag input: enters at the first point (the test frame's viewport coordinates), hovers each point and
 * drops at the last one. The browser dispatches each drag event in its own task, with a microtask checkpoint between
 * listeners, which events a test dispatches itself never have. It always drops, even when a step fails: a drag left
 * open swallows the next one's enter, and CDP's `dragCancel` reaches the page as no event at all, so it cannot stand
 * in for a cancel. The files are deleted once the command returns: read a dropped file's contents in a drop handler,
 * not after the drag.
 */
const nativeDrag: BrowserCommand<
  [
    options: {
      files?: readonly NativeDragFile[];
      frameWidth: number;
      points: readonly { x: number; y: number }[];
      text?: string;
      uriList?: string;
    },
  ]
> = async ({ context, frame, page }, { files = [], frameWidth, points, text, uriList }) => {
  const box = await (await (await frame()).frameElement()).boundingBox();
  if (!box || points.length === 0) {
    throw new Error('nativeDrag needs a visible test frame and at least one point');
  }
  const scale = box.width / frameWidth;
  const viewportPoints = points.map((point) => ({
    x: Math.round(box.x + point.x * scale),
    y: Math.round(box.y + point.y * scale),
  }));
  const directory = await mkdtemp(join(tmpdir(), 'native-drag-'));
  const session = await context.newCDPSession(page);
  // The page's next frame, so what a drag event rendered is on screen before the next one hit-tests it.
  const nextFrame = () =>
    page.evaluate(
      () =>
        new Promise((resolve) => {
          requestAnimationFrame(resolve);
        })
    );

  try {
    const paths = await Promise.all(
      files.map(async (file) => {
        const path = join(directory, file.name);
        await writeFile(path, Buffer.from(file.base64, 'base64'));
        return path;
      })
    );
    const data = {
      dragOperationsMask: 1,
      files: paths,
      items: [
        ...(uriList === undefined ? [] : [{ data: uriList, mimeType: 'text/uri-list' }]),
        ...(text === undefined ? [] : [{ data: text, mimeType: 'text/plain' }]),
      ],
    };
    const drop = { data, type: 'drop', ...viewportPoints.at(-1)! } as const;
    let isOpen = false;
    try {
      await session.send('Input.dispatchDragEvent', { data, type: 'dragEnter', ...viewportPoints[0]! });
      isOpen = true;
      for (const point of viewportPoints) {
        // Twice per point: the first hover may render the zones the second one finds.
        for (let hover = 0; hover < 2; hover++) {
          await nextFrame();
          await session.send('Input.dispatchDragEvent', { data, type: 'dragOver', ...point });
        }
      }
      await nextFrame();
      isOpen = false;
      await session.send('Input.dispatchDragEvent', drop);
    } finally {
      // A step that failed must not leave the drag open for the next test.
      if (isOpen) {
        await session.send('Input.dispatchDragEvent', drop).catch(() => undefined);
      }
    }
  } finally {
    await session.detach();
    await rm(directory, { force: true, recursive: true });
  }
};

const devicePixelRatioSessions = new Map<string, CDPSession>();

/**
 * Emulates a display's device pixel ratio, such as a desktop scaled to 125%, at the same CSS viewport size; `null`
 * restores the real one. Chromium drops the override with the CDP session that set it, so the session stays open until
 * then.
 */
const emulateDevicePixelRatio: BrowserCommand<[ratio: number | null]> = async ({ context, page, sessionId }, ratio) => {
  const previous = devicePixelRatioSessions.get(sessionId);
  devicePixelRatioSessions.delete(sessionId);
  await previous?.detach();
  if (ratio === null) {
    return;
  }
  const session = await context.newCDPSession(page);
  devicePixelRatioSessions.set(sessionId, session);
  await session.send('Emulation.setDeviceMetricsOverride', {
    deviceScaleFactor: ratio,
    height: 0,
    mobile: false,
    width: 0,
  });
};

/**
 * Captures `rect` (the test frame's CSS pixels) at the display's device pixels, under any emulated device pixel ratio.
 * Playwright's own screenshots restore its viewport emulation, which ends `emulateDevicePixelRatio`'s.
 */
const captureDevicePixels: BrowserCommand<
  [rect: { height: number; width: number; x: number; y: number }, frameWidth: number]
> = async ({ context, frame, page, sessionId }, rect, frameWidth) => {
  const box = await (await (await frame()).frameElement()).boundingBox();
  if (!box) {
    throw new Error('captureDevicePixels needs a visible test frame');
  }
  const scale = box.width / frameWidth;
  // Detaching any other session ends the emulation too, so capture through the one that set it.
  const emulating = devicePixelRatioSessions.get(sessionId);
  const session = emulating ?? (await context.newCDPSession(page));
  try {
    const { data } = await session.send('Page.captureScreenshot', {
      clip: {
        height: rect.height * scale,
        scale: 1 / scale,
        width: rect.width * scale,
        x: box.x + rect.x * scale,
        y: box.y + rect.y * scale,
      },
      format: 'png',
    });
    return data;
  } finally {
    if (!emulating) {
      await session.detach();
    }
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

/**
 * Without the dev proxy, `/api` requests fail the way they do in CI. Proxied to a developer's running backend, their
 * responses arrive late and change what tests render: a menu row inserted mid-click moves the clicked item.
 */
const { proxy: _devBackendProxy, ...server } = viteConfig.server ?? {};

export default mergeConfig(
  { ...viteConfig, server },
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
          captureDevicePixels,
          emulateDevicePixelRatio,
          endThumbnailRouteFailures,
          getThumbnailRequests,
          imeCompose,
          mouseDragPath,
          nativeDrag,
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
      setupFiles: ['./scripts/browser-test-console.ts', './scripts/browser-test-pointer.ts'],
    },
  })
);

import type { GalleryVideoItem, GeneratedImageContract } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type * as GalleryQueriesModule from '@features/gallery/queries';
import type { ImageActions } from '@workbench/image-actions';
import type { WidgetRuntimeApi } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { WHEEL_ZOOM_STEP } from '@workbench/panZoom';
import { createInstance } from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { PreviewComparisonMode } from './previewSettings';

import { PreviewCompare } from './PreviewCompare';
import { PreviewDetailsPanel } from './PreviewDetailsPanel';
import { usePreviewStageContext } from './previewHeaderStore';

vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryModule>()),
  galleryVideos: {
    workflow: () => Promise.resolve({ graph: null, workflow: null }),
  },
}));
// The transport under the gallery's shared video metadata query.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryVideoMetadata: () => Promise.resolve({ seed: 7 }),
}));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryQueriesModule>()),
  imageIndexAvailabilityOptions: () => ({
    queryFn: () => ({ modelName: null, state: 'disabled' }),
    queryKey: ['test-image-index-availability'],
  }),
}));

const detailsItem: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-16T00:00:00.000Z',
  durationSeconds: 5,
  fullUrl: '/videos/clip.mp4',
  height: 1080,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/thumbnails/clip.webp',
  width: 1920,
};

const detailsActions = {} as ImageActions;
const ignoreOpenChange = () => undefined;

/** Details as the header wires it: open, and pinned to whichever stage Preview publishes. */
const OpenDetails = () => (
  <PreviewDetailsPanel
    actions={detailsActions}
    image={null}
    isOpen
    item={detailsItem}
    position={null}
    stageElement={usePreviewStageContext().stageElement}
    onOpenChange={ignoreOpenChange}
  />
);

const i18n = createInstance();
const commandHandlers = new Map<string, () => void>();
const previewRuntime = {
  commands: {
    register: ({ handler, id }: { handler: () => void; id: string }) => {
      commandHandlers.set(id, handler);
      return () => commandHandlers.delete(id);
    },
  },
  hotkeys: { register: () => () => undefined },
} as unknown as WidgetRuntimeApi;
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { swap: 'Swap' },
        widgets: {
          preview: {
            commands: { nextComparisonMode: 'Next comparison mode' },
            compare: 'Compare',
            exitCompare: 'Exit Compare',
            hover: 'Hover',
            hoverComparisonAriaLabel: 'Reveal comparison image on hover, focus, or touch',
            sideBySide: 'Side by Side',
            slider: 'Slider',
            viewing: 'Viewing',
          },
        },
      },
    },
  },
});

const createImage = (name: string, width: number, height: number): GeneratedImageContract => ({
  height,
  imageName: name,
  imageUrl: `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"/>`,
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  thumbnailUrl: '',
  width,
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

/** Poll until transitions settle; fixed waits can read pre-transition values under compositor starvation. */
const waitForOpacity = (element: HTMLElement, expected: string, label = '', timeoutMs = 5000): Promise<void> =>
  act(async () => {
    const deadline = Date.now() + timeoutMs;

    while (getComputedStyle(element).opacity !== expected) {
      if (Date.now() > deadline) {
        throw new Error(
          `Timed out after ${timeoutMs}ms waiting for opacity ${expected} at [${label}]; last value ${getComputedStyle(element).opacity}; activeElement=${document.activeElement?.getAttribute('aria-label') ?? document.activeElement?.tagName}`
        );
      }

      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 16);
      });
    }
  });

/**
 * React derives `onPointerLeave` from `pointerout`, so this clears a hover the
 * component may have picked up from the real cursor without moving it.
 */
const clearMouseHover = (element: HTMLElement): Promise<void> =>
  interact(() =>
    element.dispatchEvent(
      new PointerEvent('pointerout', {
        bubbles: true,
        pointerId: 1,
        pointerType: 'mouse',
        relatedTarget: document.body,
      })
    )
  );

const renderComparison = async ({
  baseImage = createImage('base', 1200, 800),
  compareImage = createImage('compare', 800, 1200),
  initialMode = 'hover',
  withDetails = false,
}: {
  baseImage?: GeneratedImageContract;
  compareImage?: GeneratedImageContract;
  initialMode?: PreviewComparisonMode;
  withDetails?: boolean;
} = {}) => {
  commandHandlers.clear();
  const onExit = vi.fn();
  const onModeChange = vi.fn();
  const onSwap = vi.fn();
  const Harness = ({
    baseImage,
    compareImage,
  }: {
    baseImage: GeneratedImageContract;
    compareImage: GeneratedImageContract;
  }) => {
    const [mode, setMode] = useState(initialMode);
    const handleModeChange = useCallback((nextMode: PreviewComparisonMode) => {
      setMode(nextMode);
      onModeChange(nextMode);
    }, []);

    return (
      <PreviewCompare
        baseImage={baseImage}
        compareImage={compareImage}
        mode={mode}
        runtime={previewRuntime}
        onExit={onExit}
        onModeChange={handleModeChange}
        onSwap={onSwap}
      />
    );
  };

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  /** Renders the pair into the same tree, so a new pair reaches the mounted comparison as a selection change does. */
  const showPair = (pair: { base: GeneratedImageContract; compare: GeneratedImageContract }) =>
    interact(() => {
      root?.render(
        <QueryClientProvider client={queryClient}>
          <I18nextProvider i18n={i18n}>
            <ChakraProvider value={system}>
              {withDetails ? <OpenDetails /> : null}
              <Harness baseImage={pair.base} compareImage={pair.compare} />
            </ChakraProvider>
          </I18nextProvider>
        </QueryClientProvider>
      );
    });

  host = document.createElement('div');
  host.style.height = '500px';
  host.style.width = '800px';
  document.body.append(host);
  root = createRoot(host);
  await showPair({ base: baseImage, compare: compareImage });

  return { commandHandlers, onExit, onModeChange, onSwap, showPair };
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewCompare', () => {
  it('reveals the comparison on focus and touch press, then restores the selected image', async () => {
    await renderComparison();
    const frame = host!.querySelector<HTMLElement>('[aria-label*="Reveal comparison"]')!;
    const compareOverlay = host!.querySelector<HTMLImageElement>('img[alt="compare"]')?.parentElement as HTMLElement;

    // Clear incidental hover to isolate focus behavior, then wait for any resulting fade to settle.
    await clearMouseHover(frame);
    await waitForOpacity(compareOverlay, '0', 'initial');
    await interact(() => frame.focus());
    await waitForOpacity(compareOverlay, '1', 'after-focus');
    await interact(() => frame.blur());
    await waitForOpacity(compareOverlay, '0', 'after-blur');

    Object.defineProperties(frame, {
      hasPointerCapture: { value: () => true },
      releasePointerCapture: { value: vi.fn() },
      setPointerCapture: { value: vi.fn() },
    });
    await interact(() =>
      frame.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 7, pointerType: 'touch' }))
    );
    await waitForOpacity(compareOverlay, '1', 'after-pointerdown');
    await interact(() =>
      frame.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 7, pointerType: 'touch' }))
    );
    await waitForOpacity(compareOverlay, '0', 'after-pointerup');
    expect(getComputedStyle(compareOverlay).transitionProperty).toContain('opacity');
  });

  it('cycles all modes through the M command and keeps swap and exit separate', async () => {
    const { commandHandlers, onExit, onModeChange, onSwap } = await renderComparison({ initialMode: 'slider' });
    const nextMode = commandHandlers.get('viewer.nextComparisonMode')!;

    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('side-by-side');
    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('hover');
    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('slider');

    const buttons = Array.from(host?.querySelectorAll<HTMLButtonElement>('button') ?? []);
    await interact(() => buttons.find((button) => button.textContent?.includes('Swap'))?.click());
    await interact(() => buttons.find((button) => button.textContent?.includes('Exit Compare'))?.click());
    expect(onSwap).toHaveBeenCalledOnce();
    expect(onExit).toHaveBeenCalledOnce();
  });

  describe('zoom with differently sized images', () => {
    /** Parses `translate(Xpx, Ypx) scale(S)` into fractions of the image's frame, so differently sized panes compare. */
    const readFractionTransform = (image: HTMLImageElement) => {
      const frame = image.parentElement!;
      const match = /translate\((-?[\d.]+)px, (-?[\d.]+)px\) scale\(([\d.]+)\)/.exec(image.style.transform);

      if (!match) {
        return null;
      }

      return {
        fx: Number(match[1]) / frame.clientWidth,
        fy: Number(match[2]) / frame.clientHeight,
        scale: Number(match[3]),
      };
    };
    /** Both panes show the same region of their own picture: equal scale and equal frame-relative pan. */
    const expectLinked = (first: HTMLImageElement, second: HTMLImageElement) => {
      const a = readFractionTransform(first);
      const b = readFractionTransform(second);

      expect(a).not.toBeNull();
      expect(b?.scale).toBeCloseTo(a!.scale, 3);
      expect(b?.fx).toBeCloseTo(a!.fx, 3);
      expect(b?.fy).toBeCloseTo(a!.fy, 3);
    };
    const getImages = () => ({
      base: host!.querySelector<HTMLImageElement>('img[alt="base"]')!,
      compare: host!.querySelector<HTMLImageElement>('img[alt="compare"]')!,
    });
    const wheelAt = (frame: HTMLElement, fractionX: number, fractionY: number, deltaY: number) => {
      const rect = frame.getBoundingClientRect();

      frame.dispatchEvent(
        new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          clientX: rect.left + rect.width * fractionX,
          clientY: rect.top + rect.height * fractionY,
          deltaY,
        })
      );
    };
    const mouse = (target: HTMLElement, type: string, clientX: number, clientY: number) =>
      target.dispatchEvent(
        new PointerEvent(type, {
          bubbles: true,
          button: type === 'pointermove' ? -1 : 0,
          cancelable: true,
          clientX,
          clientY,
          isPrimary: true,
          pointerId: 1,
          pointerType: 'mouse',
        })
      );

    it('links wheel zoom, pan, and reset across side-by-side panes of different sizes and aspect ratios', async () => {
      // 1200×800 landscape beside 800×1200 portrait.
      await renderComparison({ initialMode: 'side-by-side' });
      const { base, compare } = getImages();
      const baseFrame = base.parentElement!;
      const compareFrame = compare.parentElement!;

      expect(baseFrame.clientWidth).not.toBe(compareFrame.clientWidth);

      // Zoom ×2 around a point three quarters across, one quarter down the landscape pane.
      await interact(() => wheelAt(baseFrame, 0.75, 0.25, -Math.log(2) / WHEEL_ZOOM_STEP));
      const zoomed = readFractionTransform(base);

      expect(zoomed?.scale).toBeCloseTo(2, 3);
      // The point under the cursor stays put: f + scale * 0.75 = 0.75.
      expect(zoomed?.fx).toBeCloseTo(-0.75, 3);
      expect(zoomed?.fy).toBeCloseTo(-0.25, 3);
      expectLinked(base, compare);

      // Drag the portrait pane by a tenth of its own width: both pictures follow by the same share of their frames.
      const rect = compareFrame.getBoundingClientRect();
      const startX = rect.left + rect.width / 2;
      const startY = rect.top + rect.height / 2;

      await interact(() => mouse(compareFrame, 'pointerdown', startX, startY));
      const tenth = compareFrame.clientWidth / 10;

      await interact(() => mouse(compareFrame, 'pointermove', startX + tenth, startY));
      await interact(() => mouse(compareFrame, 'pointerup', startX + tenth, startY));
      const panned = readFractionTransform(compare);

      expect(panned?.scale).toBeCloseTo(2, 3);
      expect(panned?.fx).toBeCloseTo(-0.65, 3);
      expect(panned?.fy).toBeCloseTo(-0.25, 3);
      expectLinked(compare, base);

      // Double-click resets both panes to fit.
      await interact(() =>
        baseFrame.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: startX, clientY: startY }))
      );
      expect(base.style.transform).toBe('');
      expect(compare.style.transform).toBe('');
    });

    it('caps zoom where the sharper image reaches eight times its own pixels', async () => {
      // A 300×200 image beside the 800×1200 one: the larger image sets the limit.
      await renderComparison({ baseImage: createImage('base', 300, 200), initialMode: 'side-by-side' });
      const { base, compare } = getImages();
      const compareFrame = compare.parentElement!;

      await interact(() => wheelAt(compareFrame, 0.5, 0.5, -Math.log(1000) / WHEEL_ZOOM_STEP));

      expect(readFractionTransform(compare)?.scale).toBeCloseTo((8 * 800) / compareFrame.clientWidth, 3);
      expectLinked(compare, base);
      // Past 2× of its own pixels, each pane renders crisp pixels by its own resolution.
      expect(base.style.imageRendering).toBe('pixelated');
    });

    it('keeps the zoom on a new selection against the same comparison image, within the new pair’s limit', async () => {
      const compareImage = createImage('compare', 512, 512);
      const { showPair } = await renderComparison({
        baseImage: createImage('large', 4096, 4096),
        compareImage,
        initialMode: 'side-by-side',
      });
      const compareFrame = getImages().compare.parentElement!;

      // Zoom to the 4096px image's cap, a quarter of the way across.
      await interact(() => wheelAt(compareFrame, 0.25, 0.5, -Math.log(1000) / WHEEL_ZOOM_STEP));
      expect(readFractionTransform(getImages().compare)?.scale).toBeCloseTo((8 * 4096) / compareFrame.clientWidth, 3);

      await showPair({ base: createImage('base', 512, 512), compare: compareImage });
      const { base, compare } = getImages();
      const limit = (8 * 512) / compareFrame.clientWidth;
      const kept = readFractionTransform(compare);

      expect(kept?.scale).toBeCloseTo(limit, 3);
      // Clamping zooms out around the view's centre, so the region in view stays where it was.
      expect((0.5 - kept!.fx) / kept!.scale).toBeCloseTo(0.25, 2);
      expectLinked(compare, base);
      expect(getComputedStyle(base.parentElement!).cursor).toBe('grab');
    });

    it('opens a wholly new pair at fit', async () => {
      const { showPair } = await renderComparison({ initialMode: 'side-by-side' });
      const baseFrame = getImages().base.parentElement!;

      await interact(() => wheelAt(baseFrame, 0.5, 0.5, -Math.log(2) / WHEEL_ZOOM_STEP));
      expect(readFractionTransform(getImages().base)?.scale).toBeCloseTo(2, 3);

      await showPair({ base: createImage('next-base', 1200, 800), compare: createImage('next-compare', 800, 1200) });

      expect(host!.querySelector<HTMLImageElement>('img[alt="next-base"]')!.style.transform).toBe('');
      expect(host!.querySelector<HTMLImageElement>('img[alt="next-compare"]')!.style.transform).toBe('');
      expect(getComputedStyle(host!.querySelector('img[alt="next-base"]')!.parentElement!).cursor).not.toBe('grab');
    });

    it('keeps the zoom when the pair swaps sides', async () => {
      const baseImage = createImage('base', 1200, 800);
      const compareImage = createImage('compare', 800, 1200);
      const { showPair } = await renderComparison({ baseImage, compareImage, initialMode: 'side-by-side' });

      await interact(() => wheelAt(getImages().base.parentElement!, 0.75, 0.25, -Math.log(2) / WHEEL_ZOOM_STEP));
      await showPair({ base: compareImage, compare: baseImage });
      const { base, compare } = getImages();

      expect(readFractionTransform(base)?.scale).toBeCloseTo(2, 3);
      expect(readFractionTransform(base)?.fx).toBeCloseTo(-0.75, 3);
      expectLinked(base, compare);
    });
  });

  it.each([
    ['portrait', 400, 1600],
    ['panorama', 1600, 400],
    ['square', 900, 900],
  ])('fits a %s image frame without overflow', async (_name, width, height) => {
    await renderComparison({ baseImage: createImage('base', width, height), initialMode: 'slider' });
    const frame = host?.querySelector<HTMLImageElement>('img[alt="base"]')?.parentElement as HTMLElement;
    const stage = frame.parentElement!;
    const frameRect = frame.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();

    expect(frameRect.width).toBeLessThanOrEqual(stageRect.width);
    expect(frameRect.height).toBeLessThanOrEqual(stageRect.height);
    expect(frameRect.width / frameRect.height).toBeCloseTo(width / height, 1);
  });

  it('pins open Details in the comparison stage, clear of its mode controls and slider', async () => {
    await renderComparison({ initialMode: 'slider', withDetails: true });
    const frame = host!.querySelector<HTMLImageElement>('img[alt="base"]')!.parentElement!;
    const stage = frame.parentElement!;
    const panel = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>('[data-preview-details]');
      expect(element).not.toBeNull();
      return element!;
    });
    const exit = Array.from(host!.querySelectorAll('button')).find((button) =>
      button.textContent?.includes('Exit Compare')
    )!;

    expect(panel.parentElement).toBe(stage);
    expect(stage.contains(exit)).toBe(false);
    expect(panel.getBoundingClientRect().bottom).toBeLessThanOrEqual(exit.getBoundingClientRect().top);

    // Pressing inside Details leaves the divider where it was.
    const overlay = host!.querySelector<HTMLImageElement>('img[alt="compare"]')!.parentElement!;
    const clipPath = overlay.style.clipPath;
    const rect = panel.getBoundingClientRect();
    await interact(() =>
      panel.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          clientX: rect.left + 4,
          clientY: rect.top + 4,
          pointerType: 'mouse',
        })
      )
    );
    expect(overlay.style.clipPath).toBe(clipPath);
  });
});

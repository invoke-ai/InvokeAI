import type { GalleryVideoItem, GeneratedImageContract } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type * as GalleryQueriesModule from '@features/gallery/queries';
import type { ImageActions } from '@workbench/image-actions';
import type { WidgetRuntimeApi } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
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
    metadata: () => Promise.resolve({ seed: 7 }),
    workflow: () => Promise.resolve({ graph: null, workflow: null }),
  },
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
  initialMode = 'hover',
  withDetails = false,
}: {
  baseImage?: GeneratedImageContract;
  initialMode?: PreviewComparisonMode;
  withDetails?: boolean;
} = {}) => {
  commandHandlers.clear();
  const onExit = vi.fn();
  const onModeChange = vi.fn();
  const onSwap = vi.fn();
  const compareImage = createImage('compare', 800, 1200);
  const Harness = () => {
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

  host = document.createElement('div');
  host.style.height = '500px';
  host.style.width = '800px';
  document.body.append(host);
  root = createRoot(host);
  await interact(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            {withDetails ? <OpenDetails /> : null}
            <Harness />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    );
  });

  return { commandHandlers, onExit, onModeChange, onSwap };
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

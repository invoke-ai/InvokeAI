/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasStagingSlot } from '@workbench/canvasStagingView';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import { CanvasBottomOverlay } from './CanvasBottomOverlay';
import { StagingBar } from './StagingBar';

/** Roughly the width of a canvas widget on a laptop — wide enough for the bar, far too narrow for 24 thumbnails. */
const CANVAS_WIDTH = 900;
const TRANSPARENT_PIXEL = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

const noop = () => {};

const i18n = createInstance();

const makeSlot = (index: number): CanvasStagingSlot => ({
  candidate: {
    height: 512,
    imageName: `image-${index}`,
    imageUrl: TRANSPARENT_PIXEL,
    placement: { height: 512, opacity: 1, width: 512, x: 0, y: 0 },
    queuedAt: '2026-01-01T00:00:00.000Z',
    sourceQueueItemId: 'queue-item',
    thumbnailUrl: TRANSPARENT_PIXEL,
    width: 512,
  },
  height: 512,
  id: `slot-${index}`,
  imageName: `image-${index}`,
  kind: 'candidate',
  queueItemId: 'queue-item',
  width: 512,
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

const renderStagingBar = async (
  slotCount: number,
  selectedImageIndex = 0,
  onSaveToLayerAndContinue: () => void = noop,
  onPreloadCandidate: (imageName: string) => void = noop
) => {
  const slots = Array.from({ length: slotCount }, (_, index) => makeSlot(index));
  const selectedSlot = slots[selectedImageIndex];

  if (!host) {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  }

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Box h="400px" position="relative" w={`${CANVAS_WIDTH}px`}>
            <CanvasBottomOverlay.Root>
              <CanvasBottomOverlay.Staging>
                <StagingBar
                  antialiasProgressImages={false}
                  areThumbnailsVisible
                  autoSwitchMode="off"
                  canAccept
                  hasMultipleSlots={slotCount > 1}
                  isGenerating={false}
                  isVisible
                  selectedCandidate={selectedSlot?.kind === 'candidate' ? selectedSlot.candidate : undefined}
                  selectedImageIndex={selectedImageIndex}
                  selectedSlot={selectedSlot}
                  slots={slots}
                  onAccept={noop}
                  onCancelQueueItem={noop}
                  onCycle={noop}
                  onDiscardAll={noop}
                  onDiscardSelected={noop}
                  onPreloadCandidate={onPreloadCandidate}
                  onSelectImage={noop}
                  onSaveToLayerAndContinue={onSaveToLayerAndContinue}
                  onSetAutoSwitch={noop}
                  onToggleThumbnails={noop}
                  onToggleVisibility={noop}
                />
              </CanvasBottomOverlay.Staging>
            </CanvasBottomOverlay.Root>
          </Box>
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const viewport = host.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
  const overlay = host.firstElementChild!.firstElementChild as HTMLElement;

  return { overlay, thumbnails: Array.from(viewport.querySelectorAll<HTMLElement>('button')), viewport };
};

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());

  await i18n.use(initReactI18next).init({
    fallbackLng: 'en',
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation } },
  });
});

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('StagingBar thumbnail strip', () => {
  it('keeps every thumbnail reachable by scrolling when many images are staged', async () => {
    const { overlay, thumbnails, viewport } = await renderStagingBar(24);
    const bounds = viewport.getBoundingClientRect();

    expect(thumbnails).toHaveLength(24);
    // The strip gets the full canvas widget to scroll within — not the width of
    // the options bar beneath it, and never wider than the canvas itself (the
    // overlay would silently clip the centered overflow at both edges).
    expect(bounds.width).toBe(overlay.getBoundingClientRect().width);
    expect(bounds.width).toBeLessThanOrEqual(CANVAS_WIDTH);
    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);

    // A centered overflowing row puts the leading thumbnails at a negative
    // offset, where no scroll position can reach them.
    await interact(() => {
      viewport.scrollLeft = 0;
    });
    expect(thumbnails[0]!.getBoundingClientRect().left).toBeGreaterThanOrEqual(bounds.left - 1);

    await interact(() => {
      viewport.scrollLeft = viewport.scrollWidth;
    });
    expect(thumbnails.at(-1)!.getBoundingClientRect().right).toBeLessThanOrEqual(bounds.right + 1);
  });

  it('scrolls the selected thumbnail into view when the selection moves', async () => {
    const { viewport } = await renderStagingBar(24);

    expect(viewport.scrollLeft).toBe(0);

    const { thumbnails } = await renderStagingBar(24, 20);
    const selected = thumbnails[20]!.getBoundingClientRect();
    const bounds = viewport.getBoundingClientRect();

    expect(viewport.scrollLeft).toBeGreaterThan(0);
    expect(selected.left).toBeGreaterThanOrEqual(bounds.left - 1);
    expect(selected.right).toBeLessThanOrEqual(bounds.right + 1);
  });

  it('centers a short strip instead of scrolling it', async () => {
    const { thumbnails, viewport } = await renderStagingBar(2);
    const bounds = viewport.getBoundingClientRect();

    expect(viewport.scrollWidth).toBe(viewport.clientWidth);
    const leadingGap = thumbnails[0]!.getBoundingClientRect().left - bounds.left;
    const trailingGap = bounds.right - thumbnails[1]!.getBoundingClientRect().right;
    expect(leadingGap).toBeGreaterThan(0);
    expect(Math.abs(leadingGap - trailingGap)).toBeLessThan(2);
  });

  it('warms full-resolution bytes for every candidate once its thumbnail settles', async () => {
    const onPreloadCandidate = vi.fn();
    await renderStagingBar(3, 0, noop, onPreloadCandidate);
    const images = Array.from(document.querySelectorAll<HTMLImageElement>('img'));

    await interact(() => {
      images[0]?.dispatchEvent(new Event('error'));
      images.slice(1).forEach((image) => image.dispatchEvent(new Event('load')));
    });

    expect(onPreloadCandidate).toHaveBeenCalledWith('image-0');
    expect(onPreloadCandidate).toHaveBeenCalledWith('image-1');
    expect(onPreloadCandidate).toHaveBeenCalledWith('image-2');
  });

  it('offers a split-button action that saves without ending staging', async () => {
    const onSaveToLayerAndContinue = vi.fn();
    await renderStagingBar(1, 0, onSaveToLayerAndContinue);

    const trigger = document.querySelector<HTMLButtonElement>('button[aria-label="More accept options"]');
    expect(trigger).not.toBeNull();
    await interact(() => trigger!.click());

    const menuItem = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((item) =>
      item.textContent?.includes('Save & Continue')
    );
    expect(menuItem).toBeDefined();
    await interact(() => menuItem!.click());

    expect(onSaveToLayerAndContinue).toHaveBeenCalledOnce();
  });
});

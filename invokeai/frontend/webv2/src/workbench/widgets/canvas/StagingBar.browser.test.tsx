import '@fontsource/inter/index.css';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { QueueItemProgress } from '@features/queue/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/api';
import type { CanvasStagingSlot } from '@workbench/canvasStagingView';
import type { WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { CanvasBottomOverlay } from './CanvasBottomOverlay';
import { StagingBar } from './StagingBar';

const contextMenu = vi.hoisted(() => ({
  targets: [] as { slotId: string; x: number; y: number }[],
}));
const queueProgress = vi.hoisted(() => ({ current: null as QueueItemProgress | null }));
vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useQueueItemProgress: () => queueProgress.current,
}));
vi.mock('./StagingItemContextMenu', () => ({
  StagingItemContextMenu: ({
    onClose,
    target,
  }: {
    onClose: () => void;
    target: { slot: { id: string }; x: number; y: number };
  }) => {
    contextMenu.targets.push({ slotId: target.slot.id, x: target.x, y: target.y });
    // Stands in for the menu's dismissal: the menu holds focus, then unmounts and drops it.
    return (
      <div data-testid="staging-context-menu">
        <button type="button" onClick={onClose}>
          Close menu
        </button>
      </div>
    );
  },
}));

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

const makePlaceholder = (index: number): CanvasStagingSlot => ({
  height: 512,
  id: `slot-${index}`,
  itemIndex: index,
  kind: 'placeholder',
  queueItemId: 'queue-item',
  width: 512,
  x: 0,
  y: 0,
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
  onPreloadCandidate: (imageName: string) => void = noop,
  onSelectImage: (index: number) => void = noop,
  slotAt: (index: number) => CanvasStagingSlot = makeSlot,
  {
    canvasWidth = CANVAS_WIDTH,
    isGenerating = false,
    isSavingToGallery = false,
    onAccept = noop,
    onStopGenerating = noop,
    onCycle = noop,
    onDiscardSelected = noop,
    onSaveToGallery = () => Promise.resolve(),
  }: {
    canvasWidth?: number;
    isGenerating?: boolean;
    isSavingToGallery?: boolean;
    onAccept?: () => void;
    onStopGenerating?: () => void;
    onCycle?: (direction: -1 | 1) => void;
    onDiscardSelected?: () => void;
    onSaveToGallery?: (imageName: string) => Promise<void>;
  } = {}
) => {
  const slots = Array.from({ length: slotCount }, (_, index) => slotAt(index));
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
          <Box h="400px" position="relative" w={`${canvasWidth}px`}>
            <CanvasBottomOverlay.Root>
              <CanvasBottomOverlay.Staging>
                <StagingBar
                  antialiasProgressImages={false}
                  areThumbnailsVisible
                  autoSwitchMode="off"
                  canAccept
                  hasMultipleSlots={slotCount > 1}
                  isGenerating={isGenerating}
                  isSavingToGallery={isSavingToGallery}
                  isVisible
                  selectedCandidate={selectedSlot?.kind === 'candidate' ? selectedSlot.candidate : undefined}
                  selectedImageIndex={selectedImageIndex}
                  selectedSlot={selectedSlot}
                  slots={slots}
                  onAccept={onAccept}
                  onCycle={onCycle}
                  onDiscardAll={noop}
                  onDiscardSelected={onDiscardSelected}
                  onPreloadCandidate={onPreloadCandidate}
                  onSelectImage={onSelectImage}
                  onSaveToGallery={onSaveToGallery}
                  onSaveToLayerAndContinue={onSaveToLayerAndContinue}
                  onSetAutoSwitch={noop}
                  onStopGenerating={onStopGenerating}
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

  await document.fonts.ready;

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
    // Constrain staging scroll width to the canvas, independent of the narrower options bar.
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

  it('selects a right-clicked candidate and opens its context menu at the pointer', async () => {
    const onSelectImage = vi.fn();
    const { thumbnails } = await renderStagingBar(3, 2, noop, noop, onSelectImage);

    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 });
    await interact(() => thumbnails[2]!.dispatchEvent(event));

    expect(event.defaultPrevented).toBe(true);
    expect(onSelectImage).toHaveBeenCalledWith(2);
    expect(contextMenu.targets.at(-1)).toEqual({ slotId: 'slot-2', x: 120, y: 80 });
    expect(document.querySelector('[data-testid="staging-context-menu"]')).not.toBeNull();
  });

  it('shows the spinner on the executing placeholder until its progress is known', async () => {
    const indicators = () => [...host!.querySelectorAll('[role="progressbar"]')];
    const remount = async () => {
      await interact(() => root?.unmount());
      host?.remove();
      host = null;
      await renderStagingBar(3, 0, noop, noop, noop, makePlaceholder);
    };

    try {
      // Zero reads as not started yet, as on every other progress surface.
      for (const percentage of [null, 0]) {
        queueProgress.current = { activeItemIndex: 1, message: 'Loading model', percentage };
        await remount();
        expect(indicators()).toHaveLength(1);
        expect(indicators()[0]!.classList.contains('chakra-spinner')).toBe(true);
        expect(indicators()[0]!.getAttribute('aria-label')).toBe('Generation progress');
      }

      queueProgress.current = { activeItemIndex: 1, message: 'Denoising', percentage: 0.42 };
      await remount();
      expect(indicators()).toHaveLength(1);
      expect(indicators()[0]!.classList.contains('chakra-spinner')).toBe(false);
      expect(indicators()[0]!.getAttribute('aria-valuenow')).toBe('42');
    } finally {
      queueProgress.current = null;
    }
  });

  it('leaves the native menu alone on an in-progress placeholder', async () => {
    const { thumbnails } = await renderStagingBar(3, 0, noop, noop, noop, makePlaceholder);
    const before = contextMenu.targets.length;

    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 });
    await interact(() => thumbnails[1]!.dispatchEvent(event));

    expect(event.defaultPrevented).toBe(false);
    expect(contextMenu.targets).toHaveLength(before);
  });

  it('anchors a keyboard-opened context menu to the thumbnail itself', async () => {
    const { thumbnails } = await renderStagingBar(3, 1);
    const rect = thumbnails[1]!.getBoundingClientRect();

    await interact(() => thumbnails[1]!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true })));

    expect(contextMenu.targets.at(-1)).toEqual({ slotId: 'slot-1', x: rect.left + rect.width / 2, y: rect.top });
  });

  it('closes the context menu once the selection moves off its candidate', async () => {
    const { thumbnails } = await renderStagingBar(3, 1);
    await interact(() =>
      thumbnails[1]!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: 50, clientY: 50 }))
    );
    expect(document.querySelector('[data-testid="staging-context-menu"]')).not.toBeNull();

    // An auto-switch lands a newer result: the bar re-renders with that one selected.
    await renderStagingBar(3, 2);

    expect(document.querySelector('[data-testid="staging-context-menu"]')).toBeNull();
  });

  it('offers a split-button action that saves without ending staging', async () => {
    const onSaveToLayerAndContinue = vi.fn();
    await renderStagingBar(1, 0, onSaveToLayerAndContinue);

    const trigger = document.querySelector<HTMLButtonElement>('button[aria-label="More accept options"]');
    expect(trigger).not.toBeNull();
    await interact(() => trigger!.click());

    const menuItem = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((item) =>
      item.textContent?.includes('Keep as hidden layer')
    );
    expect(menuItem).toBeDefined();
    await interact(() => menuItem!.click());

    expect(onSaveToLayerAndContinue).toHaveBeenCalledOnce();
  });
  it('accepts from the bar and keeps the hidden-layer alternative', async () => {
    const onAccept = vi.fn();
    const onSaveToLayerAndContinue = vi.fn();
    await renderStagingBar(1, 0, onSaveToLayerAndContinue, noop, noop, makeSlot, { onAccept });
    const accept = page.getByRole('button', { exact: true, name: 'Accept to layer' });
    await expect.element(accept).toBeVisible();
    await accept.click();
    expect(onAccept).toHaveBeenCalledOnce();

    await page.getByRole('button', { name: 'More accept options' }).click();
    const keepHidden = page.getByRole('menuitem', { name: 'Keep as hidden layer' });
    await expect.element(keepHidden).toBeVisible();
    await keepHidden.click();
    expect(onSaveToLayerAndContinue).toHaveBeenCalledOnce();
    expect(onAccept).toHaveBeenCalledOnce();
  });

  it.each([360, 450, 560, 640, 700, 800])(
    'keeps the accept split button and every other control inside a %ipx canvas while a batch is generating',
    async (canvasWidth) => {
      const { overlay } = await renderStagingBar(
        3,
        0,
        noop,
        noop,
        noop,
        (index) => (index === 0 ? makeSlot(index) : makePlaceholder(index)),
        { canvasWidth, isGenerating: true }
      );
      const trigger = page.getByRole('button', { name: 'More accept options' }).element();
      const accept = page.getByRole('button', { exact: true, name: 'Accept to layer' }).element();
      const bounds = overlay.getBoundingClientRect();
      expect(trigger.getBoundingClientRect().right).toBeLessThanOrEqual(bounds.right);
      expect(accept.getBoundingClientRect().left).toBeGreaterThanOrEqual(bounds.left);
      // Nothing in the bar is clipped: each control a pointer can reach sits inside the canvas.
      for (const control of overlay.querySelectorAll<HTMLElement>('button')) {
        const rect = control.getBoundingClientRect();
        // Thumbnails scroll inside their own strip, which the first test covers.
        if (rect.width > 1 && !control.closest('[data-scope="scroll-area"]')) {
          expect(rect.left, control.getAttribute('aria-label') ?? control.textContent ?? '').toBeGreaterThanOrEqual(
            bounds.left
          );
          expect(rect.right, control.getAttribute('aria-label') ?? control.textContent ?? '').toBeLessThanOrEqual(
            bounds.right
          );
        }
      }
      // The accept action keeps its full name, and Tooltip, however narrow the bar.
      expect(accept.getAttribute('aria-label')).toBe('Accept to layer');
      // Labelled or collapsed to its icon, the auto-switch trigger keeps the height of the bar's other controls.
      expect(
        page
          .getByRole('button', { name: /^Auto-switch:/ })
          .element()
          .getBoundingClientRect().height
      ).toBe(page.getByRole('button', { exact: true, name: 'Discard' }).element().getBoundingClientRect().height);
      // Collapsed or not, Discard All and the generating status keep their names.
      await expect.element(page.getByRole('button', { name: 'Discard all' })).toBeVisible();
      expect(page.getByRole('status').element().textContent).toBe('Generating, 3 results pending');
    }
  );

  // The bar's floor is a 360px canvas (see the compact breakpoints); the pending status adds nothing to it.
  it.each([false, true])(
    'keeps Accept inside the narrowest canvas (generating: %s), the pending count read but not shown',
    async (isGenerating) => {
      const { overlay } = await renderStagingBar(2, 0, noop, noop, noop, makeSlot, { canvasWidth: 360, isGenerating });
      const bounds = overlay.getBoundingClientRect();
      const trigger = page.getByRole('button', { name: 'More accept options' }).element();
      expect(trigger.getBoundingClientRect().right).toBeLessThanOrEqual(bounds.right);
      const status = page.getByRole('status').element();
      expect(status.textContent).toBe(isGenerating ? 'Generating, 2 results pending' : '2 results pending');
      // No bare count duplicating "1 of 2" beside the counter.
      const hidden = [...overlay.querySelectorAll('[aria-hidden="true"]')].map((node) => node.textContent?.trim());
      expect(hidden).not.toContain('2');
    }
  );

  it('saves the selected candidate from an icon button that keeps its name inside a compact canvas', async () => {
    const onSaveToGallery = vi.fn(() => Promise.resolve());
    const { overlay } = await renderStagingBar(3, 1, noop, noop, noop, makeSlot, {
      canvasWidth: 640,
      isGenerating: true,
      onSaveToGallery,
    });
    const save = page.getByRole('button', { exact: true, name: 'Save to gallery' });
    const bounds = overlay.getBoundingClientRect();
    const rect = save.element().getBoundingClientRect();

    expect(rect.left).toBeGreaterThanOrEqual(bounds.left);
    expect(rect.right).toBeLessThanOrEqual(bounds.right);
    await save.click();
    expect(onSaveToGallery).toHaveBeenCalledExactlyOnceWith('image-1');
  });

  it('holds the save button while a save of the candidate is in flight', async () => {
    const onSaveToGallery = vi.fn(() => Promise.resolve());
    await renderStagingBar(1, 0, noop, noop, noop, makeSlot, { isSavingToGallery: true, onSaveToGallery });

    await expect.element(page.getByRole('button', { exact: true, name: 'Save to gallery' })).toBeDisabled();
  });

  const visibleStopButtons = (overlay: HTMLElement) =>
    [...overlay.querySelectorAll<HTMLButtonElement>('button')].filter(
      (button) =>
        (button.getAttribute('aria-label') ?? button.textContent?.trim()) ===
          i18n.t('widgets.canvas.staging.stopGenerating') && button.getBoundingClientRect().width > 0
    );

  it.each([
    { canvasWidth: 1000, iconOnly: false },
    { canvasWidth: 360, iconOnly: true },
  ])(
    'shows one Stop generating at $canvasWidth px, labelled only when there is room',
    async ({ canvasWidth, iconOnly }) => {
      const { overlay } = await renderStagingBar(
        2,
        1,
        noop,
        noop,
        noop,
        (index) => (index === 0 ? makeSlot(index) : makePlaceholder(index)),
        { canvasWidth, isGenerating: true }
      );
      const stops = visibleStopButtons(overlay);

      expect(stops).toHaveLength(1);
      // The icon-only one carries its name (and so its tooltip); the labelled one shows the words themselves.
      expect(stops[0]!.hasAttribute('aria-label')).toBe(iconOnly);
      expect(stops[0]!.getBoundingClientRect().right).toBeLessThanOrEqual(overlay.getBoundingClientRect().right);
    }
  );

  it('keeps a long Stop generating label and every result action inside a wide canvas', async () => {
    i18n.addResource(
      'en',
      'translation',
      'widgets.canvas.staging.stopGenerating',
      'Stop generating the remaining images'
    );
    try {
      const { overlay } = await renderStagingBar(
        3,
        0,
        noop,
        noop,
        noop,
        (index) => (index === 0 ? makeSlot(index) : makePlaceholder(index)),
        { canvasWidth: 1000, isGenerating: true }
      );
      const bounds = overlay.getBoundingClientRect();

      expect(visibleStopButtons(overlay)[0]?.textContent?.trim()).toBe('Stop generating the remaining images');
      for (const control of overlay.querySelectorAll<HTMLElement>('button')) {
        const rect = control.getBoundingClientRect();
        if (rect.width > 1 && !control.closest('[data-scope="scroll-area"]')) {
          expect(rect.left).toBeGreaterThanOrEqual(bounds.left);
          expect(rect.right).toBeLessThanOrEqual(bounds.right);
        }
      }
    } finally {
      i18n.addResource('en', 'translation', 'widgets.canvas.staging.stopGenerating', 'Stop generating');
    }
  });

  it('stops generating while a finished result is selected and the next image of its run starts', async () => {
    const onStopGenerating = vi.fn();
    await renderStagingBar(
      2,
      0,
      noop,
      noop,
      noop,
      (index) => (index === 0 ? makeSlot(index) : makePlaceholder(index)),
      { isGenerating: true, onStopGenerating }
    );

    await page.getByRole('button', { exact: true, name: 'Stop generating' }).click();

    expect(onStopGenerating).toHaveBeenCalledOnce();
  });

  it('offers no Stop generating once every image has finished', async () => {
    const { overlay } = await renderStagingBar(2, 0);

    expect(visibleStopButtons(overlay)).toEqual([]);
  });
});

describe('StagingBar focus', () => {
  const nextFrame = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );

  /** A canvas region whose surface outlives the bar, which unmounts with its last result as the canvas's does. */
  const CanvasWithStaging = ({ slots }: { slots: CanvasStagingSlot[] }) => {
    const [remaining, setRemaining] = useState(slots);
    const selected = remaining[0];
    const discardSelected = () => setRemaining((current) => current.slice(1));

    return (
      <Box h="400px" position="relative" role="region" w={`${CANVAS_WIDTH}px`}>
        <Box data-canvas-surface="" data-testid="surface" h="full" tabIndex={-1} />
        <CanvasBottomOverlay.Root>
          {remaining.length > 0 ? (
            <CanvasBottomOverlay.Staging>
              <StagingBar
                antialiasProgressImages={false}
                areThumbnailsVisible
                autoSwitchMode="off"
                canAccept
                hasMultipleSlots={remaining.length > 1}
                isGenerating={false}
                isSavingToGallery={false}
                isVisible
                selectedCandidate={selected?.kind === 'candidate' ? selected.candidate : undefined}
                selectedImageIndex={0}
                selectedSlot={selected}
                slots={remaining}
                onAccept={noop}
                onCycle={noop}
                onDiscardAll={() => setRemaining([])}
                onDiscardSelected={discardSelected}
                onPreloadCandidate={noop}
                onSelectImage={noop}
                onSaveToGallery={() => Promise.resolve()}
                onSaveToLayerAndContinue={noop}
                onSetAutoSwitch={noop}
                onStopGenerating={() => setRemaining((current) => current.filter((slot) => slot.kind === 'candidate'))}
                onToggleThumbnails={noop}
                onToggleVisibility={noop}
              />
            </CanvasBottomOverlay.Staging>
          ) : null}
        </CanvasBottomOverlay.Root>
      </Box>
    );
  };

  const renderCanvas = async (slotCount: number, slotAt: (index: number) => CanvasStagingSlot = makeSlot) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() =>
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <CanvasWithStaging slots={Array.from({ length: slotCount }, (_, index) => slotAt(index))} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
    const thumbnails = () =>
      Array.from(host!.querySelectorAll<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"] button'));
    return { surface: host.querySelector<HTMLElement>('[data-testid="surface"]')!, thumbnails };
  };

  it('hands focus to the canvas surface when discarding the last result removes the bar', async () => {
    const { surface, thumbnails } = await renderCanvas(1);

    await act(() => thumbnails()[0]!.focus());
    await act(() => userEvent.keyboard('{Delete}'));
    await nextFrame();

    expect(host!.querySelector('[data-testid="staging-context-menu"]')).toBeNull();
    expect(document.activeElement).toBe(surface);
  });

  it('hands focus to the canvas surface after Discard all', async () => {
    const { surface } = await renderCanvas(3);

    await userEvent.click(page.getByRole('button', { exact: true, name: 'Discard all' }).first());
    await nextFrame();

    expect(document.activeElement).toBe(surface);
  });

  it('keeps focus with the results when Stop generating goes with the last image still to come', async () => {
    const { thumbnails } = await renderCanvas(2, (index) => (index === 0 ? makeSlot(index) : makePlaceholder(index)));

    await userEvent.click(page.getByRole('button', { exact: true, name: 'Stop generating' }));
    await nextFrame();

    expect(page.getByRole('button', { exact: true, name: 'Stop generating' }).elements()).toEqual([]);
    expect(document.activeElement).toBe(thumbnails()[0]);
  });

  it('returns focus to the thumbnail a dismissed menu opened from', async () => {
    const { thumbnails } = await renderCanvas(3);

    await act(() => thumbnails()[0]!.focus());
    await act(() => {
      thumbnails()[0]!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    const close = page.getByRole('button', { name: 'Close menu' });
    await act(() => (close.element() as HTMLElement).focus());
    await act(() => (close.element() as HTMLElement).click());
    await nextFrame();

    expect(document.activeElement).toBe(thumbnails()[0]);
  });
});

describe('StagingBar keys', () => {
  /** Records the keydowns that reach the window, where the workbench hotkeys listen. */
  const watchWindow = () => {
    const reached: string[] = [];
    const listener = (event: KeyboardEvent) => reached.push(event.key);
    window.addEventListener('keydown', listener);
    return { reached, stop: () => window.removeEventListener('keydown', listener) };
  };

  it('picks with the arrows, moving focus with the pick, and keeps keys from the canvas', async () => {
    const onCycle = vi.fn();
    const onSelectImage = vi.fn();
    const { thumbnails } = await renderStagingBar(3, 0, noop, noop, onSelectImage, makeSlot, { onCycle });
    const window = watchWindow();
    try {
      // Focus alone picks nothing.
      await act(() => thumbnails[0]!.focus());
      expect(onSelectImage).not.toHaveBeenCalled();
      await act(() => userEvent.keyboard('{ArrowRight}'));
      expect(document.activeElement).toBe(thumbnails[1]);
      await act(() => userEvent.keyboard('{ArrowRight}'));
      expect(document.activeElement).toBe(thumbnails[2]);
      // Wrapping, as the step buttons do.
      await act(() => userEvent.keyboard('{ArrowRight}'));
      expect(document.activeElement).toBe(thumbnails[0]);
      await act(() => userEvent.keyboard('{ArrowLeft}{ArrowUp}'));
      expect(document.activeElement).toBe(thumbnails[2]);
    } finally {
      window.stop();
    }

    expect(onSelectImage.mock.calls).toEqual([[1], [2], [0], [2]]);
    expect(onCycle).not.toHaveBeenCalled();
    expect(window.reached).toEqual([]);
  });

  it('takes one Tab stop for the strip: the selected thumbnail', async () => {
    const { thumbnails } = await renderStagingBar(3, 1);

    expect(thumbnails.map((thumbnail) => thumbnail.tabIndex)).toEqual([-1, 0, -1]);
  });

  it('steps the selection from the bar buttons, and Delete discards the selected result', async () => {
    const onCycle = vi.fn();
    const onDiscardSelected = vi.fn();
    await renderStagingBar(3, 0, noop, noop, noop, makeSlot, { onCycle, onDiscardSelected });
    const window = watchWindow();
    try {
      await act(() =>
        (page.getByRole('button', { exact: true, name: 'Discard all' }).element() as HTMLElement).focus()
      );
      await act(() => userEvent.keyboard('{ArrowRight}{ArrowLeft}{Delete}{Backspace}'));
    } finally {
      window.stop();
    }

    expect(onCycle.mock.calls).toEqual([[1], [-1]]);
    expect(onDiscardSelected).toHaveBeenCalledTimes(2);
    expect(window.reached).toEqual([]);
  });

  it('leaves keys alone inside its menus: the accept options and a thumbnail menu neither discard nor step', async () => {
    const onCycle = vi.fn();
    const onDiscardSelected = vi.fn();
    const { thumbnails } = await renderStagingBar(3, 0, noop, noop, noop, makeSlot, { onCycle, onDiscardSelected });

    await page.getByRole('button', { name: 'More accept options' }).click();
    await expect.element(page.getByRole('menu')).toBeVisible();
    await act(() => userEvent.keyboard('{Delete}{ArrowRight}{Backspace}{ArrowLeft}'));
    await act(() => userEvent.keyboard('{Escape}'));

    await act(() => {
      thumbnails[0]!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.element(page.getByRole('menu')).toBeVisible();
    await act(() => userEvent.keyboard('{Delete}{ArrowRight}{Backspace}{ArrowLeft}'));

    expect(onDiscardSelected).not.toHaveBeenCalled();
    expect(onCycle).not.toHaveBeenCalled();
  });

  it('lets modified presses and other keys through to the workbench', async () => {
    const onCycle = vi.fn();
    const { thumbnails } = await renderStagingBar(3, 0, noop, noop, noop, makeSlot, { onCycle });
    const window = watchWindow();
    try {
      await act(() => thumbnails[0]!.focus());
      await act(() => userEvent.keyboard('{Alt>}{ArrowRight}{/Alt}b'));
    } finally {
      window.stop();
    }

    expect(onCycle).not.toHaveBeenCalled();
    expect(window.reached).toEqual(['Alt', 'ArrowRight', 'b']);
  });
});

describe('StagingBar picks and Auto-switch', () => {
  type Mode = 'latest' | 'off' | 'progress';

  /** A run of three images on the canvas: two results staged, the third still generating. */
  const createStagedState = (mode: Mode): WorkbenchState => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    const result = (backendItemId: number) => ({
      height: 512,
      imageName: `result-${backendItemId}`,
      imageUrl: TRANSPARENT_PIXEL,
      placement: { height: 512, opacity: 1, width: 512, x: 0, y: 0 },
      queuedAt: '2026-01-01T00:00:00.000Z',
      sourceBackendItemId: backendItemId,
      sourceQueueItemId: 'run-1',
      thumbnailUrl: TRANSPARENT_PIXEL,
      width: 512,
    });
    const run = {
      backendItemIds: [11, 12, 13],
      cancellable: true,
      completedBackendItemIds: [11, 12],
      id: 'run-1',
      snapshot: {
        canvas: {
          document: { bbox: { height: 512, width: 512, x: 0, y: 0 }, height: 1024, width: 1024 },
          documentRevision: project.canvas.documentRevision,
        },
        destination: 'canvas',
        graph: { id: 'graph', label: 'Generate' },
        presentation: { batchCount: 3, height: 512, width: 512 },
        sourceId: 'generate',
        submittedAt: '2026-01-01T00:00:00.000Z',
      },
      status: 'running',
    } as unknown as WorkbenchQueueItem;
    const pendingImages = [result(11), result(12)];

    return {
      ...state,
      projects: [
        {
          ...project,
          canvas: {
            ...project.canvas,
            stagingArea: {
              ...project.canvas.stagingArea,
              autoSwitchMode: mode,
              isVisible: true,
              pendingImageIds: pendingImages.map((image) => image.imageName),
              pendingImages,
              selectedImageIndex: 1,
            },
          },
          queue: { items: [run] },
        },
      ],
    };
  };

  /** The bar wired to the workbench reducer, as the canvas widget wires it. */
  const WiredStagingBar = ({ initial }: { initial: WorkbenchState }) => {
    const [state, setState] = useState(initial);
    const project = state.projects[0]!;
    const { stagingArea } = project.canvas;
    const slots = getCanvasStagingSlots(project.canvas, project.queue.items);
    const selectedSlot = slots[stagingArea.selectedImageIndex];
    const mutate = (mutation: CanvasProjectMutation) =>
      setState((current) =>
        workbenchReducer(current, { mutation, projectId: project.id, type: 'applyCanvasProjectMutation' })
      );

    return (
      <Box h="400px" position="relative" w={`${CANVAS_WIDTH}px`}>
        {/* Where Tab starts from: the canvas, before the bar. */}
        <button data-testid="before-bar" type="button">
          Canvas
        </button>
        <CanvasBottomOverlay.Root>
          <CanvasBottomOverlay.Staging>
            <StagingBar
              antialiasProgressImages={false}
              areThumbnailsVisible
              autoSwitchMode={stagingArea.autoSwitchMode}
              canAccept
              hasMultipleSlots={slots.length > 1}
              isGenerating
              isSavingToGallery={false}
              isVisible
              selectedCandidate={selectedSlot?.kind === 'candidate' ? selectedSlot.candidate : undefined}
              selectedImageIndex={stagingArea.selectedImageIndex}
              selectedSlot={selectedSlot}
              slots={slots}
              onAccept={noop}
              onCycle={(direction) => mutate({ direction, type: 'cycleStagedImage' })}
              onDiscardAll={noop}
              onDiscardSelected={noop}
              onPreloadCandidate={noop}
              onSelectImage={(imageIndex) => mutate({ imageIndex, type: 'setStagedImageIndex' })}
              onSaveToGallery={() => Promise.resolve()}
              onSaveToLayerAndContinue={noop}
              onSetAutoSwitch={(mode) => mutate({ mode, type: 'setCanvasStagingAutoSwitch' })}
              onStopGenerating={noop}
              onToggleThumbnails={noop}
              onToggleVisibility={noop}
            />
          </CanvasBottomOverlay.Staging>
        </CanvasBottomOverlay.Root>
      </Box>
    );
  };

  const renderWired = async (mode: Mode) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() =>
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <WiredStagingBar initial={createStagedState(mode)} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  };
  const autoSwitch = () => page.getByRole('button', { name: /^Auto-switch:/ });
  const thumbnail = (number: number) =>
    page.getByRole('button', { exact: true, name: `Select staged candidate ${number}` });

  it.each(['latest', 'progress'] as const)(
    'clicking off the newest result under %s turns Auto-switch Off',
    async (mode) => {
      await renderWired(mode);

      await thumbnail(1).click();

      await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: Off');
      await expect.element(thumbnail(1)).toHaveAttribute('aria-current', 'true');
    }
  );

  it('passes the strip with Tab in one stop, picking nothing and leaving Auto-switch as it was', async () => {
    await renderWired('latest');
    const visited: Element[] = [];

    await act(() => host!.querySelector<HTMLElement>('[data-testid="before-bar"]')!.focus());
    for (let step = 0; step < 12; step += 1) {
      await userEvent.keyboard('{Tab}');
      if (document.activeElement) {
        visited.push(document.activeElement);
      }
    }

    const thumbnailsVisited = visited.filter((element) => element.hasAttribute('data-staging-thumbnail'));
    expect(new Set(thumbnailsVisited).size).toBe(1);
    expect(thumbnailsVisited[0]?.getAttribute('aria-label')).toBe('Select staged candidate 2');
    expect(visited.some((element) => element.getAttribute('aria-label') === 'Accept to layer')).toBe(true);
    await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: Newest');
    await expect.element(thumbnail(2)).toHaveAttribute('aria-current', 'true');
  });

  it('an arrow from the selected thumbnail is a pick: off the newest, Auto-switch turns Off', async () => {
    await renderWired('latest');

    await act(() => (thumbnail(2).element() as HTMLElement).focus());
    await userEvent.keyboard('{ArrowLeft}');

    await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: Off');
    await expect.element(thumbnail(1)).toHaveAttribute('aria-current', 'true');
    expect(document.activeElement).toBe(thumbnail(1).element());
  });

  it("right-clicking an older result's thumbnail is a pick too", async () => {
    await renderWired('progress');

    await act(() => {
      thumbnail(1)
        .element()
        .dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 }));
    });

    await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: Off');
    await expect.element(thumbnail(1)).toHaveAttribute('aria-current', 'true');
  });

  it('stepping off the newest result with the bar arrows turns Auto-switch Off', async () => {
    await renderWired('latest');

    await page.getByRole('button', { exact: true, name: 'Previous staged candidate' }).click();

    await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: Off');
    await expect.element(thumbnail(1)).toHaveAttribute('aria-current', 'true');
  });

  it.each(['off', 'latest', 'progress'] as const)(
    'clicking the image still generating under %s follows it: Auto-switch In progress',
    async (mode) => {
      await renderWired(mode);

      await thumbnail(3).click();

      await expect.element(autoSwitch()).toHaveAccessibleName('Auto-switch: In progress');
      await expect.element(thumbnail(3)).toHaveAttribute('aria-current', 'true');
    }
  );
});

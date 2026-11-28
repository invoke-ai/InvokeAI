import '@fontsource/inter/index.css';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
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
  loadRecall: vi.fn(),
  targets: [] as { slotId: string; x: number; y: number }[],
}));
const slotProgress = vi.hoisted(() => new Map<string, { message: string; percentage: number | null }>());
vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useQueueItemSlotProgress: (queueItemId: string, itemIndex: number) =>
    slotProgress.get(`${queueItemId}:${itemIndex}`) ?? null,
}));
vi.mock('./StagingItemContextMenu', () => ({
  loadStagingItemRecall: contextMenu.loadRecall,
  StagingItemContextMenu: ({
    onClose,
    target,
  }: {
    onClose: () => void;
    target: { slot: { id: string }; x: number; y: number };
  }) => {
    contextMenu.targets.push({ slotId: target.slot.id, x: target.x, y: target.y });
    return (
      <div data-testid="staging-context-menu">
        <button type="button" onClick={onClose}>
          Close menu
        </button>
      </div>
    );
  },
}));

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
  slotProgress.clear();
});

describe('StagingBar thumbnail strip', () => {
  it('shows concurrent iterations of one run independently and leaves waiting slots without a badge', async () => {
    const waiting = { message: 'Preparing', percentage: null };
    slotProgress.set('queue-item:1', waiting);
    slotProgress.set('queue-item:2', waiting);
    const slotAt = (index: number) => makePlaceholder(index + 1);
    const render = () => renderStagingBar(4, 0, noop, noop, noop, slotAt, { isGenerating: true });
    const badges = (thumbnails: HTMLElement[]) =>
      thumbnails.map((thumbnail) => thumbnail.querySelector('[role="progressbar"]') !== null);

    expect(badges((await render()).thumbnails)).toEqual([true, true, false, false]);

    // An event for a different iteration must not take over or hide a running slot's badge.
    slotProgress.set('queue-item:2', { message: 'Loading model', percentage: null });
    expect(badges((await render()).thumbnails)).toEqual([true, true, false, false]);

    slotProgress.delete('queue-item:1');
    slotProgress.set('queue-item:3', waiting);
    expect(badges((await render()).thumbnails)).toEqual([false, true, true, false]);
  });

  it('shows determinate progress only on the Canvas slot reporting progress', async () => {
    slotProgress.set('queue-item:1', { message: 'Child sampling', percentage: 0.5 });
    const slotAt = (index: number) => makePlaceholder(index + 1);
    const { thumbnails } = await renderStagingBar(2, 0, noop, noop, noop, slotAt, { isGenerating: true });
    const progressBar = thumbnails[0]!.querySelector('[role="progressbar"]');
    expect(progressBar?.getAttribute('aria-valuenow')).toBe('50');
    expect(thumbnails[1]!.querySelector('[role="progressbar"]')).toBeNull();
  });

  it('keeps every thumbnail reachable by scrolling when many images are staged', async () => {
    const { overlay, thumbnails, viewport } = await renderStagingBar(24);
    const bounds = viewport.getBoundingClientRect();

    expect(thumbnails).toHaveLength(24);
    expect(bounds.width).toBe(overlay.getBoundingClientRect().width);
    expect(bounds.width).toBeLessThanOrEqual(CANVAS_WIDTH);
    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);

    await interact(() => {
      viewport.scrollLeft = 0;
    });
    expect(thumbnails[0]!.getBoundingClientRect().left).toBeGreaterThanOrEqual(bounds.left - 1);

    await interact(() => {
      viewport.scrollLeft = viewport.scrollWidth;
    });
    expect(thumbnails.at(-1)!.getBoundingClientRect().right).toBeLessThanOrEqual(bounds.right + 1);
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
});

describe('StagingBar focus', () => {
  const nextFrame = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );

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
      await act(() => thumbnails[0]!.focus());
      expect(onSelectImage).not.toHaveBeenCalled();
      await act(() => userEvent.keyboard('{ArrowRight}'));
      expect(document.activeElement).toBe(thumbnails[1]);
      await act(() => userEvent.keyboard('{ArrowRight}'));
      expect(document.activeElement).toBe(thumbnails[2]);
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
    const onSelectImage = vi.fn();
    const { thumbnails } = await renderStagingBar(3, 0, noop, noop, onSelectImage, makeSlot, {
      onCycle,
      onDiscardSelected,
    });

    await page.getByRole('button', { name: 'More accept options' }).click();
    await expect.element(page.getByRole('menu')).toBeVisible();
    await act(() => userEvent.keyboard('{Delete}{ArrowRight}{Backspace}{ArrowLeft}'));
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(page.getByRole('menu')).not.toBeInTheDocument();

    await act(() => {
      thumbnails[0]!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.element(page.getByTestId('staging-context-menu')).toBeVisible();
    // Escape left focus on the accept options; the thumbnail's own keys are the ones its menu must hold.
    await act(() => thumbnails[0]!.focus());
    await act(() => userEvent.keyboard('{Delete}{ArrowRight}{Backspace}{ArrowLeft}'));

    expect(onDiscardSelected).not.toHaveBeenCalled();
    expect(onCycle).not.toHaveBeenCalled();
    // Only the right-click's own selection; no arrow moved it.
    expect(onSelectImage.mock.calls).toEqual([[0]]);
  });
});

describe('StagingBar picks and Auto-switch', () => {
  type Mode = 'latest' | 'off' | 'progress';

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

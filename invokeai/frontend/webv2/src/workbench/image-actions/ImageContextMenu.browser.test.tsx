/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GalleryBoard, GalleryImage, GalleryItem, GalleryItemKey } from '@features/gallery';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ModelsSnapshot } from '@features/models/data/modelsStore';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getContrastRatio } from '@platform/ui/theme/contrastRatio.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { DEFAULT_THEME_ID, THEMES } from '@theme/themes';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type * as ModelOnlyUpscaleModule from './modelOnlyUpscale';
import type { ImageActions } from './useImageActions';

import { ImageContextMenu, type ImageContextMenuTarget } from './ImageContextMenu';
import { EMPTY_IMAGE_RECALL_CAPABILITIES } from './imageRecall';

const NO_BOARDS: [] = [];
const LOAD_WORKFLOW = 'widgets.gallery.itemActions.loadWorkflow.label';
const GENERATE_PBR_MAPS = 'widgets.gallery.itemActions.pbrMaps.label';

const findGalleryItem = vi.fn();

vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => findGalleryItem }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/projects/library', () => ({
  useProjectLibrarySelector: (selector: (snapshot: { summaries: [] }) => unknown) => selector({ summaries: [] }),
}));
const commandsMock = vi.hoisted(() => ({
  generation: { setSource: vi.fn(), submitModelOnlyUpscale: vi.fn() },
  notifications: { add: vi.fn() },
  widgets: { patchValues: vi.fn() },
}));
const galleryMock = vi.hoisted(() => ({
  metadata: vi.fn((): Promise<Record<string, unknown> | null> => Promise.resolve(null)),
}));
vi.mock('@features/gallery', async (importOriginal) => {
  const original = await importOriginal<{ galleryImages: Record<string, unknown> }>();

  return { ...original, galleryImages: { ...original.galleryImages, metadata: galleryMock.metadata } };
});
// The Upscale submenu asks for the model library on mount; tests seed the snapshot instead of fetching it.
const ensureModelsLoaded = vi.hoisted(() => vi.fn(() => Promise.resolve()));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded,
}));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGalleryBoardDestinations: () => null,
}));
// The upscale command's chunk loads on pick; `fail` stands in for that load failing.
const modelOnlyLoad = vi.hoisted(() => ({ fail: false }));
vi.mock('./modelOnlyUpscale', async (importOriginal) => {
  const original = await importOriginal<typeof ModelOnlyUpscaleModule>();

  return {
    ...original,
    queueModelOnlyUpscale: (...args: Parameters<typeof original.queueModelOnlyUpscale>) =>
      modelOnlyLoad.fail
        ? Promise.reject(new Error('Failed to fetch dynamically imported module'))
        : original.queueModelOnlyUpscale(...args),
  };
});
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectName: () => 'Project',
  useWorkbenchCommands: () => commandsMock,
}));
// Keys stand in for copy; the one interpolated value a test reads, an upscale factor, is shown as the locale does.
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { key?: string; scale?: number }) =>
      // The hint renders beside its label as written, short; a key there would widen the menu past the viewport.
      key === 'widgets.gallery.itemMenu.compareShortcut'
        ? `${String(options?.key)}+click`
        : options?.scale === undefined
          ? key
          : `${options.scale}×${key.endsWith('.factorNative') ? ' (native)' : ''}`,
  }),
}));

const image = (imageName: string): GalleryImage => ({
  boardId: 'none',
  height: 512,
  imageCategory: 'general',
  imageName,
  imageUrl: `/${imageName}`,
  queuedAt: '2026-06-15T00:00:00Z',
  sourceQueueItemId: 'queue-item',
  starred: false,
  thumbnailUrl: `/thumb-${imageName}`,
  width: 512,
});

const createActions = (deleteItems: ImageActions['deleteItems']): ImageActions => ({
  canUseAsConditioningClip: { audio: false, video: false },
  canUseAsReferenceImage: false,
  canUseAsReferenceVideo: false,
  copyImage: vi.fn(() => Promise.resolve()),
  createCanvasFromImages: vi.fn(() => Promise.resolve()),
  deleteItems,
  deleteImages: vi.fn(() => Promise.resolve()),
  deriveImageRecallCapabilities: vi.fn(() => EMPTY_IMAGE_RECALL_CAPABILITIES),
  downloadItem: vi.fn(() => Promise.resolve()),
  downloadItems: vi.fn(() => Promise.resolve()),
  downloadImage: vi.fn(() => Promise.resolve()),
  downloadImages: vi.fn(() => Promise.resolve()),
  generatePbrMaps: vi.fn(() => Promise.resolve()),
  getImageRecallCapabilities: vi.fn(() => Promise.resolve(EMPTY_IMAGE_RECALL_CAPABILITIES)),
  loadImageWorkflow: vi.fn(() => Promise.resolve()),
  moveItemsToBoard: vi.fn(() => Promise.resolve()),
  moveImagesToBoard: vi.fn(() => Promise.resolve()),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  openImageInPreview: vi.fn(),
  recallImageData: vi.fn(() => Promise.resolve()),
  getVideoRecallCapabilities: vi.fn(() => Promise.resolve({ all: false, prompts: false, remix: false, seed: false })),
  hasVideoWorkflow: vi.fn(() => Promise.resolve(false)),
  loadVideoWorkflow: vi.fn(() => Promise.resolve()),
  recallVideoData: vi.fn(() => Promise.resolve()),
  selectForCompare: vi.fn(),
  sendToCanvas: vi.fn(() => Promise.resolve()),
  sendImageAsControl: vi.fn(() => Promise.resolve()),
  setItemsStarred: vi.fn(() => Promise.resolve()),
  setImagesStarred: vi.fn(() => Promise.resolve()),
  savePromptAsTemplate: vi.fn(),
  sendToInitialVideo: vi.fn(),
  useAsReferenceImage: vi.fn(),
  useAsConditioningClip: vi.fn(),
  useAsReferenceVideo: vi.fn(),
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

/** Polls inside `act` until the condition holds, so a wait tracks state rather than a fixed delay. */
const settleUntil = async (isSettled: () => boolean, description: string, timeoutMs = 2000): Promise<void> => {
  const deadline = Date.now() + timeoutMs;

  while (!isSettled()) {
    if (Date.now() > deadline) {
      throw new Error(`Timed out after ${String(timeoutMs)}ms waiting for ${description}`);
    }
    await act(async () => {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 10);
      });
    });
  }
};

/**
 * Hovers an item the way zag recognises. zag only treats a move as a real hover when the pointer position changes,
 * so approach the item's center from one pixel away instead of dispatching a static point, which reads as no move
 * at all whenever zag's last recorded position is already there.
 */
const hoverItem = async (target: HTMLElement): Promise<void> => {
  const rect = target.getBoundingClientRect();
  const clientX = rect.left + rect.width / 2;
  const clientY = rect.top + rect.height / 2;
  await interact(() => {
    for (const x of [clientX - 1, clientX]) {
      target.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, clientX: x, clientY, pointerType: 'mouse' })
      );
    }
  });
};

/** Quick icon items select through zag, which needs the item highlighted by a hover before the click. */
const pickQuickItem = async (label: string): Promise<void> => {
  const target = document.querySelector<HTMLElement>(`[aria-label="${label}"]`);
  expect(target).not.toBeNull();
  await hoverItem(target!);
  // Wait for Zag's asynchronous highlight before clicking; unhighlighted items ignore clicks.
  await settleUntil(() => target!.hasAttribute('data-highlighted'), `"${label}" to be highlighted`);
  await interact(() => target!.click());
};

/** An item's label: its first line, so a hint under it does not count. */
const itemLabel = (candidate: HTMLElement): string | undefined =>
  (candidate.querySelector('p') ?? candidate).textContent?.trim() || undefined;

const getMenuItem = (label: string): HTMLElement => {
  const item = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (candidate) => itemLabel(candidate) === label || candidate.textContent?.trim() === label
  );

  if (!item) {
    throw new Error(`Could not find menu item: ${label}`);
  }

  return item;
};

/** Items that select through zag (`onSelect`) need the item highlighted by a hover before the click. */
const pickItem = async (label: string): Promise<void> => {
  const target = getMenuItem(label);
  await hoverItem(target);
  await settleUntil(() => target.hasAttribute('data-highlighted'), `"${label}" to be highlighted`);
  await interact(() => target.click());
};

const getOpenAlertDialog = (): HTMLElement | null =>
  document.querySelector<HTMLElement>('[role="alertdialog"][data-state="open"]');

const Harness = ({ actions, images }: { actions: ImageActions; images: GalleryImage[] }) => {
  const [target, setTarget] = useState<ImageContextMenuTarget | null>({ images, x: 20, y: 20 });

  return <ImageContextMenu actions={actions} boards={NO_BOARDS} target={target} onClose={() => setTarget(null)} />;
};

const renderMenu = async (actions: ImageActions, images: GalleryImage[]) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ChakraProvider value={system}>
          <Harness actions={actions} images={images} />
        </ChakraProvider>
      </QueryClientProvider>
    );
  });
};

const item = (kind: GalleryItem['kind'], name: string): GalleryItem => {
  const base = {
    boardId: 'none',
    category: 'general' as const,
    createdAt: '2026-06-15T00:00:00Z',
    fullUrl: `/full/${name}`,
    height: 512,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumb-${name}`,
    width: 512,
  };

  return kind === 'video' ? { ...base, durationSeconds: 12, kind } : { ...base, kind };
};

const renderItemMenu = async (
  actions: ImageActions,
  target: GalleryItemContextMenuTarget,
  previewVideoActions?: {
    isCopyCurrentFrameAvailable: boolean;
    itemKey: GalleryItemKey;
    onCopyCurrentFrame: () => void;
    onOpenDetails: () => void;
  }
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ChakraProvider value={system}>
          <ImageContextMenu
            actions={actions}
            boards={NO_BOARDS}
            previewVideoActions={previewVideoActions}
            target={target as unknown as ImageContextMenuTarget}
            onClose={vi.fn()}
          />
        </ChakraProvider>
      </QueryClientProvider>
    );
  });
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ImageContextMenu deletion delegation', () => {
  it('delegates deletion to the shared image actions adapter', async () => {
    const deleteItems = vi.fn(() => Promise.resolve());
    await renderMenu(createActions(deleteItems), [image('single.png')]);

    await interact(() => getMenuItem('widgets.gallery.itemMenu.deleteImage').click());

    expect(deleteItems).toHaveBeenCalledExactlyOnceWith([{ kind: 'image', name: 'single.png' }]);
    expect(getOpenAlertDialog()).toBeNull();
  });
});

describe('ImageContextMenu find in gallery', () => {
  it('reveals a single image or video by its gallery reference', async () => {
    findGalleryItem.mockClear();
    await renderMenu(createActions(vi.fn()), [image('still.png')]);
    await interact(() => getMenuItem('widgets.gallery.findInGallery').click());
    expect(findGalleryItem).toHaveBeenCalledExactlyOnceWith({ kind: 'image', name: 'still.png' });

    await interact(() => root?.unmount());
    const video = item('video', 'clip.mp4');
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: [{ kind: 'video', name: video.name }],
      items: [video],
      x: 20,
      y: 20,
    });
    await interact(() => getMenuItem('widgets.gallery.findInGallery').click());
    expect(findGalleryItem).toHaveBeenLastCalledWith({ kind: 'video', name: 'clip.mp4' });
  });

  it('is absent from bulk menus, which have no single item to reveal', async () => {
    await renderMenu(createActions(vi.fn()), [image('a.png'), image('b.png')]);

    expect(document.body.textContent).not.toContain('widgets.gallery.findInGallery');
  });
});

describe('ImageContextMenu load workflow', () => {
  it('enables Load Workflow once the image is known to embed one and hands the image to the action', async () => {
    const actions = createActions(vi.fn());
    actions.getImageRecallCapabilities = vi.fn(() =>
      Promise.resolve({ ...EMPTY_IMAGE_RECALL_CAPABILITIES, workflow: true })
    );
    await renderMenu(actions, [image('made-by-workflow.png')]);

    await vi.waitFor(() => expect(getMenuItem(LOAD_WORKFLOW).getAttribute('aria-disabled')).not.toBe('true'));
    await interact(() => getMenuItem(LOAD_WORKFLOW).click());

    expect(actions.loadImageWorkflow).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'made-by-workflow.png' })
    );
  });

  it('keeps Load Workflow disabled for an image without one', async () => {
    const actions = createActions(vi.fn());
    await renderMenu(actions, [image('plain.png')]);

    await vi.waitFor(() => expect(actions.getImageRecallCapabilities).toHaveBeenCalled());
    expect(getMenuItem(LOAD_WORKFLOW).getAttribute('aria-disabled')).toBe('true');
    expect(actions.loadImageWorkflow).not.toHaveBeenCalled();
  });
});

describe('ImageContextMenu PBR maps', () => {
  it('hands a single image to the PBR maps action', async () => {
    const actions = createActions(vi.fn());
    await renderMenu(actions, [image('texture.png')]);

    await interact(() => getMenuItem(GENERATE_PBR_MAPS).click());

    expect(actions.generatePbrMaps).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'texture.png' })
    );
  });

  it('is absent from bulk menus, which queue nothing per image', async () => {
    await renderMenu(createActions(vi.fn()), [image('a.png'), image('b.png')]);

    expect(document.body.textContent).not.toContain(GENERATE_PBR_MAPS);
  });
});

describe('ImageContextMenu starred state', () => {
  const starIconFill = (label: string): string =>
    getComputedStyle(document.querySelector<HTMLElement>(`[aria-label="${label}"] svg`)!).fill;

  it('fills the star for a starred item and leaves it outlined otherwise', async () => {
    // Fill distinguishes starred state because Lucide outlines alone look identical.
    const unstarred = item('image', 'plain.png');
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: [{ kind: 'image', name: unstarred.name }],
      items: [unstarred],
      x: 20,
      y: 20,
    });

    expect(starIconFill('widgets.preview.starImage')).toBe('none');

    await interact(() => root?.unmount());
    host?.remove();

    const starred = { ...item('image', 'fave.png'), starred: true };
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: [{ kind: 'image', name: starred.name }],
      items: [starred],
      x: 20,
      y: 20,
    });

    expect(starIconFill('widgets.preview.unstarImage')).not.toBe('none');
  });

  it('fills the bulk star only once every selected item is starred', async () => {
    const mixed = [item('image', 'a.png'), { ...item('image', 'b.png'), starred: true }];
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: mixed.map((entry) => ({ kind: 'image' as const, name: entry.name })),
      items: mixed,
      x: 20,
      y: 20,
    });

    expect(getComputedStyle(getMenuItem('widgets.gallery.starSelection').querySelector('svg')!).fill).toBe('none');

    await interact(() => root?.unmount());
    host?.remove();

    const allStarred = mixed.map((entry) => ({ ...entry, starred: true }));
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: allStarred.map((entry) => ({ kind: 'image' as const, name: entry.name })),
      items: allStarred,
      x: 20,
      y: 20,
    });

    expect(getComputedStyle(getMenuItem('widgets.gallery.unstarSelection').querySelector('svg')!).fill).not.toBe(
      'none'
    );
  });
});

describe('ImageContextMenu canvas destinations', () => {
  const menuItemLabels = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).map(
      (candidate) => itemLabel(candidate) || candidate.getAttribute('aria-label')
    );

  /** Top-level menu entries in order, with separators as `—`, so a position within its group is observable. */
  const menuSequence = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"], [role="separator"]')).map((candidate) =>
      candidate.getAttribute('role') === 'separator'
        ? '—'
        : itemLabel(candidate) || candidate.getAttribute('aria-label')
    );
  const ADD = 'widgets.gallery.itemMenu.addToCanvasAs';
  const NEW_PROJECT = 'widgets.canvas.import.newProjectFromImage';

  it('groups the single-image menu by intent: continue working, then use and reuse, then housekeeping', async () => {
    await renderMenu(createActions(vi.fn()), [image('image-0.png')]);
    const sequence = menuSequence();
    const workflowIndex = sequence.indexOf(LOAD_WORKFLOW);

    expect(sequence.slice(workflowIndex, workflowIndex + 15)).toEqual([
      LOAD_WORKFLOW,
      '—',
      'widgets.preview.editOnCanvas',
      'widgets.gallery.itemActions.upscale.label',
      'widgets.gallery.itemMenu.sendToVideo',
      '—',
      ADD,
      NEW_PROJECT,
      '—',
      'widgets.gallery.itemMenu.useAsReferenceImage',
      'widgets.gallery.itemActions.useAsControl',
      'widgets.gallery.itemMenu.useAsPromptTemplate',
      'widgets.gallery.itemMenu.reuseFromImage',
      '—',
      'widgets.preview.selectForCompare',
    ]);
    const headings = Array.from(document.querySelectorAll('[data-scope="menu"][data-part="item-group-label"]')).map(
      (label) => label.textContent?.trim()
    );
    expect(headings).toEqual(['widgets.gallery.itemMenu.continueWorking', 'widgets.gallery.itemMenu.useImageAs']);
    // A context menu holding every group stays single-line: the step hints belong to Preview's own menus.
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.editOnCanvasHint');
  });

  it('edits the image on the current canvas as a raster layer in one step', async () => {
    const actions = createActions(vi.fn());
    await renderMenu(actions, [image('image-0.png')]);

    await pickItem('widgets.preview.editOnCanvas');

    expect(
      vi.mocked(actions.sendToCanvas).mock.calls.map(([images, destination]) => [images.length, destination])
    ).toEqual([[1, 'raster']]);
  });

  it('leaves the Edit and Use groups out of the housekeeping variant, keeping the rest in order', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() => {
      root?.render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <ChakraProvider value={system}>
            <ImageContextMenu
              actions={createActions(vi.fn())}
              boards={NO_BOARDS}
              target={{ images: [image('image-0.png')], x: 20, y: 20 }}
              variant="housekeeping"
              onClose={vi.fn()}
            />
          </ChakraProvider>
        </QueryClientProvider>
      );
    });

    expect(
      menuSequence().filter((entry) => entry !== null && !entry.startsWith('widgets.gallery.itemMenu.open'))
    ).toEqual([
      'widgets.gallery.itemMenu.copyToClipboard',
      'widgets.gallery.itemMenu.downloadImage',
      'widgets.preview.starImage',
      '—',
      LOAD_WORKFLOW,
      'widgets.preview.selectForCompare',
      GENERATE_PBR_MAPS,
      '—',
      'widgets.gallery.findInGallery',
      'widgets.gallery.changeBoard',
      '—',
      'widgets.gallery.itemMenu.deleteImage',
    ]);
  });

  it('reuses the settings from a submenu, Remix first, each gated by what the image records', async () => {
    const actions = createActions(vi.fn());
    actions.getImageRecallCapabilities = vi.fn(() =>
      Promise.resolve({ ...EMPTY_IMAGE_RECALL_CAPABILITIES, remix: true, seed: true })
    );
    await renderMenu(actions, [image('image-0.png')]);
    await hoverItem(getMenuItem('widgets.gallery.itemMenu.reuseFromImage'));
    await settleUntil(
      () => menuItemLabels().includes('widgets.gallery.itemMenu.allSettings'),
      'the reuse submenu to open',
      5000
    );
    const submenu = getMenuItem('widgets.gallery.itemMenu.allSettings').closest('[role="menu"]')!;
    await vi.waitFor(() =>
      expect(
        Array.from(submenu.querySelectorAll<HTMLElement>('[role="menuitem"]')).map(
          (entry) => `${itemLabel(entry)}${entry.getAttribute('aria-disabled') === 'true' ? ' (disabled)' : ''}`
        )
      ).toEqual([
        'widgets.gallery.itemActions.recall.verbs.remix',
        'widgets.gallery.itemMenu.allSettings (disabled)',
        'widgets.gallery.itemActions.recall.verbs.prompts (disabled)',
        'widgets.gallery.itemActions.recall.verbs.seed',
        'widgets.gallery.itemActions.recall.verbs.dimensions (disabled)',
        'widgets.gallery.itemActions.recall.verbs.clipSkip (disabled)',
      ])
    );

    await pickItem('widgets.gallery.itemActions.recall.verbs.seed');
    expect(vi.mocked(actions.recallImageData).mock.calls.map(([entry, kind]) => [entry.imageName, kind])).toEqual([
      ['image-0.png', 'seed'],
    ]);
  });

  it('opens the multi-image menu, below its header, with the two canvas destinations in a group of their own', async () => {
    await renderMenu(
      createActions(vi.fn()),
      Array.from({ length: 3 }, (_, index) => image(`image-${index}.png`))
    );

    expect(menuSequence().slice(0, 5)).toEqual(['—', ADD, NEW_PROJECT, '—', 'widgets.gallery.starSelection']);
    // The single-image hand-offs take one image; the canvas items take the whole selection.
    expect(menuSequence()).not.toContain('widgets.preview.editOnCanvas');
  });

  it.each([1, 3])('adds the %s targeted image(s) to the current canvas as the chosen layer', async (count) => {
    const actions = createActions(vi.fn());
    const images = Array.from({ length: count }, (_, index) => image(`image-${index}.png`));
    await renderMenu(actions, images);

    // A nested menu opens from a real hover on its trigger item, after zag's open delay, so the open is waited for
    // rather than slept through.
    await hoverItem(getMenuItem('widgets.gallery.itemMenu.addToCanvasAs'));
    await settleUntil(
      () => menuItemLabels().includes('widgets.canvas.import.inpaintMask'),
      'the add-to-current-canvas submenu to open',
      5000
    );
    // Raster leads, as the drop zones order them; each destination shows its layer kind's glyph, and resized control
    // a resize glyph so it reads apart from control.
    const destinations = Array.from(
      getMenuItem('widgets.canvas.import.raster')
        .closest('[role="menu"]')!
        .querySelectorAll<HTMLElement>('[role="menuitem"]')
    ).map((item) => [
      item.textContent?.trim(),
      Array.from(item.querySelector('svg')?.classList ?? []).find((name) => name.startsWith('lucide-')),
    ]);
    expect(destinations).toEqual([
      ['widgets.canvas.import.raster', 'lucide-brush'],
      ['widgets.canvas.import.control', 'lucide-sliders-horizontal'],
      ['widgets.canvas.import.inpaintMask', 'lucide-square-dashed-bottom'],
      ['widgets.canvas.import.regionalGuidance', 'lucide-map-pin'],
      ['widgets.canvas.import.regionalReference', 'lucide-image-plus'],
      ['widgets.canvas.import.resizedControl', 'lucide-scaling'],
    ]);
    // Destinations select through zag, which needs the item highlighted by a hover before the click.
    const inpaintMask = getMenuItem('widgets.canvas.import.inpaintMask');
    await hoverItem(inpaintMask);
    await settleUntil(() => inpaintMask.hasAttribute('data-highlighted'), 'the inpaint mask destination highlight');
    await interact(() => inpaintMask.click());

    const calls = vi.mocked(actions.sendToCanvas).mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0]?.[0].map((entry) => entry.imageName)).toEqual(images.map((entry) => entry.imageName));
    expect(calls[0]?.[1]).toBe('inpaint-mask');
    expect(actions.createCanvasFromImages).not.toHaveBeenCalled();
  });

  it.each([1, 3])('starts a new project from the %s targeted image(s) in one step', async (count) => {
    const actions = createActions(vi.fn());
    const images = Array.from({ length: count }, (_, index) => image(`image-${index}.png`));
    await renderMenu(actions, images);

    await pickItem('widgets.canvas.import.newProjectFromImage');

    const calls = vi.mocked(actions.createCanvasFromImages).mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0]?.[0].map((entry) => entry.imageName)).toEqual(images.map((entry) => entry.imageName));
    expect(actions.sendToCanvas).not.toHaveBeenCalled();
  });

  it('offers Use as Control beside Use as Reference Image and hands it the image', async () => {
    const actions = createActions(vi.fn());
    await renderMenu(actions, [image('image-0.png')]);
    const sequence = menuSequence();

    expect(sequence[sequence.indexOf('widgets.gallery.itemMenu.useAsReferenceImage') + 1]).toBe(
      'widgets.gallery.itemActions.useAsControl'
    );
    await pickItem('widgets.gallery.itemActions.useAsControl');

    expect(vi.mocked(actions.sendImageAsControl).mock.calls.map(([entry]) => entry.imageName)).toEqual(['image-0.png']);
  });

  it('leaves Use as Control out of a multi-image menu: one control layer takes one image', async () => {
    await renderMenu(
      createActions(vi.fn()),
      Array.from({ length: 2 }, (_, index) => image(`image-${index}.png`))
    );

    expect(menuSequence()).not.toContain('widgets.gallery.itemActions.useAsControl');
  });
});

describe('ImageContextMenu video placement', () => {
  const renderVideoMenu = (actions: ImageActions, video: GalleryItem) =>
    renderItemMenu(actions, { itemRefs: [{ kind: 'video', name: video.name }], items: [video], x: 20, y: 20 });

  it('hands the video to the Initial Video and reference actions', async () => {
    const video = item('video', 'clip.mp4');
    const actions = { ...createActions(vi.fn()), canUseAsReferenceVideo: true };
    await renderVideoMenu(actions, video);

    await interact(() => getMenuItem('widgets.gallery.itemMenu.extendInVideo').click());
    expect(actions.sendToInitialVideo).toHaveBeenCalledExactlyOnceWith(video);

    await interact(() => getMenuItem('widgets.gallery.itemMenu.useAsReferenceVideo').click());
    expect(actions.useAsReferenceVideo).toHaveBeenCalledExactlyOnceWith(video);
  });

  it('disables Use as Reference Video when the Video panel cannot take another', async () => {
    const actions = createActions(vi.fn());
    await renderVideoMenu(actions, item('video', 'clip.mp4'));

    expect(getMenuItem('widgets.gallery.itemMenu.useAsReferenceVideo').getAttribute('aria-disabled')).toBe('true');
    expect(getMenuItem('widgets.gallery.itemMenu.extendInVideo').getAttribute('aria-disabled')).not.toBe('true');
  });

  it('enables Load Workflow once the video is known to embed one and hands the video to the action', async () => {
    const video = item('video', 'made-by-workflow.mp4');
    const actions = createActions(vi.fn());
    actions.hasVideoWorkflow = vi.fn(() => Promise.resolve(true));
    await renderVideoMenu(actions, video);

    await vi.waitFor(() => expect(getMenuItem(LOAD_WORKFLOW).getAttribute('aria-disabled')).not.toBe('true'));
    await interact(() => getMenuItem(LOAD_WORKFLOW).click());

    expect(actions.loadVideoWorkflow).toHaveBeenCalledExactlyOnceWith(video);
  });

  it('aborts the video capability reads when the menu closes before they answer', async () => {
    const actions = createActions(vi.fn());
    const pending = () => new Promise<never>(() => {});
    actions.getVideoRecallCapabilities = vi.fn(pending);
    actions.hasVideoWorkflow = vi.fn(pending);
    await renderVideoMenu(actions, item('video', 'slow.mp4'));
    await vi.waitFor(() => expect(actions.hasVideoWorkflow).toHaveBeenCalled());

    await interact(() => root?.unmount());

    const signals = [
      vi.mocked(actions.getVideoRecallCapabilities).mock.calls[0]?.[1],
      vi.mocked(actions.hasVideoWorkflow).mock.calls[0]?.[1],
    ];
    expect(signals.map((signal) => signal?.aborted)).toEqual([true, true]);
  });

  it('keeps Load Workflow disabled for a video without one', async () => {
    const actions = createActions(vi.fn());
    await renderVideoMenu(actions, item('video', 'plain.mp4'));

    await vi.waitFor(() => expect(actions.hasVideoWorkflow).toHaveBeenCalled());
    expect(getMenuItem(LOAD_WORKFLOW).getAttribute('aria-disabled')).toBe('true');
    expect(actions.loadVideoWorkflow).not.toHaveBeenCalled();
  });

  const openConditioningSubmenu = async (): Promise<void> => {
    await hoverItem(getMenuItem('widgets.gallery.itemMenu.useAsConditioningClip'));
    await settleUntil(
      () =>
        Array.from(document.querySelectorAll('[role="menuitem"]')).some(
          (candidate) => candidate.textContent?.trim() === 'widgets.video.conditioningRoleAudio'
        ),
      'the conditioning clip submenu to open',
      5000
    );
  };

  it('hands the video to the conditioning clip action in the chosen role', async () => {
    const video = item('video', 'clip.mp4');
    const actions = { ...createActions(vi.fn()), canUseAsConditioningClip: { audio: true, video: true } };
    await renderVideoMenu(actions, video);
    await openConditioningSubmenu();

    await interact(() => getMenuItem('widgets.video.conditioningRoleAudio').click());

    expect(actions.useAsConditioningClip).toHaveBeenCalledExactlyOnceWith(video, 'audio');
  });

  it('enables each role from its own capability flag', async () => {
    const video = item('video', 'clip.mp4');
    const actions = { ...createActions(vi.fn()), canUseAsConditioningClip: { audio: false, video: true } };
    await renderVideoMenu(actions, video);
    await openConditioningSubmenu();

    expect(getMenuItem('widgets.video.conditioningRoleAudio').getAttribute('aria-disabled')).toBe('true');
    await interact(() => getMenuItem('widgets.video.conditioningRoleVideo').click());
    expect(actions.useAsConditioningClip).toHaveBeenCalledExactlyOnceWith(video, 'video');
  });

  it('disables the roles the Video panel cannot take, and the picture of a wrapped audio upload', async () => {
    const song = { ...item('video', 'song.mp4'), mediaOrigin: 'audio_upload' };
    const actions = { ...createActions(vi.fn()), canUseAsConditioningClip: { audio: true, video: true } };
    await renderVideoMenu(actions, song);
    await openConditioningSubmenu();

    expect(getMenuItem('widgets.video.conditioningRoleAudio').getAttribute('aria-disabled')).not.toBe('true');
    expect(getMenuItem('widgets.video.conditioningRoleVideo').getAttribute('aria-disabled')).toBe('true');

    await interact(() => root?.unmount());
    host?.remove();
    await renderVideoMenu(createActions(vi.fn()), item('video', 'clip.mp4'));
    await openConditioningSubmenu();

    expect(getMenuItem('widgets.video.conditioningRoleAudio').getAttribute('aria-disabled')).toBe('true');
    expect(getMenuItem('widgets.video.conditioningRoleVideo').getAttribute('aria-disabled')).toBe('true');
  });
});

describe('ImageContextMenu mixed-media action visibility', () => {
  it('shows common actions for a single video and hides every image-only action', async () => {
    const video = item('video', 'clip.mp4');
    await renderItemMenu(createActions(vi.fn()), {
      itemRefs: [{ kind: 'video', name: video.name }],
      items: [video],
      x: 20,
      y: 20,
    });

    expect(document.querySelector('[aria-label="widgets.gallery.itemMenu.openInNewTab"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="widgets.gallery.itemMenu.downloadVideo"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="widgets.gallery.itemMenu.openInPreview"]')).not.toBeNull();
    expect(document.querySelector('[aria-label="widgets.preview.starVideo"]')).not.toBeNull();
    expect(document.body.textContent).toContain('widgets.gallery.changeBoard');
    expect(document.body.textContent).toContain('widgets.gallery.itemMenu.deleteVideo');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.copyToClipboard');
    expect(document.body.textContent).not.toContain('widgets.preview.copyCurrentFrame');
    // Videos carry their own Recall Metadata submenu (parameter recall).
    expect(document.body.textContent).toContain('widgets.gallery.itemMenu.reuseFromVideo');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemActions.upscale.label');
    expect(document.body.textContent).not.toContain('widgets.preview.selectForCompare');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.addToCanvasAs');
    expect(document.body.textContent).not.toContain('widgets.canvas.import.newProjectFromImage');
  });

  it('shows frame copy and Details only when a Preview host opts a single video into them', async () => {
    const video = item('video', 'preview.mp4');
    const onCopyCurrentFrame = vi.fn();
    const onOpenDetails = vi.fn();
    await renderItemMenu(
      createActions(vi.fn()),
      {
        itemRefs: [{ kind: 'video', name: video.name }],
        items: [video],
        x: 20,
        y: 20,
      },
      { isCopyCurrentFrameAvailable: false, itemKey: 'video:preview.mp4', onCopyCurrentFrame, onOpenDetails }
    );

    const copy = getMenuItem('widgets.preview.copyCurrentFrame');
    const details = getMenuItem('widgets.preview.videoDetails');
    expect(copy.getAttribute('aria-disabled')).toBe('true');

    await interact(() => details.click());
    expect(onOpenDetails).toHaveBeenCalledOnce();
    expect(onCopyCurrentFrame).not.toHaveBeenCalled();
  });

  it('hides Preview video extras when the captured menu target is no longer the selected video', async () => {
    const video = item('video', 'stale-menu.mp4');
    await renderItemMenu(
      createActions(vi.fn()),
      {
        itemRefs: [{ kind: 'video', name: video.name }],
        items: [video],
        x: 20,
        y: 20,
      },
      {
        isCopyCurrentFrameAvailable: true,
        itemKey: 'video:new-selection.mp4',
        onCopyCurrentFrame: vi.fn(),
        onOpenDetails: vi.fn(),
      }
    );

    expect(document.body.textContent).not.toContain('widgets.preview.copyCurrentFrame');
    expect(document.body.textContent).not.toContain('widgets.preview.videoDetails');
  });

  it('opens the primary video from a video-only multi-selection and keeps image-only actions hidden', async () => {
    const primaryVideo = item('video', 'primary.mp4');
    const secondaryVideo = item('video', 'secondary.mp4');
    const actions = createActions(vi.fn());
    await renderItemMenu(actions, {
      itemRefs: [
        { kind: 'video', name: primaryVideo.name },
        { kind: 'video', name: secondaryVideo.name },
      ],
      items: [primaryVideo, secondaryVideo],
      x: 20,
      y: 20,
    });

    const target = {
      itemRefs: [
        { kind: 'video' as const, name: primaryVideo.name },
        { kind: 'video' as const, name: secondaryVideo.name },
      ],
      items: [primaryVideo, secondaryVideo],
      x: 20,
      y: 20,
    };
    // Selecting closes the menu, so each quick item gets its own open.
    await pickQuickItem('widgets.gallery.itemMenu.openInNewTab');
    await interact(() => root?.unmount());
    host?.remove();
    await renderItemMenu(actions, target);
    await pickQuickItem('widgets.gallery.itemMenu.openInPreview');

    expect(actions.openItemInNewTab).toHaveBeenCalledWith(primaryVideo);
    expect(actions.openItemInPreview).toHaveBeenCalledWith(primaryVideo);
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.copyToClipboard');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.reuseFromVideo');
    expect(document.body.textContent).not.toContain('widgets.preview.selectForCompare');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.addToCanvasAs');
    expect(document.body.textContent).not.toContain('widgets.canvas.import.newProjectFromImage');
  });

  it("offers Unstar All from the host's star answer for a selection with unloaded members", async () => {
    const loadedImage = { ...item('image', 'starred-still.png'), starred: true };
    const actions = createActions(vi.fn());
    const refs = [
      { kind: 'image' as const, name: loadedImage.name },
      { kind: 'image' as const, name: 'starred-unloaded.png' },
    ];
    await renderItemMenu(actions, { allStarred: true, itemRefs: refs, items: [loadedImage], x: 20, y: 20 });

    await interact(() => getMenuItem('widgets.gallery.unstarSelection').click());
    expect(actions.setItemsStarred).toHaveBeenCalledWith(refs, false);
  });

  it('keeps complete mixed refs for common bulk actions and hides image-only bulk actions when a ref is unresolved', async () => {
    const loadedImage = item('image', 'still.png');
    const actions = createActions(vi.fn());
    const refs = [
      { kind: 'image' as const, name: loadedImage.name },
      { kind: 'video' as const, name: 'unloaded.mp4' },
    ];
    await renderItemMenu(actions, {
      itemRefs: refs,
      items: [loadedImage],
      x: 20,
      y: 20,
    });

    expect(document.body.textContent).toContain('widgets.gallery.itemMenu.itemsSelected');
    expect(document.body.textContent).toContain('widgets.gallery.starSelection');
    expect(document.body.textContent).toContain('widgets.gallery.downloadSelection');
    expect(document.body.textContent).toContain('widgets.gallery.changeBoard');
    expect(document.body.textContent).toContain('widgets.gallery.deleteSelection');
    expect(document.body.textContent).not.toContain('widgets.gallery.itemMenu.addToCanvasAs');
    expect(document.body.textContent).not.toContain('widgets.canvas.import.newProjectFromImage');
    await pickQuickItem('widgets.gallery.itemMenu.openInNewTab');
    await interact(() => root?.unmount());
    host?.remove();
    await renderItemMenu(actions, { itemRefs: refs, items: [loadedImage], x: 20, y: 20 });
    await pickQuickItem('widgets.gallery.itemMenu.openInPreview');
    expect(actions.openItemInNewTab).toHaveBeenCalledWith(loadedImage);
    expect(actions.openItemInPreview).toHaveBeenCalledWith(loadedImage);

    await interact(() => getMenuItem('widgets.gallery.starSelection').click());
    expect(actions.setItemsStarred).toHaveBeenCalledWith(refs, true);
  });
});

describe('ImageContextMenu change-board destinations', () => {
  const board = (id: string, access: NonNullable<GalleryBoard['access']>): GalleryBoard => ({
    access,
    archived: false,
    assetCount: 0,
    assetVideoCount: 0,
    id,
    imageCount: 1,
    isInbox: false,
    kind: 'board',
    name: id,
    ownerName: access.role === 'owner' ? 'Me' : 'Dee',
    projectId: null,
    videoCount: 0,
  });
  const renderWithBoards = async (boards: GalleryBoard[], listedValue = 'board-theirs', target = image('one.png')) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await interact(() => {
      root?.render(
        <ChakraProvider value={system}>
          <ImageContextMenu
            actions={createActions(vi.fn())}
            boards={boards}
            target={{ images: [target], x: 20, y: 20 }}
            onClose={vi.fn()}
          />
        </ChakraProvider>
      );
    });
    const trigger = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((candidate) =>
      candidate.textContent?.includes('widgets.gallery.changeBoard')
    );
    expect(trigger).not.toBeUndefined();
    await hoverItem(trigger!);
    await settleUntil(
      () => document.querySelector(`[data-value="${listedValue}"]`) !== null,
      'the change-board submenu to list its boards'
    );

    return Array.from(
      document
        .querySelector(`[data-value="${listedValue}"]`)!
        .closest('[role="menu"]')!
        .querySelectorAll<HTMLElement>('[data-scope="menu"][data-part="item-group-label"]')
    ).map((label) => label.textContent?.trim());
  };

  it("groups the boards as the host lists them: another user's board apart, the caller's own under Library", async () => {
    expect(
      await renderWithBoards([
        board('mine', {
          actions: ['read', 'add_media'],
          adminOverride: false,
          generalAccess: 'restricted',
          role: 'owner',
        }),
        // An admin's view of someone else's board; a host without board sharing folds it before handing it over.
        board('theirs', {
          actions: ['read', 'add_media'],
          adminOverride: true,
          generalAccess: 'restricted',
          role: null,
        }),
      ])
    ).toEqual(['widgets.gallery.boardGroups.library', 'widgets.gallery.boardGroups.otherUsers']);
    expect(document.querySelector('[data-value="board-mine"]')).not.toBeNull();
  });

  it("offers what the selection bar offers: no archived board, not the item's own, and Uncategorized as a move", async () => {
    const owned = {
      actions: ['read', 'add_media'],
      adminOverride: false,
      generalAccess: 'restricted',
      role: 'owner',
    } as const;
    await renderWithBoards(
      [board('here', owned), { ...board('old', owned), archived: true }, board('there', owned)],
      'board-there',
      { ...image('one.png'), boardId: 'here' }
    );

    expect(document.querySelector('[data-value="board-here"]')).toBeNull();
    expect(document.querySelector('[data-value="board-old"]')).toBeNull();
    expect(document.querySelector('[data-value="remove-from-board"]')).not.toBeNull();
  });

  it('offers no move to Uncategorized for an item already there', async () => {
    const owned = {
      actions: ['read', 'add_media'],
      adminOverride: false,
      generalAccess: 'restricted',
      role: 'owner',
    } as const;
    await renderWithBoards([board('there', owned)], 'board-there');

    expect(document.querySelector('[data-value="remove-from-board"]')).toBeNull();
  });

  it('grows a long board list past the old 18rem cap, keeps it on screen, and scrolls to the last board by keyboard', async () => {
    const owned = {
      actions: ['read', 'add_media'],
      adminOverride: false,
      generalAccess: 'restricted',
      role: 'owner',
    } as const;
    const boards = Array.from({ length: 30 }, (_, index) => board(`b${String(index + 1).padStart(2, '0')}`, owned));
    await renderWithBoards(boards, 'board-b30');

    const last = document.querySelector<HTMLElement>('[data-value="board-b30"]')!;
    const content = last.closest<HTMLElement>('[data-scope="menu"][data-part="content"]')!;
    const scroller = last.closest<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
    const contentRect = content.getBoundingClientRect();
    expect(contentRect.height).toBeGreaterThan(18 * 16);
    expect(contentRect.top).toBeGreaterThanOrEqual(0);
    expect(contentRect.bottom).toBeLessThanOrEqual(window.innerHeight);
    // Thirty boards overflow even the taller list, so the Scrollable carries the rest.
    expect(scroller.scrollHeight).toBeGreaterThan(scroller.clientHeight);
    expect(last.getBoundingClientRect().top).toBeGreaterThanOrEqual(scroller.getBoundingClientRect().bottom);

    await interact(() => content.focus());
    await userEvent.keyboard('{End}');
    await settleUntil(() => last.hasAttribute('data-highlighted'), 'the last board to be highlighted');
    await settleUntil(() => {
      const itemRect = last.getBoundingClientRect();
      const viewRect = scroller.getBoundingClientRect();
      return itemRect.top >= viewRect.top && itemRect.bottom <= viewRect.bottom;
    }, 'the last board to scroll fully into view');
  });

  it('leaves the list where it is when the pointer highlights a row cut off at its edge', async () => {
    const owned = {
      actions: ['read', 'add_media'],
      adminOverride: false,
      generalAccess: 'restricted',
      role: 'owner',
    } as const;
    await renderWithBoards(
      Array.from({ length: 30 }, (_, index) => board(`b${String(index + 1).padStart(2, '0')}`, owned)),
      'board-b30'
    );
    const scroller = document
      .querySelector('[data-value="board-b30"]')!
      .closest<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
    // Leave the first board half scrolled away above the list's top edge.
    const first = document.querySelector<HTMLElement>('[data-value="board-b01"]')!;
    await interact(() => {
      scroller.scrollTop = first.getBoundingClientRect().top - scroller.getBoundingClientRect().top + 8;
    });
    const scrollTop = scroller.scrollTop;
    const firstRect = first.getBoundingClientRect();
    const viewTop = scroller.getBoundingClientRect().top;
    expect(firstRect.top).toBeLessThan(viewTop);
    expect(firstRect.bottom).toBeGreaterThan(viewTop + 4);

    // A real pointer on the part still showing. It targets the list, which is fully in view: Playwright would scroll a
    // cut-off target into view itself before hovering it.
    await userEvent.hover(scroller, { position: { x: 8, y: firstRect.bottom - viewTop - 3 } });
    await settleUntil(() => first.hasAttribute('data-highlighted'), 'the cut-off board to be highlighted');
    await interact(() => undefined);

    expect(scroller.scrollTop).toBe(scrollTop);
  });
});

describe('ImageContextMenu Upscale submenu', () => {
  const upscaler = (key: string, name: string, scale: number | null) => ({
    base: 'any' as const,
    file_size: 1,
    format: 'checkpoint' as const,
    hash: `${key}-hash`,
    key,
    name,
    path: key,
    scale,
    source: key,
    source_type: 'path' as const,
    type: 'spandrel_image_to_image' as const,
  });
  const SEND_TO = 'widgets.gallery.itemActions.upscale.sendTo';
  const MODEL_ONLY = 'widgets.gallery.itemActions.upscale.modelOnlyGroup';
  const NO_UPSCALERS = 'widgets.gallery.itemActions.upscale.noUpscalers';
  const NATIVE_SCALE_UNKNOWN = 'widgets.upscale.nativeScaleUnknown';
  const NATIVE_ENTRY = 'widgets.gallery.itemActions.upscale.nativeScaleUnknownFactor';
  const OTHER_FACTORS = 'widgets.gallery.itemActions.upscale.otherFactors';

  const openUpscaleSubMenu = async (
    models: ReturnType<typeof upscaler>[],
    status: ModelsSnapshot['status'] = 'loaded'
  ) => {
    setModelsSnapshotForTests({ models, status });
    await renderMenu(createActions(vi.fn()), [image('source.png')]);
    await hoverItem(getMenuItem('widgets.gallery.itemActions.upscale.label'));
    await settleUntil(() => document.querySelector('[data-value="send-to-upscale"]') !== null, 'the Upscale submenu');

    return document
      .querySelector<HTMLElement>('[data-value="send-to-upscale"]')!
      .closest<HTMLElement>('[role="menu"]')!;
  };
  /** Opens one upscaler's factor submenu; `scaleSuffix` is the native scale its trigger shows. */
  const openFactorSubMenu = async (model: ReturnType<typeof upscaler>, scaleSuffix: string) => {
    await openUpscaleSubMenu([model]);
    await hoverItem(getMenuItem(`${model.name}${scaleSuffix}`));
    await settleUntil(
      () => document.querySelector(`[data-value^="upscale-model-only-${model.key}-"]`) !== null,
      `the factors of ${model.name}`
    );

    return document
      .querySelector<HTMLElement>(`[data-value^="upscale-model-only-${model.key}-"]`)!
      .closest<HTMLElement>('[role="menu"]')!;
  };
  const entries = (menu: HTMLElement) =>
    Array.from(menu.querySelectorAll<HTMLElement>('[role="menuitem"], [data-part="item-group-label"]')).map(
      (entry) => `${entry.textContent?.trim()}${entry.getAttribute('aria-disabled') === 'true' ? ' (disabled)' : ''}`
    );

  afterEach(() => {
    setModelsSnapshotForTests({ models: [], status: 'idle' });
    ensureModelsLoaded.mockClear();
    commandsMock.generation.submitModelOnlyUpscale.mockReset();
    commandsMock.notifications.add.mockReset();
    galleryMock.metadata.mockReset();
  });

  it('lists every installed upscaler by name, with its native scale where it is recorded', async () => {
    const menu = await openUpscaleSubMenu([
      upscaler('ultra', '4x-UltraSharp', 4),
      upscaler('anime', '2x-AnimeSharp', 2),
      upscaler('legacy', 'Legacy restorer', null),
    ]);

    expect(entries(menu)).toEqual([SEND_TO, MODEL_ONLY, '2x-AnimeSharp2×', '4x-UltraSharp4×', 'Legacy restorer']);
  });

  it('still runs an upscaler of unknown native scale once at it, and says how to unlock the other factors', async () => {
    const legacy = upscaler('legacy', 'Legacy restorer', null);
    const factorMenu = await openFactorSubMenu(legacy, '');
    const otherFactors = factorMenu.querySelector<HTMLElement>('[data-value="upscale-model-only-legacy-other"]')!;

    expect(entries(factorMenu)).toEqual([NATIVE_ENTRY, `${OTHER_FACTORS}${NATIVE_SCALE_UNKNOWN} (disabled)`]);
    expect(document.getElementById(otherFactors.getAttribute('aria-describedby')!)?.textContent).toBe(
      NATIVE_SCALE_UNKNOWN
    );

    await interact(() => getMenuItem(NATIVE_ENTRY).click());
    await settleUntil(
      () => commandsMock.generation.submitModelOnlyUpscale.mock.calls.length > 0,
      'the upscale to queue'
    );

    expect(commandsMock.generation.submitModelOnlyUpscale).toHaveBeenCalledTimes(1);
    expect(commandsMock.generation.submitModelOnlyUpscale).toHaveBeenCalledWith(
      expect.objectContaining({ scale: 'native', upscaleModel: legacy })
    );
  });

  it.each([
    [4, ['1.5×', '2×', '4× (native)']],
    // A 2x model's native scale is one of the presets: it is offered once, as native.
    [2, ['1.5×', '2× (native)']],
    // A 1x model enlarges nothing, so it offers its own scale alone.
    [1, ['1× (native)']],
  ])('offers a %sx upscaler its factors, smallest first', async (scale, factors) => {
    const factorMenu = await openFactorSubMenu(upscaler('model', 'Model', scale), `${scale}×`);

    expect(entries(factorMenu)).toEqual(factors);
  });
  it.each(THEMES.map(({ id }) => id))('shows the scale suffix at AA contrast on the %s menu', async (themeId) => {
    applyThemeToRoot(themeId);

    try {
      const menu = await openUpscaleSubMenu([upscaler('ultra', '4x-UltraSharp', 4)]);
      const suffix = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"] *')].find(
        (element) => element.childElementCount === 0 && element.textContent === '4×'
      )!;

      expect(
        getContrastRatio(getComputedStyle(suffix).color, getComputedStyle(menu).backgroundColor, 1)
      ).toBeGreaterThanOrEqual(4.5);
    } finally {
      applyThemeToRoot(DEFAULT_THEME_ID);
    }
  });

  it.each([
    ['none are installed', 'loaded', NO_UPSCALERS],
    ['the library is loading', 'loading', 'models.loadingModels'],
    ['the library has not been asked for yet', 'idle', 'models.loadingModels'],
    ['the library could not load', 'error', 'models.couldNotLoad'],
  ] as const)(
    'offers Send to Upscale and one unavailable model-only entry that says why when %s',
    async (_case, status, reason) => {
      const menu = await openUpscaleSubMenu([], status);
      const modelOnly = menu.querySelector<HTMLElement>('[data-value="upscale-model-only"]')!;

      expect(ensureModelsLoaded).toHaveBeenCalled();
      expect(entries(menu)).toEqual([SEND_TO, `${MODEL_ONLY}${reason} (disabled)`]);
      expect(document.getElementById(modelOnly.getAttribute('aria-describedby')!)?.textContent).toBe(reason);
    }
  );

  it.each([
    ['its native scale', '4× (native)', 'native'],
    ['another factor', '1.5×', 1.5],
  ])("queues the chosen upscaler at %s on the image with the image's metadata", async (_case, label, scale) => {
    const ultra = upscaler('ultra', '4x-UltraSharp', 4);
    galleryMock.metadata.mockResolvedValue({ positive_prompt: 'a lighthouse' });
    await openFactorSubMenu(ultra, '4×');

    await interact(() => getMenuItem(label).click());
    await settleUntil(
      () => commandsMock.generation.submitModelOnlyUpscale.mock.calls.length > 0,
      'the upscale to queue'
    );

    expect(galleryMock.metadata).toHaveBeenCalledWith('source.png', expect.any(AbortSignal));
    expect(commandsMock.generation.submitModelOnlyUpscale).toHaveBeenCalledWith({
      backendSupportsCancellation: true,
      image: { height: 512, image_name: 'source.png', width: 512 },
      projectId: 'project-1',
      scale,
      sourceMetadata: { positive_prompt: 'a lighthouse' },
      upscaleModel: ultra,
    });
  });

  it('says the upscale could not start when its code fails to load', async () => {
    modelOnlyLoad.fail = true;

    try {
      await openFactorSubMenu(upscaler('ultra', '4x-UltraSharp', 4), '4×');
      await interact(() => getMenuItem('4× (native)').click());
      await settleUntil(() => commandsMock.notifications.add.mock.calls.length > 0, 'the load failure notice');
    } finally {
      modelOnlyLoad.fail = false;
    }

    expect(commandsMock.notifications.add).toHaveBeenCalledWith({
      kind: 'error',
      title: 'widgets.gallery.itemActions.upscale.loadFailed',
    });
    expect(commandsMock.generation.submitModelOnlyUpscale).not.toHaveBeenCalled();
  });

  it('queues nothing and names the image when its metadata cannot be read', async () => {
    galleryMock.metadata.mockRejectedValue(new Error('offline'));
    await openFactorSubMenu(upscaler('ultra', '4x-UltraSharp', 4), '4×');

    await interact(() => getMenuItem('4× (native)').click());
    await settleUntil(() => commandsMock.notifications.add.mock.calls.length > 0, 'the failure notice');

    expect(commandsMock.generation.submitModelOnlyUpscale).not.toHaveBeenCalled();
    expect(commandsMock.notifications.add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', title: 'widgets.gallery.itemActions.upscale.metadataFailed' })
    );
  });
});

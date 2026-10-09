/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';
import type { GalleryUiAdapter } from '@features/gallery/react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { DEFAULT_GALLERY_SETTINGS } from '@features/gallery/core/settings';
import { GalleryUiProvider } from '@features/gallery/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GALLERY_PINNED_FOOTER_PX, GALLERY_STARRED_HEADER_HEIGHT_PX } from './galleryGridLayout';
import { GalleryStackedLayout } from './GalleryStackedLayout';
import { GalleryWideLayout } from './GalleryWideLayout';
import { GalleryWidgetContext } from './GalleryWidgetContext';
import { EMPTY_GALLERY_STARRED_STRIP } from './useGalleryStarredStrip';

vi.mock('@features/queue/react', () => ({
  useItemProgress: () => null,
  useQueueItemProgressImage: () => null,
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) =>
      key === 'widgets.gallery.selectionCount' ? `${String(values?.count)} selected` : key,
  }),
}));

const board: GalleryBoard = {
  archived: false,
  assetCount: 3,
  assetVideoCount: 0,
  id: 'dogs',
  imageCount: 50,
  kind: 'board',
  name: 'dogs',
  projectId: null,
  videoCount: 0,
};

const createItem = (name: string) => ({
  boardId: 'dogs',
  category: 'general' as const,
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: `/full/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'image' as const,
  name,
  starred: false,
  thumbnailUrl: `data:image/svg+xml,${encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#547c83"/></svg>'
  )}`,
  width: 64,
});

const createGallery = (overrides: Partial<GalleryStateView> = {}) =>
  ({
    boards: [board],
    compareImageKey: null,
    galleryView: 'images',
    items: [createItem('a.png'), createItem('b.png')],
    pendingPlaceholders: [],
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    selectedItemKey: 'image:a.png',
    selectedItemKeys: ['image:a.png', 'image:b.png'],
    semanticImageQuery: null,
    settings: DEFAULT_GALLERY_SETTINGS,
    starredOnly: false,
    ...overrides,
  }) as unknown as GalleryStateView;

const gallery = createGallery();

let activeGallery: GalleryStateView = gallery;

const setGallery = (next: GalleryStateView) => {
  activeGallery = next;
};

const createContextValue = () =>
  ({
    ...contextBase,
    gallery: activeGallery,
    loadedItems: activeGallery.items,
    region: 'center',
  }) as unknown as GalleryWidgetContextValue;

const READY = {
  error: null,
  isFetchingMore: false,
  isRetrying: false,
  retry: () => Promise.resolve(),
  status: 'ready',
} as const;

const contextBase = {
  actions: {
    createBoard: vi.fn(),
    loadMore: vi.fn(),
    selectBoard: vi.fn(),
    selectProjectBoard: vi.fn(),
    setSearchTerm: vi.fn(),
    setStarredOnly: vi.fn(),
    setView: vi.fn(),
    updateSettings: vi.fn(),
    uploadFiles: vi.fn(),
  },
  boardsState: READY,
  filter: { boardId: 'dogs', galleryView: 'images', searchTerm: '' },
  gallery,
  isWindowTruncated: false,
  listing: READY,
  starredStrip: EMPTY_GALLERY_STARRED_STRIP,
  itemActions: {
    deleteItems: vi.fn(),
    downloadItems: vi.fn(),
    moveItemsToBoard: vi.fn(),
    setItemsStarred: vi.fn(),
  },
  projectName: 'Project',
  runtime: {
    commands: { register: () => () => undefined },
    hotkeys: { register: () => () => undefined },
  },
};

const adapter = {
  ImageContextMenu: () => null,
  progressSessions: [],
  pinnedProgressSessionId: null,
  followedProgressSessionId: null,
  liveFollowEnabled: false,
  getItemLabel: () => Promise.resolve(null),
  followProgressSession: vi.fn(),
  antialiasProgressImages: false,
  galleryValues: {},
  widgets: { openGallery: vi.fn(() => true), patchGalleryValues: vi.fn() },
} as unknown as GalleryUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderLayout = async (Layout: typeof GalleryStackedLayout | typeof GalleryWideLayout) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <GalleryUiProvider adapter={adapter}>
            <GalleryWidgetContext value={createContextValue()}>
              <DndContext>
                <Layout />
              </DndContext>
            </GalleryWidgetContext>
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

beforeEach(() => {
  vi.clearAllMocks();
  setGallery(gallery);
  adapter.progressSessions = [];
  host = document.createElement('div');
  host.style.cssText = 'height:600px;left:0;position:fixed;top:0;width:900px;';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  queryClient = null;
});

describe('gallery layout shells', () => {
  it('wraps progress into matching gallery columns inside the same vertical viewport', async () => {
    host!.style.width = '560px';
    adapter.progressSessions = Array.from({ length: 4 }, (_, index) => ({
      id: `batch:${index + 1}`,
      queueItemId: 'batch',
      backendItemId: index + 1,
      itemIndex: index + 1,
      itemCount: 4,
      label: 'Batch',
      sourceId: 'workflow',
      width: 512,
      height: 512,
      state: 'queued',
    }));
    setGallery(
      createGallery({ settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelCollapsed: true, imageDensityPercent: 0 } })
    );
    await renderLayout(GalleryStackedLayout);
    const panel = host!.querySelector<HTMLElement>('[role="tabpanel"]')!;
    await expect.poll(() => panel.querySelectorAll('[role="region"] button[aria-pressed]').length).toBe(4);
    const tiles = panel.querySelectorAll<HTMLElement>('[role="region"] button[aria-pressed]');
    const saved = panel.querySelector<HTMLElement>('[role="listitem"]')!;
    const viewport = panel.querySelector<HTMLElement>('[data-part="viewport"]')!;
    expect(panel.querySelectorAll('[data-part="viewport"]')).toHaveLength(1);
    expect(viewport.contains(tiles[0]!)).toBe(true);
    expect(tiles[0]!.getBoundingClientRect().width).toBeCloseTo(saved.getBoundingClientRect().width, 0);
    expect(tiles[3]!.getBoundingClientRect().top).toBeGreaterThan(tiles[0]!.getBoundingClientRect().top);
    expect(saved.getBoundingClientRect().top).toBeGreaterThanOrEqual(tiles[3]!.getBoundingClientRect().bottom);
    expect(viewport.scrollWidth).toBe(viewport.clientWidth);
    setGallery(
      createGallery({
        settings: {
          ...DEFAULT_GALLERY_SETTINGS,
          boardPanelCollapsed: true,
          imageDensityPercent: 0,
          progressSectionCollapsed: true,
        },
      })
    );
    await renderLayout(GalleryStackedLayout);
    expect(panel.querySelector('[role="region"] button[aria-pressed]')).toBeNull();
    // The collapsed disclosure row, then the pinned block's rule and margin.
    expect(saved.getBoundingClientRect().top - viewport.getBoundingClientRect().top).toBeCloseTo(
      GALLERY_STARRED_HEADER_HEIGHT_PX + GALLERY_PINNED_FOOTER_PX,
      0
    );
  });

  it.each(['images', 'assets'] as const)(
    'fills the stacked %s panel with media or its empty upload picker',
    async (galleryView) => {
      await page.viewport(760, 680);
      host!.style.width = '560px';
      for (const items of [[createItem('visible.png')], []]) {
        setGallery(
          createGallery({ galleryView, items, settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelCollapsed: true } })
        );
        await renderLayout(GalleryStackedLayout);
        const panel = host!.querySelector<HTMLElement>('[role="tabpanel"]')!;
        const grid = panel.firstElementChild as HTMLElement;
        expect(grid.getBoundingClientRect().width).toBeCloseTo(panel.getBoundingClientRect().width, 0);
        if (items.length) {
          await expect.poll(() => panel.querySelector('[role="listitem"]')).not.toBeNull();
          const viewport = panel.querySelector<HTMLElement>('[data-part="viewport"]')!;
          expect(viewport.clientWidth).toBeGreaterThan(500);
          expect(viewport.clientHeight).toBeGreaterThan(100);
        } else {
          const picker = panel.querySelector<HTMLElement>('[role="button"]')!;
          expect(picker.getBoundingClientRect().width).toBeGreaterThan(panel.clientWidth - 24);
        }
        await page.screenshot({
          path: `../../../../artifacts/gallery-progress/stacked-${galleryView}-${items.length ? 'media' : 'empty'}.png`,
        });
      }
    }
  );

  it('keeps the center gallery scroll area above the selection actions', async () => {
    const items = Array.from({ length: 80 }, (_, index) => createItem(`image-${index}.png`));

    setGallery(createGallery({ items }));
    await renderLayout(GalleryWideLayout);

    const list = host?.querySelector<HTMLElement>('[role="list"]');
    const viewport = list?.closest<HTMLElement>('[data-part="viewport"]');
    const selectionActions = host?.querySelector<HTMLElement>('[role="toolbar"]');

    if (!viewport || !selectionActions) {
      throw new Error('gallery viewport or selection actions did not render');
    }

    expect(viewport.getBoundingClientRect().bottom).toBeLessThanOrEqual(selectionActions.getBoundingClientRect().top);
  });

  it('keeps a long board list inside the board panel instead of spilling it over the grid', async () => {
    const boards = Array.from({ length: 40 }, (_, index) => ({
      ...board,
      id: `board-${index}`,
      name: `Board ${index}`,
    }));

    setGallery(createGallery({ boards, selectedBoardId: 'board-0' }));
    await renderLayout(GalleryStackedLayout);

    const viewport = host?.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]');

    if (!viewport) {
      throw new Error('board panel viewport did not render');
    }

    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    expect(viewport.getBoundingClientRect().height).toBeLessThanOrEqual(DEFAULT_GALLERY_SETTINGS.boardPanelHeightPx);

    const grid = host?.querySelector('[role="list"]');
    const panelBottom = viewport.getBoundingClientRect().bottom;

    expect(grid?.getBoundingClientRect().top).toBeGreaterThanOrEqual(panelBottom);
  });

  it('clamps a persisted tall stacked panel to preserve the grid, then restores it without rewriting settings', async () => {
    host!.style.height = '480px';
    setGallery(createGallery({ settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelHeightPx: 600 } }));
    await renderLayout(GalleryStackedLayout);

    const separator = host?.querySelector<HTMLElement>('[role="separator"]');
    const gridWrapper = host?.querySelector<HTMLElement>('[data-gallery-grid-wrapper]');

    await vi.waitFor(() => expect(Number(separator?.getAttribute('aria-valuemax'))).toBeLessThan(600));
    const shortMaximum = Number(separator?.getAttribute('aria-valuemax'));

    expect(Number(separator?.getAttribute('aria-valuenow'))).toBe(shortMaximum);
    expect(gridWrapper?.getBoundingClientRect().height).toBeGreaterThanOrEqual(128);
    expect(contextBase.actions.updateSettings).not.toHaveBeenCalled();

    host!.style.height = '1000px';

    await vi.waitFor(() => expect(separator?.getAttribute('aria-valuenow')).toBe('600'));
    expect(contextBase.actions.updateSettings).not.toHaveBeenCalled();
  });

  it('bounds stacked pointer and keyboard resizing by the measured maximum', async () => {
    host!.style.height = '480px';
    setGallery(createGallery({ settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelHeightPx: 150 } }));
    await renderLayout(GalleryStackedLayout);

    const separator = host?.querySelector<HTMLElement>('[role="separator"]');

    await vi.waitFor(() => expect(Number(separator?.getAttribute('aria-valuemax'))).toBeLessThan(600));
    const measuredMaximum = Number(separator?.getAttribute('aria-valuemax'));

    await act(() => separator?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));
    expect(contextBase.actions.updateSettings).toHaveBeenLastCalledWith({ boardPanelHeightPx: measuredMaximum });

    contextBase.actions.updateSettings.mockClear();
    const pointer = (type: string, clientY: number, buttons: number) =>
      new PointerEvent(type, { bubbles: true, buttons, clientY, pointerId: 1 });
    await act(() => separator?.dispatchEvent(pointer('pointerdown', 0, 1)));
    await act(() => window.dispatchEvent(pointer('pointermove', 5_000, 1)));
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );
    await act(() => window.dispatchEvent(pointer('pointerup', 5_000, 0)));

    expect(contextBase.actions.updateSettings).toHaveBeenLastCalledWith({ boardPanelHeightPx: measuredMaximum });
  });

  it('puts the upload control in the toolbar row for both shells, disabled for date boards', async () => {
    for (const Layout of [GalleryStackedLayout, GalleryWideLayout]) {
      setGallery(gallery);
      await renderLayout(Layout);

      const upload = host?.querySelector<HTMLButtonElement>('button[aria-label^="widgets.gallery.upload"]');

      expect(upload, 'upload control did not render in the toolbar').not.toBeNull();
      expect(upload?.disabled).toBe(false);

      setGallery(createGallery({ selectedBoardId: 'by_date:2026-07-30' }));
      await renderLayout(Layout);

      const uploadForDateBoard = host?.querySelector<HTMLButtonElement>('button[aria-label^="widgets.gallery.upload"]');

      expect(uploadForDateBoard?.disabled).toBe(true);
    }
  });

  it('offers the starred-only filter in both shells, pressed while active and disabled under a ranked query', async () => {
    for (const Layout of [GalleryStackedLayout, GalleryWideLayout]) {
      setGallery(createGallery({ starredOnly: false }));
      await renderLayout(Layout);

      const toggle = host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.starredOnly"]');

      expect(toggle, 'starred filter toggle did not render in the toolbar').not.toBeNull();
      expect(toggle?.getAttribute('aria-pressed')).toBe('false');

      await act(() => {
        toggle?.click();
      });
      expect(contextBase.actions.setStarredOnly).toHaveBeenLastCalledWith(true);

      setGallery(createGallery({ starredOnly: true }));
      await renderLayout(Layout);
      expect(
        host
          ?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.starredOnly"]')
          ?.getAttribute('aria-pressed')
      ).toBe('true');

      setGallery(createGallery({ semanticImageQuery: { imageName: 'ref.png', kind: 'image' }, starredOnly: true }));
      await renderLayout(Layout);

      const ranked = host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.starredOnly"]');

      // Inert but focusable, so the tooltip can say why it does not apply.
      expect(ranked?.getAttribute('aria-disabled')).toBe('true');
      expect(ranked?.disabled).toBe(false);
      expect(ranked?.getAttribute('aria-pressed')).toBe('false');
      (contextBase.actions.setStarredOnly as ReturnType<typeof vi.fn>).mockClear();
      await act(() => {
        ranked?.click();
      });
      expect(contextBase.actions.setStarredOnly).not.toHaveBeenCalled();
    }
  });

  it('moves focus to the toolbar toggle when Show all removes the strip header', async () => {
    const starredItem = { ...createItem('starred.png'), starred: true };

    contextBase.starredStrip = { items: [starredItem], state: READY, total: 9 };
    setGallery(createGallery({ items: [createItem('a.png')], starredOnly: false }));
    await renderLayout(GalleryWideLayout);

    const showAll = host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.showAllStarredItems"]');

    expect(showAll, 'Show all did not render').not.toBeNull();
    showAll?.focus();
    await act(() => {
      showAll?.click();
    });

    expect(contextBase.actions.setStarredOnly).toHaveBeenLastCalledWith(true);
    expect(document.activeElement?.getAttribute('aria-label')).toBe('widgets.gallery.starredOnly');
    contextBase.starredStrip = EMPTY_GALLERY_STARRED_STRIP;
  });

  it('hides the board column and its handle in both shells when collapsed', async () => {
    setGallery(createGallery({ settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelCollapsed: true } }));

    for (const Layout of [GalleryStackedLayout, GalleryWideLayout]) {
      await renderLayout(Layout);

      expect(host?.querySelector('input[aria-label="widgets.gallery.searchOrCreateBoards"]')).toBeNull();
      expect(host?.querySelector('[role="separator"]')).toBeNull();
      expect(host?.querySelector('[role="list"]'), 'the grid still renders').not.toBeNull();
    }
  });
});

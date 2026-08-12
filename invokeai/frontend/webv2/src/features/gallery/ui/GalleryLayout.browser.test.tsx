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

import type { GalleryStateView } from './galleryStateView';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryStackedLayout } from './GalleryStackedLayout';
import { GalleryWideLayout } from './GalleryWideLayout';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('@features/queue/react', () => ({
  useQueueItemProgress: () => null,
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
  thumbnailUrl: `/thumb/${name}`,
  width: 64,
});

const createGallery = (overrides: Partial<GalleryStateView> = {}) =>
  ({
    boards: [board],
    compareImageKey: null,
    currentItem: null,
    galleryView: 'images',
    isLoading: false,
    items: [createItem('a.png'), createItem('b.png')],
    pendingPlaceholders: [],
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    selectedItemKey: 'image:a.png',
    selectedItemKeys: ['image:a.png', 'image:b.png'],
    settings: DEFAULT_GALLERY_SETTINGS,
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
    region: 'center',
  }) as unknown as GalleryWidgetContextValue;

const contextBase = {
  actions: {
    createBoard: vi.fn(),
    loadMore: vi.fn(),
    selectBoard: vi.fn(),
    selectProjectBoard: vi.fn(),
    setSearchTerm: vi.fn(),
    setView: vi.fn(),
    updateSettings: vi.fn(),
    uploadFiles: vi.fn(),
  },
  filter: { boardId: 'dogs', galleryView: 'images', searchTerm: '' },
  gallery,
  isWindowTruncated: false,
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
  account: { enableLiveFollow: vi.fn() },
  antialiasProgressImages: false,
  widgets: { patchGalleryValues: vi.fn() },
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
    setGallery(createGallery({ settings: { ...DEFAULT_GALLERY_SETTINGS, boardPanelHeightPx: 600 } }));
    await renderLayout(GalleryStackedLayout);

    const separator = host?.querySelector<HTMLElement>('[role="separator"]');

    await vi.waitFor(() => expect(Number(separator?.getAttribute('aria-valuemax'))).toBeLessThan(600));
    const measuredMaximum = Number(separator?.getAttribute('aria-valuemax'));

    await act(() => separator?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));
    expect(contextBase.actions.updateSettings).toHaveBeenLastCalledWith({ boardPanelHeightPx: measuredMaximum });

    contextBase.actions.updateSettings.mockClear();
    await act(() => {
      separator?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientY: 0 }));
      window.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientY: 5_000 }));
      window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientY: 5_000 }));
    });

    expect(contextBase.actions.updateSettings).toHaveBeenLastCalledWith({ boardPanelHeightPx: measuredMaximum });
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

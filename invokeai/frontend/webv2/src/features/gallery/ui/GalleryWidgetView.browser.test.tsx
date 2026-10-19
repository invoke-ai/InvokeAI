import type { GalleryImageItem } from '@features/gallery/core/items';

import { ChakraProvider } from '@chakra-ui/react';
import { requestGalleryItemReveal } from '@features/gallery/core/selection';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, type ComponentProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryUiAdapter } from './GalleryUiContext';

import { GalleryItemActionsProvider, GalleryUiProvider } from './GalleryUiContext';
import { GalleryStatusChip, GalleryWidgetView } from './GalleryWidgetView';

const mocks = vi.hoisted(() => ({
  boardSharing: undefined as boolean | undefined,
  galleryData: null as unknown,
  getItemActionContext: null as null | ComponentProps<GalleryUiAdapter['ItemActionsProvider']>['getItemActionContext'],
}));

vi.mock('./GalleryBoardDragMonitor', () => ({ GalleryBoardDragMonitor: () => null }));
vi.mock('./GalleryLayout', async () => {
  const { useGalleryWidget } = await import('./GalleryWidgetContext');
  return {
    // What the board panel and menus will read: the context as the widget publishes it.
    GalleryLayout: () => {
      mocks.boardSharing = useGalleryWidget().boardSharing;
      return null;
    },
  };
});
vi.mock('./useGalleryActions', () => ({ useGalleryActions: () => ({}) }));
vi.mock('./useGalleryData', () => ({ useGalleryData: () => mocks.galleryData }));
vi.mock('./useGalleryStarredStrip', () => ({
  useGalleryStarredStrip: () => ({
    items: [],
    state: { error: null, isFetchingMore: false, isRetrying: false, retry: () => Promise.resolve(), status: 'ready' },
    total: 0,
  }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      key === 'widgets.gallery.statusChip' ? `Gallery: ${String(values?.count)} items` : key,
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const createImageItem = (name: string): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-10-05T00:00:00.000Z',
  fullUrl: `/api/v1/images/i/${name}/full`,
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name,
  starred: false,
  thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
  width: 512,
});

const itemActions = {
  deleteItems: vi.fn(),
  downloadItem: vi.fn(),
  downloadItems: vi.fn(),
  moveItemsToBoard: vi.fn(),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  setItemsStarred: vi.fn(),
};
const noop = vi.fn();

const CapturingItemActionsProvider = ({
  children,
  getItemActionContext,
}: ComponentProps<GalleryUiAdapter['ItemActionsProvider']>) => {
  // eslint-disable-next-line react/immutability -- The browser test reads the props supplied to this provider.
  mocks.getItemActionContext = getItemActionContext ?? null;
  return <GalleryItemActionsProvider actions={itemActions}>{children}</GalleryItemActionsProvider>;
};

const adapter: GalleryUiAdapter = {
  ItemActionsProvider: CapturingItemActionsProvider,
  ImageContextMenu: () => null,
  antialiasProgressImages: false,
  exportProject: noop,
  followedProgressSessionId: null,
  followProgressSession: noop,
  gallery: {
    clearSelection: noop,
    clearSearch: noop,
    commitSemanticSearch: noop,
    reconcileDeletedBoardOutcome: noop,
    selectBoard: noop,
    selectImage: noop,
    selectItem: noop,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection: noop,
    setPage: noop,
    setPageInfo: noop,
    setSearchTerm: noop,
    setSemanticSearchMode: noop,
    setSemanticSearchText: noop,
    setStarredOnly: noop,
    setView: noop,
    toggleItemSelection: noop,
    updateSettings: noop,
  },
  galleryValues: { galleryPage: 0, selectedBoardId: 'none' },
  generateValues: {},
  getItemLabel: () => Promise.resolve(null),
  liveFollowEnabled: false,
  notifications: { add: noop, reportError: noop },
  pinnedProgressSessionId: null,
  progressSessions: [],
  projectId: 'project-1',
  projectName: 'Project',
  projects: [],
  widgets: { openGallery: () => true, patchGalleryValues: noop },
};
const sharingAdapter: GalleryUiAdapter = { ...adapter, boardSharing: true };
const runtime = { commands: { register: noop }, hotkeys: { register: noop } };

it('renders the compact gallery status through the localized status-chip key', async () => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryStatusChip count={12} />
      </ChakraProvider>
    )
  );

  expect(document.body.textContent).toContain('Gallery: 12 items');
  expect(document.body.textContent).not.toContain('widgets.gallery.statusChip');
});

const createGalleryData = (deepItem: GalleryImageItem) => ({
  boards: [],
  boardsState: { error: null, isFetchingMore: false, isRetrying: false, retry: noop, status: 'ready' },
  filter: { boardId: 'none' },
  isLoadingItems: false,
  isWindowTruncated: false,
  items: [deepItem],
  listing: { error: null, isFetchingMore: false, isRetrying: false, retry: noop, status: 'ready' },
  loadMore: noop,
  queryError: null,
  selectedBoardId: 'none',
  setVisibleRange: noop,
  sparseListing: {
    itemSlots: new Map([[180, deepItem]]),
    pageStates: new Map(),
    recentItems: [],
    total: 181,
  },
  total: 181,
});

const renderGallery = () =>
  act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={new QueryClient()}>
          <GalleryUiProvider adapter={adapter}>
            <GalleryWidgetView region="left" runtime={runtime} />
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

it('exposes sparse ranking pages to item context-menu actions', async () => {
  const deepItem = createImageItem('deep-result');
  mocks.galleryData = createGalleryData(deepItem);

  await renderGallery();

  expect(mocks.getItemActionContext?.()?.getItemSelectionPage?.(deepItem)).toBe(3);
});

it('publishes whether boards can be shared to the board panel and menus', async () => {
  mocks.galleryData = {
    boards: [],
    boardsState: { error: null, isFetchingMore: false, isRetrying: false, retry: noop, status: 'ready' },
    filter: { boardId: 'none' },
    isLoadingItems: false,
    isWindowTruncated: false,
    items: [],
    listing: { error: null, isFetchingMore: false, isRetrying: false, retry: noop, status: 'ready' },
    loadMore: noop,
    queryError: null,
    selectedBoardId: 'none',
    setVisibleRange: noop,
    total: 0,
  };

  const queryClient = new QueryClient();

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <GalleryUiProvider adapter={sharingAdapter}>
            <GalleryWidgetView region="left" runtime={runtime} />
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  expect(mocks.boardSharing).toBe(true);
});

describe('external reveal announcements', () => {
  const REVEALED = 'widgets.gallery.externalGenerationRevealed';
  const accountSignal = new AbortController().signal;
  /** The failure region first, then the two reveal regions. */
  const regionTexts = () => [...document.body.querySelectorAll('[role="status"]')].map((region) => region.textContent);
  const revealExternal = (name: string) =>
    act(() => requestGalleryItemReveal(`image:${name}`, accountSignal, 0, 'external-generation'));

  beforeEach(() => {
    mocks.galleryData = createGalleryData(createImageItem('deep-result'));
  });

  it('announces each external reveal once, and none the user made', async () => {
    await renderGallery();

    await act(() => requestGalleryItemReveal('image:mine.png', accountSignal, 3));
    expect(regionTexts()).toEqual(['', '', '']);

    await revealExternal('first.png');
    const afterFirst = regionTexts();
    expect(afterFirst.filter((text) => text === REVEALED)).toHaveLength(1);

    // The second announcement lands in the region the first left blank, so it is a change and is read again.
    await revealExternal('second.png');
    const afterSecond = regionTexts();
    const announcedIn = afterSecond.indexOf(REVEALED);
    expect(afterSecond.filter((text) => text === REVEALED)).toHaveLength(1);
    expect(afterFirst[announcedIn]).toBe('');

    await act(() => requestGalleryItemReveal('image:mine.png', accountSignal, 3));
    expect(regionTexts()).toEqual(['', '', '']);
  });

  it('keeps reveal text out of the failure announcement', async () => {
    mocks.galleryData = {
      ...createGalleryData(createImageItem('deep-result')),
      listing: { error: null, isFetchingMore: false, isRetrying: false, retry: noop, status: 'stale-error' },
    };
    await renderGallery();
    await revealExternal('external.png');

    expect(regionTexts()[0]).toBe('widgets.gallery.listingRefreshFailed');
    expect(regionTexts().slice(1)).toContain(REVEALED);
  });

  it('does not repeat an earlier reveal when the Gallery mounts again', async () => {
    await renderGallery();
    await revealExternal('external.png');
    expect(regionTexts()).toContain(REVEALED);

    await act(() => root?.unmount());
    root = createRoot(host!);
    await renderGallery();

    expect(regionTexts()).toEqual(['', '', '']);
  });
});

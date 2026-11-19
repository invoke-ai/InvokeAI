/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemRef, GalleryItemsPage } from '@features/gallery/contracts';
import type { ImageIndexAvailability } from '@features/gallery/data/backend';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';
import type { QueueProgressSession } from '@features/queue/contracts';
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';
import type * as VirtualModule from 'react-hook-tanstack-virtual';

import { Box, ChakraProvider } from '@chakra-ui/react';
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useDndMonitor,
  useSensor,
  useSensors,
  type DragStartEvent,
} from '@dnd-kit/core';
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';
import { toGalleryItemRef } from '@features/gallery/core/items';
import { requestGalleryItemReveal } from '@features/gallery/core/selection';
import { getGallerySettings } from '@features/gallery/core/settings';
import {
  galleryBoardsOptions,
  galleryItemsPageOptions,
  getGalleryListingBoardsQuery,
} from '@features/gallery/data/queries';
import { invalidateGalleryItems, patchGalleryItemCaches } from '@features/gallery/data/queryCache';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { getAutoSwitchFollowedLive } from '@features/queue/contracts';
import { dragWithMouse } from '@platform/browser/mouseDrag.testing';
import { parseDateTokens } from '@platform/search/dateTokens';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { PrimaryMouseSensor } from '@workbench/shell/holdToDragSensor';
import { workbenchAutoScroll } from '@workbench/widgetDnd';
import { PreviewFilmstrip } from '@workbench/widgets/preview/PreviewFilmstrip';
import { PreviewFrame } from '@workbench/widgets/preview/PreviewFrame';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryActions, GalleryStarredStrip, GalleryWidgetContextValue } from './GalleryWidgetContext';
import type { GalleryListingState, GallerySparseListing } from './useGalleryData';

import { mergeGalleryLoadedItems } from './galleryGridLayout';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryWidgetContext } from './GalleryWidgetContext';
import { useGalleryData } from './useGalleryData';
import { EMPTY_GALLERY_STARRED_STRIP } from './useGalleryStarredStrip';

const mocks = vi.hoisted(() => ({
  itemProgress: null as { percentage: number; message: string } | null,
  progressFrame: null as { dataUrl: string; width: number; height: number } | null,
  fetchNames: vi.fn(),
  fetchSparsePage: vi.fn<(filter: GalleryItemsFilter, offset: number) => Promise<GalleryItemsPage>>(),
  fetchSparseTotal: vi.fn<(filter: GalleryItemsFilter) => Promise<number>>(),
  getGalleryItemByRef: vi.fn<(ref: { kind: string; name: string }, signal?: AbortSignal) => Promise<GalleryItem>>(),
  getItemLabel: vi.fn<GalleryUiAdapter['getItemLabel']>(),
  indexAvailability: { modelName: null, state: 'disabled' } as ImageIndexAvailability,
  measure: vi.fn(),
  scrollToIndex: vi.fn(),
  setPage: vi.fn(),
  virtualizerOptions: [] as Array<{
    count: number;
    estimateSize: (index: number) => number;
    getScrollElement: () => Element | null;
    onChange?: (instance: {
      getVirtualItems: () => readonly { index: number }[];
      range: { endIndex: number; startIndex: number } | null;
    }) => void;
    overscan: number;
  }>,
  useRealGridVirtualizer: false,
  partialRowPosition: null as { ratio: number; rowIndex: number; visibleRows: number } | null,
  partialRowScrollOffset: null as number | null,
}));

const getNamesKey = (filter: unknown) => ['test-gallery-item-names', JSON.stringify(filter)] as const;
const requestReveal = (itemKey: Parameters<typeof requestGalleryItemReveal>[0], absoluteIndex?: number) =>
  requestGalleryItemReveal(itemKey, accountLifecycle.capture().signal, absoluteIndex);

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getGalleryItemByRef: mocks.getGalleryItemByRef,
}));

vi.mock('@features/gallery/data/queries', async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  const getPageOptions = actual.galleryItemsPageOptions as (
    filter: GalleryItemsFilter,
    offset: number
  ) => Record<string, unknown>;
  const getTotalOptions = actual.galleryItemsTotalOptions as (filter: GalleryItemsFilter) => Record<string, unknown>;

  return {
    ...actual,
    galleryItemsPageOptions: (filter: GalleryItemsFilter, offset: number) => ({
      ...getPageOptions(filter, offset),
      queryFn: () => mocks.fetchSparsePage(filter, offset),
    }),
    galleryItemsTotalOptions: (filter: GalleryItemsFilter) => ({
      ...getTotalOptions(filter),
      queryFn: () => mocks.fetchSparseTotal(filter),
    }),
    imageIndexAvailabilityOptions: () => ({
      queryFn: () => mocks.indexAvailability,
      queryKey: ['test-image-index-availability'],
    }),
    galleryItemNamesOptions: (filter: unknown) => ({
      queryFn: () => mocks.fetchNames(filter),
      queryKey: getNamesKey(filter),
      staleTime: Infinity,
    }),
  };
});

vi.mock('react-hook-tanstack-virtual', async (importOriginal) => {
  const actual = await importOriginal<typeof VirtualModule>();
  return {
    ...actual,
    useVirtualizer: (options: {
      count: number;
      horizontal?: boolean;
      scrollMargin?: number;
      estimateSize: (index: number) => number;
      getScrollElement: () => Element | null;
      onChange?: (instance: {
        getVirtualItems: () => readonly { index: number }[];
        range: { endIndex: number; startIndex: number } | null;
      }) => void;
      overscan: number;
    }) => {
      if (options.overscan === 2 || mocks.useRealGridVirtualizer) {
        return actual.useVirtualizer(options);
      }
      mocks.virtualizerOptions.push(options);
      const sizes = Array.from({ length: options.count }, (_, index) => options.estimateSize(index));
      const starts = sizes.map(
        (_, index) => (options.scrollMargin ?? 0) + sizes.slice(0, index).reduce((total, size) => total + size, 0)
      );
      const partialRow = mocks.partialRowPosition;
      const rowSize = sizes[partialRow?.rowIndex ?? 0] ?? 0;
      const scrollOffset = partialRow ? (starts[partialRow.rowIndex] ?? 0) + rowSize * partialRow.ratio : null;
      const scrollHeight = partialRow
        ? (options.getScrollElement()?.clientHeight ?? rowSize * partialRow.visibleRows)
        : null;
      mocks.partialRowScrollOffset = scrollOffset;

      return {
        measure: mocks.measure,
        range: partialRow
          ? { endIndex: partialRow.rowIndex + partialRow.visibleRows, startIndex: partialRow.rowIndex }
          : null,
        scrollOffset,
        scrollRect:
          scrollHeight === null ? null : { height: scrollHeight, width: options.getScrollElement()?.clientWidth ?? 0 },
        scrollToIndex: mocks.scrollToIndex,
        totalSize: sizes.reduce((total, size) => total + size, 0),
        virtualItems: Array.from({ length: options.count }, (_, index) => ({
          end: (starts[index] ?? 0) + (sizes[index] ?? 0),
          index,
          key: index,
          size: sizes[index] ?? 0,
          start: starts[index] ?? 0,
        })),
      };
    },
  };
});

vi.mock('@features/queue/react', () => ({
  useItemProgress: () => mocks.itemProgress,
  useQueueItemProgressImage: () => mocks.progressFrame,
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { generating: 'Generating', retry: 'Retry' },
        widgets: {
          gallery: {
            inProgress: 'In progress',
            progressShared: 'Active gallery runs across all boards',
            progressPreparing: 'Preparing',
            progressQueued: 'Queued',
            progressSettling: 'Finishing',
            progressSession: '{{name}} · {{index}} of {{total}}',
            commands: {
              clearSelection: 'Clear selection',
              deleteSelection: 'Delete selection',
              navigationDown: 'Down',
              navigationLeft: 'Left',
              navigationRight: 'Right',
              navigationUp: 'Up',
              selectAllOnPage: 'Select all',
              toggleStarImage: 'Toggle star',
              toggleStarredOnly: 'Toggle starred filter',
            },
            generationProgress: 'Generation progress',
            generationProgressPercent: 'Generation {{percentage}}%',
            itemsAriaLabel: 'Gallery items',
            loadingBackendGallery: 'Loading gallery',
            noImagesMatch: 'No items',
            noMatchingImages: 'No images match',
            noMatchingVideos: 'No videos match',
            showAllMedia: 'Show all media',
            noStarredItemsMatch: 'No starred items',
            showAllStarred: 'Show all',
            showAllStarredItems: 'Show all starred items',
            dropMediaToUploadToBoard: 'Drop media to {{name}}',
            emptyBoardUploadHint: 'Drop media here or click to upload',
            selectImageForPreview: 'Select {{name}} for preview',
            selectVideoForPreview: 'Select video {{name}}, duration {{duration}}, for preview',
            selectedBoardFallback: 'selected board',
            starImage: 'Star {{name}}',
            starredItems: 'Starred',
            unstarImage: 'Unstar {{name}}',
            uncategorized: 'Uncategorized',
          },
          preview: {
            compare: 'Compare',
            showInProgressDiffusion: 'Show progress',
            viewing: 'Viewing',
          },
        },
      },
    },
  },
});

const createItem = (kind: GalleryItem['kind'], name: string, overrides: Partial<GalleryItem> = {}): GalleryItem => {
  const base = {
    boardId: 'board-a',
    category: 'general' as const,
    createdAt: '2026-07-30T00:00:00.000Z',
    fullUrl: `/full/${kind}/${name}`,
    height: 96,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumbnail/${kind}/${name}`,
    width: 128,
    ...overrides,
  };

  return kind === 'video'
    ? ({
        ...base,
        durationSeconds: 'durationSeconds' in base ? (base.durationSeconds ?? 65.2) : 65.2,
        kind,
      } as GalleryItem)
    : ({ ...base, kind } as GalleryItem);
};

const previewSource: StreamingImageSource = {
  alt: 'shared',
  height: 96,
  kind: 'final',
  src: '/full/image/shared',
  width: 128,
};

const board = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'board-a',
  imageCount: 3,
  kind: 'board',
  name: 'Board A',
  isInbox: false,
  projectId: null,
  videoCount: 1,
} as const;

const createFilter = (gallery: GalleryStateView): GalleryItemsFilter => {
  const parse = parseDateTokens(gallery.searchTerm);

  return {
    boardId: gallery.selectedBoardId,
    ...(parse.range?.from ? { createdFrom: parse.range.from } : {}),
    ...(parse.range?.to ? { createdTo: parse.range.to } : {}),
    galleryView: gallery.galleryView,
    orderDir: gallery.settings.imageOrderDir,
    searchTerm: parse.text,
    ...(gallery.starredOnly ? { starred: true } : {}),
  };
};

const stableFilters = new Map<string, GalleryItemsFilter>();
const getStableFilter = (gallery: GalleryStateView): GalleryItemsFilter => {
  const filter = createFilter(gallery);
  const identity = JSON.stringify(filter);

  if (!stableFilters.has(identity)) {
    stableFilters.set(identity, filter);
  }

  return stableFilters.get(identity)!;
};

const DENSE_SETTINGS = { ...getGallerySettings({}), imageDensityPercent: 0 };

const createGallery = (overrides: Partial<GalleryStateView> = {}): GalleryStateView => {
  const items = overrides.items ?? [
    createItem('image', 'first.png'),
    createItem('video', 'shared'),
    createItem('image', 'last.png'),
  ];

  return {
    anchoredWindowPage: 0,
    boards: [board],
    compareImageKey: null,
    galleryView: 'images',
    isComparisonActive: false,
    isCompareArmed: false,
    isSearchActive: false,
    items,
    listingBoardId: overrides.selectedBoardId ?? board.id,
    page: 0,
    primarySelectedItemKey: overrides.selectedItemKey === undefined ? 'image:first.png' : overrides.selectedItemKey,
    revealTargetPage: null,
    projectBoardId: null,
    searchScope: 'board',
    searchTerm: '',
    selectedBoardId: board.id,
    selectedItemKey: 'image:first.png',
    selectedItemKeys: ['image:first.png'],
    selectionStarredOnly: false,
    semanticImageQuery: null,
    semanticSearchText: null,
    settings: { ...getGallerySettings({ paginationMode: 'paginated' }), imageDensityPercent: 0 },
    starredOnly: false,
    ...overrides,
  };
};

const actionMocks = {
  loadMore: vi.fn(),
  selectItem: vi.fn(),
  selectItemRange: vi.fn(),
  setCompareItem: vi.fn(),
  setStarredOnly: vi.fn(),
  toggleItemInSelection: vi.fn(),
  updateSettings: vi.fn(),
  uploadFiles: vi.fn(),
};
const imageActionMocks = {
  deleteItems: vi.fn(),
  deleteImages: vi.fn(),
  downloadItem: vi.fn(),
  downloadItems: vi.fn(),
  moveImagesToBoard: vi.fn(),
  moveItemsToBoard: vi.fn(),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  setItemsStarred: vi.fn(),
  setImagesStarred: vi.fn(),
};
const noop = vi.fn();
const registeredCommands = new Map<string, () => unknown>();
const runtime = {
  commands: {
    register: ({ handler, id }: { handler: () => unknown; id: string }) => {
      registeredCommands.set(id, handler);
      return () => registeredCommands.delete(id);
    },
  },
  hotkeys: {
    register: ({ commandId, defaultKeys }: { commandId: string; defaultKeys: string[] }) => {
      const handler = (event: KeyboardEvent) => {
        if (defaultKeys.includes(event.key.toLowerCase())) {
          registeredCommands.get(commandId)?.();
        }
      };
      window.addEventListener('keydown', handler);
      return () => window.removeEventListener('keydown', handler);
    },
  },
};

const createActions = (): GalleryActions =>
  ({
    archiveBoard: vi.fn(),
    createBoard: vi.fn(),
    deleteBoard: vi.fn(),
    downloadBoard: vi.fn(),
    loadMore: actionMocks.loadMore,
    refresh: noop,
    renameBoard: vi.fn(),
    selectBoard: noop,
    selectImage: actionMocks.selectItem,
    selectImageRange: actionMocks.selectItemRange,
    selectItem: actionMocks.selectItem,
    selectItemRange: actionMocks.selectItemRange,
    selectProjectBoard: vi.fn(),
    setCompareItem: actionMocks.setCompareItem,
    setSearchTerm: noop,
    setStarredOnly: actionMocks.setStarredOnly,
    setView: noop,
    toggleImageInSelection: actionMocks.toggleItemInSelection,
    toggleItemInSelection: actionMocks.toggleItemInSelection,
    updateSettings: actionMocks.updateSettings,
    uploadFiles: actionMocks.uploadFiles,
  }) as unknown as GalleryActions;

type CanonicalContextTarget = {
  allStarred?: boolean;
  itemRefs?: GalleryItemRef[];
  items: GalleryItem[];
  returnFocus?: () => HTMLElement | null;
  x: number;
  y: number;
} | null;
const ContextMenuProbe = ({ target }: { target: CanonicalContextTarget }) => (
  <>
    <output data-testid="context-target">
      {JSON.stringify(
        target
          ? {
              itemRefs: target.itemRefs ?? null,
              items: target.items.map(({ kind, name }) => ({ kind, name })),
              ...(target.allStarred === undefined ? {} : { allStarred: target.allStarred }),
            }
          : null
      )}
    </output>
    {/* Stands in for a dialog the menu opened, closing: focus goes where the target says. */}
    <button data-testid="context-return-focus" type="button" onClick={() => target?.returnFocus?.()?.focus()}>
      Close dialog
    </button>
    {/* The open menu's portalled layer: a React child of the grid, outside its DOM. */}
    {target ? createPortal(<div data-testid="context-menu-layer">Menu</div>, document.body) : null}
  </>
);
const NoopProvider = ({ children }: { children: ReactNode }) => children;

// Passed as arguments so the compiler's memoization sees them change.
const createAdapter = (
  progressSessions: QueueProgressSession[],
  liveFollowEnabled: boolean,
  pinnedProgressSessionId: string | null,
  galleryValues: Record<string, unknown>
): GalleryUiAdapter =>
  ({
    ItemActionsProvider: NoopProvider,
    ImageContextMenu: ContextMenuProbe,
    antialiasProgressImages: false,
    gallery: {
      clearSelection: noop,
      reconcileDeletedBoardOutcome: noop,
      selectBoard: noop,
      selectImage: noop,
      selectItem: noop,
      setCompareImage: noop,
      setCompareItem: noop,
      setItemMultiSelection: noop,
      setPage: mocks.setPage,
      setPageInfo: noop,
      setSearchTerm: noop,
      setView: noop,
      toggleItemSelection: noop,
      updateSettings: noop,
    },
    galleryValues,
    generateValues: {},
    getItemLabel: mocks.getItemLabel,
    liveFollowEnabled,
    progressSessions,
    pinnedProgressSessionId,
    followedProgressSessionId: liveFollowEnabled
      ? // Every live session in these fixtures is running and has shown a frame of its own.
        (getAutoSwitchFollowedLive(
          'progress',
          progressSessions.map((session) => ({
            hasOwnFrame: true,
            id: session.id,
            state: session.state,
          })),
          pinnedProgressSessionId
        )?.id ?? null)
      : null,
    followProgressSession,
    notifications: { add: noop, reportError: noop },
    projectId: 'project-1',
    projectName: 'Project',
    widgets: { openGallery: () => true, patchGalleryValues: noop },
  }) as unknown as GalleryUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
let currentGallery = createGallery();
let currentGalleryValues: Record<string, unknown> = {};
let currentLiveFollowEnabled = false;
let currentPinnedSessionId: string | null = null;
let currentProgressSessions: QueueProgressSession[] = [];
const followProgressSession = vi.fn();
let currentStrip: GalleryStarredStrip = EMPTY_GALLERY_STARRED_STRIP;
let currentSparseListing: GallerySparseListing | undefined;
const setVisibleRange = vi.fn();
const pinRevealIndex = vi.fn();
const READY_LISTING: GalleryListingState = {
  error: null,
  isFetchingMore: false,
  isRetrying: false,
  retry: () => Promise.resolve(),
  status: 'ready',
};
let currentListing = READY_LISTING;
let onDragStart = vi.fn();
let withShellMouseSensor = false;

const setStrip = (items: GalleryItem[], total = items.length) => {
  currentStrip = { items, state: { ...READY_LISTING, status: items.length > 0 ? 'ready' : 'empty' }, total };
};
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const DragMonitor = () => {
  useDndMonitor({
    onDragStart: (event: DragStartEvent) => onDragStart({ data: event.active.data.current, id: event.active.id }),
  });

  return null;
};

const Harness = ({
  background = 'bg',
  coMountPreviewSources = false,
  gallery,
  liveFollowEnabled,
  pinnedSessionId,
  progressSessions,
}: {
  background?: 'bg' | 'bg.panel';
  coMountPreviewSources?: boolean;
  gallery: GalleryStateView;
  liveFollowEnabled: boolean;
  pinnedSessionId: string | null;
  progressSessions: QueueProgressSession[];
}) => {
  const sensors = useSensors(
    useSensor(withShellMouseSensor ? PrimaryMouseSensor : PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const contextValue: GalleryWidgetContextValue = {
    actions: createActions(),
    boardsState: READY_LISTING,
    filter: getStableFilter(gallery),
    gallery,
    itemActions: imageActionMocks,
    listing: currentListing,
    loadedItems: mergeGalleryLoadedItems(currentStrip.items, gallery.items),
    projectName: 'Project',
    region: 'right',
    pinRevealIndex,
    runtime,
    setVisibleRange,
    sparseListing: currentSparseListing,
    starredStrip: currentStrip,
  } as unknown as GalleryWidgetContextValue;

  return (
    <I18nextProvider i18n={i18n}>
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <GalleryUiProvider
            adapter={createAdapter(progressSessions, liveFollowEnabled, pinnedSessionId, currentGalleryValues)}
          >
            <GalleryWidgetContext value={contextValue}>
              <DndContext autoScroll={workbenchAutoScroll} sensors={sensors}>
                <DragMonitor />
                <Box bg={background} data-testid="gallery-surface" h="full">
                  <GalleryImageGrid />
                </Box>
                {coMountPreviewSources ? (
                  <>
                    <PreviewFrame
                      dragItem={{ kind: 'image', name: 'shared' }}
                      frameHeight={96}
                      frameWidth={128}
                      isLive={false}
                      shouldAntialiasLiveImage
                      source={{ itemKey: 'image:shared', kind: 'image', source: previewSource }}
                      variant="framed"
                    />
                    <PreviewFilmstrip
                      density="full"
                      items={[createItem('image', 'shared'), createItem('image', 'other.png')]}
                      selectedItemKey="image:shared"
                      onSelect={noop}
                    />
                  </>
                ) : null}
              </DndContext>
            </GalleryWidgetContext>
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    </I18nextProvider>
  );
};

const QueryBackedGalleryHarness = ({
  gallery,
  liveFollowEnabled,
  pinnedSessionId,
  progressSessions,
}: {
  gallery: GalleryStateView;
  liveFollowEnabled: boolean;
  pinnedSessionId: string | null;
  progressSessions: QueueProgressSession[];
}) => {
  const sparseData = useGalleryData({
    galleryView: gallery.galleryView,
    page: gallery.page,
    projectBoardId: gallery.projectBoardId,
    recentImages: [],
    searchTerm: gallery.searchTerm,
    selectedBoardId: gallery.selectedBoardId,
    semanticQuery: gallery.semanticImageQuery,
    settings: gallery.settings,
    starred: gallery.starredOnly ? true : undefined,
    sparseViewport: true,
  });
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const contextValue: GalleryWidgetContextValue = {
    actions: createActions(),
    boardsState: sparseData.boardsState,
    filter: sparseData.filter,
    gallery,
    itemActions: imageActionMocks,
    listing: sparseData.listing,
    loadedItems: mergeGalleryLoadedItems(currentStrip.items, sparseData.items ?? []),
    projectName: 'Project',
    region: 'right',
    runtime,
    setVisibleRange: sparseData.setVisibleRange,
    sparseListing: sparseData.sparseListing,
    starredStrip: currentStrip,
  } as unknown as GalleryWidgetContextValue;

  return (
    <GalleryUiProvider
      adapter={createAdapter(progressSessions, liveFollowEnabled, pinnedSessionId, currentGalleryValues)}
    >
      <GalleryWidgetContext value={contextValue}>
        <DndContext autoScroll={workbenchAutoScroll} sensors={sensors}>
          <DragMonitor />
          <Box data-testid="gallery-surface" h="full">
            <GalleryImageGrid />
          </Box>
        </DndContext>
      </GalleryWidgetContext>
    </GalleryUiProvider>
  );
};

const interact = (action: () => void, delay = 0): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const renderGallery = async (
  gallery = currentGallery,
  coMountPreviewSources = false,
  background: 'bg' | 'bg.panel' = 'bg'
) => {
  currentGallery = gallery;
  await interact(() =>
    root?.render(
      <Harness
        progressSessions={currentProgressSessions}
        background={background}
        coMountPreviewSources={coMountPreviewSources}
        gallery={gallery}
        liveFollowEnabled={currentLiveFollowEnabled}
        pinnedSessionId={currentPinnedSessionId}
      />
    )
  );
};

const renderQueryBackedGallery = async (gallery: GalleryStateView) => {
  currentGallery = gallery;
  await interact(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient!}>
            <QueryBackedGalleryHarness
              gallery={gallery}
              liveFollowEnabled={currentLiveFollowEnabled}
              pinnedSessionId={currentPinnedSessionId}
              progressSessions={currentProgressSessions}
            />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const getButton = (label: string): HTMLButtonElement => {
  const button = host?.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

  if (!button) {
    throw new Error(`Expected button "${label}"`);
  }

  return button;
};

const findStarredToggle = (expanded: boolean): HTMLButtonElement | null =>
  [...(host?.querySelectorAll<HTMLButtonElement>(`button[aria-expanded="${String(expanded)}"]`) ?? [])].find((button) =>
    button.textContent?.startsWith('Starred')
  ) ?? null;

const getStarredToggle = (expanded: boolean): HTMLButtonElement => {
  const button = findStarredToggle(expanded);

  if (!button) {
    throw new Error(`Expected the ${expanded ? 'expanded' : 'collapsed'} starred disclosure`);
  }

  return button;
};

const click = (button: HTMLButtonElement, init: MouseEventInit = {}): Promise<void> =>
  interact(() => button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, ...init })));

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

beforeEach(() => {
  // Drain singleton reveal intent so later tests cannot adopt it.
  requestReveal('image:__drained__');
  accountLifecycle.activate('grid-user');
  vi.clearAllMocks();
  mocks.fetchSparsePage.mockReset();
  mocks.fetchSparseTotal.mockReset();
  mocks.getGalleryItemByRef.mockReset();
  registeredCommands.clear();
  currentGallery = createGallery();
  currentGalleryValues = { selectedBoardId: board.id };
  stableFilters.clear();
  mocks.itemProgress = null;
  mocks.indexAvailability = { modelName: null, state: 'disabled' };
  mocks.getItemLabel.mockReset();
  currentProgressSessions = [];
  currentLiveFollowEnabled = false;
  currentPinnedSessionId = null;
  mocks.progressFrame = null;
  currentStrip = EMPTY_GALLERY_STARRED_STRIP;
  currentSparseListing = undefined;
  mocks.useRealGridVirtualizer = false;
  mocks.partialRowPosition = null;
  mocks.partialRowScrollOffset = null;
  mocks.virtualizerOptions.length = 0;
  currentListing = READY_LISTING;
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  host.style.cssText = 'height:480px;left:20px;position:fixed;top:20px;width:600px;';
  document.body.append(host);
  root = createRoot(host);
  onDragStart = vi.fn();
  withShellMouseSensor = false;
});

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  queryClient?.clear();
  host = null;
  queryClient = null;
  root = null;
});

describe('GalleryImageGrid mixed item cells', () => {
  it('keeps the gallery scroll position when an image is dragged upward out of a long grid', async () => {
    const items = Array.from({ length: 60 }, (_, index) => createItem('image', `image-${index}.png`));
    await renderGallery(createGallery({ items, settings: DENSE_SETTINGS }));

    const viewport = host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
    await interact(() => {
      viewport.scrollTop = viewport.scrollHeight;
    }, 50);
    const initialScrollTop = viewport.scrollTop;
    expect(initialScrollTop).toBeGreaterThan(viewport.clientHeight);

    const tile = getButton('Select image-59.png for preview');
    const rect = tile.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    await interact(() => pointer('pointerdown', tile, x, y));
    await interact(() => pointer('pointermove', document, x, y - 20), 50);
    expect(onDragStart).toHaveBeenCalledOnce();

    const edgeY = viewport.getBoundingClientRect().top + 2;
    await interact(() => pointer('pointermove', document, x, edgeY), 250);
    expect(viewport.scrollTop).toBe(initialScrollTop);

    await interact(() => pointer('pointermove', document, x, edgeY - 20), 100);
    // Let dnd-kit's post-drag click guard expire before the next test.
    await interact(() => pointer('pointerup', document, x, edgeY - 20), 300);
    expect(viewport.scrollTop).toBe(initialScrollTop);
  });

  const starred = createItem('image', 'starred.png', { starred: true });
  const sectionOrder = () =>
    Array.from(host?.querySelectorAll('[data-gallery-section]') ?? []).map((row) =>
      row.getAttribute('data-gallery-section')
    );

  it('puts the starred strip in an expanded disclosure above the unstarred listing', async () => {
    setStrip([starred]);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }));

    const trigger = getStarredToggle(true);
    const starredSection = host?.querySelector('[data-gallery-section="starred"]');
    const regularSection = host?.querySelector('[data-gallery-section="regular"]');

    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    // WCAG 2.5.3.
    await expect.element(trigger).toHaveAccessibleName(/^Starred\s+1$/);
    expect(trigger.closest('[role="list"]')).toBeNull();
    expect(starredSection?.querySelector('button[aria-label="Select starred.png for preview"]')).not.toBeNull();
    expect(regularSection?.querySelector('button[aria-label="Select regular.png for preview"]')).not.toBeNull();
    expect(host?.querySelectorAll('button[aria-label="Select starred.png for preview"]')).toHaveLength(1);
    expect(sectionOrder()).toEqual(['starred', 'regular']);
    expect(host?.querySelector('button[aria-label="Show all starred items"]')).toBeNull();
  });

  it('stars from the strip for a selection the listing window has not loaded', async () => {
    setStrip([starred]);
    await renderGallery(
      createGallery({
        items: [createItem('image', 'regular.png')],
        selectedItemKey: 'image:starred.png',
        selectedItemKeys: ['image:starred.png'],
        settings: DENSE_SETTINGS,
      })
    );

    registeredCommands.get('gallery.starImage')?.();
    expect(imageActionMocks.setItemsStarred).toHaveBeenCalledWith([{ kind: 'image', name: 'starred.png' }], false);

    await click(getButton('Unstar starred.png'));
    expect(imageActionMocks.setItemsStarred).toHaveBeenCalledTimes(2);

    registeredCommands.get('gallery.toggleStarredOnly')?.();
    expect(actionMocks.setStarredOnly).toHaveBeenCalledExactlyOnceWith(true);
  });

  it('renders same-name media independently and gives a video a static accessible poster', async () => {
    const gallery = createGallery({
      items: [createItem('image', 'shared'), createItem('video', 'shared')],
      selectedItemKey: 'image:shared',
      selectedItemKeys: ['image:shared'],
    });

    await renderGallery(gallery);

    const list = host?.querySelector('[role="list"]');
    const imageButton = getButton('Select shared for preview');
    const videoButton = getButton('Select video shared, duration 1:06, for preview');
    const videoCell = videoButton.closest<HTMLElement>('[role="listitem"]');
    const videoPoster = videoButton.querySelector<HTMLImageElement>('img');
    const playIcon = videoCell?.querySelector('svg.lucide-play');

    expect(list?.getAttribute('aria-label')).toBe('Gallery items');
    expect(host?.querySelectorAll('[role="listitem"]')).toHaveLength(2);
    expect(imageButton.getAttribute('aria-pressed')).toBe('true');
    expect(videoButton.getAttribute('aria-pressed')).toBe('false');
    expect(videoPoster?.getAttribute('src')).toContain('/thumbnail/video/shared');
    expect(videoPoster?.getAttribute('decoding')).toBe('async');
    expect(videoPoster?.hasAttribute('loading')).toBe(false);
    expect(videoCell?.textContent).toContain('1:06');
    expect(playIcon?.getAttribute('aria-hidden')).toBe('true');
    expect(imageButton.closest('[role="listitem"]')?.textContent).toContain('128x96');
    expect(host?.querySelector('button[aria-label="Star shared"]')).not.toBeNull();
    expect(host?.querySelector('video')).toBeNull();
  });

  it('uses qualified video drag identity and emits a video-capable item payload', async () => {
    await renderGallery(createGallery({ items: [createItem('video', 'shared')] }));
    const videoButton = getButton('Select video shared, duration 1:06, for preview');

    await interact(() => pointer('pointerdown', videoButton, 80, 80), 20);
    await interact(() => pointer('pointermove', videoButton.ownerDocument, 120, 80), 50);

    expect(onDragStart).toHaveBeenCalledWith({
      data: { items: [{ kind: 'video', name: 'shared' }], kind: 'gallery-item' },
      id: 'gallery-grid#right:video:shared',
    });

    // Let dnd-kit's post-drag click guard expire before the next test.
    await interact(() => pointer('pointerup', videoButton.ownerDocument, 120, 80), 300);
  });

  it('opens the focused video in Preview with Enter, once, without activating keyboard DnD', async () => {
    const video = createItem('video', 'keyboard.mp4');

    await renderGallery(createGallery({ items: [video] }));
    const videoButton = getButton('Select video keyboard.mp4, duration 1:06, for preview');

    await interact(() => videoButton.focus());
    await act(() => userEvent.keyboard('{Enter}'));

    expect(imageActionMocks.openItemInPreview).toHaveBeenCalledExactlyOnceWith(video);
    expect(actionMocks.selectItem).not.toHaveBeenCalled();
    expect(onDragStart).not.toHaveBeenCalled();
  });

  it('opens a double-clicked thumbnail in Preview after its clicks select it', async () => {
    const image = createItem('image', 'first.png');

    await renderGallery(createGallery({ selectedItemKey: 'image:last.png', selectedItemKeys: ['image:last.png'] }));
    await act(() => userEvent.dblClick(getButton('Select first.png for preview')));

    expect(actionMocks.selectItem).toHaveBeenCalledWith(image);
    expect(imageActionMocks.openItemInPreview).toHaveBeenCalledExactlyOnceWith(image);
  });

  describe.each([
    { item: createItem('image', 'last.png'), label: 'Select last.png for preview' },
    { item: createItem('video', 'shared'), label: 'Select video shared, duration 1:06, for preview' },
  ])('browser middle-click on $item.kind', ({ item, label }) => {
    it.each([false, true])(
      'respects middleClickOpensInNewTab=%s without selecting, previewing, or dragging',
      async (enabled) => {
        const gallery = createGallery({ items: [item], selectedItemKey: null, selectedItemKeys: [] });
        await renderGallery({ ...gallery, settings: { ...gallery.settings, middleClickOpensInNewTab: enabled } });
        const button = getButton(label);
        const events: MouseEvent[] = [];
        const recordEvent = (event: MouseEvent) => events.push(event);
        // Keep the native events until React has handled them, even if it stops their propagation.
        button.addEventListener('mousedown', recordEvent, { capture: true, once: true });
        button.addEventListener('auxclick', recordEvent, { capture: true, once: true });

        await act(() => userEvent.click(button, { button: 'middle' }));

        expect(
          events.map(({ type, button, isTrusted, defaultPrevented }) => ({ type, button, isTrusted, defaultPrevented }))
        ).toEqual([
          { type: 'mousedown', button: 1, isTrusted: true, defaultPrevented: enabled },
          { type: 'auxclick', button: 1, isTrusted: true, defaultPrevented: enabled },
        ]);
        if (enabled) {
          expect(imageActionMocks.openItemInNewTab).toHaveBeenCalledExactlyOnceWith(item);
        } else {
          expect(imageActionMocks.openItemInNewTab).not.toHaveBeenCalled();
        }
        expect(actionMocks.selectItem).not.toHaveBeenCalled();
        expect(actionMocks.selectItemRange).not.toHaveBeenCalled();
        expect(actionMocks.toggleItemInSelection).not.toHaveBeenCalled();
        expect(actionMocks.setCompareItem).not.toHaveBeenCalled();
        expect(imageActionMocks.openItemInPreview).not.toHaveBeenCalled();
        expect(onDragStart).not.toHaveBeenCalled();
      }
    );
  });

  it('starts a thumbnail drag through the shell mouse sensor', async () => {
    withShellMouseSensor = true;
    await renderGallery(createGallery({ items: [createItem('image', 'a.png'), createItem('image', 'b.png')] }));

    await dragWithMouse(getButton('Select a.png for preview'), getButton('Select b.png for preview'));

    expect(onDragStart).toHaveBeenCalledWith(expect.objectContaining({ id: 'gallery-grid#right:image:a.png' }));
    // Let dnd-kit's post-drag click guard expire before the next test.
    await interact(noop, 300);
  });

  it('preserves the ordered full selection when an unloaded video is dragged from a loaded image', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'loaded.png')],
        selectedItemKey: 'image:loaded.png',
        selectedItemKeys: ['image:loaded.png', 'video:unloaded.mp4'],
      })
    );
    const imageButton = getButton('Select loaded.png for preview');

    await interact(() => pointer('pointerdown', imageButton, 80, 80), 20);
    await interact(() => pointer('pointermove', imageButton.ownerDocument, 120, 80), 50);

    const drag = onDragStart.mock.calls[0]?.[0];
    await interact(() => pointer('pointerup', imageButton.ownerDocument, 120, 80), 300);

    expect(drag).toEqual({
      data: {
        items: [
          { kind: 'image', name: 'loaded.png' },
          { kind: 'video', name: 'unloaded.mp4' },
        ],
        kind: 'gallery-item',
      },
      id: 'gallery-grid#right:image:loaded.png',
    });
    expect(isGalleryImageDragData(drag?.data)).toBe(false);
  });

  it('marks the image Select for compare waits with as the comparison, and only it', async () => {
    const tileText = (name: string) =>
      getButton(`Select ${name} for preview`).closest('[role="listitem"]')?.textContent ?? '';
    await renderGallery(
      createGallery({
        compareImageKey: 'image:first.png',
        isCompareArmed: true,
        items: [createItem('image', 'first.png'), createItem('image', 'second.png')],
        selectedItemKey: 'image:first.png',
        selectedItemKeys: ['image:first.png'],
      })
    );

    expect(tileText('first.png')).toContain('Compare');
    expect(tileText('second.png')).not.toContain('Compare');
    expect(host!.textContent).not.toContain('Viewing');

    registeredCommands.get('gallery.exitCompare')?.();
    expect(actionMocks.setCompareItem).toHaveBeenLastCalledWith(null);

    currentProgressSessions = [
      {
        backendItemId: 10,
        height: 768,
        id: 'run:1',
        itemCount: 1,
        itemIndex: 1,
        label: 'Workflow A',
        queueItemId: 'run',
        sourceId: 'workflow',
        state: 'running',
        width: 512,
      },
    ];
    currentLiveFollowEnabled = true;
    await renderGallery();
    expect(tileText('first.png')).toContain('Compare');
    expect(registeredCommands.has('gallery.exitCompare')).toBe(true);
  });

  it('returns focus from a dialog its context menu opens to the right-clicked thumbnail, or the grid once it is gone', async () => {
    const first = createItem('image', 'first.png');
    const second = createItem('image', 'second.png');
    await renderGallery(createGallery({ items: [first, second], selectedItemKey: null, selectedItemKeys: [] }));
    const secondButton = getButton('Select second.png for preview');

    await interact(() =>
      secondButton.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 23, clientY: 41 })
      )
    );

    const closeDialog = host!.querySelector<HTMLButtonElement>('[data-testid="context-return-focus"]')!;

    await interact(() => closeDialog.click());
    expect(document.activeElement).toBe(secondButton);

    secondButton.remove();
    await interact(() => closeDialog.click());
    const gridViewport = getButton('Select first.png for preview').closest<HTMLElement>(
      '[data-scope="scroll-area"][data-part="viewport"]'
    );
    expect(gridViewport).not.toBeNull();
    expect(document.activeElement).toBe(gridViewport);
  });

  it('says a hotkey deletion reaches an item off screen when a selected member is on no loaded page', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'loaded.png')],
        selectedItemKey: 'image:loaded.png',
        selectedItemKeys: ['image:loaded.png', 'video:unloaded.mp4'],
      })
    );

    registeredCommands.get('gallery.deleteSelection')?.();

    expect(imageActionMocks.deleteItems).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'loaded.png' },
        { kind: 'video', name: 'unloaded.mp4' },
      ],
      expect.objectContaining({ targetsOffScreen: true })
    );
  });

  it('select-all and common hotkeys target ordered same-name mixed refs independently', async () => {
    const items = [createItem('image', 'shared'), createItem('video', 'shared')];
    await renderGallery(
      createGallery({
        items,
        selectedItemKey: 'video:shared',
        selectedItemKeys: ['image:shared', 'video:shared'],
      })
    );

    registeredCommands.get('gallery.selectAllOnPage')?.();
    registeredCommands.get('gallery.deleteSelection')?.();
    registeredCommands.get('gallery.starImage')?.();

    expect(actionMocks.selectItemRange).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'shared' },
      ],
      items[0]
    );
    expect(imageActionMocks.deleteItems).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'shared' },
      ],
      { returnFocus: undefined, targetsOffScreen: false }
    );
    expect(imageActionMocks.setItemsStarred).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'shared' },
      ],
      true
    );
  });
});

describe('GalleryImageGrid image-map labels', () => {
  const getTile = (name: string): HTMLElement => {
    const tile = getButton(`Select ${name} for preview`).closest<HTMLElement>('[role="listitem"]');

    if (!tile) {
      throw new Error(`Expected tile for ${name}`);
    }

    return tile;
  };
  const findBadge = (tile: HTMLElement, text: string) =>
    Array.from(tile.querySelectorAll<HTMLElement>('.gallery-thumb-overlay')).find(
      (element) => element.textContent === text
    );
  const opacityOf = (element: HTMLElement | undefined) => (element ? getComputedStyle(element).opacity : null);

  // Seeded so a reveal right after mount does not race the availability fetch.
  const setIndexAvailability = (availability: ImageIndexAvailability) => {
    mocks.indexAvailability = availability;
    queryClient?.setQueryData(['test-image-index-availability'], availability);
  };

  beforeEach(() => setIndexAvailability({ modelName: null, state: 'ready' }));

  it('reveals the hovered item label in step with its dimensions and star', async () => {
    mocks.getItemLabel.mockImplementation((item) => Promise.resolve(item.name === 'a.png' ? 'sunset' : 'forest'));
    await renderGallery(createGallery({ items: [createItem('image', 'a.png'), createItem('image', 'b.png')] }));

    expect(mocks.getItemLabel).not.toHaveBeenCalled();

    const tile = getTile('a.png');
    await userEvent.hover(tile);

    expect(mocks.getItemLabel).toHaveBeenCalledWith({ kind: 'image', name: 'a.png' });
    const dimensions = findBadge(tile, '128x96');
    const star = tile.querySelector<HTMLElement>('button[aria-label="Star a.png"]') ?? undefined;
    await vi.waitFor(() =>
      expect([findBadge(tile, 'sunset'), dimensions, star].map(opacityOf)).toEqual(['1', '1', '1'])
    );

    await userEvent.hover(getTile('b.png'));

    await vi.waitFor(() =>
      expect([findBadge(tile, 'sunset'), dimensions, star].map(opacityOf)).toEqual(['0', '0', '0'])
    );
    await vi.waitFor(() => expect(opacityOf(findBadge(getTile('b.png'), 'forest'))).toBe('1'));
  });
});

describe('GalleryImageGrid range selection', () => {
  const orderedRefs: GalleryItemRef[] = [
    { kind: 'image', name: 'first.png' },
    { kind: 'video', name: 'middle.mp4' },
    { kind: 'image', name: 'last.png' },
  ];
  const rangeItems = [
    createItem('image', 'first.png'),
    createItem('video', 'middle.mp4'),
    createItem('image', 'last.png'),
  ];

  it('does not load names on render and lazily selects the backend-ordered mixed range on Shift-click', async () => {
    mocks.fetchNames.mockResolvedValue({ items: orderedRefs, total: orderedRefs.length });
    const gallery = createGallery({ items: rangeItems });

    await renderGallery(gallery);
    expect(mocks.fetchNames).not.toHaveBeenCalled();

    await click(getButton('Select last.png for preview'), { shiftKey: true });
    await vi.waitFor(() => expect(actionMocks.selectItemRange).toHaveBeenCalledWith(orderedRefs, rangeItems[2]));
    expect(mocks.fetchNames).toHaveBeenCalledOnce();
  });

  it('ignores a names response after the account epoch changes', async () => {
    let resolveNames: ((value: { items: GalleryItemRef[]; total: number }) => void) | null = null;
    mocks.fetchNames.mockReturnValue(
      new Promise((resolve) => {
        resolveNames = resolve;
      })
    );

    await renderGallery(createGallery({ items: rangeItems }));
    await click(getButton('Select last.png for preview'), { shiftKey: true });
    accountLifecycle.activate('other-grid-user');
    await interact(() => resolveNames?.({ items: orderedRefs, total: orderedRefs.length }));

    expect(actionMocks.selectItemRange).not.toHaveBeenCalled();
  });
});

describe('GalleryImageGrid upload drop zone', () => {
  it('turns a true-empty, non-searching, non-virtual board into a click/drop upload target', async () => {
    await renderGallery(createGallery({ items: [] }));

    const target = host?.querySelector<HTMLElement>('[role="button"]');
    const input = host?.querySelector<HTMLInputElement>('input[type="file"]');

    expect(target).not.toBeNull();
    expect(target?.getAttribute('tabIndex')).toBe('0');
    expect(target?.textContent).toContain('Drop media here or click to upload');
    expect(input).not.toBeNull();
    expect(host?.textContent).not.toContain('No items');

    const clickSpy = vi.spyOn(input!, 'click');

    await interact(() => target?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));

    expect(clickSpy).toHaveBeenCalledOnce();
  });

  it('offers every container and audio format the upload route ingests', async () => {
    await renderGallery(createGallery({ items: [] }));

    const accept = host?.querySelector<HTMLInputElement>('input[type="file"]')?.accept.split(',');

    expect(accept).toEqual(expect.arrayContaining(['image/png', 'video/*', 'audio/*', '.mov', '.mkv', '.mp3', '.wav']));
  });
});

describe('GalleryImageGrid reveal requests', () => {
  it('never scrolls on selection changes alone', async () => {
    const gallery = createGallery();

    await renderGallery(gallery);
    await renderGallery({ ...gallery, selectedItemKey: 'image:last.png', selectedItemKeys: ['image:last.png'] });

    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });

  it('scrolls to a revealed item, and again when the same item is re-revealed', async () => {
    const gallery = createGallery();

    await renderGallery(gallery);
    await interact(() => requestReveal('image:first.png'));
    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);

    await interact(() => requestReveal('image:first.png'));
    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(2);
  });

  it('ignores an absolute-index reveal from the previous account before this grid mounts', async () => {
    accountLifecycle.activate('grid-user-a');
    await interact(() => requestReveal('image:account-a-deep.png', 70));
    accountLifecycle.activate('grid-user-b');

    await renderGallery(
      createGallery({
        items: [],
        selectedItemKey: null,
        selectedItemKeys: [],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );

    expect(setVisibleRange).not.toHaveBeenCalled();
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });

  it('waits for a loading page instead of stepping over it', async () => {
    const item59 = createItem('image', 'item-59.png');
    const item60 = createItem('image', 'item-60.png');
    const item120 = createItem('image', 'item-120.png');
    const settled = { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) };
    currentSparseListing = {
      itemSlots: new Map([
        [59, item59],
        [120, item120],
      ]),
      pageStates: new Map([
        [0, settled],
        [60, { ...settled, isLoading: true }],
        [120, settled],
      ]),
      recentItems: [],
      total: 180,
    };
    const gallery = createGallery({
      items: [item59, item120],
      selectedItemKey: 'image:item-59.png',
      selectedItemKeys: ['image:item-59.png'],
      settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
    });
    await renderGallery(gallery);
    setVisibleRange.mockClear();

    await interact(() => registeredCommands.get('gallery.galleryNavRight')?.());

    expect(actionMocks.selectItem).not.toHaveBeenCalled();
    expect(setVisibleRange).not.toHaveBeenCalledWith({ endIndexExclusive: 120, startIndex: 60 });

    currentSparseListing = {
      ...currentSparseListing,
      itemSlots: new Map([
        [59, item59],
        [60, item60],
        [120, item120],
      ]),
      pageStates: new Map([
        [0, settled],
        [60, settled],
        [120, settled],
      ]),
    };
    await renderGallery({ ...gallery, items: [item59, item60, item120] });

    expect(actionMocks.selectItem).toHaveBeenCalledExactlyOnceWith(item60, 1);
  });

  const createOffPageGallery = (revealTargetPage: number | null) =>
    createGallery({
      items: [createItem('image', 'page-zero.png')],
      revealTargetPage,
      selectedItemKey: null,
      selectedItemKeys: ['image:deep.png'],
    });

  it('follows the selection onto its paginated page when the revealed item is not loaded', async () => {
    const gallery = createOffPageGallery(2);

    await renderGallery(gallery);
    await interact(() => requestReveal('image:deep.png'));

    expect(mocks.setPage).toHaveBeenCalledWith(2);
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();

    await renderGallery({
      ...gallery,
      items: [createItem('image', 'deep.png')],
      page: 2,
      selectedItemKey: 'image:deep.png',
    });

    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);
  });
});

describe('GalleryImageGrid virtualization', () => {
  it('stamps a selected semantic result with its absolute sparse page', async () => {
    const result = createItem('image', 'semantic-180.png');
    currentSparseListing = {
      itemSlots: new Map([[180, result]]),
      pageStates: new Map([[180, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
      recentItems: [],
      total: 181,
    };

    await renderGallery(
      createGallery({
        items: [result],
        semanticImageQuery: { kind: 'text', query: 'sunset' },
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );
    await click(getButton('Select semantic-180.png for preview'));

    expect(actionMocks.selectItem).toHaveBeenCalledWith(result, 3);
  });

  it('keeps the rest of the selection when Ctrl-click removes a primary whose next primary is unloaded', async () => {
    const pageOne = createItem('image', 'page-one.png');
    const pageSix = createItem('image', 'page-six.png');
    currentSparseListing = {
      itemSlots: new Map([[300, pageSix]]),
      pageStates: new Map([[300, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
      recentItems: [],
      total: 400,
    };
    mocks.getGalleryItemByRef.mockResolvedValue(pageOne);

    await renderGallery(
      createGallery({
        items: [pageSix],
        selectedItemKey: 'image:page-six.png',
        selectedItemKeys: ['image:page-one.png', 'image:page-six.png'],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );
    await click(getButton('Select page-six.png for preview'), { ctrlKey: true });

    await vi.waitFor(() => expect(actionMocks.toggleItemInSelection).toHaveBeenCalledExactlyOnceWith(pageSix, pageOne));
    expect(mocks.getGalleryItemByRef).toHaveBeenCalledWith({ kind: 'image', name: 'page-one.png' }, expect.anything());
  });

  it('abandons an unloaded next-primary lookup when the grid unmounts', async () => {
    const pageSix = createItem('image', 'page-six.png');
    let lookupSignal: AbortSignal | undefined;
    currentSparseListing = {
      itemSlots: new Map([[300, pageSix]]),
      pageStates: new Map([[300, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
      recentItems: [],
      total: 400,
    };
    mocks.getGalleryItemByRef.mockImplementation((_ref, signal?: AbortSignal) => {
      lookupSignal = signal;
      return new Promise<GalleryItem>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
      });
    });

    await renderGallery(
      createGallery({
        items: [pageSix],
        selectedItemKey: 'image:page-six.png',
        selectedItemKeys: ['image:page-one.png', 'image:page-six.png'],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );
    await click(getButton('Select page-six.png for preview'), { ctrlKey: true });
    await act(async () => {
      root?.unmount();
      await Promise.resolve();
    });
    root = null;

    expect(lookupSignal?.aborted).toBe(true);
    expect(actionMocks.toggleItemInSelection).not.toHaveBeenCalled();
  });

  it('renders absolute sparse slots, page-local retry, and recent overlays at the sort edge', async () => {
    const first = createItem('image', 'sparse-120.png');
    const third = createItem('image', 'sparse-122.png');
    const recent = createItem('image', 'recent.png');
    const retry = vi.fn(() => Promise.resolve());
    const pageError = new Error('Page unavailable');
    currentSparseListing = {
      itemSlots: new Map([
        [120, first],
        [122, third],
      ]),
      pageStates: new Map([[120, { error: pageError, isLoading: false, retry }]]),
      recentItems: [recent],
      total: 130,
    };

    await renderGallery(
      createGallery({
        items: [first, third, recent],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite', imageOrderDir: 'DESC' },
      })
    );

    const descendingSections = [...(host?.querySelectorAll<HTMLElement>('[data-gallery-section]') ?? [])];
    expect(descendingSections[0]?.dataset.gallerySection).toBe('recent');
    expect(descendingSections.at(-1)?.dataset.gallerySection).toBe('regular');
    expect(host?.querySelector('img[alt="sparse-120.png"]')).not.toBeNull();
    expect(host?.querySelector('img[alt="sparse-122.png"]')).not.toBeNull();
    expect(host?.querySelectorAll('[data-gallery-slot-state="error"]')).toHaveLength(1);
    expect(host?.querySelectorAll('[data-gallery-slot-state="error"] button')).toHaveLength(1);
    expect(host?.querySelectorAll('[data-gallery-page-error] button')).toHaveLength(0);

    const localRetry = host!.querySelector<HTMLButtonElement>('[data-gallery-slot-state="error"] button')!;
    expect(localRetry.closest('[role="listitem"]')?.textContent).toContain('Page unavailable');
    await click(localRetry);
    expect(retry).toHaveBeenCalledTimes(1);

    await renderGallery(
      createGallery({
        items: [first, third, recent],
        settings: { ...DENSE_SETTINGS, imageOrderDir: 'ASC', paginationMode: 'infinite' },
      })
    );
    const ascendingSections = [...(host?.querySelectorAll<HTMLElement>('[data-gallery-section]') ?? [])];
    expect(ascendingSections[0]?.dataset.gallerySection).toBe('regular');
    expect(ascendingSections.at(-1)?.dataset.gallerySection).toBe('recent');
  });

  it('loads and selects the next absolute slot on keyboard navigation, while select-all excludes gaps', async () => {
    const lastInPage = createItem('image', 'page-0-last.png');
    const firstNextPage = createItem('image', 'page-1-first.png');
    currentSparseListing = {
      itemSlots: new Map([[59, lastInPage]]),
      pageStates: new Map([[0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
      recentItems: [],
      total: 120,
    };

    await renderGallery(
      createGallery({
        items: [lastInPage],
        selectedItemKey: 'image:page-0-last.png',
        selectedItemKeys: ['image:page-0-last.png'],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );

    const lastTile = getButton('Select page-0-last.png for preview');
    await interact(() => lastTile.focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(setVisibleRange).toHaveBeenLastCalledWith({ endIndexExclusive: 120, startIndex: 0 });
    expect(mocks.scrollToIndex).toHaveBeenCalled();
    expect(document.activeElement).toBe(lastTile);

    currentSparseListing = {
      itemSlots: new Map([
        [59, lastInPage],
        [60, firstNextPage],
      ]),
      pageStates: new Map([
        [0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
        [60, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
      ]),
      recentItems: [],
      total: 120,
    };
    await renderGallery({ ...currentGallery, items: [lastInPage, firstNextPage] });
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(firstNextPage, 1);
    expect(document.activeElement).toBe(getButton('Select page-1-first.png for preview'));

    registeredCommands.get('gallery.selectAllOnPage')?.();
    expect(actionMocks.selectItemRange).toHaveBeenLastCalledWith(
      [
        { kind: 'image', name: 'page-0-last.png' },
        { kind: 'image', name: 'page-1-first.png' },
      ],
      lastInPage,
      0
    );
  });

  it('loads and extends a sparse range across an unloaded page boundary', async () => {
    const lastInPage = createItem('image', 'range-page-0-last.png');
    const firstNextPage = createItem('image', 'range-page-1-first.png');
    mocks.fetchNames.mockResolvedValue({
      items: [toGalleryItemRef(lastInPage), toGalleryItemRef(firstNextPage)],
      total: 2,
    });
    currentSparseListing = {
      itemSlots: new Map([[59, lastInPage]]),
      pageStates: new Map([[0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
      recentItems: [],
      total: 120,
    };

    await renderGallery(
      createGallery({
        items: [lastInPage],
        selectedItemKey: 'image:range-page-0-last.png',
        selectedItemKeys: ['image:range-page-0-last.png'],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );
    await interact(() => getButton('Select range-page-0-last.png for preview').focus());
    registeredCommands.get('gallery.extendSelectionRight')?.();

    expect(setVisibleRange).toHaveBeenLastCalledWith({ endIndexExclusive: 120, startIndex: 0 });
    currentSparseListing = {
      itemSlots: new Map([
        [59, lastInPage],
        [60, firstNextPage],
      ]),
      pageStates: new Map([
        [0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
        [60, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
      ]),
      recentItems: [],
      total: 120,
    };
    await renderGallery({ ...currentGallery, items: [lastInPage, firstNextPage] });

    await vi.waitFor(() =>
      expect(actionMocks.selectItemRange).toHaveBeenLastCalledWith(
        [toGalleryItemRef(lastInPage), toGalleryItemRef(firstNextPage)],
        firstNextPage,
        1
      )
    );
    expect(document.activeElement).toBe(getButton('Select range-page-1-first.png for preview'));
  });

  it.each(['backend-ordered names', 'materialized fallback', 'error fallback'] as const)(
    'cancels a hydrated range when focus moves during the %s path',
    async (resolutionPath) => {
      const previousItem = createItem('image', 'hydrated-cancel-page-0-previous.png');
      const anchorItem = createItem('image', 'hydrated-cancel-page-0-anchor.png');
      const pendingItem = createItem('image', 'hydrated-cancel-page-1-pending.png');
      let resolveNames: ((value: { items: GalleryItemRef[]; total: number }) => void) | null = null;
      let rejectNames: ((reason?: unknown) => void) | null = null;
      mocks.fetchNames.mockReturnValue(
        new Promise((resolve, reject) => {
          resolveNames = resolve;
          rejectNames = reject;
        })
      );
      currentSparseListing = {
        itemSlots: new Map([
          [58, previousItem],
          [59, anchorItem],
        ]),
        pageStates: new Map([[0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }]]),
        recentItems: [],
        total: 120,
      };

      await renderGallery(
        createGallery({
          items: [previousItem, anchorItem],
          selectedItemKey: 'image:hydrated-cancel-page-0-anchor.png',
          selectedItemKeys: ['image:hydrated-cancel-page-0-anchor.png'],
          settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
        })
      );
      await interact(() => getButton('Select hydrated-cancel-page-0-anchor.png for preview').focus());
      registeredCommands.get('gallery.extendSelectionRight')?.();

      currentSparseListing = {
        itemSlots: new Map([
          [58, previousItem],
          [59, anchorItem],
          [60, pendingItem],
        ]),
        pageStates: new Map([
          [0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
          [60, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
        ]),
        recentItems: [],
        total: 120,
      };
      await renderGallery({ ...currentGallery, items: [previousItem, anchorItem, pendingItem] });
      await vi.waitFor(() => expect(mocks.fetchNames).toHaveBeenCalledOnce());

      registeredCommands.get('gallery.moveFocusLeft')?.();
      expect(document.activeElement).toBe(getButton('Select hydrated-cancel-page-0-anchor.png for preview'));
      if (resolutionPath === 'error fallback') {
        await interact(() => rejectNames?.(new Error('Names unavailable')));
      } else {
        await interact(() =>
          resolveNames?.({
            items:
              resolutionPath === 'backend-ordered names'
                ? [toGalleryItemRef(previousItem), toGalleryItemRef(anchorItem), toGalleryItemRef(pendingItem)]
                : [],
            total: 120,
          })
        );
      }

      expect(actionMocks.selectItemRange).not.toHaveBeenCalled();
      expect(document.activeElement).toBe(getButton('Select hydrated-cancel-page-0-anchor.png for preview'));
    }
  );

  it('hands focus to the first tile in view when the focused tile leaves with its page', async () => {
    const items = Array.from({ length: 120 }, (_, index) => createItem('image', `evicted-focus-${index}.png`));
    const settled = { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) };
    currentSparseListing = {
      itemSlots: new Map(items.map((item, index) => [index, item])),
      pageStates: new Map([
        [0, settled],
        [60, settled],
      ]),
      recentItems: [],
      total: items.length,
    };
    await renderGallery(
      createGallery({
        items,
        selectedItemKey: null,
        selectedItemKeys: [],
        settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
      })
    );
    await interact(() => getButton('Select evicted-focus-11.png for preview').focus());

    const viewport = host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
    const deepTile = getButton('Select evicted-focus-71.png for preview');
    await interact(() => {
      viewport.scrollTop += deepTile.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    });
    currentSparseListing = {
      itemSlots: new Map(items.slice(60).map((item, index) => [index + 60, item])),
      pageStates: new Map([[60, settled]]),
      recentItems: [],
      total: items.length,
    };
    await renderGallery({ ...currentGallery, items: items.slice(60) });

    expect(document.activeElement).toBe(
      deepTile.closest('[data-gallery-section="regular"]')?.querySelector('button[aria-pressed]')
    );
  });

  it('bounds mounted media and keeps sparse row identity through page load, eviction, and refetch', async () => {
    mocks.useRealGridVirtualizer = true;
    const items = Array.from({ length: 120 }, (_, index) => createItem('image', `lifecycle-${index}.png`));
    const pageStates = new Map([
      [0, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
      [60, { error: null, isLoading: false, retry: vi.fn(() => Promise.resolve()) }],
    ]);
    const loadedSlots = new Map(items.map((item, index) => [index, item]));
    currentSparseListing = { itemSlots: loadedSlots, pageStates, recentItems: [], total: items.length };

    await renderGallery(createGallery({ items: [], settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' } }));

    const viewport = host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
    const gridRows = host!.querySelectorAll('[data-gallery-section="regular"]');
    expect(gridRows.length).toBeGreaterThan(0);
    expect(gridRows.length).toBeLessThan(Math.ceil(items.length / 2));
    expect(host!.querySelectorAll('img').length).toBeLessThan(items.length);

    const target = [...host!.querySelectorAll<HTMLButtonElement>('button[aria-label^="Select lifecycle-"]')].find(
      (button) => {
        const rect = button.getBoundingClientRect();
        const viewportRect = viewport.getBoundingClientRect();
        return rect.top >= viewportRect.top && rect.bottom <= viewportRect.bottom;
      }
    );
    expect(target).toBeDefined();
    const targetRow = target!.closest<HTMLElement>('[data-gallery-section="regular"]')!;
    const targetName = target!.querySelector('img')?.getAttribute('alt');

    currentSparseListing = { itemSlots: new Map(), pageStates, recentItems: [], total: items.length };
    await renderGallery({ ...currentGallery });

    expect(targetRow.isConnected).toBe(true);
    expect(host?.querySelectorAll('[data-gallery-slot-state="empty"]').length).toBeGreaterThan(0);
    expect(host?.querySelectorAll('img')).toHaveLength(0);

    currentSparseListing = { itemSlots: loadedSlots, pageStates, recentItems: [], total: items.length };
    await renderGallery({ ...currentGallery });

    expect(targetRow.isConnected).toBe(true);
    expect(targetRow.querySelector(`img[alt="${targetName}"]`)).not.toBeNull();

    const getAnchor = () =>
      host!.querySelector<HTMLImageElement>(`img[alt="${targetName}"]`)!.closest<HTMLButtonElement>('button')!;
    const anchorOffset = () => getAnchor().getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    const columns = () =>
      getComputedStyle(getAnchor().closest<HTMLElement>('[data-gallery-section="regular"]')!)
        .gridTemplateColumns.trim()
        .split(/\s+/).length;
    const initialOffset = anchorOffset();
    const initialColumns = columns();

    host!.style.width = '900px';
    await vi.waitFor(() => expect(columns()).toBeGreaterThan(initialColumns));
    expect(anchorOffset()).toBeCloseTo(initialOffset, 0);

    const resizedOffset = anchorOffset();
    const resizedColumns = columns();
    await renderGallery({
      ...currentGallery,
      settings: { ...currentGallery.settings, imageDensityPercent: 75 },
    });
    await vi.waitFor(() => expect(columns()).not.toBe(resizedColumns));
    expect(anchorOffset()).toBeCloseTo(resizedOffset, 0);

    const densityOffset = anchorOffset();
    host!.style.zoom = '1.2';
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
    expect(getAnchor().isConnected).toBe(true);
    expect(anchorOffset()).toBeCloseTo(densityOffset, 0);
  });

  it('repairs offsets after a partial delete and keeps the visible item anchored through a failed page retry', async () => {
    const items = Array.from({ length: 96 }, (_, index) => createItem('image', `mutation-item-${index}.png`));
    const gallery = createGallery({
      items,
      selectedItemKey: 'image:mutation-item-12.png',
      selectedItemKeys: ['image:mutation-item-12.png'],
      settings: { ...DENSE_SETTINGS, paginationMode: 'infinite' },
    });
    const filter = createFilter(gallery);
    const pageOptions = galleryItemsPageOptions(filter, 0);
    const laterPageOptions = galleryItemsPageOptions(filter, 60);
    const boardOptions = galleryBoardsOptions(getGalleryListingBoardsQuery(gallery.settings));
    const successfulDelete = { kind: 'image' as const, name: items[3]!.name };
    const failedDelete = { kind: 'image' as const, name: items[8]!.name };
    const repairedItems = items.filter((item) => item.name !== successfulDelete.name);
    queryClient!.setQueryData(boardOptions.queryKey, [board]);
    queryClient!.setQueryData<GalleryItemsPage>(pageOptions.queryKey, {
      items: items.slice(0, 60),
      itemIndices: Array.from({ length: 60 }, (_, index) => index),
      offset: 0,
      total: items.length,
    });
    queryClient!.setQueryData<GalleryItemsPage>(laterPageOptions.queryKey, {
      items: items.slice(60),
      itemIndices: Array.from({ length: 36 }, (_, index) => index + 60),
      offset: 60,
      total: items.length,
    });
    mocks.fetchSparsePage
      .mockRejectedValueOnce(new Error('Page repair unavailable'))
      .mockResolvedValueOnce({
        items: repairedItems.slice(0, 60),
        itemIndices: Array.from({ length: 60 }, (_, index) => index),
        offset: 0,
        total: repairedItems.length,
      })
      .mockResolvedValueOnce({
        items: repairedItems.slice(60),
        itemIndices: Array.from({ length: 35 }, (_, index) => index + 60),
        offset: 60,
        total: repairedItems.length,
      });

    await renderQueryBackedGallery(gallery);

    const viewport = host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
    viewport.scrollTop = 1_000;
    const initialOptions = mocks.virtualizerOptions.at(-1)!;
    await interact(() =>
      initialOptions.onChange?.({
        getVirtualItems: () => [{ index: 4 }, { index: 5 }],
        range: { endIndex: 5, startIndex: 4 },
      })
    );
    const anchor = host!.querySelector<HTMLButtonElement>(
      'button[aria-label="Select mutation-item-12.png for preview"]'
    )!;
    const beforeOffset = anchor.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    let rollback: () => void = () => undefined;
    const cachedPage = (offset = 0) =>
      queryClient!.getQueryData<GalleryItemsPage>((offset === 0 ? pageOptions : laterPageOptions).queryKey)!;

    expect(cachedPage().itemIndices).toEqual(Array.from({ length: 60 }, (_, index) => index));
    expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 36 }, (_, index) => index + 60));

    await interact(() => {
      rollback = patchGalleryItemCaches(queryClient!, {
        kind: 'delete',
        result: { failed: [], succeeded: [successfulDelete, failedDelete] },
      });
    });
    expect(queryClient!.getQueryData<GalleryItemsPage>(pageOptions.queryKey)?.total).toBe(94);
    expect(cachedPage().itemIndices).toEqual([
      0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    ]);
    expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 36 }, (_, index) => index + 60));
    expect(host?.querySelector('img[alt="mutation-item-3.png"]')).toBeNull();
    expect(host?.querySelector('img[alt="mutation-item-8.png"]')).toBeNull();

    await interact(() => {
      rollback();
      expect(cachedPage().itemIndices).toEqual(Array.from({ length: 60 }, (_, index) => index));
      expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 36 }, (_, index) => index + 60));
      patchGalleryItemCaches(queryClient!, {
        kind: 'delete',
        result: { failed: [failedDelete], succeeded: [successfulDelete] },
      });
    });
    expect(queryClient!.getQueryData<GalleryItemsPage>(pageOptions.queryKey)?.total).toBe(95);
    expect(cachedPage().itemIndices).toEqual([
      0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    ]);
    expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 36 }, (_, index) => index + 60));
    expect(host?.querySelector('img[alt="mutation-item-3.png"]')).toBeNull();
    expect(host?.querySelector('img[alt="mutation-item-8.png"]')).not.toBeNull();

    await act(async () => {
      await invalidateGalleryItems(queryClient!);
    });
    expect(mocks.fetchSparsePage).toHaveBeenCalledTimes(1);
    expect(mocks.fetchSparsePage.mock.calls.map(([, offset]) => offset)).toEqual([0]);
    await vi.waitFor(() => expect(host?.querySelectorAll('[data-gallery-slot-state="error"] button')).toHaveLength(1));
    const retryOffset =
      host!
        .querySelector<HTMLButtonElement>('button[aria-label="Select mutation-item-12.png for preview"]')!
        .getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    expect(Math.abs(retryOffset - beforeOffset)).toBeLessThan(1);

    await click(host!.querySelector<HTMLButtonElement>('[data-gallery-slot-state="error"] button')!);
    await vi.waitFor(() => {
      expect(mocks.fetchSparsePage).toHaveBeenCalledTimes(2);
      expect(host?.querySelector('[data-gallery-slot-state="error"]')).toBeNull();
    });

    const repairedAnchor = host!.querySelector<HTMLButtonElement>(
      'button[aria-label="Select mutation-item-12.png for preview"]'
    )!;
    const repairedOffset = repairedAnchor.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    expect(Math.abs(repairedOffset - beforeOffset)).toBeLessThan(1);
    expect(viewport.scrollTop).toBeLessThan(1_000);
    expect(host?.querySelector('img[alt="mutation-item-8.png"]')).not.toBeNull();
    expect(queryClient!.getQueryData<GalleryItemsPage>(pageOptions.queryKey)?.total).toBe(95);
    expect(cachedPage().itemIndices).toEqual(Array.from({ length: 60 }, (_, index) => index));
    expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 36 }, (_, index) => index + 60));
    await act(async () => {
      await queryClient!.fetchQuery(laterPageOptions);
    });
    expect(mocks.fetchSparsePage.mock.calls.map(([, offset]) => offset)).toEqual([0, 0, 60]);
    expect(cachedPage(60).items.map((item) => item.name)).toEqual(repairedItems.slice(60).map((item) => item.name));
    expect(cachedPage(60).itemIndices).toEqual(Array.from({ length: 35 }, (_, index) => index + 60));
  });
});

describe('shared gallery progress section', () => {
  const session: QueueProgressSession = {
    id: 'run:1',
    queueItemId: 'run',
    itemIndex: 1,
    backendItemId: 10,
    label: 'Workflow A',
    sourceId: 'workflow',
    width: 512,
    height: 768,
    itemCount: 1,
    state: 'running',
  };
  it('stays visible above empty and filtered boards and follows the clicked session', async () => {
    currentProgressSessions = [session];
    const cases: [Partial<GalleryStateView>, GalleryListingState['status']][] = [
      [{ selectedBoardId: 'board-other', items: [] }, 'empty'],
      [{ searchTerm: 'unmatched', items: [] }, 'empty'],
      [{ starredOnly: true, items: [] }, 'empty'],
      [{ galleryView: 'assets', items: [] }, 'loading'],
      [{ items: [] }, 'error'],
      [{ page: 5, anchoredWindowPage: 5 }, 'ready'],
    ];

    for (const [boardState, status] of cases) {
      currentListing = { ...READY_LISTING, status };
      await renderGallery(createGallery(boardState));
      expect(host?.querySelector('button[aria-label^="Workflow A ·"]')).not.toBeNull();
    }
    await click(host!.querySelector<HTMLButtonElement>('button[aria-label^="Workflow A ·"]')!);
    expect(followProgressSession).toHaveBeenCalledWith('run:1', { revealPreview: true });
    expect(host?.querySelectorAll('[role="listitem"]')).toHaveLength(currentGallery.items.length);
  });
  it('bounds mounted tiles for large batches and unmounts them when collapsed', async () => {
    currentProgressSessions = Array.from({ length: 1000 }, (_, index) => ({
      ...session,
      id: `run:${index + 1}`,
      itemCount: 1000,
      itemIndex: index + 1,
      state: 'queued',
    }));
    await renderGallery();
    const tiles = () => host!.querySelectorAll('[aria-label="In progress"] button[aria-pressed]');
    await expect.poll(() => tiles().length).toBeGreaterThan(0);
    expect(tiles().length).toBeLessThan(60);
    const contentId = host!.querySelector('[data-progress-disclosure]')!.getAttribute('aria-controls')!;
    const viewport = document.getElementById(contentId)!.closest<HTMLElement>('[data-part="viewport"]')!;
    await act(() => {
      viewport.scrollTop = viewport.scrollHeight;
    });
    await expect
      .poll(() => host!.querySelector('button[aria-label="Workflow A · 1000 of 1000 · Queued"]'))
      .not.toBeNull();
    expect(tiles().length).toBeLessThan(60);
    await renderGallery(createGallery({ settings: { ...DENSE_SETTINGS, progressSectionCollapsed: true } }));
    expect(tiles()).toHaveLength(0);
  });
  it('tabs through running previews and skips queued and settling tiles', async () => {
    currentProgressSessions = [
      { ...session, id: 'run:2', state: 'queued' },
      session,
      { ...session, id: 'run:3', state: 'settling' },
    ];
    await renderGallery();
    host!.querySelector<HTMLButtonElement>('[data-progress-disclosure]')!.focus();
    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(host!.querySelector('button[aria-label$="Preparing"]'));
    await act(() => userEvent.tab());
    expect(document.activeElement?.closest('[role="listitem"]')).not.toBeNull();
    for (const tile of host!.querySelectorAll<HTMLButtonElement>(
      '[aria-label="In progress"] button[aria-disabled="true"]'
    )) {
      expect(tile.tabIndex).toBe(-1);
    }
  });
});

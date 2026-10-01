/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
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
import { requestGalleryItemReveal } from '@features/gallery/core/selection';
import { getGallerySettings } from '@features/gallery/core/settings';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { GALLERY_PINNED_FOOTER_PX } from '@features/gallery/ui/galleryGridLayout';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { getFollowedProgressSession } from '@features/queue/contracts';
import { parseDateTokens } from '@platform/search/dateTokens';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getContrastRatio } from '@platform/ui/theme/contrastRatio.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { workbenchAutoScroll } from '@workbench/widgetDnd';
import { PreviewFilmstrip } from '@workbench/widgets/preview/PreviewFilmstrip';
import { PreviewFrame } from '@workbench/widgets/preview/PreviewFrame';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryActions, GalleryStarredStrip, GalleryWidgetContextValue } from './GalleryWidgetContext';

import { mergeGalleryLoadedItems } from './galleryGridLayout';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryWidgetContext } from './GalleryWidgetContext';
import { EMPTY_GALLERY_STARRED_STRIP } from './useGalleryStarredStrip';

const mocks = vi.hoisted(() => ({
  itemProgress: null as { percentage: number; message: string } | null,
  progressFrame: null as { dataUrl: string; width: number; height: number } | null,
  fetchNames: vi.fn(),
  getItemLabel: vi.fn<GalleryUiAdapter['getItemLabel']>(),
  indexAvailability: { modelName: null, state: 'disabled' } as ImageIndexAvailability,
  measure: vi.fn(),
  scrollToIndex: vi.fn(),
  setPage: vi.fn(),
  virtualizerOptions: [] as Array<{
    count: number;
    estimateSize: (index: number) => number;
    getScrollElement: () => Element | null;
    overscan: number;
  }>,
}));

const getNamesKey = (filter: unknown) => ['test-gallery-item-names', JSON.stringify(filter)] as const;

vi.mock('@features/gallery/data/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  imageIndexAvailabilityOptions: () => ({
    queryFn: () => mocks.indexAvailability,
    queryKey: ['test-image-index-availability'],
  }),
  galleryItemNamesOptions: (filter: unknown) => ({
    queryFn: () => mocks.fetchNames(filter),
    queryKey: getNamesKey(filter),
    staleTime: Infinity,
  }),
}));

vi.mock('react-hook-tanstack-virtual', async (importOriginal) => {
  const actual = await importOriginal<typeof VirtualModule>();
  return {
    useVirtualizer: (options: {
      count: number;
      horizontal?: boolean;
      scrollMargin?: number;
      estimateSize: (index: number) => number;
      getScrollElement: () => Element | null;
      overscan: number;
    }) => {
      if (options.overscan === 2) {
        return actual.useVirtualizer(options);
      }
      mocks.virtualizerOptions.push(options);
      const sizes = Array.from({ length: options.count }, (_, index) => options.estimateSize(index));
      const starts = sizes.map(
        (_, index) => (options.scrollMargin ?? 0) + sizes.slice(0, index).reduce((total, size) => total + size, 0)
      );

      return {
        measure: mocks.measure,
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
        common: { generating: 'Generating' },
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
            noStarredItemsMatch: 'No starred items',
            showAllStarred: 'Show all',
            showAllStarredItems: 'Show all starred items',
            collapseStarredItems: 'Collapse starred items',
            dropMediaToUploadToBoard: 'Drop media to {{name}}',
            emptyBoardUploadHint: 'Drop media here or click to upload',
            expandStarredItems: 'Expand starred items',
            selectImageForPreview: 'Select {{name}} for preview',
            selectVideoForPreview: 'Select video {{name}}, duration {{duration}}, for preview',
            selectedBoardFallback: 'selected board',
            starImage: 'Star {{name}}',
            starredItems: 'Starred',
            unstarImage: 'Unstar {{name}}',
            uncategorized: 'Uncategorized',
            windowLimit: 'Limited to {{count}}',
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
  projectId: null,
  videoCount: 1,
} as const;

/** Mirrors how `useGalleryData` derives the filter the widget publishes on context. */
const createFilter = (gallery: GalleryStateView): GalleryItemsFilter => {
  const parse = parseDateTokens(gallery.searchTerm);

  return {
    boardId: gallery.selectedBoardId,
    ...(parse.range?.from ? { createdFrom: parse.range.from } : {}),
    ...(parse.range?.to ? { createdTo: parse.range.to } : {}),
    galleryView: gallery.galleryView,
    orderDir: gallery.settings.imageOrderDir,
    searchTerm: parse.text,
  };
};

/** Infinite-mode settings at the sparsest density, so few columns fit the harness. */
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
    isLoading: false,
    items,
    page: 0,
    revealTargetPage: null,
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: board.id,
    selectedItemKey: 'image:first.png',
    selectedItemKeys: ['image:first.png'],
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
  hotkeys: { register: () => () => undefined },
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
  itemRefs?: GalleryItemRef[];
  items: GalleryItem[];
  x: number;
  y: number;
} | null;
const ContextMenuProbe = ({ target }: { target: CanonicalContextTarget }) => (
  <output data-testid="context-target">
    {JSON.stringify(
      target
        ? {
            itemRefs: target.itemRefs ?? null,
            items: target.items.map(({ kind, name }) => ({ kind, name })),
          }
        : null
    )}
  </output>
);
const NoopProvider = ({ children }: { children: ReactNode }) => children;

// Live-follow state travels as arguments so the compiler's memoization sees it change.
const createAdapter = (
  progressSessions: QueueProgressSession[],
  liveFollowEnabled: boolean,
  pinnedProgressSessionId: string | null
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
    galleryValues: {},
    generateValues: {},
    getItemLabel: mocks.getItemLabel,
    liveFollowEnabled,
    progressSessions,
    pinnedProgressSessionId,
    followedProgressSessionId: liveFollowEnabled
      ? (getFollowedProgressSession(progressSessions, pinnedProgressSessionId)?.id ?? null)
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
let currentLiveFollowEnabled = false;
let currentPinnedSessionId: string | null = null;
let currentProgressSessions: QueueProgressSession[] = [];
const followProgressSession = vi.fn();
let currentStrip: GalleryStarredStrip = EMPTY_GALLERY_STARRED_STRIP;
let onDragStart = vi.fn();

/** The strip the next renders show; `total` defaults to the item count. */
const setStrip = (items: GalleryItem[], total = items.length) => {
  currentStrip = { items, total };
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
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const contextValue: GalleryWidgetContextValue = {
    actions: createActions(),
    filter: createFilter(gallery),
    gallery,
    itemActions: imageActionMocks,
    isWindowTruncated: false,
    loadedItems: mergeGalleryLoadedItems(currentStrip.items, gallery.items),
    projectName: 'Project',
    region: 'right',
    runtime,
    starredStrip: currentStrip,
  } as unknown as GalleryWidgetContextValue;

  return (
    <I18nextProvider i18n={i18n}>
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <GalleryUiProvider adapter={createAdapter(progressSessions, liveFollowEnabled, pinnedSessionId)}>
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

const getButton = (label: string): HTMLButtonElement => {
  const button = host?.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

  if (!button) {
    throw new Error(`Expected button "${label}"`);
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
  // Drain singleton reveal intent with an unmatchable request so later tests cannot adopt it.
  requestGalleryItemReveal('image:__drained__');
  accountLifecycle.activate('grid-user');
  vi.clearAllMocks();
  registeredCommands.clear();
  currentGallery = createGallery();
  mocks.itemProgress = null;
  mocks.indexAvailability = { modelName: null, state: 'disabled' };
  mocks.getItemLabel.mockReset();
  currentProgressSessions = [];
  currentLiveFollowEnabled = false;
  currentPinnedSessionId = null;
  mocks.progressFrame = null;
  currentStrip = EMPTY_GALLERY_STARRED_STRIP;
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  host.style.cssText = 'height:480px;left:20px;position:fixed;top:20px;width:600px;';
  document.body.append(host);
  root = createRoot(host);
  onDragStart = vi.fn();
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
    await interact(() => pointer('pointerup', document, x, edgeY - 20));
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

    const trigger = getButton('Collapse starred items');
    const starredSection = host?.querySelector('[data-gallery-section="starred"]');
    const regularSection = host?.querySelector('[data-gallery-section="regular"]');

    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(trigger.textContent).toContain('Starred');
    expect(trigger.closest('[role="list"]')).toBeNull();
    expect(starredSection?.querySelector('button[aria-label="Select starred.png for preview"]')).not.toBeNull();
    expect(regularSection?.querySelector('button[aria-label="Select regular.png for preview"]')).not.toBeNull();
    expect(host?.querySelectorAll('button[aria-label="Select starred.png for preview"]')).toHaveLength(1);
    expect(sectionOrder()).toEqual(['starred', 'regular']);
    // Every starred item is on screen, so there is nothing more to show.
    expect(host?.querySelector('button[aria-label="Show all starred items"]')).toBeNull();
  });

  it('offers Show all when the board holds more starred items than the strip, switching to the starred listing', async () => {
    setStrip([starred], 7);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }));

    const trigger = getButton('Collapse starred items');
    expect([...trigger.querySelectorAll<HTMLElement>('span')].some((span) => span.textContent === '7')).toBe(true);

    await click(getButton('Show all starred items'));

    expect(actionMocks.setStarredOnly).toHaveBeenCalledExactlyOnceWith(true);
  });

  it.each(['bg', 'bg.panel'] as const)('keeps the starred count readable on the %s surface', async (background) => {
    setStrip([starred]);
    await renderGallery(
      createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }),
      false,
      background
    );

    const trigger = getButton('Collapse starred items');
    const count = [...trigger.querySelectorAll<HTMLElement>('span')].find((span) => span.textContent === '1');
    const surface = host?.querySelector<HTMLElement>('[data-testid="gallery-surface"]');

    if (!count || !surface) {
      throw new Error('Expected the starred count and gallery surface');
    }

    const countStyle = getComputedStyle(count);
    const ratio = getContrastRatio(
      countStyle.color,
      getComputedStyle(surface).backgroundColor,
      Number(countStyle.opacity)
    );

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });

  it('matches board disclosure chrome while retaining the star marker', async () => {
    setStrip([starred], 3);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }));

    const trigger = getButton('Collapse starred items');
    const header = trigger.parentElement;

    expect(header?.getBoundingClientRect().height).toBe(24);
    expect(trigger.querySelector('svg.lucide-star')).not.toBeNull();
    // The Show all control shares the row without growing it.
    expect(getButton('Show all starred items').getBoundingClientRect().height).toBeLessThanOrEqual(24);
  });

  it('pins the strip in a ruled block above the listing, open or collapsed', async () => {
    setStrip([starred]);
    const gallery = createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS });
    await renderGallery(gallery);

    const pinned = host!.querySelector<HTMLElement>('[data-gallery-pinned]')!;
    const listingTop = () => host!.querySelector('[data-gallery-section="regular"]')!.getBoundingClientRect().top;
    const headerRect = getButton('Collapse starred items').parentElement?.getBoundingClientRect();
    const starredRect = getButton('Select starred.png for preview').getBoundingClientRect();

    expect(pinned.contains(getButton('Collapse starred items'))).toBe(true);
    expect(pinned.contains(getButton('Select starred.png for preview'))).toBe(true);
    expect(pinned.contains(getButton('Select regular.png for preview'))).toBe(false);
    expect(starredRect.top - (headerRect?.bottom ?? 0)).toBeLessThan(4);
    // The block closes with a rule and a margin; the listing starts after them.
    expect(getComputedStyle(pinned).borderBottomWidth).toBe('1px');
    expect(listingTop() - pinned.getBoundingClientRect().bottom).toBeCloseTo(GALLERY_PINNED_FOOTER_PX - 1, 0);

    await click(getButton('Collapse starred items'));
    expect(actionMocks.updateSettings).toHaveBeenCalledExactlyOnceWith({ starredSectionCollapsed: true });
    await renderGallery({ ...gallery, settings: { ...DENSE_SETTINGS, starredSectionCollapsed: true } });

    const collapsedPinned = host!.querySelector<HTMLElement>('[data-gallery-pinned]')!;
    const collapsedHeaderRect = getButton('Expand starred items').parentElement?.getBoundingClientRect();

    expect(collapsedPinned.getBoundingClientRect().bottom - (collapsedHeaderRect?.bottom ?? 0)).toBeCloseTo(1, 0);
    expect(listingTop() - collapsedPinned.getBoundingClientRect().bottom).toBeCloseTo(GALLERY_PINNED_FOOTER_PX - 1, 0);
  });

  it('collapses only the strip cells, keeps the count, and omits the disclosure when the strip is empty', async () => {
    setStrip([starred], 4);
    await renderGallery(
      createGallery({
        items: [createItem('image', 'regular.png')],
        settings: { ...DENSE_SETTINGS, starredSectionCollapsed: true },
      })
    );

    const trigger = getButton('Expand starred items');

    expect(trigger.getAttribute('aria-expanded')).toBe('false');
    expect([...trigger.querySelectorAll<HTMLElement>('span')].some((span) => span.textContent === '4')).toBe(true);
    expect(host?.querySelector('button[aria-label="Select starred.png for preview"]')).toBeNull();
    expect(host?.querySelector('button[aria-label="Select regular.png for preview"]')).not.toBeNull();
    // Collapsed, nothing is shown, so Show all still leads to the rest.
    expect(host?.querySelector('button[aria-label="Show all starred items"]')).not.toBeNull();

    await click(trigger);
    expect(actionMocks.updateSettings).toHaveBeenLastCalledWith({ starredSectionCollapsed: false });

    setStrip([]);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }));

    expect(host?.querySelector('button[aria-label="Expand starred items"]')).toBeNull();
    expect(host?.querySelector('button[aria-label="Collapse starred items"]')).toBeNull();
  });

  it('shows the strip on paginated pages too, and no chrome when the strip is empty', async () => {
    setStrip([starred]);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')] }));

    expect(host?.querySelector('button[aria-label="Collapse starred items"]')).not.toBeNull();
    expect(sectionOrder()).toEqual(['starred', 'regular']);

    setStrip([]);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')] }));

    expect(host?.querySelector('button[aria-label="Collapse starred items"]')).toBeNull();
    expect(sectionOrder()).toEqual(['regular']);
  });

  it('reads a ranking that matched nothing as a search result, not an empty board', async () => {
    setStrip([]);
    await renderGallery(
      createGallery({
        items: [],
        searchTerm: '',
        semanticImageQuery: { kind: 'text', query: 'sunset' },
        semanticSearchText: 'sunset',
        settings: DENSE_SETTINGS,
      })
    );

    expect(host?.textContent).toContain('No items');
    expect(host?.textContent).not.toContain('Drop media');
  });

  it('keeps showing the strip when every item on the board is starred', async () => {
    setStrip([starred]);
    await renderGallery(createGallery({ items: [], settings: DENSE_SETTINGS }));

    expect(getButton('Collapse starred items')).not.toBeNull();
    expect(host?.querySelector('button[aria-label="Select starred.png for preview"]')).not.toBeNull();
    expect(host?.querySelector('[role="button"]')).toBeNull();
    expect(host?.textContent).not.toContain('Drop media');
  });

  it('caps the strip at three rows of the current column count', async () => {
    const starredItems = Array.from({ length: 40 }, (_, index) =>
      createItem('image', `starred-${index}.png`, { starred: true })
    );
    setStrip(starredItems, 40);
    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS }));

    const stripRows = host?.querySelectorAll('[data-gallery-section="starred"]') ?? [];
    const stripCells = host?.querySelectorAll('[data-gallery-section="starred"] [role="listitem"]').length ?? 0;
    const columnCount = stripRows[0]?.querySelectorAll('[role="listitem"]').length ?? 0;

    expect(stripRows).toHaveLength(3);
    expect(columnCount).toBeGreaterThan(0);
    expect(stripCells).toBe(3 * columnCount);
    expect(host?.querySelector('button[aria-label="Show all starred items"]')).not.toBeNull();
  });

  it('walks the arrow keys from the strip into the listing and back across the seam', async () => {
    const regular = createItem('image', 'regular.png');
    setStrip([starred]);
    await renderGallery(
      createGallery({
        items: [regular],
        selectedItemKey: 'image:starred.png',
        selectedItemKeys: ['image:starred.png'],
        settings: DENSE_SETTINGS,
      })
    );

    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);

    await renderGallery({
      ...currentGallery,
      selectedItemKey: 'image:regular.png',
      selectedItemKeys: ['image:regular.png'],
    });
    registeredCommands.get('gallery.galleryNavLeft')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(starred);
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

    // dnd-kit briefly suppresses the click following a completed drag. Let
    // that document-level guard expire before the next interaction test.
    await interact(() => pointer('pointerup', videoButton.ownerDocument, 120, 80), 300);
  });

  it.each([
    { key: '{Enter}', label: 'Enter' },
    { key: ' ', label: 'Space' },
  ])('opens a video with $label without activating keyboard DnD', async ({ key }) => {
    const video = createItem('video', 'keyboard.mp4');

    await renderGallery(createGallery({ items: [video] }));
    const videoButton = getButton('Select video keyboard.mp4, duration 1:06, for preview');

    await interact(() => videoButton.focus());
    await act(() => userEvent.keyboard(key));

    expect(actionMocks.selectItem).toHaveBeenCalledWith(video);
    expect(onDragStart).not.toHaveBeenCalled();
  });

  it.each([
    { chord: '{Control>}{Enter}{/Control}', modifier: 'ctrlKey' },
    { chord: '{Meta>}{Enter}{/Meta}', modifier: 'metaKey' },
  ])(
    'lets $modifier+Enter reach app hotkeys from a focused tile without activating it or starting a drag',
    async ({ chord, modifier }) => {
      const image = createItem('image', 'chord.png');
      const reachedWindow = vi.fn();
      window.addEventListener('keydown', reachedWindow);

      try {
        await renderGallery(createGallery({ items: [image] }));
        const imageButton = getButton('Select chord.png for preview');

        await interact(() => imageButton.focus());
        await act(() => userEvent.keyboard(chord));

        expect(reachedWindow).toHaveBeenCalledWith(expect.objectContaining({ key: 'Enter', [modifier]: true }));
        expect(actionMocks.selectItem).not.toHaveBeenCalled();
        expect(onDragStart).not.toHaveBeenCalled();
      } finally {
        window.removeEventListener('keydown', reachedWindow);
      }
    }
  );

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

  it('keeps grid multi-selection data when preview sources for the same image are co-mounted', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'shared'), createItem('video', 'selected.mp4')],
        selectedItemKey: 'image:shared',
        selectedItemKeys: ['image:shared', 'video:selected.mp4'],
      }),
      true
    );
    const imageButton = getButton('Select shared for preview');

    await interact(() => pointer('pointerdown', imageButton, 80, 80), 20);
    await interact(() => pointer('pointermove', imageButton.ownerDocument, 120, 80), 50);

    const drag = onDragStart.mock.calls[0]?.[0];
    await interact(() => pointer('pointerup', imageButton.ownerDocument, 120, 80), 300);

    expect(drag?.data).toEqual({
      items: [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'selected.mp4' },
      ],
      kind: 'gallery-item',
    });
  });

  it('keeps Alt comparison image-only and forms a one-video context target outside selection', async () => {
    const image = createItem('image', 'first.png');
    const video = createItem('video', 'clip.mp4', { durationSeconds: 2 });
    await renderGallery(
      createGallery({
        compareImageKey: 'image:compare.png',
        isComparisonActive: true,
        items: [image, video],
        selectedItemKey: 'image:first.png',
        selectedItemKeys: ['image:first.png'],
      })
    );

    await click(getButton('Select first.png for preview'), { altKey: true });
    expect(actionMocks.setCompareItem).toHaveBeenCalledWith(image);
    expect(actionMocks.selectItem).not.toHaveBeenCalled();

    const videoButton = getButton('Select video clip.mp4, duration 0:02, for preview');
    await click(videoButton, { altKey: true });
    expect(actionMocks.selectItem).toHaveBeenCalledWith(video);

    await interact(() =>
      videoButton.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 23, clientY: 41 })
      )
    );
    expect(host?.querySelector('[data-testid="context-target"]')?.textContent).toBe(
      JSON.stringify({
        itemRefs: [{ kind: 'video', name: 'clip.mp4' }],
        items: [{ kind: 'video', name: 'clip.mp4' }],
      })
    );
  });

  it('retains an unloaded video ref when opening the context menu inside a mixed selection', async () => {
    const image = createItem('image', 'loaded.png');
    await renderGallery(
      createGallery({
        items: [image],
        selectedItemKey: 'image:loaded.png',
        selectedItemKeys: ['image:loaded.png', 'video:unloaded.mp4'],
      })
    );
    const imageButton = getButton('Select loaded.png for preview');

    await interact(() =>
      imageButton.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 23, clientY: 41 })
      )
    );

    expect(host?.querySelector('[data-testid="context-target"]')?.textContent).toBe(
      JSON.stringify({
        itemRefs: [
          { kind: 'image', name: 'loaded.png' },
          { kind: 'video', name: 'unloaded.mp4' },
        ],
        items: [{ kind: 'image', name: 'loaded.png' }],
      })
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
    expect(imageActionMocks.deleteItems).toHaveBeenCalledWith([
      { kind: 'image', name: 'shared' },
      { kind: 'video', name: 'shared' },
    ]);
    expect(imageActionMocks.setItemsStarred).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'shared' },
        { kind: 'video', name: 'shared' },
      ],
      true
    );
  });

  it('stars a video from its grid affordance through the common qualified action', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('video', 'clip.mp4')],
        selectedItemKey: 'video:clip.mp4',
        selectedItemKeys: ['video:clip.mp4'],
      })
    );

    await click(getButton('Star clip.mp4'));

    expect(imageActionMocks.setItemsStarred).toHaveBeenCalledWith([{ kind: 'video', name: 'clip.mp4' }], true);
  });
});

describe('GalleryImageGrid image-map labels', () => {
  const getTile = (name: string): HTMLElement => {
    const tile = getButton(`Select ${name} for preview`).parentElement;

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

    // Nothing is requested for tiles that were only rendered.
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

  it('reveals the label for keyboard focus as well as hover', async () => {
    mocks.getItemLabel.mockResolvedValue('sunset');
    await renderGallery(createGallery({ items: [createItem('image', 'a.png')] }));

    await interact(() => getButton('Select a.png for preview').focus());

    await vi.waitFor(() => expect(opacityOf(findBadge(getTile('a.png'), 'sunset'))).toBe('1'));
  });

  it('adds nothing for an unlabeled item', async () => {
    mocks.getItemLabel.mockResolvedValue(null);
    await renderGallery(createGallery({ items: [createItem('image', 'a.png')] }));

    await userEvent.hover(getTile('a.png'));
    await vi.waitFor(() => expect(mocks.getItemLabel).toHaveBeenCalled());
    await interact(noop);

    // Dimensions and star only.
    expect(getTile('a.png').querySelectorAll('.gallery-thumb-overlay')).toHaveLength(2);
  });

  it('never asks for labels while the image index is not ready', async () => {
    setIndexAvailability({ modelName: 'clip', state: 'model_missing' });
    mocks.getItemLabel.mockResolvedValue('sunset');
    await renderGallery(createGallery({ items: [createItem('image', 'a.png')] }));

    await userEvent.hover(getTile('a.png'));
    await interact(noop);

    expect(mocks.getItemLabel).not.toHaveBeenCalled();
    expect(findBadge(getTile('a.png'), 'sunset')).toBeUndefined();
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

  it('reuses the date board name list already in the query cache', async () => {
    const gallery = createGallery({
      boards: [{ ...board, id: 'by_date:2026-07-30', kind: 'date' }],
      items: rangeItems,
      selectedBoardId: 'by_date:2026-07-30',
    });
    const filter = {
      boardId: gallery.selectedBoardId,
      galleryView: gallery.galleryView,
      orderDir: gallery.settings.imageOrderDir,
      searchTerm: '',
    };
    queryClient?.setQueryData(getNamesKey(filter), { items: orderedRefs, total: orderedRefs.length });

    await renderGallery(gallery);
    await click(getButton('Select last.png for preview'), { shiftKey: true });

    expect(actionMocks.selectItemRange).toHaveBeenCalledWith(orderedRefs, rangeItems[2]);
    expect(mocks.fetchNames).not.toHaveBeenCalled();
  });

  it('ignores a names response after the filter identity changes', async () => {
    let resolveNames: ((value: { items: GalleryItemRef[]; total: number }) => void) | null = null;
    mocks.fetchNames.mockReturnValue(
      new Promise((resolve) => {
        resolveNames = resolve;
      })
    );
    const gallery = createGallery({ items: rangeItems });

    await renderGallery(gallery);
    await click(getButton('Select last.png for preview'), { shiftKey: true });
    await renderGallery({ ...gallery, searchTerm: 'different filter' });
    await interact(() => resolveNames?.({ items: orderedRefs, total: orderedRefs.length }));

    expect(actionMocks.selectItemRange).not.toHaveBeenCalled();
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

  it('falls back to the materialized mixed range when the names request fails', async () => {
    mocks.fetchNames.mockRejectedValue(new Error('names unavailable'));

    await renderGallery(createGallery({ items: rangeItems }));
    await click(getButton('Select last.png for preview'), { shiftKey: true });

    await vi.waitFor(() => expect(actionMocks.selectItemRange).toHaveBeenCalledWith(orderedRefs, rangeItems[2]));
  });
});

describe('GalleryImageGrid upload drop zone', () => {
  it('uses the localized label for the Uncategorized upload target', async () => {
    const uncategorizedBoard = { ...board, id: 'none', kind: 'uncategorized' as const, name: '' };

    await renderGallery(createGallery({ boards: [uncategorizedBoard], selectedBoardId: 'none' }));

    const gridRoot = host?.querySelector('[role="list"]')?.closest('[data-scope="scroll-area"]')?.parentElement;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(new File(['image'], 'image.png', { type: 'image/png' }));

    await interact(() => gridRoot?.dispatchEvent(new DragEvent('dragenter', { bubbles: true, dataTransfer })));

    expect(host?.textContent).toContain('Drop media to Uncategorized');
  });

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

    // Picker acceptance must include the video and audio formats supported by reference uploads.
    expect(accept).toEqual(expect.arrayContaining(['image/png', 'video/*', 'audio/*', '.mov', '.mkv', '.mp3', '.wav']));
  });

  // Drops bypass the accept list; downstream classification alone decides media kind.
  it('hands a dropped audio file to the upload action', async () => {
    await renderGallery(createGallery({ items: [] }));

    const dropTarget = host?.querySelector('[role="button"]');
    const dataTransfer = new DataTransfer();
    const song = new File(['audio'], 'song.mp3', { type: 'audio/mpeg' });

    dataTransfer.items.add(song);
    await interact(() => dropTarget?.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer })));

    expect(actionMocks.uploadFiles).toHaveBeenCalledWith([song]);
  });

  it('keeps the no-match message for a search with no results instead of the upload target', async () => {
    await renderGallery(createGallery({ items: [], searchTerm: 'nope' }));

    expect(host?.textContent).toContain('No items');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });

  it('keeps the no-match message for an empty virtual (date) board instead of the upload target', async () => {
    await renderGallery(
      createGallery({
        boards: [{ ...board, id: 'by_date:2026-07-30', kind: 'date' }],
        items: [],
        selectedBoardId: 'by_date:2026-07-30',
      })
    );

    expect(host?.textContent).toContain('No items');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });
});

describe('GalleryImageGrid reveal requests', () => {
  it('reports the starred empty state instead of the upload target under the starred-only listing', async () => {
    await renderGallery(createGallery({ items: [], starredOnly: true }));

    expect(host?.textContent).toContain('No starred items');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });

  it('keeps the loading message while an empty board is still loading', async () => {
    await renderGallery(createGallery({ isLoading: true, items: [] }));

    expect(host?.textContent).toContain('Loading gallery');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });

  it('never scrolls on selection changes alone', async () => {
    // Only explicit reveal intent may scroll; generation-driven selection must preserve the user's browsing
    // position.
    const gallery = createGallery();

    await renderGallery(gallery);
    await renderGallery({ ...gallery, selectedItemKey: 'image:last.png', selectedItemKeys: ['image:last.png'] });

    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });

  it('scrolls to a revealed item, and again when the same item is re-revealed', async () => {
    const gallery = createGallery();

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:first.png'));
    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);

    // Re-clicking the same map point after scrolling away must reveal again
    // even though the selection is unchanged.
    await interact(() => requestGalleryItemReveal('image:first.png'));
    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(2);
  });

  it('keeps a reveal pending until its item materializes with a late-loading page', async () => {
    const items = [createItem('image', 'first.png')];
    const gallery = createGallery({ items, selectedItemKey: null, selectedItemKeys: [] });

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:deep.png'));
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();

    // The page lands and the revealed item appears: scroll exactly then.
    await renderGallery({
      ...gallery,
      items: [...items, createItem('image', 'deep.png')],
      selectedItemKey: 'image:deep.png',
      selectedItemKeys: ['image:deep.png'],
    });
    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);
  });

  it('honors a reveal requested before this grid mounted, while its item is still selected', async () => {
    // A grid mounted by a reveal must honor the request that preceded its mount.
    await interact(() => requestGalleryItemReveal('image:last.png'));
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();

    await renderGallery(createGallery({ selectedItemKey: 'image:last.png', selectedItemKeys: ['image:last.png'] }));

    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);
  });

  it('ignores a reveal requested before mount once the selection has moved on', async () => {
    // Age is not what makes a request stale — a superseding selection is.
    await interact(() => requestGalleryItemReveal('image:last.png'));
    await renderGallery(createGallery({ selectedItemKey: 'image:first.png', selectedItemKeys: ['image:first.png'] }));

    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });

  it('reveals a starred item in the strip, and keeps the reveal pending while the strip is collapsed', async () => {
    const starred = createItem('image', 'starred.png', { starred: true });
    setStrip([starred]);
    const gallery = createGallery({
      items: [createItem('image', 'regular.png')],
      selectedItemKey: 'image:starred.png',
      selectedItemKeys: ['image:starred.png'],
      settings: { ...DENSE_SETTINGS, starredSectionCollapsed: true },
    });

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:starred.png'));
    // Loaded but row-less under the collapsed disclosure: consuming the
    // reveal here would silently drop it.
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();

    const viewport = host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
    const scrollTo = vi.spyOn(viewport, 'scrollTo');

    await renderGallery({ ...gallery, settings: DENSE_SETTINGS });
    // The strip is pinned above the listing, so the reveal scrolls to the top
    // of the viewport rather than to a listing row.
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
    expect(scrollTo).toHaveBeenCalledExactlyOnceWith({ top: 0 });
  });

  /** A persisted off-page selection of deep.png with page-zero content loaded. */
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
    await interact(() => requestGalleryItemReveal('image:deep.png'));

    expect(mocks.setPage).toHaveBeenCalledWith(2);
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();

    // The page arrives; the still-pending reveal settles by scrolling.
    await renderGallery({
      ...gallery,
      items: [createItem('image', 'deep.png')],
      page: 2,
      selectedItemKey: 'image:deep.png',
    });

    expect(mocks.scrollToIndex).toHaveBeenCalledTimes(1);
  });

  it('follows a reveal onto its page at most once, so a missing item cannot pull the user back', async () => {
    const gallery = createOffPageGallery(2);

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:deep.png'));
    expect(mocks.setPage).toHaveBeenCalledTimes(1);

    // The stamped page arrives without the item, then the user pages away.
    await renderGallery({ ...gallery, items: [createItem('image', 'page-two.png')], page: 2 });
    await renderGallery({ ...gallery, items: [createItem('image', 'page-four.png')], page: 4 });

    expect(mocks.setPage).toHaveBeenCalledTimes(1);
  });

  it('does not page-follow a selection stamped for a different listing', async () => {
    const gallery = createOffPageGallery(null);

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:deep.png'));

    expect(mocks.setPage).not.toHaveBeenCalled();
  });

  it('retires a pending reveal once the persisted selection moves to another off-page item', async () => {
    const gallery = createOffPageGallery(null);

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:deep.png'));

    // An off-page auto-select replaces the persisted selection.
    await renderGallery({
      ...gallery,
      items: [createItem('image', 'page-zero.png')],
      selectedItemKeys: ['image:fresh.png'],
    });

    // The revealed item arriving later must not scroll a retired reveal.
    await renderGallery({
      ...gallery,
      items: [createItem('image', 'deep.png')],
      selectedItemKeys: ['image:fresh.png'],
    });

    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });

  it('retires a pending reveal once a different selection lands', async () => {
    const items = [createItem('image', 'first.png')];
    const gallery = createGallery({ items, selectedItemKey: null, selectedItemKeys: [] });

    await renderGallery(gallery);
    await interact(() => requestGalleryItemReveal('image:deep.png'));

    // The user picks something else before the reveal's page arrives.
    await renderGallery({ ...gallery, selectedItemKey: 'image:first.png', selectedItemKeys: ['image:first.png'] });

    // The revealed item arriving later must not scroll away from that choice.
    await renderGallery({
      ...gallery,
      items: [...items, createItem('image', 'deep.png')],
      selectedItemKey: 'image:first.png',
      selectedItemKeys: ['image:first.png'],
    });
    expect(mocks.scrollToIndex).not.toHaveBeenCalled();
  });
});

describe('GalleryImageGrid virtualization', () => {
  it('keeps external-store option callbacks stable across equivalent renders', async () => {
    const gallery = createGallery({ items: [createItem('video', 'clip.mp4')] });

    await renderGallery(gallery);
    const firstOptions = mocks.virtualizerOptions.at(-1);
    await renderGallery({ ...gallery });
    const secondOptions = mocks.virtualizerOptions.at(-1);

    expect(secondOptions?.estimateSize).toBe(firstOptions?.estimateSize);
    expect(secondOptions?.getScrollElement).toBe(firstOptions?.getScrollElement);
  });

  it('re-measures when the row model changes without a resize, and only then', async () => {
    const starred = createItem('image', 'starred.png', { starred: true });
    setStrip([starred]);
    const gallery = createGallery({ items: [createItem('image', 'regular.png')], settings: DENSE_SETTINGS });

    await renderGallery(gallery);

    // Collapsing starred requires measurement even if visible indices stay unchanged, or stale offsets leave a
    // gap.
    mocks.measure.mockClear();
    await renderGallery({ ...gallery, settings: { ...DENSE_SETTINGS, starredSectionCollapsed: true } });
    expect(mocks.measure).toHaveBeenCalled();

    mocks.measure.mockClear();
    await renderGallery(createGallery({ items: [createItem('image', 'other.png')] }));
    expect(mocks.measure).toHaveBeenCalled();

    // An equivalent render leaves the row model alone and must not thrash.
    mocks.measure.mockClear();
    await renderGallery({ ...currentGallery });
    expect(mocks.measure).not.toHaveBeenCalled();
  });

  it('retains constant row estimates, overscan, and the near-end infinite-load trigger', async () => {
    const items = Array.from({ length: 14 }, (_, index) => createItem('image', `image-${index}.png`));

    await renderGallery(
      createGallery({
        items,
        settings: DENSE_SETTINGS,
      })
    );
    await vi.waitFor(() => expect(actionMocks.loadMore).toHaveBeenCalled());

    // Assert every cell is covered once; measured viewport width determines row count.
    const renderedRows = host?.querySelectorAll('[role="list"] [role="presentation"]').length ?? 0;
    const renderedCells = host?.querySelectorAll('[role="listitem"]').length ?? 0;

    const options = mocks.virtualizerOptions.at(-1);
    expect(options?.count).toBe(renderedRows);
    expect(renderedCells).toBe(items.length);
    expect(options?.overscan).toBe(4);
    expect(options?.estimateSize(0)).toBe(options?.estimateSize(0));
    expect(host?.querySelector('[role="listitem"]')).toHaveStyle({ aspectRatio: '1 / 1' });
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
    for (const boardState of [
      { selectedBoardId: 'board-other', items: [] },
      { searchTerm: 'unmatched', items: [] },
      { starredOnly: true, items: [] },
      { galleryView: 'assets' as const, items: [], isLoading: true },
      { page: 5, anchoredWindowPage: 5 },
    ]) {
      await renderGallery(createGallery(boardState));
      expect(host?.querySelector('button[title^="Workflow A ·"]')).not.toBeNull();
    }
    await click(host!.querySelector<HTMLButtonElement>('button[title^="Workflow A ·"]')!);
    expect(followProgressSession).toHaveBeenCalledWith('run:1', { revealPreview: true });
    expect(host?.querySelectorAll('[role="listitem"]')).toHaveLength(currentGallery.items.length);
  });
  it('steps the arrow keys between the strip, the followed tile and the listing as one sequence', async () => {
    const starred = createItem('image', 'starred.png', { starred: true });
    const regular = createItem('image', 'regular.png');
    currentProgressSessions = [session, { ...session, id: 'run:2', itemIndex: 2, backendItemId: 11 }];
    currentLiveFollowEnabled = true;
    currentPinnedSessionId = 'run:2';
    setStrip([starred]);
    // A saved selection is still there while following live; the followed tile is the cursor, not it.
    await renderGallery(
      createGallery({ items: [regular], selectedItemKey: 'image:regular.png', selectedItemKeys: ['image:regular.png'] })
    );

    // Up from the second tile lands on the strip above; right steps off the tiles into the listing.
    registeredCommands.get('gallery.galleryNavUp')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(starred);
    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);
    registeredCommands.get('gallery.galleryNavLeft')?.();
    expect(followProgressSession).toHaveBeenLastCalledWith('run:1', { revealPreview: false });

    // From the strip, down and right follow a tile; up and left have nowhere to go.
    currentLiveFollowEnabled = false;
    currentPinnedSessionId = null;
    await renderGallery(
      createGallery({ items: [regular], selectedItemKey: 'image:starred.png', selectedItemKeys: ['image:starred.png'] })
    );
    registeredCommands.get('gallery.galleryNavDown')?.();
    expect(followProgressSession).toHaveBeenLastCalledWith('run:1', { revealPreview: false });
    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(followProgressSession).toHaveBeenLastCalledWith('run:1', { revealPreview: false });
    registeredCommands.get('gallery.galleryNavUp')?.();
    registeredCommands.get('gallery.galleryNavLeft')?.();
    expect(followProgressSession).toHaveBeenCalledTimes(3);
    expect(actionMocks.selectItem).toHaveBeenCalledTimes(2);
  });
  it('skips waiting tiles, which cannot be followed, and steps past a collapsed section', async () => {
    const starred = createItem('image', 'starred.png', { starred: true });
    const regular = createItem('image', 'regular.png');
    currentProgressSessions = [
      session,
      { ...session, id: 'run:2', itemIndex: 2, backendItemId: null, state: 'queued' },
    ];
    currentLiveFollowEnabled = true;
    setStrip([starred]);
    await renderGallery(createGallery({ items: [regular], selectedItemKey: null, selectedItemKeys: [] }));

    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);
    expect(followProgressSession).not.toHaveBeenCalled();

    // A collapsed in-progress section shows no tiles, so the strip sits directly above the listing.
    currentLiveFollowEnabled = false;
    await renderGallery(
      createGallery({
        items: [regular],
        selectedItemKey: 'image:starred.png',
        selectedItemKeys: ['image:starred.png'],
        settings: { ...getGallerySettings({}), progressSectionCollapsed: true },
      })
    );
    registeredCommands.get('gallery.galleryNavDown')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);
    expect(followProgressSession).not.toHaveBeenCalled();
    expect(actionMocks.selectItem).toHaveBeenCalledTimes(2);
  });
  it('steps out of a starred selection the strip does not show instead of resetting', async () => {
    const shown = Array.from({ length: 12 }, (_, index) =>
      createItem('image', `starred-${index}.png`, { starred: true })
    );
    const hidden = createItem('image', 'starred-hidden.png', { starred: true });
    const regular = createItem('image', 'regular.png');
    setStrip([...shown, hidden], 13);
    await renderGallery(
      createGallery({
        items: [regular],
        selectedItemKey: 'image:starred-hidden.png',
        selectedItemKeys: ['image:starred-hidden.png'],
        settings: DENSE_SETTINGS,
      })
    );
    const shownCount = host!.querySelectorAll('[data-gallery-section="starred"] [role="listitem"]').length;
    expect(shownCount).toBeLessThan(13);

    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);
    registeredCommands.get('gallery.galleryNavLeft')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(shown[shownCount - 1]);

    // Under a collapsed disclosure the whole strip is hidden; the selection still steps into the listing.
    await renderGallery({ ...currentGallery, settings: { ...DENSE_SETTINGS, starredSectionCollapsed: true } });
    registeredCommands.get('gallery.galleryNavRight')?.();
    expect(actionMocks.selectItem).toHaveBeenLastCalledWith(regular);
    registeredCommands.get('gallery.galleryNavLeft')?.();
    expect(actionMocks.selectItem).toHaveBeenCalledTimes(3);
  });
  it('keeps the saved image selected while the queue only contains waiting slots', async () => {
    currentLiveFollowEnabled = true;
    currentProgressSessions = [{ ...session, state: 'queued' }];
    await renderGallery();
    const image = host!.querySelector<HTMLButtonElement>('[role="listitem"] button[aria-current="true"]');
    expect(image).not.toBeNull();
    expect(image!.getAttribute('aria-pressed')).toBe('true');
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
  it('keeps surviving tile nodes and focus through resizing, reflow and earlier completion', async () => {
    mocks.progressFrame = {
      dataUrl: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="64" height="64"/%3E',
      width: 64,
      height: 64,
    };
    currentProgressSessions = Array.from({ length: 6 }, (_, index) => ({
      ...session,
      id: `run:${index + 1}`,
      itemIndex: index + 1,
      label: `Slot ${index + 1}`,
    }));
    await renderGallery();
    const tile = host!.querySelector<HTMLButtonElement>('button[title^="Slot 4 ·"]')!;
    const image = tile.querySelector('img');
    expect(image).not.toBeNull();
    tile.focus();
    host!.style.width = '520px';
    await renderGallery(createGallery({ settings: { ...DENSE_SETTINGS, imageDensityPercent: 100 } }));
    expect(host!.querySelector('button[title^="Slot 4 ·"]')).toBe(tile);
    expect(tile.querySelector('img')).toBe(image);
    expect(document.activeElement).toBe(tile);
    currentProgressSessions = currentProgressSessions.slice(1);
    await renderGallery();
    expect(host!.querySelector('button[title^="Slot 4 ·"]')).toBe(tile);
    expect(tile.querySelector('img')).toBe(image);
    expect(document.activeElement).toBe(tile);
  });
  it('shows per-session progress and distinct queued and finishing indicators without captions', async () => {
    mocks.itemProgress = { percentage: 0.42, message: 'Denoising' };
    currentProgressSessions = [
      session,
      { ...session, id: 'run:2', state: 'queued' },
      { ...session, id: 'run:3', state: 'settling' },
    ];
    await renderGallery();
    const region = host!.querySelector('[aria-label="In progress"]')!;
    expect(region.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('42');
    expect(region.querySelector('svg[aria-label="Queued"]')).not.toBeNull();
    expect(region.querySelector('svg[aria-label="Finishing"]')).not.toBeNull();
    const finishing = region.querySelector<HTMLButtonElement>('button[aria-label$="Finishing"]')!;
    finishing.focus();
    await click(finishing);
    expect(followProgressSession).not.toHaveBeenCalled();
    expect(finishing.getAttribute('aria-disabled')).toBe('true');
  });
  it('updates tile spacing when density changes without changing the batch', async () => {
    currentProgressSessions = [session, { ...session, id: 'run:2', itemIndex: 2 }];
    for (const imageDensityPercent of [100, 0, 100]) {
      await renderGallery(createGallery({ settings: { ...DENSE_SETTINGS, imageDensityPercent } }));
      await expect.poll(() => host!.querySelectorAll('[aria-label="In progress"] button[aria-pressed]').length).toBe(2);
      const tiles = host!.querySelectorAll<HTMLElement>('[aria-label="In progress"] button[aria-pressed]');
      expect(tiles[1]!.getBoundingClientRect().left - tiles[0]!.getBoundingClientRect().right).toBeCloseTo(4, 0);
    }
  });
  it('shows all three batch tiles while only the first is executing', async () => {
    host!.style.width = '320px';
    currentProgressSessions = [
      { ...session, itemCount: 3 },
      { ...session, id: 'run:2', itemIndex: 2, itemCount: 3, backendItemId: 11, state: 'queued' },
      { ...session, id: 'run:3', itemIndex: 3, itemCount: 3, backendItemId: 12, state: 'queued' },
    ];
    await renderGallery();
    const tiles = host!.querySelectorAll<HTMLButtonElement>('[aria-label="In progress"] button[aria-pressed]');
    expect(tiles).toHaveLength(3);
    expect(tiles[2]!.getBoundingClientRect().top).toBeGreaterThan(tiles[0]!.getBoundingClientRect().top);
    expect(tiles[2]!.getBoundingClientRect().left).toBeCloseTo(tiles[0]!.getBoundingClientRect().left, 0);
    expect(tiles[0]!.disabled).toBe(false);
    for (const tile of [tiles[1]!, tiles[2]!]) {
      expect(tile.getAttribute('aria-disabled')).toBe('true');
      expect(tile.disabled).toBe(false);
      expect(tile.textContent).toBe('');
      expect(tile.getAttribute('aria-label')).toContain('Queued');
    }
  });
  it('restores focus to the current image ahead of an earlier starred thumbnail', async () => {
    currentProgressSessions = [session];
    currentLiveFollowEnabled = true;
    setStrip([createItem('image', 'starred.png', { starred: true })]);
    const gallery = createGallery({ selectedItemKey: 'image:last.png', selectedItemKeys: ['image:last.png'] });
    await renderGallery(gallery);
    host!.querySelector<HTMLButtonElement>('button[title^="Workflow A ·"]')!.focus();
    currentProgressSessions = [];
    await renderGallery(gallery);
    const current = host!.querySelector('button[aria-current="true"]');
    expect(current).not.toBeNull();
    expect(current).not.toBe(host!.querySelector('[role="listitem"] button'));
    expect(document.activeElement).toBe(current);
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
  it('collapses independently and restores focus when the last session disappears', async () => {
    currentProgressSessions = [session];
    await renderGallery();
    await click(host!.querySelector<HTMLButtonElement>('[data-progress-disclosure]')!);
    expect(actionMocks.updateSettings).toHaveBeenCalledWith({ progressSectionCollapsed: true });
    await renderGallery(createGallery({ settings: { ...DENSE_SETTINGS, progressSectionCollapsed: true } }));
    expect(host?.querySelector('[data-progress-disclosure]')?.getAttribute('aria-expanded')).toBe('false');
    await renderGallery(createGallery());
    host!.querySelector<HTMLButtonElement>('button[title^="Workflow A ·"]')!.focus();
    currentProgressSessions = [];
    mocks.progressFrame = null;
    await renderGallery();
    expect(host?.querySelector('[data-progress-disclosure]')).toBeNull();
    expect(host?.querySelector('[aria-label="In progress"]')).toBeNull();
    expect(document.activeElement?.closest('[role="listitem"]')).not.toBeNull();
  });
  it('keeps concurrent progress separate from starred and saved items at narrow widths', async () => {
    await page.viewport(760, 580);
    const imageUrl = `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96"><rect width="128" height="96" fill="#587074"/><path d="M0 96L50 25L85 60L105 35L128 96" fill="#b7cbc8"/></svg>')}`;
    const items = ['Image A', 'Image B', 'Image C'].map((name) =>
      createItem('image', name, { fullUrl: imageUrl, thumbnailUrl: imageUrl })
    );
    mocks.progressFrame = { dataUrl: imageUrl, width: 128, height: 96 };
    currentProgressSessions = [
      session,
      { ...session, id: 'run:2', itemIndex: 2, backendItemId: 11, label: 'Workflow B' },
      { ...session, id: 'run:3', backendItemId: 12, label: 'A workflow with a long descriptive name' },
    ];
    setStrip([createItem('image', 'Starred image', { starred: true, fullUrl: imageUrl, thumbnailUrl: imageUrl })]);
    host!.style.width = '320px';
    await renderGallery(createGallery({ items }));
    expect(host!.scrollWidth).toBeLessThanOrEqual(host!.clientWidth);
    expect(host!.querySelectorAll('[aria-label="In progress"] button[aria-pressed]')).toHaveLength(3);
    for (const tile of host!.querySelectorAll<HTMLElement>('[aria-label="In progress"] button[aria-pressed]')) {
      expect(tile.textContent).toBe('');
      expect(tile.getBoundingClientRect().height).toBeCloseTo(tile.getBoundingClientRect().width, 0);
      expect(tile.getAttribute('aria-label')).toContain('Preparing');
    }
    await page.screenshot({ path: '../../../../artifacts/gallery-progress/narrow.png' });
    host!.style.width = '600px';
    await renderGallery(createGallery({ items }));
    await page.screenshot({ path: '../../../../artifacts/gallery-progress/wide.png' });
  });

  it('honors the existing hidden preference without changing board images', async () => {
    currentProgressSessions = [session];
    await renderGallery(createGallery({ settings: { ...DENSE_SETTINGS, showPendingItems: false } }));
    expect(host?.querySelector('[data-progress-disclosure]')).toBeNull();
    expect(host?.querySelectorAll('[role="listitem"]')).toHaveLength(currentGallery.items.length);
  });
});

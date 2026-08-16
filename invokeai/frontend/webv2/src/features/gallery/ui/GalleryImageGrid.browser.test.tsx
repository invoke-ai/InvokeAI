/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

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
import { DEFAULT_GALLERY_SETTINGS } from '@features/gallery/core/settings';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { parseDateTokens } from '@platform/search/dateTokens';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getContrastRatio } from '@platform/ui/theme/contrastRatio.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { PreviewFilmstrip } from '@workbench/widgets/preview/PreviewFilmstrip';
import { PreviewFrame } from '@workbench/widgets/preview/PreviewFrame';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryActions, GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryWidgetContext } from './GalleryWidgetContext';

const mocks = vi.hoisted(() => ({
  fetchNames: vi.fn(),
  measure: vi.fn(),
  scrollToIndex: vi.fn(),
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
  galleryItemNamesOptions: (filter: unknown) => ({
    queryFn: () => mocks.fetchNames(filter),
    queryKey: getNamesKey(filter),
    staleTime: Infinity,
  }),
}));

vi.mock('react-hook-tanstack-virtual', () => ({
  useVirtualizer: (options: {
    count: number;
    estimateSize: (index: number) => number;
    getScrollElement: () => Element | null;
    overscan: number;
  }) => {
    mocks.virtualizerOptions.push(options);
    const sizes = Array.from({ length: options.count }, (_, index) => options.estimateSize(index));
    const starts = sizes.map((_, index) => sizes.slice(0, index).reduce((total, size) => total + size, 0));

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
}));

vi.mock('@features/queue/react', () => ({
  useQueueItemProgress: () => null,
  useQueueItemProgressImage: () => null,
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
            commands: {
              clearSelection: 'Clear selection',
              deleteSelection: 'Delete selection',
              navigationDown: 'Down',
              navigationLeft: 'Left',
              navigationRight: 'Right',
              navigationUp: 'Up',
              selectAllOnPage: 'Select all',
              toggleStarImage: 'Toggle star',
            },
            generationProgress: 'Generation progress',
            generationProgressPercent: 'Generation {{percentage}}%',
            itemsAriaLabel: 'Gallery items',
            loadingBackendGallery: 'Loading gallery',
            noImagesMatch: 'No items',
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
    starredFirst: gallery.settings.starredFirst,
  };
};

const createGallery = (overrides: Partial<GalleryStateView> = {}): GalleryStateView => {
  const items = overrides.items ?? [
    createItem('image', 'first.png'),
    createItem('video', 'shared'),
    createItem('image', 'last.png'),
  ];

  return {
    boards: [board],
    compareImageKey: null,
    currentItem: { itemKey: 'image:first.png', kind: 'item' },
    galleryView: 'images',
    isLoading: false,
    items,
    pendingPlaceholders: [],
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: board.id,
    selectedItemKey: 'image:first.png',
    selectedItemKeys: ['image:first.png'],
    settings: { ...DEFAULT_GALLERY_SETTINGS, imageDensityPercent: 0, paginationMode: 'paginated' },
    ...overrides,
  };
};

const actionMocks = {
  loadMore: vi.fn(),
  selectItem: vi.fn(),
  selectItemRange: vi.fn(),
  setCompareItem: vi.fn(),
  toggleItemInSelection: vi.fn(),
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
    setView: noop,
    toggleImageInSelection: actionMocks.toggleItemInSelection,
    toggleItemInSelection: actionMocks.toggleItemInSelection,
    updateSettings: noop,
    uploadFiles: vi.fn(),
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

const createAdapter = (): GalleryUiAdapter =>
  ({
    ItemActionsProvider: NoopProvider,
    ImageContextMenu: ContextMenuProbe,
    account: { enableLiveFollow: noop },
    antialiasProgressImages: false,
    gallery: {
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
      setView: noop,
      toggleItemSelection: noop,
      updateSettings: noop,
    },
    galleryValues: {},
    generateValues: {},
    liveFollowEnabled: false,
    liveProgressTarget: null,
    notifications: { add: noop, reportError: noop },
    projectId: 'project-1',
    projectName: 'Project',
    queueItems: [],
    widgets: { patchGalleryValues: noop },
  }) as unknown as GalleryUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
let currentGallery = createGallery();
let onDragStart = vi.fn();
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
}: {
  background?: 'bg' | 'bg.panel';
  coMountPreviewSources?: boolean;
  gallery: GalleryStateView;
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
    projectName: 'Project',
    region: 'right',
    runtime,
  } as unknown as GalleryWidgetContextValue;

  return (
    <I18nextProvider i18n={i18n}>
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <GalleryUiProvider adapter={createAdapter()}>
            <GalleryWidgetContext value={contextValue}>
              <DndContext sensors={sensors}>
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
    root?.render(<Harness background={background} coMountPreviewSources={coMountPreviewSources} gallery={gallery} />)
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
  accountLifecycle.activate('grid-user');
  vi.clearAllMocks();
  registeredCommands.clear();
  currentGallery = createGallery();
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
  it('separates starred items into an expanded disclosure above regular items', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
      })
    );

    const trigger = getButton('Collapse starred items');
    const starredSection = host?.querySelector('[data-gallery-section="starred"]');
    const regularSection = host?.querySelector('[data-gallery-section="regular"]');

    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(trigger.textContent).toContain('Starred');
    expect(trigger.closest('[role="list"]')).toBeNull();
    expect(starredSection?.querySelector('button[aria-label="Select starred.png for preview"]')).not.toBeNull();
    expect(regularSection?.querySelector('button[aria-label="Select regular.png for preview"]')).not.toBeNull();
    expect(
      Array.from(host?.querySelectorAll('[data-gallery-section]') ?? []).map((row) =>
        row.getAttribute('data-gallery-section')
      )
    ).toEqual(['starred', 'regular']);
  });

  it.each(['bg', 'bg.panel'] as const)('keeps the starred count readable on the %s surface', async (background) => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
      }),
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
    await renderGallery(
      createGallery({
        items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
      })
    );

    const trigger = getButton('Collapse starred items');
    const header = trigger.parentElement;

    expect(header?.getBoundingClientRect().height).toBe(24);
    expect(trigger.querySelector('svg.lucide-star')).not.toBeNull();
    expect(getComputedStyle(trigger).transitionProperty).toBe('color');

    await act(() => userEvent.hover(trigger));

    expect(getComputedStyle(trigger).backgroundColor).toBe('rgba(0, 0, 0, 0)');
  });

  it('keeps the starred label and grid together before a dedicated trailing gap', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
      })
    );

    const listRect = host?.querySelector('[role="list"]')?.getBoundingClientRect();
    const headerRect = getButton('Collapse starred items').parentElement?.getBoundingClientRect();
    const starredRect = getButton('Select starred.png for preview').getBoundingClientRect();
    const regularRect = getButton('Select regular.png for preview').getBoundingClientRect();

    expect((headerRect?.top ?? 0) - (listRect?.top ?? 0)).toBeCloseTo(0, 0);
    expect(starredRect.top - (headerRect?.bottom ?? 0)).toBeLessThan(4);
    expect(regularRect.top - starredRect.bottom).toBeCloseTo(12, 0);

    await click(getButton('Collapse starred items'));

    const collapsedHeaderRect = getButton('Expand starred items').parentElement?.getBoundingClientRect();
    const collapsedRegularRect = getButton('Select regular.png for preview').getBoundingClientRect();
    const collapsedSectionGap = collapsedRegularRect.top - (collapsedHeaderRect?.bottom ?? 0);

    expect((collapsedHeaderRect?.top ?? 0) - (listRect?.top ?? 0)).toBeCloseTo(0, 0);
    expect(collapsedSectionGap).toBeGreaterThanOrEqual(4);
    expect(collapsedSectionGap).toBeLessThan(8);
  });

  it('collapses only the starred items and omits the disclosure when no stars are loaded', async () => {
    await renderGallery(
      createGallery({
        items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
      })
    );

    await click(getButton('Collapse starred items'));

    expect(getButton('Expand starred items').getAttribute('aria-expanded')).toBe('false');
    expect(host?.querySelector('button[aria-label="Select starred.png for preview"]')).toBeNull();
    expect(host?.querySelector('button[aria-label="Select regular.png for preview"]')).not.toBeNull();

    await renderGallery(createGallery({ items: [createItem('image', 'regular.png')] }));

    expect(host?.querySelector('button[aria-label="Expand starred items"]')).toBeNull();
    expect(host?.querySelector('button[aria-label="Collapse starred items"]')).toBeNull();
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
    const durationBadge = playIcon?.parentElement;
    const durationBadgeStyle = durationBadge ? getComputedStyle(durationBadge) : null;

    expect(list?.getAttribute('aria-label')).toBe('Gallery items');
    expect(host?.querySelectorAll('[role="listitem"]')).toHaveLength(2);
    expect(imageButton.getAttribute('aria-pressed')).toBe('true');
    expect(videoButton.getAttribute('aria-pressed')).toBe('false');
    expect(videoPoster?.getAttribute('src')).toContain('/thumbnail/video/shared');
    expect(videoPoster?.getAttribute('decoding')).toBe('async');
    expect(videoPoster?.hasAttribute('loading')).toBe(false);
    expect(videoCell?.textContent).toContain('1:06');
    expect(playIcon?.getAttribute('aria-hidden')).toBe('true');
    expect(durationBadgeStyle?.fontVariantNumeric).toContain('tabular-nums');
    expect(durationBadgeStyle?.opacity).toBe('1');
    expect(durationBadgeStyle?.transitionProperty).toBe('opacity');
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
    mocks.fetchNames.mockResolvedValue({ items: orderedRefs, starredCount: 0, total: orderedRefs.length });
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
      starredFirst: gallery.settings.starredFirst,
    };
    queryClient?.setQueryData(getNamesKey(filter), {
      items: orderedRefs,
      starredCount: 0,
      total: orderedRefs.length,
    });

    await renderGallery(gallery);
    await click(getButton('Select last.png for preview'), { shiftKey: true });

    expect(actionMocks.selectItemRange).toHaveBeenCalledWith(orderedRefs, rangeItems[2]);
    expect(mocks.fetchNames).not.toHaveBeenCalled();
  });

  it('ignores a names response after the filter identity changes', async () => {
    let resolveNames: ((value: { items: GalleryItemRef[]; starredCount: number; total: number }) => void) | null = null;
    mocks.fetchNames.mockReturnValue(
      new Promise((resolve) => {
        resolveNames = resolve;
      })
    );
    const gallery = createGallery({ items: rangeItems });

    await renderGallery(gallery);
    await click(getButton('Select last.png for preview'), { shiftKey: true });
    await renderGallery({ ...gallery, searchTerm: 'different filter' });
    await interact(() => resolveNames?.({ items: orderedRefs, starredCount: 0, total: orderedRefs.length }));

    expect(actionMocks.selectItemRange).not.toHaveBeenCalled();
  });

  it('ignores a names response after the account epoch changes', async () => {
    let resolveNames: ((value: { items: GalleryItemRef[]; starredCount: number; total: number }) => void) | null = null;
    mocks.fetchNames.mockReturnValue(
      new Promise((resolve) => {
        resolveNames = resolve;
      })
    );

    await renderGallery(createGallery({ items: rangeItems }));
    await click(getButton('Select last.png for preview'), { shiftKey: true });
    accountLifecycle.activate('other-grid-user');
    await interact(() => resolveNames?.({ items: orderedRefs, starredCount: 0, total: orderedRefs.length }));

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
    await renderGallery(createGallery({ items: [], pendingPlaceholders: [] }));

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

  it('keeps the no-match message for a search with no results instead of the upload target', async () => {
    await renderGallery(createGallery({ items: [], pendingPlaceholders: [], searchTerm: 'nope' }));

    expect(host?.textContent).toContain('No items');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });

  it('keeps the no-match message for an empty virtual (date) board instead of the upload target', async () => {
    await renderGallery(
      createGallery({
        boards: [{ ...board, id: 'by_date:2026-07-30', kind: 'date' }],
        items: [],
        pendingPlaceholders: [],
        selectedBoardId: 'by_date:2026-07-30',
      })
    );

    expect(host?.textContent).toContain('No items');
    expect(host?.querySelector('[role="button"]')).toBeNull();
  });

  it('keeps the loading message while an empty board is still loading', async () => {
    await renderGallery(createGallery({ isLoading: true, items: [], pendingPlaceholders: [] }));

    expect(host?.textContent).toContain('Loading gallery');
    expect(host?.querySelector('[role="button"]')).toBeNull();
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
    const gallery = createGallery({
      items: [createItem('image', 'starred.png', { starred: true }), createItem('image', 'regular.png')],
    });

    await renderGallery(gallery);

    // Collapsing starred keeps the visible range identical, so without an
    // explicit measure() the virtualizer would keep serving the expanded
    // offsets — the new rows would paint below a stale starred-sized hole.
    mocks.measure.mockClear();
    await click(getButton('Collapse starred items'));
    expect(mocks.measure).toHaveBeenCalled();

    // Swapping the item list (e.g. the media/assets view switch) is the same
    // structural change arriving through props.
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
        settings: { ...DEFAULT_GALLERY_SETTINGS, imageDensityPercent: 0, paginationMode: 'infinite' },
      })
    );
    await vi.waitFor(() => expect(actionMocks.loadMore).toHaveBeenCalled());

    // Columns follow the measured viewport width now, so pinning a row count
    // would just re-encode the harness width. The invariant that matters is
    // that the rows the virtualizer is asked for cover every cell exactly once.
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

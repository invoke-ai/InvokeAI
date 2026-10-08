/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryItemsPage, GalleryVideoItem } from '@features/gallery';
import type { InvocationProgressEvent, QueueItem, QueueItemStatusChangedEvent } from '@features/queue/contracts';
import type * as DeletionConfirmationModule from '@workbench/image-actions/useDeletionConfirmation';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, useSensor, useSensors, type DndContextProps } from '@dnd-kit/core';
import { requestGalleryItemReveal } from '@features/gallery/contracts';
import { createQueueCoordinator, type QueueCoordinatorBackendPort } from '@features/queue/runtime/coordinator';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { QueryClient, QueryClientProvider, type InfiniteData } from '@tanstack/react-query';
import { system } from '@theme/system';
import { HoldToDragSensor, PrimaryMouseSensor } from '@workbench/shell/holdToDragSensor';
import i18next from 'i18next';
import { act, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { LivePreviewFollowProvider, useLivePreviewFollow } from './livePreviewFollow';

const queueItem: QueueItem = {
  backendItemIds: [1],
  cancellable: true,
  id: 'queue-item-live',
  snapshot: {
    backendSubmission: { batchCount: 1, graph: { edges: [], id: 'graph-1', nodes: {} }, kind: 'workflow' },
    destination: 'gallery',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph-1', label: 'Live generation' },
    presentation: { batchCount: 1, height: 64, width: 64 },
    sourceId: 'workflow',
    submittedAt: '2026-07-21T00:00:00.000Z',
  },
  status: 'running',
};

const createImageItem = (name: string, createdAt: string): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt,
  fullUrl: `/images/${name}/full`,
  height: 720,
  isIntermediate: false,
  kind: 'image',
  name,
  sourceQueueItemId: `queue-${name}`,
  starred: false,
  thumbnailUrl: `/images/${name}/thumbnail`,
  width: 1280,
});

const createVideoItem = (name: string, createdAt: string): GalleryVideoItem => ({
  boardId: 'none',
  category: 'general',
  createdAt,
  durationSeconds: 65.1,
  fps: 23.976,
  fullUrl: `data:video/mp4;base64,${name}`,
  height: 1080,
  isIntermediate: false,
  kind: 'video',
  name,
  starred: false,
  thumbnailUrl: `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" data-name="${name}"/>`,
  width: 1920,
});

const mocks = vi.hoisted(() => {
  const recentImages = [
    {
      height: 64,
      imageName: 'newest',
      imageUrl: '/images/newest/full',
      queuedAt: '2026-07-21T00:00:00.000Z',
      sourceQueueItemId: 'queue-item-done',
      thumbnailUrl: '/images/newest/thumbnail',
      width: 64,
    },
    {
      height: 64,
      imageName: 'oldest',
      imageUrl: '/images/oldest/full',
      queuedAt: '2026-07-21T00:00:00.000Z',
      sourceQueueItemId: 'queue-item-done',
      thumbnailUrl: '/images/oldest/thumbnail',
      width: 64,
    },
  ];

  return {
    commands: {
      account: { updateProjectPreferences: vi.fn() },
      gallery: { selectImage: vi.fn(), selectItem: vi.fn(), setCompareImage: vi.fn(), setCompareItem: vi.fn() },
      notifications: { reportError: vi.fn() },
      widgets: { patchValues: vi.fn() },
    },
    project: {
      id: 'project-1',
      queue: { items: [] as unknown[] },
      settings: { antialiasProgressImages: false, showProgressImagesInViewer: false },
      widgetInstances: {
        gallery: {
          state: {
            values: {
              recentImages,
              selectedImage: { ...recentImages[0], boardId: 'none' },
              selectedImageName: 'newest',
            },
          },
          typeId: 'gallery',
        },
        preview: { state: { values: {} }, typeId: 'preview' },
      },
    },
    galleryItemFilters: [] as Array<{ boardId: string; starred?: boolean }>,
    galleryStripFetches: [] as Array<{ boardId: string; starred?: boolean }>,
    galleryStripItems: [] as Array<GalleryImageItem | GalleryVideoItem>,
    galleryItemPageOffsets: [] as number[],
    galleryItemPageQueryKeys: [] as Array<{ offset: number; queryKey: readonly unknown[] }>,
    galleryPageFetchSignals: [] as AbortSignal[],
    deferredPageFetches: new Map<
      number,
      {
        ignoreAbort?: boolean;
        promise: Promise<GalleryItemsPage>;
        reject?: (error: Error) => void;
        resolve: (page: GalleryItemsPage) => void;
      }
    >(),
    verifiedGalleryPage: null as null | { index: number; offset: number; page: GalleryItemsPage; total: number },
    verifiedGalleryPageFetches: [] as Array<{ ref: { kind: 'image' | 'video'; name: string }; signal: AbortSignal }>,
    galleryItemPages: [] as GalleryItemsPage[],
    galleryItemNames: [] as Array<{ kind: 'image' | 'video'; name: string }>,
    galleryItemNamesOptionCalls: 0,
    imageActionOptions: null as null | {
      getItemActionContext?: () => {
        getItemSelectionPage?: (item: GalleryImageItem | GalleryVideoItem) => number;
        items: Array<GalleryImageItem | GalleryVideoItem>;
        loadOrderedRefs: (signal: AbortSignal) => Promise<Array<{ kind: 'image' | 'video'; name: string }>>;
        selectedItemKey: string | null;
      };
      onImagesDeleted?: (imageNames: string[]) => void;
      requestDeletionConfirmation?: DeletionConfirmationModule.RequestDeletionConfirmation;
    },
    recentImages,
    bridgeProgressImage: null as unknown,
    runningProgressTargets: undefined as { queueItemId: string; itemIndex: number }[] | undefined,
    slotProgressImage: undefined as unknown,
    useActiveProgressTarget: vi.fn(() => null as unknown),
    useProgressImage: vi.fn(() => null as unknown),
  };
});

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectSelector: (selector: (project: typeof mocks.project) => unknown) => selector(mocks.project),
  useWidgetValuesSelector: () => ({}),
  useWorkbenchCommands: () => mocks.commands,
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: mocks.project }) }),
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));

const mockProgressTargets = () => {
  const target = mocks.useActiveProgressTarget() as { queueItemId: string; itemIndex: number } | null;

  return target ? [target] : [];
};

vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProgressTarget: () => mocks.useActiveProgressTarget(),
  useActiveProgressTargets: () => mocks.runningProgressTargets ?? mockProgressTargets(),
  useFollowedProgressTargets: () =>
    [...(mocks.runningProgressTargets ?? []), ...mockProgressTargets()].filter(
      (target, index, all) =>
        all.findIndex((other) => other.queueItemId === target.queueItemId && other.itemIndex === target.itemIndex) ===
        index
    ),
  useProgressImage: () => mocks.useProgressImage(),
  useQueueItemBridgeProgressImage: () => mocks.bridgeProgressImage,
  // The slot's own frame: derived from the "latest" mock by target unless a test overrides it.
  useQueueItemProgressImage: (queueItemId: string, itemIndex: number) => {
    if (mocks.slotProgressImage !== undefined) {
      return mocks.slotProgressImage;
    }

    const latest = mocks.useProgressImage() as { target?: { itemIndex: number; queueItemId: string } } | null;

    return latest?.target?.queueItemId === queueItemId && latest.target.itemIndex === itemIndex ? latest : null;
  },
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_MAX_ROWS: 600,
  GALLERY_PAGE_SIZE: 60,
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  flattenGalleryItemsData: (data: InfiniteData<GalleryItemsPage, number> | undefined) =>
    data?.pages.flatMap((page) => page.items) ?? [],
  galleryBoardsOptions: () => ({ queryFn: () => [], queryKey: ['test-boards'], staleTime: Infinity }),
  getGalleryListingBoardsQuery: () => ({}),
  galleryStarredStripOptions: (query: { boardId: string; starred?: boolean }) => ({
    queryFn: () => {
      mocks.galleryStripFetches.push(query);

      return Promise.resolve({ items: mocks.galleryStripItems, total: mocks.galleryStripItems.length });
    },
    queryKey: ['test-strip', query.boardId, mocks.galleryStripItems.map((item) => item.name).join(',')],
    staleTime: Infinity,
  }),
  galleryItemsInfiniteOptions: (
    query: { boardId: string; orderDir?: 'ASC' | 'DESC'; starred?: boolean },
    window: { kind: 'anchor' | 'infinite' | 'page'; offset?: number } = { kind: 'infinite' }
  ) => {
    mocks.galleryItemFilters.push(query);
    const pages = mocks.galleryItemPages.map((page) => {
      const items = page.items.filter(
        (item) => item.boardId === query.boardId && (query.starred === undefined || item.starred === query.starred)
      );

      return {
        ...page,
        items: query.orderDir === 'ASC' ? items.reverse() : items,
      };
    });
    const initialOffset = window.offset ?? 0;
    const initialPage = pages[initialOffset / 60] ?? { items: [], total: 0 };

    return {
      getNextPageParam: (_lastPage: GalleryItemsPage, _allPages: GalleryItemsPage[], lastPageParam: number) =>
        pages[lastPageParam / 60 + 1] ? lastPageParam + 60 : undefined,
      getPreviousPageParam: (_firstPage: GalleryItemsPage, _allPages: GalleryItemsPage[], firstPageParam: number) =>
        // An anchored infinite window cannot grow upward past its anchor.
        firstPageParam >= 60 && (window.kind !== 'infinite' || firstPageParam - 60 >= initialOffset)
          ? firstPageParam - 60
          : undefined,
      initialData: { pageParams: [initialOffset], pages: [initialPage] },
      initialPageParam: initialOffset,
      queryFn: ({ pageParam }: { pageParam: number }) => {
        mocks.galleryItemPageOffsets.push(pageParam);
        return Promise.resolve(pages[pageParam / 60] ?? { items: [], total: 0 });
      },
      queryKey: ['test-items', query.boardId, query.orderDir, window.kind, initialOffset],
      staleTime: Infinity,
    };
  },
  galleryItemsPageOptions: (
    query: { boardId: string; orderDir?: 'ASC' | 'DESC'; starred?: boolean },
    offset: number
  ) => {
    mocks.galleryItemFilters.push(query);
    const storedPage = mocks.galleryItemPages[offset / 60];
    const items =
      storedPage?.items.filter(
        (item) => item.boardId === query.boardId && (query.starred === undefined || item.starred === query.starred)
      ) ?? [];
    const orderedItems = query.orderDir === 'ASC' ? [...items].reverse() : items;

    const queryKey = ['test-items-page', query, offset] as const;
    mocks.galleryItemPageQueryKeys.push({ offset, queryKey });

    return {
      queryKey,
      queryFn: () => {
        mocks.galleryItemPageOffsets.push(offset);
        const deferred = mocks.deferredPageFetches.get(offset);

        if (deferred) {
          return deferred.promise;
        }

        return Promise.resolve({
          ...storedPage,
          items: orderedItems,
          offset,
          total: Math.max(storedPage?.total ?? 0, offset + orderedItems.length),
        });
      },
      staleTime: Infinity,
    };
  },
  fetchGalleryItemsPage: (
    queryClient: QueryClient,
    query: { boardId: string; orderDir?: 'ASC' | 'DESC'; starred?: boolean },
    offset: number,
    { signal }: { signal?: AbortSignal }
  ) => {
    if (signal) {
      mocks.galleryPageFetchSignals.push(signal);
    }
    const storedPage = mocks.galleryItemPages[offset / 60];
    const items =
      storedPage?.items.filter(
        (item) => item.boardId === query.boardId && (query.starred === undefined || item.starred === query.starred)
      ) ?? [];
    const orderedItems = query.orderDir === 'ASC' ? [...items].reverse() : items;

    return queryClient.fetchQuery({
      queryKey: ['test-items-page', query, offset],
      queryFn: () => {
        mocks.galleryItemPageOffsets.push(offset);
        const deferred = mocks.deferredPageFetches.get(offset);
        if (deferred && signal) {
          if (deferred.ignoreAbort) {
            return deferred.promise;
          }

          return Promise.race([
            deferred.promise,
            new Promise<never>((_resolve, reject) => {
              const abort = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
              signal.addEventListener('abort', abort, { once: true });
              if (signal.aborted) {
                abort();
              }
            }),
          ]);
        }
        return Promise.resolve({
          ...storedPage,
          items: orderedItems,
          offset,
          total: Math.max(storedPage?.total ?? 0, offset + orderedItems.length),
        });
      },
      staleTime: Infinity,
    });
  },
  fetchVerifiedGalleryItemPage: (
    _queryClient: QueryClient,
    _query: { boardId: string; orderDir?: 'ASC' | 'DESC'; starred?: boolean },
    ref: { kind: 'image' | 'video'; name: string },
    _owner: unknown,
    signal: AbortSignal
  ) => {
    mocks.verifiedGalleryPageFetches.push({ ref, signal });
    const result = mocks.verifiedGalleryPage;
    return new Promise((resolve, reject) => {
      const abort = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) {
        abort();
      } else {
        resolve(result);
      }
    });
  },
  galleryItemNamesOptions: (query: { boardId: string; starred?: boolean }) => {
    mocks.galleryItemNamesOptionCalls++;

    return {
      queryKey: ['test-item-names', query],
      queryFn: () => Promise.resolve({ items: mocks.galleryItemNames, total: mocks.galleryItemNames.length }),
      staleTime: Infinity,
    };
  },
}));

vi.mock('@features/gallery/contracts', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGalleryItemReveal: vi.fn(),
}));

vi.mock('@workbench/image-actions', async () => ({
  EMPTY_IMAGE_RECALL_CAPABILITIES: {},
  ImageContextMenu: () => null,
  RecallActionButtons: () => null,
  buildImageRecallSettings: () => ({}),
  executeImageRecall: () => {},
  getCurrentGenerateValues: () => ({}),
  getImageRecallVerb: () => ({ icon: () => null, label: '' }),
  getGalleryCanvasImportMenuItems: () => [],
  getImageContextMenuImages: () => [],
  getImageContextMenuRecallRequestKey: () => null,
  getImageRecallCapabilities: () => ({}),
  getImageRecallMessage: () => '',
  getImageRecallTitle: () => '',
  getSelectedGalleryImage: () => null,
  getSelectedGalleryImageFromValues: () => null,
  useDeletionConfirmation: (
    await vi.importActual<typeof DeletionConfirmationModule>('@workbench/image-actions/useDeletionConfirmation')
  ).useDeletionConfirmation,
  useImageActions: (options: typeof mocks.imageActionOptions) => {
    mocks.imageActionOptions = options;
    return {};
  },
}));

vi.mock('@features/generation/react', () => ({
  GenerationUiProvider: ({ children }: { children?: unknown }) => children,
  adjustFocusedPromptAttention: () => {},
  createGenerateFormValuesSelector: () => () => ({}),
  flushGenerateDrafts: () => {},
  focusPositivePrompt: () => {},
  promptHistoryNavigation: {},
  useDebouncedDraftValue: () => ({}),
  useRegisterGenerateDraftFlusher: () => {},
}));

import { PreviewWidgetView } from './PreviewWidgetView';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { countOfTotal: '{{count}} of {{total}}', generating: 'Generating' },
        widgets: {
          preview: {
            framesPerSecond: '{{count}} fps',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            videoDuration: 'Duration {{duration}}',
          },
        },
      },
    },
  },
});

const registeredCommands = new Map<string, () => void>();
const registeredHotkeys = new Map<string, readonly string[]>();
const runtime = {
  commands: {
    register: ({ handler, id }: { handler: () => void; id: string }) => {
      registeredCommands.set(id, handler);
      return () => {
        if (registeredCommands.get(id) === handler) {
          registeredCommands.delete(id);
        }
      };
    },
  },
  hotkeys: {
    register: ({ defaultKeys, id }: { defaultKeys: readonly string[]; id: string }) => {
      registeredHotkeys.set(id, defaultKeys);
      return () => {
        if (registeredHotkeys.get(id) === defaultKeys) {
          registeredHotkeys.delete(id);
        }
      };
    },
  },
  instanceId: 'preview-instance',
  workbench: { closeWidgetInstance: () => {} },
} as unknown as WidgetViewProps['runtime'];
const manifest = { id: 'preview', label: 'Preview' } as unknown as WidgetViewProps['manifest'];
const instance = { id: 'preview-instance', typeId: 'preview' } as unknown as WidgetViewProps['instance'];

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let followControls: ReturnType<typeof useLivePreviewFollow>;
const FollowProbe = () => {
  const controls = useLivePreviewFollow();
  const ref = useCallback(() => {
    followControls = controls;
  }, [controls]);
  return <span ref={ref} />;
};

// The shell's sensors, so touch on the preview arbitrates between swipe and drag as it does in the app.
const ShellDndContext = ({ children }: Pick<DndContextProps, 'children'>) => {
  const sensors = useSensors(
    useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(HoldToDragSensor)
  );

  return <DndContext sensors={sensors}>{children}</DndContext>;
};

const renderTree = async (client: QueryClient) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <ShellDndContext>
              <LivePreviewFollowProvider>
                <FollowProbe />
                <PreviewWidgetView instance={instance} manifest={manifest} region="center" runtime={runtime} />
              </LivePreviewFollowProvider>
            </ShellDndContext>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const render = async (client = new QueryClient()) => {
  host = document.createElement('div');
  host.style.cssText = 'height:320px;width:480px;';
  document.body.append(host);
  root = createRoot(host);

  queryClient = client;
  await renderTree(client);
};

// Rerender with the same query client to preserve real cache behavior; supply fresh widget-value identities.
const rerender = async () => {
  if (!queryClient) {
    throw new Error('Expected render() to have created a query client.');
  }

  await renderTree(queryClient);
};

const setGalleryValues = (patch: Record<string, unknown>) => {
  const state = mocks.project.widgetInstances.gallery.state as { values: Record<string, unknown> };

  state.values = { ...state.values, ...patch };
};

// Apply mocked selection results back to Gallery values so subsequent navigation starts from the newly selected
// item.
const commitLastSelection = async () => {
  const lastCall = mocks.commands.gallery.selectItem.mock.lastCall as
    | [GalleryImageItem, unknown, number | undefined, boolean | undefined]
    | undefined;

  if (!lastCall) {
    throw new Error('Expected a selection to commit.');
  }

  const [item, , selectionPage] = lastCall;
  const state = mocks.project.widgetInstances.gallery.state as { values: Record<string, unknown> };
  const selectedImageQuery = (state.values.selectedImageQuery ?? {}) as Record<string, unknown>;

  setGalleryValues({
    selectedImage: {
      boardId: item.boardId,
      height: item.height,
      imageName: item.name,
      imageUrl: item.fullUrl,
      queuedAt: item.createdAt,
      sourceQueueItemId: item.sourceQueueItemId,
      thumbnailUrl: item.thumbnailUrl,
      width: item.width,
    },
    selectedImageName: item.name,
    selectedImageQuery: { ...selectedImageQuery, page: selectionPage ?? selectedImageQuery.page },
  });
  await rerender();
};

const deepQuery = {
  boardId: 'none',
  galleryView: 'images',
  imageOrderDir: 'DESC',
  page: 30,
  paginationMode: 'infinite',
  searchTerm: '',
};

const legacyImage = (name: string, queuedAt: string, sourceQueueItemId = `queue-${name}`) => ({
  boardId: 'none',
  height: 64,
  imageName: name,
  imageUrl: `/images/${name}/full`,
  queuedAt,
  sourceQueueItemId,
  thumbnailUrl: `/images/${name}/thumbnail`,
  width: 64,
});

// A board whose page 30 holds `deep` (newest first) and whose page 31 holds
// `next`; every other page is empty. Enough to anchor a window at row 1800
// and to have a boundary to cross.
const deepBoardPages = (deep: GalleryImageItem[], next: GalleryImageItem[] = []) =>
  Array.from({ length: 32 }, (_unused, index) => {
    if (index === 30) {
      return { items: deep, total: 31 * 60 + deep.length + next.length };
    }

    return index === 31
      ? { items: next, total: 31 * 60 + deep.length + next.length }
      : { items: [], total: 31 * 60 + deep.length + next.length };
  });

const getBoundary = (): HTMLElement => {
  const boundary = host?.querySelector<HTMLElement>('[tabindex="0"]');

  if (!boundary) {
    throw new Error('Expected the preview keyboard boundary to be rendered.');
  }

  return boundary;
};

/** A one-finger flick across the preview image: 90px in three quick moves. */
const flickPreview = async (direction: -1 | 1) => {
  const image = host!.querySelector<HTMLImageElement>('img[alt]:not([alt=""])')!;
  const rect = image.getBoundingClientRect();
  const y = rect.top + rect.height / 2;
  let x = rect.left + rect.width / 2;
  // Stamped 16ms apart on a virtual clock: release velocity comes from event timestamps, not the runner's speed.
  let at = performance.now();
  const touch = (type: string, target: EventTarget) => {
    const event = new PointerEvent(type, {
      bubbles: true,
      button: type === 'pointermove' ? -1 : 0,
      clientX: x,
      clientY: y,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'touch',
    });

    Object.defineProperty(event, 'timeStamp', { value: (at += 16) });
    target.dispatchEvent(event);
  };
  const step = async (type: string, target: EventTarget) => {
    await act(async () => {
      touch(type, target);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 16);
      });
    });
  };

  await step('pointerdown', image);

  for (let move = 0; move < 3; move += 1) {
    x -= direction * 30;
    await step('pointermove', image.ownerDocument);
  }

  await step('pointerup', image.ownerDocument);
};

const pressArrow = async (key: 'ArrowLeft' | 'ArrowRight') => {
  await act(async () => {
    getBoundary().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key }));
    await Promise.resolve();
  });
};

beforeEach(() => {
  registeredCommands.clear();
  registeredHotkeys.clear();
  mocks.commands.account.updateProjectPreferences.mockClear();
  mocks.commands.gallery.selectImage.mockClear();
  mocks.commands.gallery.selectItem.mockClear();
  vi.mocked(requestGalleryItemReveal).mockClear();
  mocks.project.queue.items = [];
  mocks.project.settings.showProgressImagesInViewer = false;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).compareImage;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).galleryPage;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).imageOrderDir;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).paginationMode;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).semanticImageQuery;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).selectedBoardId;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).selectedImageQuery;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).starredOnly;
  mocks.project.widgetInstances.gallery.state.values.recentImages = mocks.recentImages;
  mocks.project.widgetInstances.gallery.state.values.selectedImage = {
    ...mocks.recentImages[0],
    boardId: 'none',
  };
  mocks.project.widgetInstances.gallery.state.values.selectedImageName = 'newest';
  mocks.galleryItemFilters.length = 0;
  mocks.galleryStripFetches.length = 0;
  mocks.galleryStripItems = [];
  mocks.galleryItemPageOffsets.length = 0;
  mocks.galleryItemPageQueryKeys.length = 0;
  mocks.galleryPageFetchSignals.length = 0;
  mocks.deferredPageFetches.clear();
  mocks.verifiedGalleryPage = null;
  mocks.verifiedGalleryPageFetches.length = 0;
  mocks.galleryItemNames = [];
  mocks.galleryItemNamesOptionCalls = 0;
  mocks.imageActionOptions = null;
  mocks.galleryItemPages = [
    {
      items: mocks.recentImages.map((image) => ({
        boardId: 'none',
        category: 'general' as const,
        createdAt: image.queuedAt,
        fullUrl: image.imageUrl,
        height: image.height,
        isIntermediate: false,
        kind: 'image' as const,
        name: image.imageName,
        sourceQueueItemId: image.sourceQueueItemId,
        starred: false,
        thumbnailUrl: image.thumbnailUrl,
        width: image.width,
      })),
      total: mocks.recentImages.length,
    },
  ];
  mocks.useActiveProgressTarget.mockReturnValue(null);
  mocks.useProgressImage.mockReturnValue(null);
  mocks.bridgeProgressImage = null;
  mocks.runningProgressTargets = undefined;
  mocks.slotProgressImage = undefined;
});

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

/** The filmstrip's current thumb: the one visible trace of where the navigation cursor sits. */
const selectedThumb = (): string | null | undefined =>
  host?.querySelector<HTMLButtonElement>('button[aria-current]')?.getAttribute('aria-label');

describe('preview keyboard navigation boundary', () => {
  it.each(['starred strip', 'local recent'] as const)(
    'waits for a missing preceding page before stepping to a %s item',
    async (interveningItem) => {
      const previousItems = Array.from({ length: 60 }, (_, index) =>
        createImageItem(`page-item-${index + 60}`, new Date(Date.UTC(2026, 7, 1) - (index + 60) * 1_000).toISOString())
      );
      const selected = createImageItem('page-item-120', new Date(Date.UTC(2026, 7, 1) - 120 * 1_000).toISOString());
      const starred = { ...createImageItem('starred-older', '2026-07-01T00:00:00.000Z'), starred: true };
      const recent = legacyImage('recent-newer', '2026-09-01T00:00:00.000Z');
      let resolvePrevious!: (page: GalleryItemsPage) => void;
      const previousPage = new Promise<GalleryItemsPage>((resolve) => {
        resolvePrevious = resolve;
      });

      mocks.galleryItemPages = [
        { items: [], total: 180 },
        { items: previousItems, total: 180 },
        { items: [selected], total: 180 },
      ];
      mocks.galleryStripItems = interveningItem === 'starred strip' ? [starred] : [];
      mocks.deferredPageFetches.set(60, { promise: previousPage, resolve: resolvePrevious });
      setGalleryValues({
        galleryPage: 2,
        recentImages: interveningItem === 'local recent' ? [recent] : [],
        selectedImage: selected,
        selectedImageName: selected.name,
        selectedImageQuery: { ...deepQuery, page: 2 },
      });

      await render();
      await vi.waitFor(() => expect(mocks.galleryItemPageOffsets).toContain(60));
      await pressArrow('ArrowLeft');

      expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
      expect(mocks.galleryPageFetchSignals).toHaveLength(1);

      await act(async () => {
        resolvePrevious({ items: previousItems, offset: 60, total: 180 });
        await Promise.resolve();
      });

      await vi.waitFor(() =>
        expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
          expect.objectContaining({ name: 'page-item-119' }),
          undefined,
          1,
          true
        )
      );
    }
  );

  it('retries a failed preceding page before entering a recent section', async () => {
    const previousItems = Array.from({ length: 60 }, (_, index) =>
      createImageItem(`page-item-${index + 60}`, new Date(Date.UTC(2026, 7, 1) - (index + 60) * 1_000).toISOString())
    );
    const selected = createImageItem('page-item-120', new Date(Date.UTC(2026, 7, 1) - 120 * 1_000).toISOString());
    const recent = legacyImage('recent-newer', '2026-09-01T00:00:00.000Z');
    let rejectPrevious!: (error: Error) => void;
    const failedPage = new Promise<GalleryItemsPage>((_resolve, reject) => {
      rejectPrevious = reject;
    });
    let resolveRetry!: (page: GalleryItemsPage) => void;
    const retriedPage = new Promise<GalleryItemsPage>((resolve) => {
      resolveRetry = resolve;
    });

    mocks.galleryItemPages = [
      { items: [], total: 180 },
      { items: previousItems, total: 180 },
      { items: [selected], total: 180 },
    ];
    mocks.deferredPageFetches.set(60, { promise: failedPage, reject: rejectPrevious, resolve: () => {} });
    setGalleryValues({
      galleryPage: 2,
      recentImages: [recent],
      selectedImage: selected,
      selectedImageName: selected.name,
      selectedImageQuery: { ...deepQuery, page: 2 },
    });

    await render(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
    await vi.waitFor(() => expect(mocks.galleryItemPageOffsets).toContain(60));
    await act(async () => {
      rejectPrevious(new Error('temporary page failure'));
      await Promise.resolve();
    });
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();

    mocks.deferredPageFetches.set(60, { promise: retriedPage, resolve: resolveRetry });
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    await vi.waitFor(() =>
      expect(mocks.galleryItemPageOffsets.filter((pageOffset) => pageOffset === 60)).toHaveLength(2)
    );
    await act(async () => {
      resolveRetry({ items: previousItems, offset: 60, total: 180 });
      await Promise.resolve();
    });

    await vi.waitFor(() =>
      expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
        expect.objectContaining({ name: 'page-item-119' }),
        undefined,
        1,
        true
      )
    );
  });

  it('uses refreshed absolute positions when retrying a failed page with retained data', async () => {
    const previousItems = Array.from({ length: 60 }, (_, index) =>
      createImageItem(`page-item-${index + 60}`, new Date(Date.UTC(2026, 7, 1) - (index + 60) * 1_000).toISOString())
    );
    const selected = createImageItem('page-item-120', new Date(Date.UTC(2026, 7, 1) - 120 * 1_000).toISOString());
    const recent = legacyImage('recent-newer', '2026-09-01T00:00:00.000Z');
    let rejectRefetch!: (error: Error) => void;
    const failedRefetch = new Promise<GalleryItemsPage>((_resolve, reject) => {
      rejectRefetch = reject;
    });
    let resolveRetry!: (page: GalleryItemsPage) => void;
    const retriedPage = new Promise<GalleryItemsPage>((resolve) => {
      resolveRetry = resolve;
    });
    mocks.galleryItemPages = [
      { items: [], total: 180 },
      { itemIndices: previousItems.map((_item, index) => index + 60), items: previousItems, total: 180 },
      { items: [selected], total: 180 },
    ];
    setGalleryValues({
      galleryPage: 2,
      recentImages: [recent],
      selectedImage: selected,
      selectedImageName: selected.name,
      selectedImageQuery: { ...deepQuery, page: 2 },
    });

    await render(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
    await vi.waitFor(() => expect(mocks.galleryItemPageOffsets).toContain(60));
    const previousPageQueryKey = mocks.galleryItemPageQueryKeys.find(({ offset }) => offset === 60)?.queryKey;
    expect(previousPageQueryKey).toBeDefined();
    expect(queryClient?.getQueryData<GalleryItemsPage>(previousPageQueryKey!)?.itemIndices?.at(-1)).toBe(119);

    mocks.deferredPageFetches.set(60, { promise: failedRefetch, reject: rejectRefetch, resolve: () => {} });
    await act(async () => {
      void queryClient?.invalidateQueries({ queryKey: previousPageQueryKey });
      await Promise.resolve();
    });
    await vi.waitFor(() =>
      expect(mocks.galleryItemPageOffsets.filter((pageOffset) => pageOffset === 60)).toHaveLength(2)
    );
    await act(async () => {
      rejectRefetch(new Error('temporary refresh failure'));
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(queryClient?.getQueryState(previousPageQueryKey!)?.error).toBeInstanceOf(Error));
    expect(queryClient?.getQueryData(previousPageQueryKey!)).toBeDefined();

    mocks.deferredPageFetches.set(60, { promise: retriedPage, resolve: resolveRetry });
    await pressArrow('ArrowLeft');
    await vi.waitFor(() =>
      expect(mocks.galleryItemPageOffsets.filter((pageOffset) => pageOffset === 60)).toHaveLength(3)
    );
    await act(async () => {
      resolveRetry({
        itemIndices: previousItems.map((_item, index) => (index === 59 ? 138 : index + 60)),
        items: previousItems,
        offset: 60,
        total: 180,
      });
      await Promise.resolve();
    });

    await vi.waitFor(() =>
      expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
        expect.objectContaining({ name: 'page-item-119' }),
        undefined,
        1,
        true
      )
    );
    expect(requestGalleryItemReveal).toHaveBeenLastCalledWith(expect.any(String), expect.any(AbortSignal), 138);
  });

  it('re-resolves a deleted-prefix selection by key before stepping and stamps its updated page', async () => {
    const selected = createImageItem('survivor', '2026-07-20T00:00:01.000Z');
    const neighbor = createImageItem('preceding-survivor', '2026-07-20T00:00:02.000Z');
    const resolvedPage = { items: [neighbor, selected], offset: 60, total: 62 };
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('survivor', selected.createdAt),
      selectedImageName: 'survivor',
      selectedImageQuery: { ...deepQuery, page: 5 },
    });
    mocks.galleryItemPages = Array.from({ length: 6 }, (_unused, index) => ({
      items: index === 5 ? [] : [],
      total: 62,
    }));
    mocks.verifiedGalleryPage = { index: 61, offset: 60, page: resolvedPage, total: 62 };

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.verifiedGalleryPageFetches.map(({ ref }) => ref)).toEqual([{ kind: 'image', name: 'survivor' }]);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'preceding-survivor' }),
      undefined,
      1,
      true
    );
  });

  it('re-resolves a reordered selection when its page stamp is stale although total is unchanged', async () => {
    const selected = createImageItem('reordered', '2026-07-20T00:00:01.000Z');
    const neighbor = createImageItem('reordered-neighbor', '2026-07-20T00:00:02.000Z');
    const resolvedPage = { items: [neighbor, selected], offset: 0, total: 600 };
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('reordered', selected.createdAt),
      selectedImageName: 'reordered',
      selectedImageQuery: { ...deepQuery, page: 5 },
    });
    mocks.galleryItemPages = Array.from({ length: 10 }, () => ({ items: [], total: 600 }));
    mocks.verifiedGalleryPage = { index: 1, offset: 0, page: resolvedPage, total: 600 };

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.verifiedGalleryPageFetches).toHaveLength(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'reordered-neighbor' }),
      undefined,
      0,
      true
    );
  });

  it('continues across the adjacent page when the verified selection is at a page boundary', async () => {
    const selected = createImageItem('boundary-reordered', '2026-07-20T00:00:01.000Z');
    const neighbor = createImageItem('boundary-reordered-neighbor', '2026-07-19T00:00:00.000Z');
    const resolvedPage = { items: [selected], offset: 60, total: 180 };
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('boundary-reordered', selected.createdAt),
      selectedImageName: 'boundary-reordered',
      selectedImageQuery: { ...deepQuery, page: 5 },
    });
    mocks.galleryItemPages = Array.from({ length: 6 }, (_unused, index) => ({
      items: index === 2 ? [neighbor] : [],
      total: 180,
    }));
    mocks.verifiedGalleryPage = { index: 119, offset: 60, page: resolvedPage, total: 180 };

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.verifiedGalleryPageFetches).toHaveLength(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'boundary-reordered-neighbor' }),
      undefined,
      2,
      true
    );
  });

  it('does not select a missing key when verified lookup returns no result', async () => {
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('gone', '2026-07-20T00:00:01.000Z'),
      selectedImageName: 'gone',
      selectedImageQuery: { ...deepQuery, page: 5 },
    });
    mocks.galleryItemPages = Array.from({ length: 6 }, () => ({ items: [], total: 62 }));

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.verifiedGalleryPageFetches).toHaveLength(1);
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
  });

  it('keeps date-board selections off the ordinary listing locator', async () => {
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('date-board-item', '2026-07-20T00:00:01.000Z'),
      selectedImageName: 'date-board-item',
      selectedImageQuery: { ...deepQuery, boardId: 'by_date:2026-07-20', page: 5 },
    });
    mocks.galleryItemPages = Array.from({ length: 6 }, () => ({ items: [], total: 62 }));

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.verifiedGalleryPageFetches).toHaveLength(0);
  });

  it.each([
    ['search filter', () => ({ selectedImageQuery: { ...deepQuery, page: 0, searchTerm: 'changed' } })],
    ['sort order', () => ({ selectedImageQuery: { ...deepQuery, page: 0, imageOrderDir: 'ASC' } })],
    [
      'selected item',
      () => ({
        selectedImage: legacyImage('context-changed', '2026-07-19T00:00:00.000Z'),
        selectedImageName: 'context-changed',
      }),
    ],
  ] as Array<[string, () => Record<string, unknown>]>)(
    'aborts a deferred boundary read when the %s changes and ignores its late result',
    async (_transition, getPatch) => {
      const selected = createImageItem('boundary-selected', '2026-07-20T00:00:01.000Z');
      let resolve!: (page: GalleryItemsPage) => void;
      const promise = new Promise<GalleryItemsPage>((done) => {
        resolve = done;
      });
      setGalleryValues({
        galleryPage: 0,
        recentImages: [],
        selectedImage: legacyImage('boundary-selected', selected.createdAt),
        selectedImageName: 'boundary-selected',
        selectedImageQuery: { ...deepQuery, page: 0 },
      });
      mocks.galleryItemPages = Array.from({ length: 3 }, (_unused, index) => ({
        items: index === 0 ? [selected] : [],
        total: 180,
      }));
      mocks.deferredPageFetches.set(120, { promise, resolve });

      await render();
      await pressArrow('ArrowRight');
      await expect.poll(() => mocks.galleryPageFetchSignals.length > 0).toBe(true);
      const signal = mocks.galleryPageFetchSignals.at(-1)!;
      expect(signal.aborted).toBe(false);

      setGalleryValues(getPatch());
      await rerender();
      expect(signal.aborted).toBe(true);
      resolve({ items: [createImageItem('late-boundary', '2026-07-19T00:00:00.000Z')], offset: 120, total: 180 });
      await act(() => Promise.resolve());

      expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    }
  );

  it('aborts a deferred boundary read on unmount and ignores its late result', async () => {
    const selected = createImageItem('unmount-selected', '2026-07-20T00:00:01.000Z');
    let resolve!: (page: GalleryItemsPage) => void;
    const promise = new Promise<GalleryItemsPage>((done) => {
      resolve = done;
    });
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('unmount-selected', selected.createdAt),
      selectedImageName: 'unmount-selected',
      selectedImageQuery: { ...deepQuery, page: 0 },
    });
    mocks.galleryItemPages = Array.from({ length: 3 }, (_unused, index) => ({
      items: index === 0 ? [selected] : [],
      total: 180,
    }));
    mocks.deferredPageFetches.set(120, { promise, resolve });

    await render();
    await pressArrow('ArrowRight');
    await expect.poll(() => mocks.galleryPageFetchSignals.length > 0).toBe(true);
    const signal = mocks.galleryPageFetchSignals.at(-1)!;

    await act(() => {
      root?.unmount();
    });
    expect(signal.aborted).toBe(true);
    resolve({ items: [createImageItem('late-unmount', '2026-07-19T00:00:00.000Z')], offset: 120, total: 180 });
    await act(() => Promise.resolve());

    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
  });

  it('ignores an already-resolved boundary result after the account scope changes', async () => {
    const firstScope = accountLifecycle.activate('preview-navigation-account-a');
    const selected = createImageItem('account-selected', '2026-07-20T00:00:01.000Z');
    let resolve!: (page: GalleryItemsPage) => void;
    const promise = new Promise<GalleryItemsPage>((done) => {
      resolve = done;
    });
    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('account-selected', selected.createdAt),
      selectedImageName: 'account-selected',
      selectedImageQuery: { ...deepQuery, page: 0 },
    });
    mocks.galleryItemPages = Array.from({ length: 3 }, (_unused, index) => ({
      items: index === 0 ? [selected] : [],
      total: 180,
    }));
    mocks.deferredPageFetches.set(120, { ignoreAbort: true, promise, resolve });

    await render();
    await act(() => {
      getBoundary().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowRight' }));
    });
    await expect.poll(() => mocks.galleryPageFetchSignals.length > 0).toBe(true);
    const signal = mocks.galleryPageFetchSignals.at(-1)!;
    expect(signal.aborted).toBe(false);

    await act(async () => {
      // Resolve the request first, then rotate the account before the awaiting Preview continuation runs.
      resolve({
        items: [createImageItem('late-account-boundary', '2026-07-19T00:00:00.000Z')],
        offset: 120,
        total: 180,
      });
      accountLifecycle.activate('preview-navigation-account-b');
      await new Promise<void>((done) => {
        setTimeout(() => done(), 0);
      });
    });

    expect(firstScope.signal.aborted).toBe(true);
    expect(signal.aborted).toBe(true);
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    accountLifecycle.invalidate();
  });

  it('walks the starred strip into the unstarred listing and back, as the grid lays them out', async () => {
    const starredTop = { ...createImageItem('starred-top', '2026-07-23T00:00:00.000Z'), starred: true };
    const starredNext = { ...createImageItem('starred-next', '2026-07-22T00:00:00.000Z'), starred: true };

    mocks.galleryStripItems = [starredTop, starredNext];
    setGalleryValues({
      selectedImage: { ...legacyImage('starred-next', '2026-07-22T00:00:00.000Z'), starred: true },
      selectedImageName: 'starred-next',
    });
    await render();

    // The listing stays the unstarred one; the strip supplies the starred neighbors.
    expect(mocks.galleryItemFilters.length).toBeGreaterThan(0);
    expect(mocks.galleryItemFilters.every((query) => query.starred === false)).toBe(true);
    await expect.poll(() => selectedThumb()).toBe('starred-next');
    expect(mocks.galleryStripFetches.length).toBeGreaterThan(0);

    await pressArrow('ArrowRight');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'starred-top' }),
      undefined,
      expect.any(Number),
      true
    );

    // Back from the listing's first item, left lands on the strip's last.
    setGalleryValues({ selectedImage: legacyImage('newest', '2026-07-21T00:00:00.000Z'), selectedImageName: 'newest' });
    await render();
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'starred-next' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('shows an item just starred once, in the strip, until the listing refetch drops it', async () => {
    const justStarred = { ...createImageItem('newest', '2026-07-21T00:00:00.000Z'), starred: true };

    mocks.galleryStripItems = [justStarred];
    setGalleryValues({
      recentImages: [],
      selectedImage: { ...legacyImage('newest', '2026-07-21T00:00:00.000Z'), starred: true },
      selectedImageName: 'newest',
    });
    await render();

    await expect.poll(() => selectedThumb()).toBe('newest');
    await pressArrow('ArrowRight');
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ name: 'oldest' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('keeps a starred selection beyond the strip reachable, and drops the strip under the starred filter', async () => {
    const starredTop = { ...createImageItem('starred-top', '2026-07-23T00:00:00.000Z'), starred: true };
    const starredDeep = { ...createImageItem('starred-deep', '2026-07-01T00:00:00.000Z'), starred: true };

    mocks.galleryStripItems = [starredTop];
    setGalleryValues({
      recentImages: [],
      selectedImage: { ...legacyImage('starred-deep', '2026-07-01T00:00:00.000Z'), starred: true },
      selectedImageName: 'starred-deep',
    });
    await render();
    await expect.poll(() => selectedThumb()).toBe('starred-deep');
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'starred-top' }),
      undefined,
      expect.any(Number),
      true
    );

    mocks.commands.gallery.selectItem.mockClear();
    mocks.galleryItemFilters.length = 0;
    mocks.galleryStripFetches.length = 0;
    mocks.galleryItemPages = [{ items: [starredTop, starredDeep], total: 2 }];
    setGalleryValues({
      recentImages: [],
      selectedImage: { ...legacyImage('starred-deep', '2026-07-01T00:00:00.000Z'), starred: true },
      selectedImageName: 'starred-deep',
      selectedImageQuery: { ...deepQuery, page: 0, starredOnly: true },
      starredOnly: true,
    });
    await render();
    await expect.poll(() => selectedThumb()).toBe('starred-deep');
    expect(mocks.galleryStripFetches).toHaveLength(0);
    expect(mocks.galleryItemFilters.every((query) => query.starred === true)).toBe(true);
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ name: 'starred-top' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('handles one arrow press as exactly one selection and stops propagation', async () => {
    const documentKeydown = vi.fn();
    document.addEventListener('keydown', documentKeydown);

    try {
      await render();
      await pressArrow('ArrowRight');

      expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'image', name: 'oldest' }),
        undefined,
        expect.any(Number),
        true
      );
      expect(documentKeydown).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener('keydown', documentKeydown);
    }
  });

  it('reveals each navigated item so the gallery grid can follow', async () => {
    await render();
    await pressArrow('ArrowRight');

    expect(vi.mocked(requestGalleryItemReveal)).toHaveBeenCalledWith('image:oldest', expect.any(AbortSignal), 1);
  });

  it('reveals an ordinary deep-page navigation at its absolute sparse listing index', async () => {
    const deepNewer = createImageItem('deep-newer', '2026-07-20T00:00:02.000Z');
    const deepOlder = createImageItem('deep-older', '2026-07-20T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', deepNewer.createdAt),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepNewer, deepOlder]);
    mocks.galleryItemPages[30] = { ...mocks.galleryItemPages[30]!, itemIndices: [1801, 1806] };

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(deepOlder, undefined, 30, true);
    expect(vi.mocked(requestGalleryItemReveal)).toHaveBeenCalledWith('image:deep-older', expect.any(AbortSignal), 1806);
  });

  it('keeps a just-completed batch navigable before the backend refetch lands', async () => {
    // Bridge completed batches through recentImages until stale backend listings catch up.
    mocks.project.queue.items = [];
    mocks.project.widgetInstances.gallery.state.values.recentImages = [
      {
        height: 64,
        imageName: 'batch-2',
        imageUrl: '/images/batch-2/full',
        queuedAt: '2026-07-22T00:00:02.000Z',
        sourceQueueItemId: 'queue-item-done',
        thumbnailUrl: '/images/batch-2/thumbnail',
        width: 64,
      },
      {
        height: 64,
        imageName: 'batch-1',
        imageUrl: '/images/batch-1/full',
        queuedAt: '2026-07-22T00:00:01.000Z',
        sourceQueueItemId: 'queue-item-done',
        thumbnailUrl: '/images/batch-1/thumbnail',
        width: 64,
      },
    ];
    mocks.project.widgetInstances.gallery.state.values.selectedImage = {
      boardId: 'none',
      height: 64,
      imageName: 'batch-2',
      imageUrl: '/images/batch-2/full',
      queuedAt: '2026-07-22T00:00:02.000Z',
      sourceQueueItemId: 'queue-item-done',
      thumbnailUrl: '/images/batch-2/thumbnail',
      width: 64,
    };
    mocks.project.widgetInstances.gallery.state.values.selectedImageName = 'batch-2';
    mocks.galleryItemPages = [{ items: [createImageItem('pre-batch', '2026-07-20T00:00:00.000Z')], total: 1 }];

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'batch-1' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('walks the starred-only listing the selection was made in, and keeps recents out of it', async () => {
    // Preview must follow the starred listing and exclude new unstarred generations.
    const starredNewer = { ...createImageItem('starred-newer', '2026-07-20T00:00:02.000Z'), starred: true };
    const starredOlder = { ...createImageItem('starred-older', '2026-07-20T00:00:01.000Z'), starred: true };

    setGalleryValues({
      recentImages: [legacyImage('fresh-generation', '2026-07-23T00:00:00.000Z', 'queue-item-done')],
      selectedImage: legacyImage('starred-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'starred-newer',
      selectedImageQuery: { ...deepQuery, page: 0, starredOnly: true },
      starredOnly: true,
    });
    mocks.galleryItemPages = [{ items: [starredNewer, starredOlder], total: 2 }];

    await render();

    expect(mocks.galleryItemFilters.at(-1)).toMatchObject({ boardId: 'none', starred: true });
    expect(mocks.galleryItemFilters.every((query) => query.starred === true)).toBe(true);

    await pressArrow('ArrowRight');
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'starred-older' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('keeps settled recents out of an infinite window anchored at a deep reveal', async () => {
    // Do not merge top-of-board recents into deep windows or stamp them with unrelated page positions.
    const deepNewer = createImageItem('deep-newer', '2026-07-20T00:00:02.000Z');
    const deepOlder = createImageItem('deep-older', '2026-07-20T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [legacyImage('fresh-generation', '2026-07-23T00:00:00.000Z', 'queue-item-done')],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepNewer, deepOlder]);

    await render();
    // Descending: Right steps deeper into the anchored slice...
    await pressArrow('ArrowRight');
    // ...and Left, off the top of it, must not find the recent above it.
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'deep-older' }),
      undefined,
      30,
      true
    );
  });

  it('keeps recents out of a deep window whose stamp outlives a switch to paginated mode', async () => {
    // The live setting says paginated while the stamp still describes the deep
    // infinite window Preview is querying. The window is what the list is made
    // of, so it is what the exclusion follows.
    const deepNewer = createImageItem('deep-newer', '2026-07-20T00:00:02.000Z');
    const deepOlder = createImageItem('deep-older', '2026-07-20T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      paginationMode: 'paginated',
      recentImages: [legacyImage('fresh-generation', '2026-07-23T00:00:00.000Z', 'queue-item-done')],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepNewer, deepOlder]);

    await render();
    await pressArrow('ArrowRight');
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'deep-older' }),
      undefined,
      30,
      true
    );
  });

  it('stamps the top of the listing on an in-flight image a deep window does not hold', async () => {
    // In-flight entries in a deep window must not inherit its unrelated board page.
    mocks.project.queue.items = [queueItem];
    setGalleryValues({
      galleryPage: 0,
      recentImages: [legacyImage('in-flight', '2026-07-23T00:00:00.000Z', 'queue-item-live')],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'in-flight' }),
      undefined,
      0,
      true
    );
  });

  it('stamps the top of the listing when the compare image is swapped in', async () => {
    // Unknown comparison positions must not inherit a deep window's page.
    setGalleryValues({
      compareImage: legacyImage('compare-top', '2026-07-24T00:00:00.000Z'),
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);

    await render();

    const swap = registeredCommands.get('viewer.swapImages');

    expect(swap).toBeDefined();
    await act(async () => {
      swap?.();
      await Promise.resolve();
    });

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'compare-top' }),
      undefined,
      0,
      true
    );
  });

  it('loads only selected and adjacent absolute pages while the cursor crosses a deep page boundary', async () => {
    const deepA = createImageItem('deep-a', '2026-07-20T00:00:04.000Z');
    const deepB = createImageItem('deep-b', '2026-07-20T00:00:03.000Z');
    const deepC = createImageItem('deep-c', '2026-07-20T00:00:02.000Z');
    const deepD = createImageItem('deep-d', '2026-07-20T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-b', '2026-07-20T00:00:03.000Z'),
      selectedImageName: 'deep-b',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepA, deepB], [deepC, deepD]);

    await render();
    await pressArrow('ArrowRight');
    await commitLastSelection();
    // One more step inside the new page.
    await pressArrow('ArrowRight');
    await commitLastSelection();
    await pressArrow('ArrowLeft');
    await commitLastSelection();
    await pressArrow('ArrowLeft');

    const selected = mocks.commands.gallery.selectItem.mock.calls.map(([item, , page]) => [
      (item as { name: string }).name,
      page,
    ]);

    expect(selected).toEqual([
      ['deep-c', 31],
      ['deep-d', 31],
      ['deep-c', 31],
      ['deep-b', 30],
    ]);
    expect(mocks.galleryItemPageOffsets.sort((a, b) => a - b)).toEqual([1740, 1800, 1860]);
  });

  it('moves to the top of the listing when a selection is made there from outside Preview', async () => {
    // Follow newly stamped selection pages even when query identity is unchanged; a grid click can move a deep
    // window back to zero.
    const topNewer = createImageItem('top-newer', '2026-07-22T00:00:02.000Z');
    const topOlder = createImageItem('top-older', '2026-07-22T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);
    mocks.galleryItemPages[0] = { items: [topNewer, topOlder], total: 3 };

    await render();

    expect(mocks.galleryItemPageOffsets).toContain(1800);

    mocks.galleryItemPageOffsets.length = 0;
    setGalleryValues({
      selectedImage: legacyImage('top-newer', '2026-07-22T00:00:02.000Z'),
      selectedImageName: 'top-newer',
      selectedImageQuery: { ...deepQuery, page: 0 },
    });
    await rerender();
    await pressArrow('ArrowRight');

    expect(mocks.galleryItemPageOffsets).not.toContain(1800);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'top-older' }),
      undefined,
      0,
      true
    );
  });

  it('hands image actions a page that keeps a deletion successor in the deep window', async () => {
    // Deletion successors from Preview need its own page context rather than the grid's unrelated page.
    const deepNewer = createImageItem('deep-newer', '2026-07-20T00:00:02.000Z');
    const deepOlder = createImageItem('deep-older', '2026-07-20T00:00:01.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [legacyImage('fresh-generation', '2026-07-23T00:00:00.000Z', 'queue-item-done')],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepNewer, deepOlder]);

    await render();

    await expect
      .poll(() => mocks.imageActionOptions?.getItemActionContext?.().items.map((item) => item.name))
      .toContain('deep-older');
    const context = mocks.imageActionOptions?.getItemActionContext?.();

    expect(context?.getItemSelectionPage?.(deepOlder)).toBe(30);
    // A recent is not in the window; it lives at the top of the listing.
    expect(context?.getItemSelectionPage?.(createImageItem('fresh-generation', '2026-07-23T00:00:00.000Z'))).toBe(0);
  });

  it('swaps the compare image in and back out without losing the deep window', async () => {
    // Remember comparison navigation context so swapping back restores the deep window.
    setGalleryValues({
      compareImage: legacyImage('compare-top', '2026-07-24T00:00:00.000Z'),
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);
    mocks.galleryItemPages[0] = { items: [createImageItem('compare-top', '2026-07-24T00:00:00.000Z')], total: 2 };

    await render();

    const swap = () =>
      act(async () => {
        registeredCommands.get('viewer.swapImages')?.();
        await Promise.resolve();
      });

    await swap();

    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'image', name: 'compare-top' }),
      undefined,
      0,
      true
    );

    await commitLastSelection();
    setGalleryValues({ compareImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z') });
    await rerender();
    await swap();

    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'image', name: 'deep-newer' }),
      undefined,
      30,
      true
    );
  });

  it('does not restore a remembered page into a different listing', async () => {
    // Do not reuse a comparison page after changing to another board query.
    setGalleryValues({
      compareImage: legacyImage('compare-top', '2026-07-24T00:00:00.000Z'),
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);
    mocks.galleryItemPages[0] = { items: [createImageItem('compare-top', '2026-07-24T00:00:00.000Z')], total: 2 };

    await render();

    const swap = () =>
      act(async () => {
        registeredCommands.get('viewer.swapImages')?.();
        await Promise.resolve();
      });

    await swap();
    await commitLastSelection();
    setGalleryValues({
      compareImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImage: { ...legacyImage('other-board-image', '2026-07-25T00:00:00.000Z'), boardId: 'board-b' },
      selectedImageName: 'other-board-image',
      selectedImageQuery: { ...deepQuery, boardId: 'board-b', page: 0 },
    });
    await rerender();
    await swap();

    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'image', name: 'deep-newer' }),
      undefined,
      0,
      true
    );
  });

  it('only restores a remembered page for the item it was remembered for', async () => {
    setGalleryValues({
      compareImage: legacyImage('compare-top', '2026-07-24T00:00:00.000Z'),
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);
    mocks.galleryItemPages[0] = { items: [createImageItem('compare-top', '2026-07-24T00:00:00.000Z')], total: 2 };

    await render();

    const swap = () =>
      act(async () => {
        registeredCommands.get('viewer.swapImages')?.();
        await Promise.resolve();
      });

    await swap();
    await commitLastSelection();
    // A different image lands in the compare slot before the swap back.
    setGalleryValues({ compareImage: legacyImage('another-top', '2026-07-24T00:00:01.000Z') });
    await rerender();
    await swap();

    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'image', name: 'another-top' }),
      undefined,
      0,
      true
    );
  });

  it('hands image actions the selected item absolute page', async () => {
    const deepA = createImageItem('deep-a', '2026-07-20T00:00:04.000Z');
    const deepB = createImageItem('deep-b', '2026-07-20T00:00:03.000Z');
    const deepC = createImageItem('deep-c', '2026-07-20T00:00:02.000Z');

    setGalleryValues({
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-b', '2026-07-20T00:00:03.000Z'),
      selectedImageName: 'deep-b',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([deepA, deepB], [deepC]);

    await render();
    // Cross the boundary so page 31 is in the shared page cache.
    await pressArrow('ArrowRight');
    await commitLastSelection();

    const context = mocks.imageActionOptions?.getItemActionContext?.();

    expect(context?.items.map((item) => item.name)).toContain('deep-c');
    expect(context?.getItemSelectionPage?.(deepC)).toBe(31);
  });

  it('does not restore a remembered page for an item since moved to another board', async () => {
    // Invalidate remembered position when the comparison item moves boards, even if selection-query identity
    // remains unchanged.
    setGalleryValues({
      compareImage: legacyImage('compare-top', '2026-07-24T00:00:00.000Z'),
      galleryPage: 0,
      recentImages: [],
      selectedImage: legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'),
      selectedImageName: 'deep-newer',
      selectedImageQuery: deepQuery,
    });
    mocks.galleryItemPages = deepBoardPages([createImageItem('deep-newer', '2026-07-20T00:00:02.000Z')]);
    mocks.galleryItemPages[0] = { items: [createImageItem('compare-top', '2026-07-24T00:00:00.000Z')], total: 2 };

    await render();

    const swap = () =>
      act(async () => {
        registeredCommands.get('viewer.swapImages')?.();
        await Promise.resolve();
      });

    await swap();
    await commitLastSelection();
    // The deep image, now in the compare slot, is moved to another board.
    setGalleryValues({
      compareImage: { ...legacyImage('deep-newer', '2026-07-20T00:00:02.000Z'), boardId: 'board-b' },
    });
    await rerender();
    await swap();

    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'image', name: 'deep-newer', boardId: 'board-b' }),
      undefined,
      0,
      true
    );
  });

  it('fetches the next infinite page before stepping past the loaded backend boundary', async () => {
    const newest: GalleryImage = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general',
      starred: false,
    };
    const oldest: GalleryImage = {
      ...mocks.recentImages[1],
      boardId: 'none',
      imageCategory: 'general',
      starred: false,
    };
    mocks.project.widgetInstances.gallery.state.values.recentImages = [mocks.recentImages[0]];
    mocks.galleryItemPages = [
      {
        items: [
          {
            boardId: newest.boardId,
            category: newest.imageCategory,
            createdAt: newest.queuedAt,
            fullUrl: newest.imageUrl,
            height: newest.height,
            isIntermediate: false,
            kind: 'image',
            name: newest.imageName,
            sourceQueueItemId: newest.sourceQueueItemId,
            starred: newest.starred,
            thumbnailUrl: newest.thumbnailUrl,
            width: newest.width,
          },
        ],
        total: 61,
      },
      {
        items: [
          {
            boardId: oldest.boardId,
            category: oldest.imageCategory,
            createdAt: oldest.queuedAt,
            fullUrl: oldest.imageUrl,
            height: oldest.height,
            isIntermediate: false,
            kind: 'image',
            name: oldest.imageName,
            sourceQueueItemId: oldest.sourceQueueItemId,
            starred: oldest.starred,
            thumbnailUrl: oldest.thumbnailUrl,
            width: oldest.width,
          },
        ],
        total: 61,
      },
    ];

    await render();
    await pressArrow('ArrowRight');

    await vi.waitFor(() => {
      expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'image', name: 'oldest' }),
        undefined,
        expect.any(Number),
        true
      );
    });
    expect(mocks.galleryItemPageOffsets.sort((a, b) => a - b)).toEqual([0, 60]);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
  });

  it('anchors Preview navigation to the selected paginated Gallery page', async () => {
    const pageZero = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'page-zero',
      starred: false,
    };
    const selected = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'page-one-selected',
      starred: false,
    };
    const neighbor = {
      ...mocks.recentImages[1],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'page-one-neighbor',
      starred: false,
    };
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    galleryValues.galleryPage = 1;
    galleryValues.paginationMode = 'paginated';
    galleryValues.recentImages = [];
    galleryValues.selectedImage = selected;
    galleryValues.selectedImageName = selected.imageName;
    mocks.galleryItemPages = [
      {
        items: [pageZero].map((image) => ({
          boardId: image.boardId,
          category: image.imageCategory,
          createdAt: image.queuedAt,
          fullUrl: image.imageUrl,
          height: image.height,
          isIntermediate: false,
          kind: 'image' as const,
          name: image.imageName,
          sourceQueueItemId: image.sourceQueueItemId,
          starred: image.starred,
          thumbnailUrl: image.thumbnailUrl,
          width: image.width,
        })),
        total: 3,
      },
      {
        items: [selected, neighbor].map((image) => ({
          boardId: image.boardId,
          category: image.imageCategory,
          createdAt: image.queuedAt,
          fullUrl: image.imageUrl,
          height: image.height,
          isIntermediate: false,
          kind: 'image' as const,
          name: image.imageName,
          sourceQueueItemId: image.sourceQueueItemId,
          starred: image.starred,
          thumbnailUrl: image.thumbnailUrl,
          width: image.width,
        })),
        total: 3,
      },
    ];

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: neighbor.imageName }),
      undefined,
      1,
      true
    );
  });

  it('keeps paginated Preview inside the unstarred listing', async () => {
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;
    const selected = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'freshly-selected',
      queuedAt: '2026-07-21T12:02:30.000Z',
      starred: false,
    };

    galleryValues.galleryPage = 0;
    galleryValues.paginationMode = 'paginated';
    galleryValues.recentImages = [];
    galleryValues.selectedImage = selected;
    galleryValues.selectedImageName = selected.imageName;
    mocks.galleryItemPages = [
      {
        items: [
          createImageItem('newest', '2026-07-21T12:03:00.000Z'),
          { ...createImageItem('starred-mid', '2026-07-21T12:02:00.000Z'), starred: true },
          createImageItem('oldest', '2026-07-21T12:01:00.000Z'),
        ],
        total: 3,
      },
    ];

    await render();
    await pressArrow('ArrowRight');

    // The regular Gallery page query excludes starred items; the strip is not merged into this paginated result.
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'oldest' }),
      undefined,
      0,
      true
    );
  });

  it('does not carry the page the preview opened on onto a ranked pick', async () => {
    const selected = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'ranked-selected',
      starred: false,
    };
    const neighbor = {
      ...mocks.recentImages[1],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'ranked-neighbor',
      starred: false,
    };
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    // Ranked picks must drop stale board-page positions so clearing similarity search cannot anchor far from
    // selection.
    galleryValues.galleryPage = 0;
    galleryValues.paginationMode = 'infinite';
    galleryValues.recentImages = [];
    galleryValues.semanticImageQuery = { kind: 'text', query: 'sunset' };
    galleryValues.selectedImage = selected;
    galleryValues.selectedImageName = selected.imageName;
    galleryValues.selectedImageQuery = {
      boardId: 'none',
      galleryView: 'images',
      imageOrderDir: 'DESC',
      page: 30,
      paginationMode: 'infinite',
      searchTerm: '',
    };
    mocks.galleryItemPages = [
      {
        items: [selected, neighbor].map((image) => ({
          boardId: image.boardId,
          category: image.imageCategory,
          createdAt: image.queuedAt,
          fullUrl: image.imageUrl,
          height: image.height,
          isIntermediate: false,
          kind: 'image' as const,
          name: image.imageName,
          sourceQueueItemId: image.sourceQueueItemId,
          starred: image.starred,
          thumbnailUrl: image.thumbnailUrl,
          width: image.width,
        })),
        total: 2,
      },
    ];

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: neighbor.imageName }),
      undefined,
      0,
      true
    );
  });

  it('ranks the filmstrip within the board the gallery shows, not the board the selection was made in', async () => {
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    // Picked while board A was shown; the gallery has since moved to board B with the search still active.
    galleryValues.selectedBoardId = 'board-b';
    galleryValues.selectedImageQuery = {
      boardId: 'board-a',
      galleryView: 'images',
      imageOrderDir: 'DESC',
      page: 0,
      paginationMode: 'infinite',
      searchTerm: '',
    };
    galleryValues.semanticImageQuery = { kind: 'text', query: 'sunset' };

    await render();

    expect(mocks.galleryItemFilters.at(-1)).toMatchObject({
      boardId: 'board-b',
      semanticQuery: { kind: 'text', query: 'sunset' },
    });

    // Without a search the filmstrip keeps stepping through the listing the item was picked from.
    delete galleryValues.semanticImageQuery;
    await rerender();

    expect(mocks.galleryItemFilters.at(-1)).toMatchObject({ boardId: 'board-a' });
    expect(mocks.galleryItemFilters.at(-1)).not.toHaveProperty('semanticQuery');
  });

  it('stamps the selected ranking page when the footer paginates semantic results', async () => {
    const filler = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'ranked-page-zero',
      starred: false,
    };
    const selected = {
      ...mocks.recentImages[0],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'ranked-page-one-selected',
      starred: false,
    };
    const neighbor = {
      ...mocks.recentImages[1],
      boardId: 'none',
      imageCategory: 'general' as const,
      imageName: 'ranked-page-one-neighbor',
      starred: false,
    };
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;
    const toItem = (image: typeof filler) => ({
      boardId: image.boardId,
      category: image.imageCategory,
      createdAt: image.queuedAt,
      fullUrl: image.imageUrl,
      height: image.height,
      isIntermediate: false,
      kind: 'image' as const,
      name: image.imageName,
      sourceQueueItemId: image.sourceQueueItemId,
      starred: image.starred,
      thumbnailUrl: image.thumbnailUrl,
      width: image.width,
    });

    // The ranking page centers Preview subscriptions while semantic search remains active. Its semantic identity
    // prevents the position from becoming an ordinary board page after the search ends.
    galleryValues.galleryPage = 1;
    galleryValues.paginationMode = 'paginated';
    galleryValues.recentImages = [];
    galleryValues.semanticImageQuery = { kind: 'text', query: 'sunset' };
    galleryValues.selectedImage = selected;
    galleryValues.selectedImageName = selected.imageName;
    galleryValues.selectedImageQuery = {
      boardId: 'none',
      galleryView: 'images',
      imageOrderDir: 'DESC',
      page: 1,
      paginationMode: 'paginated',
      searchTerm: '',
      semanticKey: 'text:sunset',
    };
    mocks.galleryItemPages = [
      { items: [filler].map(toItem), total: 3 },
      { items: [selected, neighbor].map(toItem), total: 3 },
    ];

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: neighbor.imageName }),
      undefined,
      1,
      true
    );
  });

  it('steps left off the first saved item onto the running session, following it', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });
    mocks.commands.account.updateProjectPreferences.mockImplementationOnce((settings: object) => {
      Object.assign(mocks.project.settings, settings);
    });

    await render();
    // Right has saved neighbors; only the leftmost step reaches the session.
    await pressArrow('ArrowRight');
    expect(mocks.commands.account.updateProjectPreferences).not.toHaveBeenCalled();
    setGalleryValues({ selectedImage: legacyImage('newest', '2026-07-21T00:00:00.000Z'), selectedImageName: 'newest' });
    await render();
    await pressArrow('ArrowLeft');
    expect(mocks.commands.account.updateProjectPreferences).toHaveBeenCalledExactlyOnceWith({
      showProgressImagesInViewer: true,
    });
    await rerender();
    expect(followControls.pinnedSessionId).toBe('queue-item-live:1');
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
  });

  it('steps right off the followed session onto the first saved item, and never onto waiting slots', async () => {
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2, 3], completedBackendItemIds: [1, 2] }];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.project.widgetInstances.gallery.state.values.recentImages = mocks.recentImages.map((image) => ({
      ...image,
      sourceQueueItemId: 'queue-item-live',
    }));
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 3, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 3, queueItemId: 'queue-item-live' },
      width: 64,
    });

    await render();
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    await pressArrow('ArrowRight');
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('renders the live frame with the standard media chrome: footer up, no badge, item border', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });

    await render();

    // Live status reports the requested size without suggesting board navigation.
    expect(host?.querySelector<HTMLImageElement>('img[src^="data:image/png"]')).not.toBeNull();
  });

  it('keeps board navigation in its saved-image context while another board generates', async () => {
    const liveBoardImage = {
      ...mocks.project.widgetInstances.gallery.state.values.recentImages[0],
      boardId: 'board-live',
      imageName: 'live-board-image',
      imageUrl: '/images/live-board-image/full',
      sourceQueueItemId: 'queue-item-live',
      starred: false,
      thumbnailUrl: '/images/live-board-image/thumbnail',
    };
    mocks.project.widgetInstances.gallery.state.values.recentImages = [...mocks.recentImages, liveBoardImage];
    mocks.project.queue.items = [{ ...queueItem, snapshot: { ...queueItem.snapshot, galleryBoardId: 'board-live' } }];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });

    await render();
    // Stepping off the live session lands in the SAVED selection's listing, not the generating board's.
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ boardId: 'none', name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
    expect(mocks.galleryItemFilters.every((filter) => filter.boardId !== 'board-live')).toBe(true);
  });

  it('keeps following a completed slot while its result is still routing', async () => {
    // Keep settling results in the single live frame after backend completion and before gallery arrival; do not
    // tile them.
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.runningProgressTargets = [];
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });

    await render();

    expect(
      host?.querySelectorAll<HTMLImageElement>('img[src^="data:image/png"]:not([data-preview-filmstrip] img)')
    ).toHaveLength(1);
  });

  it('keeps the same-root preview through child gaps and browser focus changes until the next denoising step', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    const visibilityStateDescriptor = Object.getOwnPropertyDescriptor(document, 'visibilityState');

    const listeners = new Map<string, Set<(payload: never) => void>>();
    const fire = (event: string, payload: unknown) => {
      for (const handler of listeners.get(event) ?? []) {
        handler(payload as never);
      }
    };
    const getItem = vi.fn(() => Promise.resolve({ id: 1, status: 'waiting' as const }));
    const backend = {
      enqueueWorkflow: () => Promise.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 }),
      getItem,
      on: (event: string, handler: (payload: never) => void) => {
        const handlers = listeners.get(event) ?? new Set();
        handlers.add(handler);
        listeners.set(event, handlers);
        return () => handlers.delete(handler);
      },
      onConnectionChange: () => () => {},
    } as unknown as QueueCoordinatorBackendPort;
    let activeTarget: { itemIndex: number; queueItemId: string } | null = null;
    const isActiveTarget = (target: { itemIndex: number; queueItemId: string }) =>
      activeTarget?.itemIndex === target.itemIndex && activeTarget.queueItemId === target.queueItemId;
    const coordinator = createQueueCoordinator(
      { onGalleryRefresh: () => {} },
      {
        activeProgressTarget: {
          clear: (target) => {
            if (!target || isActiveTarget(target)) {
              activeTarget = null;
              mocks.useActiveProgressTarget.mockReturnValue(null);
              mocks.runningProgressTargets = [];
            }
          },
          set: (target) => {
            activeTarget = target;
            mocks.useActiveProgressTarget.mockReturnValue(target);
            mocks.runningProgressTargets = [target];
          },
          waitForChild: (target: { itemIndex: number; queueItemId: string }) => {
            activeTarget = target;
            mocks.useActiveProgressTarget.mockReturnValue(target);
            mocks.runningProgressTargets = [target];
          },
          settle: () => {},
        } as NonNullable<Parameters<typeof createQueueCoordinator>[1]['activeProgressTarget']>,
        backend,
        modelLoads: { completed: () => {}, reset: () => {}, started: () => {} },
        nodeExecution: {
          clearAll: () => {},
          completed: () => {},
          failed: () => {},
          progress: () => {},
          setOrigin: () => {},
          settleRunning: () => {},
          started: () => {},
        },
        progress: { clear: () => {}, clearAll: () => {}, set: () => {} },
        progressImage: {
          bindSwapImages: () => {},
          clear: (target) => {
            if (!target || isActiveTarget(target)) {
              mocks.useProgressImage.mockReturnValue(null);
              mocks.slotProgressImage = null;
            }
          },
          clearHeld: () => {},
          hold: () => {},
          set: (image, target) => {
            const frame = target ? { ...image, target } : image;
            mocks.useProgressImage.mockReturnValue(frame);
            mocks.slotProgressImage = frame;
          },
        },
        sweepIntervalMs: 60_000,
      }
    );
    const target = { itemIndex: 1, queueItemId: 'queue-item-live' };
    const createProgressEvent = (overrides: Partial<InvocationProgressEvent>): InvocationProgressEvent => ({
      batch_id: 'batch-1',
      destination: 'gallery',
      image: null,
      invocation_source_id: 'call-node',
      item_id: 1,
      message: 'Calling saved workflow',
      origin: null,
      percentage: 0.4,
      queue_id: 'default',
      revision: 1,
      session_id: 'root-session',
      timestamp: 1,
      user_id: 'user-1',
      ...overrides,
    });
    const stageFrame = () =>
      host?.querySelector<HTMLImageElement>('img[src^="data:image/png"]:not([data-preview-filmstrip] img)');

    coordinator.connect();
    try {
      await coordinator.submitWorkflow('queue-item-live', {
        batchCount: 1,
        destination: 'gallery',
        graph: {
          edges: [],
          id: 'parent-workflow',
          nodes: { 'call-node': { id: 'call-node', type: 'call_saved_workflow' } },
        },
        projectId: 'project-1',
        sourceQueueItemId: 'queue-item-live',
      });
      await render();

      fire(
        'invocation_progress',
        createProgressEvent({
          image: { dataURL: 'data:image/png;base64,root-step', height: 64, width: 64 },
          invocation_source_id: 'call-node',
        })
      );
      await rerender();
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,root-step');

      // The coordinator's waiting path consumes these fields from the status event.
      const waitingEvent: Pick<QueueItemStatusChangedEvent, 'item_id' | 'status' | 'status_sequence'> = {
        item_id: 1,
        status: 'waiting',
        status_sequence: 2,
      };
      fire('queue_item_status_changed', waitingEvent);
      await rerender();
      expect(mocks.useActiveProgressTarget()).toEqual(target);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,root-step');

      fire('queue_item_status_changed', waitingEvent);
      await rerender();
      expect(mocks.useActiveProgressTarget()).toEqual(target);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,root-step');

      fire(
        'invocation_progress',
        createProgressEvent({
          image: null,
          invocation_source_id: 'child-node',
          item_id: 2,
          message: 'Child started without a frame',
          percentage: null,
          root_item_id: 1,
          session_id: 'child-session',
          workflow_call_parent_source_id: 'call-node',
        })
      );
      await rerender();
      expect(mocks.useActiveProgressTarget()).toEqual(target);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,root-step');

      fire(
        'invocation_progress',
        createProgressEvent({
          image: { dataURL: 'data:image/png;base64,child-step', height: 64, width: 64 },
          invocation_source_id: 'child-node',
          item_id: 2,
          message: 'Child denoising',
          percentage: 0.6,
          root_item_id: 1,
          session_id: 'child-session',
          workflow_call_parent_source_id: 'call-node',
        })
      );
      await rerender();
      expect(mocks.useActiveProgressTarget()).toEqual(target);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,child-step');

      await act(async () => {
        window.dispatchEvent(new FocusEvent('blur'));
        window.dispatchEvent(new FocusEvent('focus'));
        await Promise.resolve();
      });
      await rerender();
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,child-step');

      Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
      await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'));
        await Promise.resolve();
      });
      await rerender();
      expect(getItem).not.toHaveBeenCalled();
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,child-step');

      Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
      await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'));
        await Promise.resolve();
        await Promise.resolve();
      });
      await rerender();
      expect(getItem).toHaveBeenCalledTimes(1);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,child-step');

      // A nested child carries the same root ID, even though its parent is the first child item.
      fire(
        'invocation_progress',
        createProgressEvent({
          image: { dataURL: 'data:image/png;base64,grandchild-step', height: 64, width: 64 },
          invocation_source_id: 'grandchild-node',
          item_id: 3,
          message: 'Nested child denoising',
          parent_item_id: 2,
          percentage: 0.7,
          root_item_id: 1,
          session_id: 'grandchild-session',
          workflow_call_parent_source_id: 'nested-call-node',
        })
      );
      await rerender();
      expect(mocks.useActiveProgressTarget()).toEqual(target);
      expect(stageFrame()?.getAttribute('src')).toBe('data:image/png;base64,grandchild-step');
    } finally {
      coordinator.dispose();
      if (visibilityStateDescriptor) {
        Object.defineProperty(document, 'visibilityState', visibilityStateDescriptor);
      } else {
        Reflect.deleteProperty(document, 'visibilityState');
      }
    }
  });

  it("shows the followed slot's own frame even when the store-wide latest frame is gone", async () => {
    // Retain another live slot's frame when a neighboring batch releases its latest frame.
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue(null);
    mocks.slotProgressImage = { dataUrl: 'data:image/png;base64,video-step', height: 64, width: 64 };

    await render();

    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,video-step"]')).not.toBeNull();
  });

  it('keeps the selected gallery image when no root is followed and latest frame belongs to another root', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = false;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: queueItem.id });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,other-root',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'other-root' },
      width: 64,
    });

    await render();

    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,other-root"]')).toBeNull();
    expect([...host!.querySelectorAll('img')].map((image) => image.getAttribute('src'))).toContain(
      '/images/newest/full'
    );
  });

  it('follows a running slot over a settling one so a concurrent session is never hidden', async () => {
    // Multi-GPU: slot 1 completed and is settling, slot 2 is still streaming.
    // The single-frame preview must show slot 2 live, not slot 1's static frame.
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2] }];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.runningProgressTargets = [{ itemIndex: 2, queueItemId: 'queue-item-live' }];
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,slot-two',
      height: 64,
      target: { itemIndex: 2, queueItemId: 'queue-item-live' },
      width: 64,
    });
    mocks.bridgeProgressImage = { dataUrl: 'data:image/png;base64,slot-one', height: 64, width: 64 };

    await render();

    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,slot-two"]')).not.toBeNull();
    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,slot-one"]')).toBeNull();
  });

  it("bridges to the next slot of a batch with the previous slot's last frame", async () => {
    // Bridge a newly followed frameless slot with the previous frame until its first progress arrives.
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2], completedBackendItemIds: [1] }];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 2, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,slot-one',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });
    mocks.bridgeProgressImage = { dataUrl: 'data:image/png;base64,bridge', height: 64, width: 64 };

    await render();

    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,bridge"]')).not.toBeNull();
    expect(host?.querySelector<HTMLImageElement>('img[src="data:image/png;base64,slot-one"]')).toBeNull();
  });

  it('keeps every concurrent session in the filmstrip, follows the newest session, and pins on request', async () => {
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2] }];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.runningProgressTargets = [
      { queueItemId: queueItem.id, itemIndex: 1 },
      { queueItemId: queueItem.id, itemIndex: 2 },
    ];
    mocks.slotProgressImage = { dataUrl: 'data:image/png;base64,live', width: 64, height: 64 };
    await render();
    const liveThumbs = () => [...host!.querySelectorAll<HTMLElement>('[data-preview-live-thumb]')];
    const followedThumb = () =>
      host!.querySelector<HTMLElement>('[data-preview-live-thumb][aria-current="true"]')?.dataset.previewLiveThumb;
    // One stage, two thumbs: never a grid. The session that started last is on the stage.
    expect(liveThumbs()).toHaveLength(2);
    expect(host?.querySelectorAll('[data-preview-filmstrip] img[src^="data:image/png"]')).toHaveLength(2);
    expect(followedThumb()).toBe('queue-item-live:2');

    // Frames from other slots must not change the followed session.
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,live',
      height: 64,
      target: { itemIndex: 1, queueItemId: queueItem.id },
      width: 64,
    });
    await rerender();
    expect(followedThumb()).toBe('queue-item-live:2');
    expect(followControls.pinnedSessionId).toBeNull();

    await act(() => followControls.pin('queue-item-live:1'));
    expect(followedThumb()).toBe('queue-item-live:1');
    expect(host!.querySelector('[data-preview-live-thumb="queue-item-live:1"]')?.getAttribute('aria-pressed')).toBe(
      'true'
    );
    expect(host!.querySelector<HTMLElement>('[data-preview-live-pinned]')?.dataset.previewLiveThumb).toBe(
      'queue-item-live:1'
    );
    await act(() => followControls.showAll());
    expect(followedThumb()).toBe('queue-item-live:2');
    expect(host!.querySelector('[data-preview-live-pinned]')).toBeNull();

    await act(() => followControls.pin('queue-item-live:2'));
    mocks.runningProgressTargets = [{ queueItemId: queueItem.id, itemIndex: 1 }];
    await rerender();
    // The pinned session settled, so the pin releases and the stage moves on.
    expect(followControls.pinnedSessionId).toBeNull();
    expect(followedThumb()).toBe('queue-item-live:1');
  });

  it('keeps live previews available during similarity search and temporary comparison override', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.runningProgressTargets = [{ queueItemId: queueItem.id, itemIndex: 1 }];
    mocks.slotProgressImage = { dataUrl: 'data:image/png;base64,live', width: 64, height: 64 };
    const values = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;
    values.semanticImageQuery = { kind: 'text', query: 'blue sky' };
    values.compareImage = mocks.recentImages[1];
    await render();
    expect(host?.querySelectorAll('img[src^="data:image/png"]:not([data-preview-filmstrip] img)')).toHaveLength(1);
    await act(() => followControls.pin('queue-item-live:1'));
    mocks.project.id = 'project-2';
    mocks.project.queue.items = [];
    await rerender();
    expect(followControls.pinnedSessionId).toBeNull();
    expect(followControls.sessions).toEqual([]);
    mocks.project.id = 'project-1';
  });

  it('orders local images oldest-first when the gallery is ascending', async () => {
    (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).imageOrderDir = 'ASC';

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'oldest' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('restores comparison after live activity ends and allows comparison with live preference enabled while idle', async () => {
    const values = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;
    values.compareImage = mocks.recentImages[1];
    mocks.project.settings.showProgressImagesInViewer = true;
    await render();
    expect(host!.textContent).toContain('widgets.preview.exitCompare');
    mocks.project.queue.items = [queueItem];
    mocks.runningProgressTargets = [{ queueItemId: queueItem.id, itemIndex: 1 }];
    await rerender();
    expect(host!.textContent).not.toContain('widgets.preview.exitCompare');
    mocks.runningProgressTargets = [];
    mocks.project.queue.items = [];
    await rerender();
    expect(host!.textContent).toContain('widgets.preview.exitCompare');
  });
  it('keeps pinned live controls and footer inside the widget without saved-image arrows or zero count', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.runningProgressTargets = [{ queueItemId: queueItem.id, itemIndex: 1 }];
    await render();
    await act(() => followControls.pin('queue-item-live:1'));
    const boundary = host!.querySelector<HTMLElement>('[role="region"]')!;
    expect(boundary.getBoundingClientRect().bottom).toBeLessThanOrEqual(host!.getBoundingClientRect().bottom);
  });
  it('does not consume arrow keys in comparison mode', async () => {
    (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).compareImage = {
      ...mocks.project.widgetInstances.gallery.state.values.recentImages[1],
    };
    const documentKeydown = vi.fn();
    document.addEventListener('keydown', documentKeydown);

    try {
      await render();
      await pressArrow('ArrowRight');

      expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
      expect(documentKeydown).toHaveBeenCalledTimes(1);
    } finally {
      document.removeEventListener('keydown', documentKeydown);
    }
  });

  it('renders and navigates same-name image and video items independently in server order', async () => {
    const sameNameImage = createImageItem('shared', '2026-07-30T13:00:00Z');
    const sameNameVideo = createVideoItem('shared', '2026-07-30T12:00:00Z');
    const oldestImage = createImageItem('oldest-mixed', '2026-07-30T11:00:00Z');
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    galleryValues.compareImage = {
      ...mocks.recentImages[0],
      imageName: 'comparison.png',
    };
    galleryValues.recentImages = [];
    galleryValues.selectedImage = sameNameVideo;
    galleryValues.selectedImageName = 'video:shared';
    mocks.galleryItemPages = [{ items: [sameNameImage, sameNameVideo, oldestImage], total: 3 }];

    await render();

    const video = host?.querySelector<HTMLVideoElement>('video');

    expect(video?.getAttribute('src')).toBe(sameNameVideo.fullUrl);
    expect(video?.getAttribute('poster')).toBe(sameNameVideo.thumbnailUrl);
    await expect.poll(() => host?.querySelectorAll<HTMLImageElement>('button img').length).toBe(3);
    expect(host?.textContent).not.toContain('Drop to compare');

    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(sameNameImage, undefined, 0, true);
  });

  it('leaves native video keys untouched and omits image-only hotkey registrations', async () => {
    const videoItem = createVideoItem('native-controls', '2026-07-30T12:00:00Z');
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    galleryValues.recentImages = [];
    galleryValues.selectedImage = videoItem;
    galleryValues.selectedImageName = 'video:native-controls';
    mocks.galleryItemPages = [{ items: [videoItem], total: 1 }];

    await render();

    const video = host?.querySelector<HTMLVideoElement>('video');
    expect(video).not.toBeNull();

    for (const key of ['ArrowLeft', 'ArrowRight', 'f', '1']) {
      const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key });
      await act(async () => {
        video?.dispatchEvent(event);
        await Promise.resolve();
      });
      expect(event.defaultPrevented).toBe(false);
    }

    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.swapImages');
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.zoomToActual');
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.zoomToFit');
  });

  it('retains compare and zoom hotkey registrations for images', async () => {
    await render();

    expect([...registeredHotkeys.keys()]).toEqual(
      expect.arrayContaining(['viewer.swapImages', 'viewer.zoomToActual', 'viewer.zoomToFit'])
    );
    expect([...registeredCommands.keys()]).toEqual(
      expect.arrayContaining(['viewer.swapImages', 'viewer.zoomToActual', 'viewer.zoomToFit'])
    );
  });

  it('uses the common mixed deletion context without the legacy image successor callback', async () => {
    const sameNameImage = createImageItem('shared', '2026-07-30T13:00:00Z');
    const sameNameVideo = createVideoItem('shared', '2026-07-30T12:00:00Z');
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    galleryValues.recentImages = [];
    galleryValues.selectedImage = sameNameVideo;
    galleryValues.selectedImageName = 'video:shared';
    mocks.galleryItemPages = [{ items: [sameNameImage, sameNameVideo], total: 2 }];
    mocks.galleryItemNames = [sameNameImage, sameNameVideo].map(({ kind, name }) => ({ kind, name }));

    await render();

    expect(mocks.imageActionOptions?.onImagesDeleted).toBeUndefined();
    await expect
      .poll(() => mocks.imageActionOptions?.getItemActionContext?.().items)
      .toEqual([sameNameImage, sameNameVideo]);
    const context = mocks.imageActionOptions?.getItemActionContext?.();
    expect(context?.selectedItemKey).toBe('video:shared');
    await expect(context?.loadOrderedRefs(new AbortController().signal)).resolves.toEqual([
      { kind: 'image', name: 'shared' },
      { kind: 'video', name: 'shared' },
    ]);
  });

  it('lazily loads full ordered refs for a range extending past the sparse adjacent-page window', async () => {
    const orderedItems = Array.from({ length: 240 }, (_unused, index) =>
      createImageItem(
        `range-${String(index).padStart(3, '0')}`,
        new Date(Date.UTC(2026, 6, 30, 0, 0, 240 - index)).toISOString()
      )
    );

    mocks.galleryItemPages = Array.from({ length: 4 }, (_unused, page) => ({
      items: orderedItems.slice(page * 60, (page + 1) * 60),
      total: orderedItems.length,
    }));
    mocks.galleryItemNames = orderedItems.map(({ kind, name }) => ({ kind, name }));
    setGalleryValues({
      recentImages: [],
      selectedImage: legacyImage('range-060', orderedItems[60].createdAt),
      selectedImageName: 'range-060',
      selectedImageQuery: { ...deepQuery, page: 1 },
    });

    await render();

    await expect.poll(() => mocks.imageActionOptions?.getItemActionContext?.().items.length).toBe(180);
    const context = mocks.imageActionOptions?.getItemActionContext?.();

    expect(mocks.galleryItemNamesOptionCalls).toBe(0);
    await expect(context?.loadOrderedRefs(new AbortController().signal)).resolves.toHaveLength(240);
    expect(mocks.galleryItemNamesOptionCalls).toBe(1);

    const refs = await context?.loadOrderedRefs(new AbortController().signal);

    expect(refs?.slice(178, 182)).toEqual([
      { kind: 'image', name: 'range-178' },
      { kind: 'image', name: 'range-179' },
      { kind: 'image', name: 'range-180' },
      { kind: 'image', name: 'range-181' },
    ]);
  });

  it('prefetches an image neighbor but never assigns a full video URL to Image', async () => {
    const previousImage = createImageItem('prefetch-previous', '2026-07-30T13:00:00Z');
    const selectedImage = createImageItem('prefetch-selected', '2026-07-30T12:00:00Z');
    const nextVideo = createVideoItem('prefetch-next', '2026-07-30T11:00:00Z');
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;
    const preloadedSources: string[] = [];
    const NativeImage = globalThis.Image;

    class PreloadImage {
      set src(value: string) {
        preloadedSources.push(value);
      }
    }

    Object.defineProperty(globalThis, 'Image', { configurable: true, value: PreloadImage, writable: true });

    try {
      galleryValues.recentImages = [];
      galleryValues.selectedImage = selectedImage;
      galleryValues.selectedImageName = 'image:prefetch-selected';
      mocks.galleryItemPages = [{ items: [previousImage, selectedImage, nextVideo], total: 3 }];

      await render();

      await expect.poll(() => preloadedSources).toContain(previousImage.fullUrl);
      expect(preloadedSources).not.toContain(nextVideo.fullUrl);
    } finally {
      Object.defineProperty(globalThis, 'Image', { configurable: true, value: NativeImage, writable: true });
    }
  });

  it('flicks the preview image onto the neighbor the arrow keys would select', async () => {
    await render();
    await expect.poll(() => selectedThumb()).toBe('newest');
    await flickPreview(1);

    await vi.waitFor(() => {
      expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ name: 'oldest' }),
        undefined,
        expect.any(Number),
        true
      );
    });
    // The swipe and the step agree: the panel that slid in shows that same item at full size.
    expect(
      [...host!.querySelectorAll('[data-swipe-neighbor="next"] img')].map((image) => image.getAttribute('src'))
    ).toContain('/images/oldest/full');
    expect(host?.querySelector('[data-swipe-neighbor="previous"] img')).toBeNull();
  });
});

describe('preview deletion confirmation', () => {
  it('stays open while a live session takes over the preview, then animates out on close', async () => {
    await render();
    await act(() => {
      void mocks.imageActionOptions!.requestDeletionConfirmation!([{ kind: 'image', name: 'newest' }], () =>
        Promise.resolve()
      );
    });
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirmation = dialog.textContent;

    mocks.project.queue.items = [queueItem];
    mocks.project.settings.showProgressImagesInViewer = true;
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });
    await rerender();
    expect(host?.querySelector('img[src^="data:image/png"]')).not.toBeNull();
    expect(dialog.isConnected).toBe(true);
    expect(dialog).toHaveAttribute('data-state', 'open');

    const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toBe(confirmation);
    }
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
  });
});

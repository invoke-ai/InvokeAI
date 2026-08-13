/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryItemsPage, GalleryVideoItem } from '@features/gallery';
import type { QueueItem } from '@features/queue/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { QueryClient, QueryClientProvider, type InfiniteData } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const queueItem: QueueItem = {
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
    galleryItemPageOffsets: [] as number[],
    galleryItemPages: [] as GalleryItemsPage[],
    imageActionOptions: null as null | {
      getItemActionContext?: () => {
        items: Array<GalleryImageItem | GalleryVideoItem>;
        loadOrderedRefs: (signal: AbortSignal) => Promise<Array<{ kind: 'image' | 'video'; name: string }>>;
        selectedItemKey: string | null;
      };
      onImagesDeleted?: (imageNames: string[]) => void;
    },
    recentImages,
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
}));

vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProgressTarget: () => mocks.useActiveProgressTarget(),
  useProgressImage: () => mocks.useProgressImage(),
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_MAX_ROWS: 600,
  GALLERY_PAGE_SIZE: 60,
  flattenGalleryItemsData: (data: InfiniteData<GalleryItemsPage, number> | undefined) =>
    data?.pages.flatMap((page) => page.items) ?? [],
  galleryBoardsOptions: () => ({ queryFn: () => [], queryKey: ['test-boards'], staleTime: Infinity }),
  galleryItemsInfiniteOptions: (
    query: { boardId: string; orderDir?: 'ASC' | 'DESC' },
    window: { kind: 'anchor' | 'infinite' | 'page'; offset?: number } = { kind: 'infinite' }
  ) => {
    const pages = mocks.galleryItemPages.map((page) => {
      const items = page.items.filter((item) => item.boardId === query.boardId);

      return {
        ...page,
        items: query.orderDir === 'ASC' ? items.reverse() : items,
      };
    });
    const initialOffset = window.kind === 'infinite' ? 0 : (window.offset ?? 0);
    const initialPage = pages[initialOffset / 60] ?? { items: [], total: 0 };

    return {
      getNextPageParam: (_lastPage: GalleryItemsPage, _allPages: GalleryItemsPage[], lastPageParam: number) =>
        pages[lastPageParam / 60 + 1] ? lastPageParam + 60 : undefined,
      getPreviousPageParam: (_firstPage: GalleryItemsPage, _allPages: GalleryItemsPage[], firstPageParam: number) =>
        firstPageParam >= 60 ? firstPageParam - 60 : undefined,
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
}));

vi.mock('@workbench/image-actions', () => ({
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
  useDeletionConfirmation: () => ({
    dialog: null,
    requestDeletionConfirmation: (_itemRefs: unknown, executeDeletion: () => Promise<void>) => executeDeletion(),
  }),
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
        common: { countOfTotal: '{{count}} of {{total}}' },
        widgets: {
          preview: {
            framesPerSecond: '{{count}} fps',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            nextItemInBoard: 'Next item in board',
            previousItemInBoard: 'Previous item in board',
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
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async () => {
  host = document.createElement('div');
  host.style.cssText = 'height:320px;width:480px;';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <DndContext>
              <PreviewWidgetView instance={instance} manifest={manifest} region="center" runtime={runtime} />
            </DndContext>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const getBoundary = (): HTMLElement => {
  const boundary = host?.querySelector<HTMLElement>('[tabindex="0"]');

  if (!boundary) {
    throw new Error('Expected the preview keyboard boundary to be rendered.');
  }

  return boundary;
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
  mocks.project.queue.items = [];
  mocks.project.settings.showProgressImagesInViewer = false;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).compareImage;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).galleryPage;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).imageOrderDir;
  delete (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).paginationMode;
  mocks.project.widgetInstances.gallery.state.values.recentImages = mocks.recentImages;
  mocks.project.widgetInstances.gallery.state.values.selectedImage = {
    ...mocks.recentImages[0],
    boardId: 'none',
  };
  mocks.project.widgetInstances.gallery.state.values.selectedImageName = 'newest';
  mocks.galleryItemPageOffsets.length = 0;
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

describe('preview keyboard navigation boundary', () => {
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
        total: 2,
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
        total: 2,
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
    expect(mocks.galleryItemPageOffsets).toEqual([60]);
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

  it('enables live-follow when stepping onto the active placeholder', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,',
      height: 64,
      target: { itemIndex: 1, queueItemId: 'queue-item-live' },
      width: 64,
    });

    await render();
    // Descending order: the live placeholder occupies the newest position, so
    // ArrowLeft from the newest image steps onto it.
    await pressArrow('ArrowLeft');

    expect(mocks.commands.account.updateProjectPreferences).toHaveBeenCalledTimes(1);
    expect(mocks.commands.account.updateProjectPreferences).toHaveBeenCalledWith({
      showProgressImagesInViewer: true,
    });
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
  });

  it('keeps arrow navigation working while following live', async () => {
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
    // While following live the cursor sits on the placeholder; ArrowRight
    // steps back onto the newest completed image.
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('uses the active placeholder board while following live, even before an image frame arrives', async () => {
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
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ boardId: 'board-live', kind: 'image', name: 'live-board-image' }),
      undefined,
      expect.any(Number),
      true
    );
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
    const filmstripPosters = host?.querySelectorAll<HTMLImageElement>('button img');

    expect(video?.getAttribute('src')).toBe(sameNameVideo.fullUrl);
    expect(video?.getAttribute('poster')).toBe(sameNameVideo.thumbnailUrl);
    expect(filmstripPosters).toHaveLength(3);
    expect(host?.textContent).toContain('2 of 3');
    expect(host?.textContent).toContain('1920 × 1080');
    expect(host?.textContent).toContain('Duration 1:06');
    expect(host?.textContent).toContain('23.976 fps');
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

    await render();

    expect(mocks.imageActionOptions?.onImagesDeleted).toBeUndefined();
    const context = mocks.imageActionOptions?.getItemActionContext?.();
    expect(context?.selectedItemKey).toBe('video:shared');
    expect(context?.items).toEqual([sameNameImage, sameNameVideo]);
    await expect(context?.loadOrderedRefs(new AbortController().signal)).resolves.toEqual([
      { kind: 'image', name: 'shared' },
      { kind: 'video', name: 'shared' },
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

      expect(preloadedSources).toContain(previousImage.fullUrl);
      expect(preloadedSources).not.toContain(nextVideo.fullUrl);
    } finally {
      Object.defineProperty(globalThis, 'Image', { configurable: true, value: NativeImage, writable: true });
    }
  });
});

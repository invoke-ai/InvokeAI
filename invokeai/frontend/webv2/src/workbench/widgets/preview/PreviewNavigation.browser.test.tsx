/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryImage, GalleryImagesPage } from '@features/gallery';
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
      gallery: { selectImage: vi.fn(), setCompareImage: vi.fn() },
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
    galleryImagePageOffsets: [] as number[],
    galleryImagePages: [] as GalleryImagesPage[],
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
}));

vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProgressTarget: () => mocks.useActiveProgressTarget(),
  useProgressImage: () => mocks.useProgressImage(),
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_MAX_ROWS: 600,
  GALLERY_PAGE_SIZE: 60,
  flattenGalleryImagesData: (data: InfiniteData<GalleryImagesPage, number> | undefined): GalleryImage[] =>
    data?.pages.flatMap((page) => page.images) ?? [],
  galleryBoardsOptions: () => ({ queryFn: () => [], queryKey: ['test-boards'], staleTime: Infinity }),
  galleryImagesInfiniteOptions: (
    query: { boardId: string; orderDir?: 'ASC' | 'DESC' },
    window: { kind: 'anchor' | 'infinite' | 'page'; offset?: number } = { kind: 'infinite' }
  ) => {
    const pages = mocks.galleryImagePages.map((page) => {
      const images = page.images.filter((image) => image.boardId === query.boardId);

      return {
        ...page,
        images: query.orderDir === 'ASC' ? images.reverse() : images,
      };
    });
    const initialOffset = window.kind === 'infinite' ? 0 : (window.offset ?? 0);
    const initialPage = pages[initialOffset / 60] ?? { images: [], total: 0 };

    return {
      getNextPageParam: (_lastPage: GalleryImagesPage, _allPages: GalleryImagesPage[], lastPageParam: number) =>
        pages[lastPageParam / 60 + 1] ? lastPageParam + 60 : undefined,
      getPreviousPageParam: (_firstPage: GalleryImagesPage, _allPages: GalleryImagesPage[], firstPageParam: number) =>
        firstPageParam >= 60 ? firstPageParam - 60 : undefined,
      initialData: { pageParams: [initialOffset], pages: [initialPage] },
      initialPageParam: initialOffset,
      queryFn: ({ pageParam }: { pageParam: number }) => {
        mocks.galleryImagePageOffsets.push(pageParam);
        return Promise.resolve(pages[pageParam / 60] ?? { images: [], total: 0 });
      },
      queryKey: ['test-images', query.boardId, query.orderDir, window.kind, initialOffset],
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
  getGalleryCanvasImportMenuItems: () => [],
  getImageContextMenuImages: () => [],
  getImageContextMenuRecallRequestKey: () => null,
  getImageRecallCapabilities: () => ({}),
  getImageRecallMessage: () => '',
  getImageRecallTitle: () => '',
  getSelectedGalleryImage: () => null,
  getSelectedGalleryImageFromValues: () => null,
  saveBlobToDisk: () => {},
  useImageActions: () => ({}),
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
await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

const disposer = () => {};
const runtime = {
  commands: { register: () => disposer },
  hotkeys: { register: () => disposer },
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
  mocks.commands.account.updateProjectPreferences.mockClear();
  mocks.commands.gallery.selectImage.mockClear();
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
  mocks.galleryImagePageOffsets.length = 0;
  mocks.galleryImagePages = [
    {
      images: mocks.recentImages.map((image) => ({
        ...image,
        boardId: 'none',
        imageCategory: 'general',
        starred: false,
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

      expect(mocks.commands.gallery.selectImage).toHaveBeenCalledTimes(1);
      expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
        expect.objectContaining({ imageName: 'oldest' }),
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
    mocks.galleryImagePages = [
      { images: [newest], total: 2 },
      { images: [oldest], total: 2 },
    ];

    await render();
    await pressArrow('ArrowRight');

    await vi.waitFor(() => {
      expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
        expect.objectContaining({ imageName: 'oldest' }),
        undefined,
        expect.any(Number),
        true
      );
    });
    expect(mocks.galleryImagePageOffsets).toEqual([60]);
    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledTimes(1);
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
    mocks.galleryImagePages = [
      { images: [pageZero], total: 3 },
      { images: [selected, neighbor], total: 3 },
    ];

    await render();
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
      expect.objectContaining({ imageName: neighbor.imageName }),
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
    expect(mocks.commands.gallery.selectImage).not.toHaveBeenCalled();
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

    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
      expect.objectContaining({ imageName: 'newest' }),
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

    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
      expect.objectContaining({ boardId: 'board-live', imageName: 'live-board-image' }),
      undefined,
      expect.any(Number),
      true
    );
  });

  it('orders local images oldest-first when the gallery is ascending', async () => {
    (mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>).imageOrderDir = 'ASC';

    await render();
    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectImage).toHaveBeenCalledWith(
      expect.objectContaining({ imageName: 'oldest' }),
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

      expect(mocks.commands.gallery.selectImage).not.toHaveBeenCalled();
      expect(documentKeydown).toHaveBeenCalledTimes(1);
    } finally {
      document.removeEventListener('keydown', documentKeydown);
    }
  });
});

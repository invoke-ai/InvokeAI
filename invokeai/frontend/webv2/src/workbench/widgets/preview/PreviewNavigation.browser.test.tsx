/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryImage, GalleryImageItem, GalleryItemsPage, GalleryVideoItem } from '@features/gallery';
import type { InvocationProgressEvent, QueueItem, QueueItemStatusChangedEvent } from '@features/queue/contracts';
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { ImageActions } from '@workbench/image-actions';
import type * as ImageContextMenuModule from '@workbench/image-actions/ImageContextMenu';
import type * as DeletionConfirmationModule from '@workbench/image-actions/useDeletionConfirmation';
import type * as projectsApi from '@workbench/projects/api';
import type { WidgetContributionSource, WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, useSensor, useSensors, type DndContextProps } from '@dnd-kit/core';
import { requestGalleryItemReveal } from '@features/gallery/contracts';
import { createQueueCoordinator, type QueueCoordinatorBackendPort } from '@features/queue/runtime/coordinator';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { QueryClient, QueryClientProvider, type InfiniteData } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { EMPTY_IMAGE_RECALL_CAPABILITIES } from '@workbench/image-actions/imageRecall';
import { HoldToDragSensor, PrimaryMouseSensor } from '@workbench/shell/holdToDragSensor';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import i18next from 'i18next';
import { act, useCallback, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands, page, userEvent } from 'vitest/browser';

import { LivePreviewFollowProvider, useLivePreviewFollow } from './livePreviewFollow';
import { PreviewActionStrip } from './PreviewActionStrip';
import { usePreviewHeaderContext, type PreviewItemPosition } from './previewHeaderStore';
import { PREVIEW_NAV_ATTRIBUTE } from './PreviewNavButtons';
import { PreviewWidgetLabel } from './PreviewWidgetChrome';

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

// Preview loads the player with its first video; loading it up front keeps a slow first transform out of these tests'
// timing. PreviewFrameLazyPlayer covers the lazy path itself.
await import('@platform/ui/video-player/VideoPlayer');

const FIXTURE_VIDEO_URL = '/scripts/mock-assets/fixture-video.mp4';

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

const contextMenuProps = vi.hoisted(() => ({ current: null as null | { target: unknown; variant?: string } }));
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
    openWidget: vi.fn(),
    revealPlacedWidget: vi.fn((..._args: unknown[]): unknown => null),
    commands: {
      account: { updateProjectPreferences: vi.fn() },
      gallery: {
        clearSelection: vi.fn(),
        selectImage: vi.fn(),
        selectItem: vi.fn(),
        setCompareImage: vi.fn(),
        setCompareItem: vi.fn(),
      },
      notifications: { reportError: vi.fn() },
      widgets: { patchValues: vi.fn() },
    },
    extensions: null as unknown as ExtensionRegistry,
    project: {
      id: 'project-1',
      widgetRegions: { center: { activeInstanceId: 'preview-instance', instanceIds: ['preview-instance'] } },
      queue: { items: [] as unknown[] },
      settings: { antialiasProgressImages: false, viewerAutoSwitch: 'off' },
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
        'preview-instance': { id: 'preview-instance', typeId: 'preview' },
      },
    },
    galleryItemFilters: [] as Array<{ boardId: string; mediaKind?: string; starred?: boolean }>,
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
    galleryItemNamesGate: null as Promise<void> | null,
    galleryItemNamesOptionCalls: 0,
    widgetValues: { listeners: new Set<() => void>(), version: 0 },
    imageActions: null as ImageActions | null,
    realImageContextMenu: null as typeof ImageContextMenuModule.ImageContextMenu | null,
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
    framedProgressTargets: undefined as { queueItemId: string; itemIndex: number }[] | undefined,
    runningProgressTargets: undefined as { queueItemId: string; itemIndex: number }[] | undefined,
    slotProgressImage: undefined as unknown,
    useActiveProgressTarget: vi.fn(() => null as unknown),
    useProgressImage: vi.fn(() => null as unknown),
  };
});

vi.mock('@workbench/widgetPlacementCommands', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  revealPlacedWidget: (...args: unknown[]) => mocks.revealPlacedWidget(...args),
}));

vi.mock('@workbench/useOpenWorkbenchWidget', () => ({
  useOpenWorkbenchWidget: () => mocks.openWidget,
}));

const subscribeWidgetValues = (listener: () => void) => {
  mocks.widgetValues.listeners.add(listener);
  return () => mocks.widgetValues.listeners.delete(listener);
};
const getWidgetValuesVersion = () => mocks.widgetValues.version;

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectName: () => 'Project 1',
  useActiveProjectSelector: (selector: (project: typeof mocks.project) => unknown) => selector(mocks.project),
  useOptionalWorkbenchCommands: () => mocks.commands,
  useOptionalWorkbenchExtensions: () => mocks.extensions,
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) => {
    useSyncExternalStore(subscribeWidgetValues, getWidgetValuesVersion);

    return selector(
      widgetId === 'gallery' || widgetId === 'preview'
        ? ((
            Object.values(mocks.project.widgetInstances).find((instance) => instance.typeId === widgetId) as
              | { state?: { values?: Record<string, unknown> } }
              | undefined
          )?.state?.values ?? {})
        : {}
    );
  },
  useWorkbenchCommands: () => mocks.commands,
  useWorkbenchExtensions: () => mocks.extensions,
  useWorkbenchInternalStore: () => null,
  useWorkbenchLiveCanvasEngines: () => ({ commitFrame: () => false }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: mocks.project }) }),
  useWorkbenchSubscription: () => () => () => {},
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ backendConnection: { status: 'connected' } }),
}));

const mockProgressTargets = () => {
  const target = mocks.useActiveProgressTarget() as { queueItemId: string; itemIndex: number } | null;

  return target ? [target] : [];
};

const mockFollowedProgressTargets = () =>
  [...(mocks.runningProgressTargets ?? []), ...mockProgressTargets()].filter(
    (target, index, all) =>
      all.findIndex((other) => other.queueItemId === target.queueItemId && other.itemIndex === target.itemIndex) ===
      index
  );

vi.mock('@features/queue/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProgressTarget: () => mocks.useActiveProgressTarget(),
  useActiveProgressTargets: () => mocks.runningProgressTargets ?? mockProgressTargets(),
  useFollowedProgressTargets: () => mockFollowedProgressTargets(),
  useFramedProgressTargets: () => mocks.framedProgressTargets ?? mockFollowedProgressTargets(),
  useProgressImage: () => mocks.useProgressImage(),
  useQueueItemProgressImage: (queueItemId: string, itemIndex: number) => {
    if (mocks.slotProgressImage !== undefined) {
      return mocks.slotProgressImage;
    }

    const latest = mocks.useProgressImage() as { target?: { itemIndex: number; queueItemId: string } } | null;

    return latest?.target?.queueItemId === queueItemId && latest.target.itemIndex === itemIndex ? latest : null;
  },
}));

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

vi.mock('@features/gallery/queries', () => ({
  galleryImageExportNamesOptions: vi.fn(),
  galleryImageExportOptions: vi.fn(),
  galleryImageMetadataOptions: (_owner: unknown, imageName: string) => ({
    queryFn: () => null,
    queryKey: ['test-image-metadata', imageName],
    staleTime: Infinity,
  }),
  galleryKeys: {
    imageHasWorkflow: (_owner: unknown, imageName: string) => ['test-image-has-workflow', imageName],
    videoHasWorkflow: (_owner: unknown, videoName: string) => ['test-video-has-workflow', videoName],
  },
  galleryVideoMetadataOptions: (_owner: unknown, videoName: string) => ({
    queryFn: () => null,
    queryKey: ['test-video-metadata', videoName],
    staleTime: Infinity,
  }),
  GALLERY_MAX_ROWS: 600,
  GALLERY_PAGE_SIZE: 60,
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  flattenGalleryItemsData: (data: InfiniteData<GalleryItemsPage, number> | undefined) =>
    data?.pages.flatMap((page) => page.items) ?? [],
  canAddToCachedGalleryBoard: () => true,
  galleryBoardExists: () => Promise.resolve(true),
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
    { signal, staleTime }: { signal?: AbortSignal; staleTime?: number }
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
      staleTime: staleTime ?? 60_000,
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
      queryFn: async () => {
        await mocks.galleryItemNamesGate;
        return { items: mocks.galleryItemNames, total: mocks.galleryItemNames.length };
      },
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
  EXPORT_IMAGE_COMMAND_ID: 'viewer.exportImage',
  ImageContextMenu: (props: Parameters<typeof ImageContextMenuModule.ImageContextMenu>[0]) => {
    const { realImageContextMenu: RealImageContextMenu } = mocks;

    contextMenuProps.current = props;
    return mocks.imageActions && RealImageContextMenu ? (
      <RealImageContextMenu {...props} actions={mocks.imageActions} />
    ) : null;
  },
  ImageContinueGroup: () => null,
  ImageExportGroup: () => null,
  ImageUseGroups: () => null,
  VideoUseGroups: () => null,
  RecallActionButtons: () => null,
  buildImageRecallSettings: () => ({}),
  executeImageRecall: () => {},
  getCurrentGenerateValues: () => ({}),
  getImageRecallVerb: () => ({ icon: () => null, label: '' }),
  getImageContextMenuImages: () => [],
  getImageRecallMessage: () => '',
  getImageRecallTitle: () => '',
  getSelectedGalleryImage: () => null,
  getSelectedGalleryImageFromValues: () => null,
  useDeletionConfirmation: (
    await vi.importActual<typeof DeletionConfirmationModule>('@workbench/image-actions/useDeletionConfirmation')
  ).useDeletionConfirmation,
  useImageActions: (options: typeof mocks.imageActionOptions) => {
    mocks.imageActionOptions = options;
    return mocks.imageActions ?? {};
  },
  useImageExportDialog: () => ({ dialog: null, requestImageExport: () => {} }),
  useImageRecallCapabilities: () => ({ capabilities: {}, isLoading: false }),
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

mocks.realImageContextMenu = (
  await vi.importActual<typeof ImageContextMenuModule>('@workbench/image-actions/ImageContextMenu')
).ImageContextMenu;

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: {
          cancel: 'Cancel',
          countOfTotal: '{{count}} of {{total}}',
          generating: 'Generating',
          videoPlayer: {
            frame: 'Frame {{frame}}',
            frameOfLast: 'Frame {{frame}}, last {{last}}',
            loop: 'Loop',
            mute: 'Mute',
            nextFrame: 'Next frame',
            previousFrame: 'Previous frame',
            seek: 'Seek',
            volume: 'Volume',
          },
        },
        widgets: {
          gallery: { commands: { clearSelection: 'Clear gallery selection' } },
          preview: {
            cancelCompare: 'Cancel compare',
            commands: {
              exitCompare: 'Cancel or exit compare',
              nextItem: 'Show next item',
              previousItem: 'Show previous item',
            },
            compareArmedHint: 'Pick another image to compare with this one',
            imageActions: 'Item actions',
            noGallerySelection: 'No gallery selection',
            showGallery: 'Show Gallery',
            framesPerSecond: '{{count}} fps',
            itemCount_one: '{{count}} item',
            itemCount_other: '{{count}} items',
            videoDuration: 'Duration {{duration}}',
            videoLabel: 'Video {{name}}',
          },
        },
      },
    },
  },
});

const registeredCommands = new Map<string, () => void>();
const registeredHotkeys = new Map<string, readonly string[]>();
const previewSource: WidgetContributionSource = {
  instanceId: 'preview-instance',
  projectId: 'project-1',
  region: 'center',
  typeId: 'preview',
};
const runtime = {
  commands: {
    register: (command: { handler: () => void; id: string; title: string }) => {
      const { handler, id } = command;
      const unregister = mocks.extensions.commands.register({ ...command, source: previewSource });

      registeredCommands.set(id, handler);
      return () => {
        unregister();
        if (registeredCommands.get(id) === handler) {
          registeredCommands.delete(id);
        }
      };
    },
  },
  hotkeys: {
    register: (hotkey: { commandId: string; defaultKeys: readonly string[]; id: string; title: string }) => {
      const { defaultKeys, id } = hotkey;
      const unregister = mocks.extensions.hotkeys.register({
        ...hotkey,
        defaultKeys: [...defaultKeys],
        scope: 'widget',
        source: previewSource,
      });

      registeredHotkeys.set(id, defaultKeys);
      return () => {
        unregister();
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

let headerPosition: PreviewItemPosition | null = null;
const HeaderPositionProbe = () => {
  const { position } = usePreviewHeaderContext();
  const ref = useCallback(() => {
    headerPosition = position;
  }, [position]);
  return <span ref={ref} />;
};

let showsHeaderMenuButton = false;
const HeaderMenuButton = () => {
  const { actionItem, actions, getRegion, openItemMenu } = usePreviewHeaderContext();

  return actionItem && actions ? (
    <PreviewActionStrip
      actions={actions}
      density="minimal"
      getRegion={getRegion ?? undefined}
      item={actionItem}
      onOpenMenu={openItemMenu}
    />
  ) : null;
};

const ShellDndContext = ({ children }: Pick<DndContextProps, 'children'>) => {
  const sensors = useSensors(
    useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(HoldToDragSensor)
  );

  return <DndContext sensors={sensors}>{children}</DndContext>;
};

let showsHeaderLabel = false;

const CenterRegion = ({ children }: Pick<DndContextProps, 'children'>) => (
  <section data-testid="center-region" {...useFocusRegionProps('center')} tabIndex={-1}>
    <div
      data-hotkey-widget-instance-id="preview-instance"
      data-hotkey-widget-region="center"
      data-hotkey-widget-type-id="preview"
    >
      {children}
    </div>
    {showsHeaderLabel ? (
      <div data-testid="center-label">
        <PreviewWidgetLabel region="center" />
      </div>
    ) : null}
  </section>
);

const renderTree = async (client: QueryClient) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <WorkbenchFocusProvider>
              <WorkbenchHotkeyRuntime />
              <ShellDndContext>
                <LivePreviewFollowProvider>
                  <FollowProbe />
                  <HeaderPositionProbe />
                  {showsHeaderMenuButton ? <HeaderMenuButton /> : null}
                  <CenterRegion>
                    <PreviewWidgetView instance={instance} manifest={manifest} region="center" runtime={runtime} />
                  </CenterRegion>
                </LivePreviewFollowProvider>
              </ShellDndContext>
            </WorkbenchFocusProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const render = async (client = new QueryClient()) => {
  // A fresh mount replaces the last one: two mounted hotkey runtimes would each handle every key.
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = document.createElement('div');
  host.style.cssText = 'height:320px;width:480px;';
  document.body.append(host);
  root = createRoot(host);

  queryClient = client;
  await renderTree(client);
};

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

const publishGalleryValues = (patch: Record<string, unknown>) => {
  setGalleryValues(patch);
  mocks.widgetValues.version += 1;
  mocks.widgetValues.listeners.forEach((listener) => listener());
};

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

const pressKeyOn = async (target: HTMLElement, key: string): Promise<KeyboardEvent> => {
  let pressed: KeyboardEvent | null = null;
  const capture = (event: KeyboardEvent) => {
    pressed = event;
  };

  target.addEventListener('keydown', capture, { once: true });
  await act(async () => {
    target.focus();
    await userEvent.keyboard(`{${key}}`);
  });

  if (!pressed) {
    throw new Error(`Expected ${key} to reach the focused element.`);
  }

  return pressed;
};

const pressArrow = (key: 'ArrowLeft' | 'ArrowRight') => pressKeyOn(getBoundary(), key);

beforeEach(() => {
  mocks.extensions = createExtensionRegistry();
  registeredCommands.clear();
  registeredHotkeys.clear();
  mocks.commands.account.updateProjectPreferences.mockClear();
  mocks.commands.gallery.selectImage.mockClear();
  mocks.commands.gallery.selectItem.mockClear();
  mocks.commands.gallery.clearSelection.mockReset();
  showsHeaderLabel = false;
  showsHeaderMenuButton = false;
  vi.mocked(requestGalleryItemReveal).mockClear();
  mocks.project.queue.items = [];
  mocks.project.settings.viewerAutoSwitch = 'off';
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
  mocks.galleryItemNamesGate = null;
  mocks.galleryItemNamesOptionCalls = 0;
  mocks.imageActionOptions = null;
  mocks.imageActions = null;
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
  mocks.framedProgressTargets = undefined;
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

const selectedThumb = (): string | null | undefined =>
  host?.querySelector<HTMLButtonElement>('button[aria-current]')?.getAttribute('aria-label');

describe('preview arrow buttons', () => {
  const arrow = (name: 'Show next item' | 'Show previous item') =>
    host!.querySelector<HTMLButtonElement>(`button[aria-label="${name}"]`);
  const click = (button: HTMLButtonElement) =>
    act(async () => {
      await userEvent.hover(button.closest('[data-preview-nav]')!.parentElement!);
      await userEvent.click(button);
      await Promise.resolve();
    });

  it('steps to the item the arrow key would, and leaves focus with Preview so the keys still work', async () => {
    await render();
    const before = document.activeElement;

    expect(arrow('Show previous item')).toBeNull();
    expect(arrow('Show next item')).toHaveAttribute('aria-keyshortcuts', 'ArrowRight');
    expect(arrow('Show next item')).toHaveAttribute('tabindex', '-1');
    await click(arrow('Show next item')!);

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ kind: 'image', name: 'oldest' }),
      undefined,
      expect.any(Number),
      true
    );
    expect(document.activeElement).toBe(before);

    await commitLastSelection();
    await pressArrow('ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
  });
});

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

  it('retries a failed succeeding page before stepping to a newer local item in oldest-first order', async () => {
    const pageItem = (index: number) =>
      createImageItem(`page-item-${index}`, new Date(Date.UTC(2026, 7, 1) + index * 1_000).toISOString());
    const storedPage = (start: number) => ({
      items: Array.from({ length: 60 }, (_, index) => pageItem(start + 59 - index)),
      total: 180,
    });
    const nextItems = Array.from({ length: 60 }, (_, index) => pageItem(index + 120));
    const selected = pageItem(119);
    let rejectNext!: (error: Error) => void;
    const failedPage = new Promise<GalleryItemsPage>((_resolve, reject) => {
      rejectNext = reject;
    });
    let resolveRetry!: (page: GalleryItemsPage) => void;
    const retriedPage = new Promise<GalleryItemsPage>((resolve) => {
      resolveRetry = resolve;
    });

    mocks.galleryItemPages = [storedPage(0), storedPage(60), storedPage(120)];
    mocks.deferredPageFetches.set(120, { promise: failedPage, reject: rejectNext, resolve: () => {} });
    setGalleryValues({
      galleryPage: 1,
      recentImages: [legacyImage('recent-newer', '2026-09-01T00:00:00.000Z')],
      selectedImage: selected,
      selectedImageName: selected.name,
      selectedImageQuery: { ...deepQuery, imageOrderDir: 'ASC', page: 1 },
    });

    await render(new QueryClient({ defaultOptions: { queries: { retry: false } } }));
    await vi.waitFor(() => expect(mocks.galleryItemPageOffsets).toContain(120));
    await act(async () => {
      rejectNext(new Error('temporary page failure'));
      await Promise.resolve();
    });

    mocks.deferredPageFetches.set(120, { promise: retriedPage, resolve: resolveRetry });
    await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();
    await vi.waitFor(() =>
      expect(mocks.galleryItemPageOffsets.filter((pageOffset) => pageOffset === 120)).toHaveLength(2)
    );
    await act(async () => {
      resolveRetry({ items: nextItems, offset: 120, total: 180 });
      await Promise.resolve();
    });

    await vi.waitFor(() =>
      expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ name: 'page-item-120' }),
        undefined,
        2,
        true
      )
    );
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
    await pressArrow('ArrowRight');
    await expect.poll(() => mocks.galleryPageFetchSignals.length > 0).toBe(true);
    const signal = mocks.galleryPageFetchSignals.at(-1)!;
    expect(signal.aborted).toBe(false);

    await act(async () => {
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

  it('handles one arrow press as exactly one selection and claims the key', async () => {
    await render();
    const event = await pressArrow('ArrowRight');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'image', name: 'oldest' }),
      undefined,
      expect.any(Number),
      true
    );
    expect(event.defaultPrevented).toBe(true);
  });

  it('steps with the arrows while focus rests on the region Preview was opened in', async () => {
    await render();
    const region = host!.querySelector<HTMLElement>('[data-testid="center-region"]')!;

    await pressKeyOn(region, 'ArrowRight');
    expect(document.activeElement).toBe(region);
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'oldest' }),
      undefined,
      expect.any(Number),
      true
    );

    await commitLastSelection();
    await pressKeyOn(region, 'ArrowLeft');
    expect(mocks.commands.gallery.selectItem).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'newest' }),
      undefined,
      expect.any(Number),
      true
    );
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(2);
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

  it('hands image actions a page that keeps a deletion successor in the deep window', async () => {
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
    expect(context?.getItemSelectionPage?.(createImageItem('fresh-generation', '2026-07-23T00:00:00.000Z'))).toBe(0);
  });

  it('does not restore a remembered page into a different listing', async () => {
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

  it('reports the selection position in the whole board while only the pages around it are loaded', async () => {
    const pageItem = (index: number) =>
      createImageItem(`position-${index}`, new Date(Date.UTC(2026, 6, 1) - index * 1_000).toISOString());
    const selected = pageItem(330);

    mocks.galleryItemPages = Array.from({ length: 7 }, (_unused, page) => ({
      items: page >= 4 ? Array.from({ length: 60 }, (_item, index) => pageItem(page * 60 + index)) : [],
      total: 420,
    }));
    setGalleryValues({
      recentImages: [],
      selectedImage: legacyImage(selected.name, selected.createdAt),
      selectedImageName: selected.name,
      selectedImageQuery: { ...deepQuery, page: 5 },
    });

    await render();

    await expect.poll(() => headerPosition).toEqual({ boardItemCount: 420, isLoadingBoard: false, selectedIndex: 330 });
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
    await pressArrow('ArrowRight');
    expect(mocks.commands.account.updateProjectPreferences).not.toHaveBeenCalled();
    setGalleryValues({ selectedImage: legacyImage('newest', '2026-07-21T00:00:00.000Z'), selectedImageName: 'newest' });
    await render();
    await pressArrow('ArrowLeft');
    expect(mocks.commands.account.updateProjectPreferences).toHaveBeenCalledExactlyOnceWith({
      viewerAutoSwitch: 'progress',
    });
    await rerender();
    expect(followControls.pinnedSessionId).toBe('queue-item-live:1');
    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledTimes(1);
  });

  it("holds the finished slot's last frame while its result routes, then shows the result until the next first frame", async () => {
    const slotOne = { itemIndex: 1, queueItemId: 'queue-item-live' };
    const slotTwo = { itemIndex: 2, queueItemId: 'queue-item-live' };
    const resultOfSlotOne = { ...mocks.recentImages[0]!, sourceQueueItemId: 'queue-item-live' };
    const stageSources = () =>
      [...host!.querySelectorAll<HTMLImageElement>('img:not([data-preview-filmstrip] img)')].map((image) =>
        image.getAttribute('src')
      );
    const statusLines = () => [...host!.querySelectorAll('[role="status"]')].map((line) => line.textContent);
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2], completedBackendItemIds: [1] }];
    mocks.project.settings.viewerAutoSwitch = 'progress';
    mocks.useActiveProgressTarget.mockReturnValue(slotOne);
    mocks.runningProgressTargets = [slotTwo];
    mocks.framedProgressTargets = [slotOne];
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,slot-one-last',
      height: 64,
      target: slotOne,
      width: 64,
    });
    setGalleryValues({ selectedImage: legacyImage('oldest', '2026-07-20T00:00:00.000Z'), selectedImageName: 'oldest' });

    await render();
    expect(stageSources().filter((src) => src?.startsWith('data:image/png'))).toEqual([
      'data:image/png;base64,slot-one-last',
    ]);

    mocks.useActiveProgressTarget.mockReturnValue(null);
    mocks.framedProgressTargets = [];
    mocks.useProgressImage.mockReturnValue(null);
    publishGalleryValues({
      recentImages: [resultOfSlotOne, ...mocks.recentImages.slice(1)],
      selectedImage: legacyImage('newest', resultOfSlotOne.queuedAt),
      selectedImageName: 'newest',
    });
    await rerender();

    expect(stageSources()).toContain(resultOfSlotOne.imageUrl);
    expect(stageSources().some((src) => src?.startsWith('data:image/png'))).toBe(false);
    expect(host!.querySelector('[data-preview-starting-cue]')?.textContent).toBe('widgets.preview.starting');
    expect(statusLines()).toContain('widgets.preview.startingAnnouncement');
  });

  it('keeps the same-root preview through child gaps and browser focus changes until the next denoising step', async () => {
    mocks.project.queue.items = [queueItem];
    mocks.project.settings.viewerAutoSwitch = 'progress';
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
          markFramed: (target) => {
            activeTarget = target;
            mocks.useActiveProgressTarget.mockReturnValue(target);
            mocks.runningProgressTargets = [target];
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

  it('follows a running slot with a frame of its own over a settling one so a concurrent session is never hidden', async () => {
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2] }];
    mocks.project.settings.viewerAutoSwitch = 'progress';
    mocks.useActiveProgressTarget.mockReturnValue({ itemIndex: 1, queueItemId: 'queue-item-live' });
    mocks.runningProgressTargets = [{ itemIndex: 2, queueItemId: 'queue-item-live' }];
    mocks.useProgressImage.mockReturnValue({
      dataUrl: 'data:image/png;base64,slot-two',
      height: 64,
      target: { itemIndex: 2, queueItemId: 'queue-item-live' },
      width: 64,
    });

    await render();

    expect(
      [...host!.querySelectorAll<HTMLImageElement>('img[src^="data:image/png"]:not([data-preview-filmstrip] img)')].map(
        (image) => image.getAttribute('src')
      )
    ).toEqual(['data:image/png;base64,slot-two']);
  });

  it('keeps every concurrent session in the filmstrip, follows the newest session, and pins on request', async () => {
    mocks.project.queue.items = [{ ...queueItem, backendItemIds: [1, 2] }];
    mocks.project.settings.viewerAutoSwitch = 'progress';
    mocks.runningProgressTargets = [
      { queueItemId: queueItem.id, itemIndex: 1 },
      { queueItemId: queueItem.id, itemIndex: 2 },
    ];
    mocks.slotProgressImage = { dataUrl: 'data:image/png;base64,live', width: 64, height: 64 };
    await render();
    const liveThumbs = () => [...host!.querySelectorAll<HTMLElement>('[data-preview-live-thumb]')];
    const followedThumb = () =>
      host!.querySelector<HTMLElement>('[data-preview-live-thumb][aria-current="true"]')?.dataset.previewLiveThumb;
    expect(liveThumbs()).toHaveLength(2);
    expect(host?.querySelectorAll('[data-preview-filmstrip] img[src^="data:image/png"]')).toHaveLength(2);
    expect(followedThumb()).toBe('queue-item-live:2');

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
    expect(followControls.pinnedSessionId).toBeNull();
    expect(followedThumb()).toBe('queue-item-live:1');
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

    const video = await vi.waitFor(() => {
      const element = host?.querySelector<HTMLVideoElement>('video');

      expect(element).toBeTruthy();
      return element!;
    });

    expect(video.getAttribute('src')).toBe(sameNameVideo.fullUrl);
    expect(video.getAttribute('poster')).toBe(sameNameVideo.thumbnailUrl);
    await expect.poll(() => host?.querySelectorAll<HTMLImageElement>('button img').length).toBe(3);
    expect(host?.textContent).not.toContain('Drop to compare');

    await pressArrow('ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledWith(sameNameImage, undefined, 0, true);
  });

  it('steps items with the arrows from the player, leaves them to its scrubber, and omits image-only hotkeys', async () => {
    const videoItem = { ...createVideoItem('player-keys', '2026-07-30T12:00:00Z'), fullUrl: FIXTURE_VIDEO_URL };
    const newerNeighbor = createImageItem('newer-neighbor', '2026-07-30T13:00:00Z');
    const galleryValues = mocks.project.widgetInstances.gallery.state.values as Record<string, unknown>;

    galleryValues.recentImages = [];
    galleryValues.selectedImage = videoItem;
    galleryValues.selectedImageName = 'video:player-keys';
    mocks.galleryItemPages = [
      {
        items: [newerNeighbor, videoItem, createImageItem('older-neighbor', '2026-07-30T11:00:00Z')],
        total: 3,
      },
    ];

    await render();
    await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();

    for (const key of ['ArrowLeft', 'ArrowRight']) {
      const event = await pressKeyOn(page.getByRole('slider', { name: 'Seek' }).element() as HTMLElement, key);

      expect(event.defaultPrevented).toBe(true);
    }

    // So does the volume slider, here in the popover a player this narrow opens it from. The popover moves focus to its
    // first control as it opens; let that land before focusing the slider, or it takes focus back mid-press.
    await act(() => (page.getByRole('button', { exact: true, name: 'Volume' }).element() as HTMLElement).click());
    await expect.element(page.getByRole('button', { exact: true, name: 'Mute' })).toHaveFocus();

    for (const key of ['ArrowLeft', 'ArrowRight']) {
      const event = await pressKeyOn(page.getByRole('slider', { name: 'Volume' }).element() as HTMLElement, key);

      expect(event.defaultPrevented).toBe(true);
    }

    await userEvent.keyboard('{Escape}');
    expect(mocks.commands.gallery.selectItem).not.toHaveBeenCalled();

    await pressKeyOn(page.getByRole('group', { name: 'Video player-keys' }).element() as HTMLElement, 'ArrowLeft');

    expect(mocks.commands.gallery.selectItem).toHaveBeenCalledExactlyOnceWith(newerNeighbor, undefined, 0, true);
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.swapImages');
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.zoomToActual');
    expect([...registeredHotkeys.keys()]).not.toContain('viewer.zoomToFit');
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

  it('keeps loading the ordered listing for a deletion that clears the selection first', async () => {
    const before = createImageItem('before', '2026-07-30T13:00:00Z');
    const selected = createImageItem('selected', '2026-07-30T12:00:00Z');
    const after = createImageItem('after', '2026-07-30T11:00:00Z');
    let openNames!: () => void;

    mocks.galleryItemPages = [{ items: [before, selected, after], total: 3 }];
    mocks.galleryItemNames = [before, selected, after].map(({ kind, name }) => ({ kind, name }));
    mocks.galleryItemNamesGate = new Promise<void>((resolve) => {
      openNames = resolve;
    });
    setGalleryValues({ recentImages: [], selectedImage: selected, selectedImageName: 'image:selected' });

    await render();
    await expect.poll(() => mocks.imageActionOptions?.getItemActionContext?.().items.length).toBe(3);
    const refs = mocks.imageActionOptions!.getItemActionContext!().loadOrderedRefs(new AbortController().signal);

    setGalleryValues({ selectedImage: null, selectedImageName: null });
    await rerender();
    openNames();

    await expect(refs).resolves.toEqual(mocks.galleryItemNames);
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
    mocks.project.settings.viewerAutoSwitch = 'progress';
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

const createItemMenuActions = (overrides: Partial<ImageActions>): ImageActions => ({
  canUseAsConditioningClip: { audio: false, video: false },
  canUseAsReferenceImage: false,
  canUseAsReferenceVideo: false,
  copyImage: vi.fn(() => Promise.resolve()),
  createCanvasFromImages: vi.fn(() => Promise.resolve()),
  deleteItems: vi.fn(() => Promise.resolve()),
  deleteImages: vi.fn(() => Promise.resolve()),
  deriveImageRecallCapabilities: vi.fn(() => EMPTY_IMAGE_RECALL_CAPABILITIES),
  downloadItem: vi.fn(() => Promise.resolve()),
  downloadItems: vi.fn(() => Promise.resolve()),
  downloadImage: vi.fn(() => Promise.resolve()),
  downloadImages: vi.fn(() => Promise.resolve()),
  exportItems: vi.fn(),
  generatePbrMaps: vi.fn(() => Promise.resolve()),
  loadImageWorkflow: vi.fn(() => Promise.resolve()),
  moveItemsToBoard: vi.fn(() => Promise.resolve()),
  moveImagesToBoard: vi.fn(() => Promise.resolve()),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  openImageExport: vi.fn(),
  openImageInPreview: vi.fn(),
  quickExportImages: vi.fn(() => Promise.resolve()),
  recallImageData: vi.fn(() => Promise.resolve()),
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
  ...overrides,
});

describe('preview item menu focus', () => {
  const successor = { ...mocks.recentImages[1], boardId: 'none' };
  let confirmDeletion: (() => void) | null = null;
  const deleteShownItem = () => {
    publishGalleryValues({ selectedImage: null, selectedImageName: null });

    return new Promise<void>((resolve) => {
      confirmDeletion = () => {
        publishGalleryValues({ selectedImage: successor, selectedImageName: successor.imageName });
        resolve();
      };
    });
  };
  const renderWithItemMenu = async ({ asks = true, ...overrides }: Partial<ImageActions> & { asks?: boolean } = {}) => {
    const deleteItems = vi.fn<ImageActions['deleteItems']>((items, options) =>
      asks
        ? mocks.imageActionOptions!.requestDeletionConfirmation!(items, deleteShownItem, {
            canSkipFutureConfirmations: true,
            returnFocus: options?.returnFocus,
          })
        : deleteShownItem()
    );

    confirmDeletion = null;
    mocks.imageActions = createItemMenuActions({ deleteItems, ...overrides });
    await render();
    return deleteItems;
  };
  const findByText = (selector: string, text: string) =>
    vi.waitFor(() => {
      const element = [...document.querySelectorAll<HTMLElement>(selector)].find((candidate) =>
        candidate.textContent?.includes(text)
      );

      expect(element).toBeDefined();
      return element!;
    });
  const openMenuOn = (target: Element) =>
    act(async () => {
      target.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 40 })
      );
      await Promise.resolve();
    });
  const chooseMenuItem = async (label: string) => {
    const item = await findByText('[role="menuitem"]', label);

    await act(() => userEvent.click(item));
  };
  const deleteThroughMenu = async (asks: boolean) => {
    await chooseMenuItem('widgets.gallery.itemMenu.deleteImage');

    if (asks) {
      const confirm = await findByText('[role="alertdialog"] button', 'widgets.gallery.deleteConfirmLabel');
      await expect.poll(() => document.activeElement?.closest('[role="alertdialog"]')).not.toBeNull();
      await act(() => userEvent.click(confirm));
    }

    await vi.waitFor(() => expect(confirmDeletion).not.toBeNull());
    await act(async () => {
      confirmDeletion!();
      await Promise.resolve();
    });
    expect(host?.querySelector(`img[alt="${successor.imageName}"]`)).not.toBeNull();
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    await expect.poll(() => document.querySelector('[role="menu"]')).toBeNull();
  };
  const stageImage = () => host!.querySelector(`[${PREVIEW_NAV_ATTRIBUTE}]`)!.parentElement!.querySelector('img')!;

  it.each([
    { asks: true, preference: 'deletions confirmed' },
    { asks: false, preference: "don't ask again chosen" },
  ])('closes for good after Delete on the stage, and leaves focus with Preview ($preference)', async ({ asks }) => {
    const deleteItems = await renderWithItemMenu({ asks });

    await openMenuOn(stageImage());
    await deleteThroughMenu(asks);

    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(deleteItems).toHaveBeenCalledExactlyOnceWith([{ kind: 'image', name: 'newest' }], expect.anything());
    expect(document.activeElement).toBe(getBoundary());
  });

  it('gives focus to Preview after Delete without a confirmation takes away a neighbouring thumb', async () => {
    const [firstPage] = mocks.galleryItemPages;
    const listed = { ...firstPage!.items[1]!, name: 'listed', thumbnailUrl: '/images/listed/thumbnail' };

    mocks.galleryItemPages = [{ ...firstPage!, items: [...firstPage!.items, listed], total: 3 }];
    const deleteItems = vi.fn<ImageActions['deleteItems']>(() => {
      queryClient!.setQueriesData<GalleryItemsPage>({ queryKey: ['test-items-page'] }, (page) =>
        page ? { ...page, items: page.items.filter((item) => item.name !== listed.name), total: page.total - 1 } : page
      );
      return Promise.resolve();
    });

    mocks.imageActions = createItemMenuActions({ deleteItems });
    await render();
    const thumb = await vi.waitFor(() => {
      const element = host!.querySelector<HTMLButtonElement>(`button[aria-label="${listed.name}"]`);

      expect(element).not.toBeNull();
      return element!;
    });

    thumb.focus();
    await openMenuOn(thumb);
    await chooseMenuItem('widgets.gallery.itemMenu.deleteImage');
    await expect.poll(() => thumb.isConnected).toBe(false);

    expect(deleteItems).toHaveBeenCalledExactlyOnceWith([{ kind: 'image', name: listed.name }], expect.anything());
    await expect.poll(() => document.activeElement).toBe(getBoundary());
  });
});

describe('preview image recovery', () => {
  const imageRouteCommands = commands as typeof commands & {
    endThumbnailRouteFailures: () => Promise<void>;
    getThumbnailRequests: () => Promise<{ statuses: number[]; types: string[]; urls: string[] }>;
    startThumbnailRoute: (pathnames: string[], options?: { failures?: number; status?: number }) => Promise<void>;
    stopThumbnailRoute: () => Promise<void>;
  };
  const shownImage = () => host?.querySelector<HTMLImageElement>('img[alt="newest"]') ?? null;

  it('reloads the full image an image move refused once the move ends', async () => {
    accountLifecycle.activate('preview-image-recovery');
    const imagePath = '/images/newest/full';
    // The recovery probe's reads are answered in the page, so they settle within the fake clock's steps; the image
    // element's own reads go to the network route.
    let isMoving = true;
    const isProbe = (input: unknown) => new URL(String(input), window.location.href).pathname === imagePath;
    const browserFetch = globalThis.fetch;
    const probes = vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) =>
      isProbe(input)
        ? Promise.resolve(
            isMoving
              ? new Response('moving', { status: 409 })
              : new Response('<svg xmlns="http://www.w3.org/2000/svg"/>', {
                  headers: { 'content-type': 'image/svg+xml' },
                })
          )
        : browserFetch(input, init)
    );
    await imageRouteCommands.startThumbnailRoute([imagePath], { failures: 1_000_000, status: 409 });
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setTimeout', 'clearTimeout'] });

    try {
      await render();
      await expect.poll(() => shownImage()?.complete === true && shownImage()?.naturalWidth === 0).toBe(true);
      expect((await imageRouteCommands.getThumbnailRequests()).statuses).toEqual([409]);

      await act(() => vi.advanceTimersByTimeAsync(2 * 60_000));
      expect(probes.mock.calls.filter(([input]) => isProbe(input)).length).toBeGreaterThan(1);
      expect(shownImage()?.naturalWidth).toBe(0);
      expect((await imageRouteCommands.getThumbnailRequests()).statuses).toEqual([409]);

      isMoving = false;
      await imageRouteCommands.endThumbnailRouteFailures();
      await act(() => vi.advanceTimersByTimeAsync(30_000));
      await expect.poll(() => shownImage()?.naturalWidth).toBe(64);
      expect(shownImage()?.getAttribute('src')).toContain('gallery_image_recovery=');
    } finally {
      vi.useRealTimers();
      probes.mockRestore();
      await imageRouteCommands.stopThumbnailRoute();
    }
  }, 20_000);
});

describe('preview label clear selection', () => {
  const clearButton = () =>
    host!.querySelector<HTMLButtonElement>('[data-testid="center-label"] button[aria-label="Clear gallery selection"]');
  const centerRegion = () => host!.querySelector<HTMLElement>('[data-testid="center-region"]');

  beforeEach(() => {
    showsHeaderLabel = true;
    mocks.commands.gallery.clearSelection.mockImplementation(() =>
      setGalleryValues({ selectedImage: null, selectedImageName: null, selectedImageNames: [] })
    );
  });

  const expectCleared = async () => {
    expect(mocks.commands.gallery.clearSelection).toHaveBeenCalledOnce();
    await rerender();

    expect(clearButton()).toBeNull();
    expect(host!.querySelector('[data-testid="center-label"]')?.textContent).toBe('');
    expect(host!.textContent).toContain('No gallery selection');
    await expect.poll(() => document.activeElement).toBe(centerRegion());
  };

  it('clears the selection from the image name, leaving the empty state though recent results exist', async () => {
    await render();

    await expect.poll(() => host!.querySelector('[data-testid="center-label"]')?.textContent).toContain('newest');
    expect(host!.textContent).not.toContain('No gallery selection');

    await userEvent.click(clearButton()!);

    await expectCleared();
  });
});

describe('preview select for compare', () => {
  const shown = { ...mocks.recentImages[0], boardId: 'none' };
  const other = { ...mocks.recentImages[1], boardId: 'none' };
  const status = () => host!.querySelector('[role="status"][aria-live="polite"]')?.textContent ?? '';
  const hint = () => host!.querySelector<HTMLElement>('[data-preview-compare-hint]');
  const cancelButton = () => hint()?.querySelector<HTMLButtonElement>('button[aria-label="Cancel compare"]') ?? null;
  const isComparing = () =>
    Array.from(host!.querySelectorAll('button')).some(
      (button) => (button.getAttribute('aria-label') ?? button.textContent) === 'widgets.preview.exitCompare'
    );

  beforeEach(() => {
    mocks.commands.gallery.setCompareItem.mockClear();
    setGalleryValues({ compareImage: shown });
  });

  it('says it is waiting, from the persisted state alone, and calls it off from the hint', async () => {
    await render();

    expect(status()).toBe('Pick another image to compare with this one');
    expect(hint()).not.toBeNull();
    expect(isComparing()).toBe(false);

    await act(() => userEvent.click(cancelButton()!));
    expect(mocks.commands.gallery.setCompareItem).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('closes an open comparison on Esc, and leaves Esc alone once nothing is compared', async () => {
    setGalleryValues({ compareImage: other });
    await render();
    expect(isComparing()).toBe(true);

    await pressKeyOn(getBoundary(), 'Escape');
    expect(mocks.commands.gallery.setCompareItem).toHaveBeenCalledExactlyOnceWith(null);

    await act(async () => {
      publishGalleryValues({ compareImage: null });
      await Promise.resolve();
    });
    const press = await pressKeyOn(getBoundary(), 'Escape');

    expect(mocks.commands.gallery.setCompareItem).toHaveBeenCalledOnce();
    expect(press.defaultPrevented).toBe(false);
  });
});

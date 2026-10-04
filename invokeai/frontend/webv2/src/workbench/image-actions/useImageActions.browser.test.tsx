import type { GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { CreateCanvasFromImagesResult } from '@workbench/canvas-operations/api';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { useImageActions } from './useImageActions';

const mocks = vi.hoisted(() => ({
  addToBoard: vi.fn(),
  createCanvasFromImages: vi.fn((..._args: unknown[]): Promise<CreateCanvasFromImagesResult> =>
    Promise.resolve({ projectId: null, status: 'empty' })
  ),
  createProject: vi.fn(),
  deleteImages: vi.fn(),
  downloadArchive: vi.fn(),
  downloadBlob: vi.fn(),
  galleryRemoveItems: vi.fn(),
  galleryPatchItems: vi.fn(),
  galleryWidgetsPatchValues: vi.fn(),
  getItemBoardIds: vi.fn((..._args: unknown[]) => new Map<string, string>()),
  getItemStarred: vi.fn((..._args: unknown[]) => new Map<string, boolean>()),
  getSnapshot: vi.fn(),
  gallerySelectItem: vi.fn(),
  gallerySetItemMultiSelection: vi.fn(),
  imageMetadata: vi.fn(),
  imageResolve: vi.fn((..._args: unknown[]): Promise<unknown> => Promise.reject(new Error('no record'))),
  imageResolveMany: vi.fn((..._args: unknown[]) => Promise.resolve([])),
  imageWorkflow: vi.fn((..._args: unknown[]): Promise<{ graph: string | null; workflow: string | null }> =>
    Promise.resolve({ graph: null, workflow: null })
  ),
  requestWorkflowDocumentLoad: vi.fn(),
  videoMetadata: vi.fn((..._args: unknown[]) => Promise.resolve(null)),
  invalidateGallery: vi.fn(),
  invalidateGalleryItems: vi.fn(),
  itemDelete: vi.fn(),
  itemMoveToBoard: vi.fn(),
  itemSetStarred: vi.fn(),
  notificationsAdd: vi.fn(),
  onImagesDeleted: vi.fn(),
  openWorkbenchWidget: vi.fn((..._args: unknown[]): { ok: boolean; reason?: string; region?: string } => ({
    ok: true,
    region: 'center',
  })),
  patchGalleryItemCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  removeFromBoard: vi.fn(),
  reportError: vi.fn(),
  requestDeletionConfirmation: vi.fn(),
  resolveItem: vi.fn(),
  setStarred: vi.fn(),
}));
const preferences = vi.hoisted(() => ({ confirmImageDeletion: false }));

vi.mock('@features/gallery', () => ({
  galleryImages: {
    metadata: (...args: unknown[]) => mocks.imageMetadata(...args),
    resolve: (...args: unknown[]) => mocks.imageResolve(...args),
    resolveMany: (...args: unknown[]) => mocks.imageResolveMany(...args),
    workflow: (...args: unknown[]) => mocks.imageWorkflow(...args),
  },
  galleryItemOrganization: {
    delete: (...args: unknown[]) => mocks.itemDelete(...args),
    moveToBoard: (...args: unknown[]) => mocks.itemMoveToBoard(...args),
    setStarred: (...args: unknown[]) => mocks.itemSetStarred(...args),
  },
  galleryItems: { resolve: (...args: unknown[]) => mocks.resolveItem(...args) },
  galleryOrganization: {
    addToBoard: (...args: unknown[]) => mocks.addToBoard(...args),
    deleteImages: (...args: unknown[]) => mocks.deleteImages(...args),
    removeFromBoard: (...args: unknown[]) => mocks.removeFromBoard(...args),
    setStarred: (...args: unknown[]) => mocks.setStarred(...args),
  },
  galleryTransfers: { downloadArchive: (...args: unknown[]) => mocks.downloadArchive(...args) },
  galleryVideos: { metadata: (...args: unknown[]) => mocks.videoMetadata(...args) },
  legacyGeneratedImageToGalleryItem: (image: { boardId?: string; imageName: string; starred?: boolean }) => ({
    boardId: image.boardId ?? 'none',
    kind: 'image' as const,
    name: image.imageName,
    starred: image.starred ?? false,
  }),
  toGalleryItemKey: ({ kind, name }: { kind: string; name: string }) => `${kind}:${name}`,
  toGalleryItemRef: ({ kind, name }: { kind: 'image' | 'video'; name: string }) => ({ kind, name }),
}));

vi.mock('@features/workflow/react', () => ({
  requestWorkflowDocumentLoad: (...args: unknown[]) => mocks.requestWorkflowDocumentLoad(...args),
}));

vi.mock('@features/gallery/queries', () => ({
  getGalleryItemBoardIdsFromCaches: (...args: unknown[]) => mocks.getItemBoardIds(...args),
  getGalleryItemStarredFromCaches: (...args: unknown[]) => mocks.getItemStarred(...args),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
  invalidateGalleryItems: (...args: unknown[]) => mocks.invalidateGalleryItems(...args),
  patchGalleryItemCaches: (...args: unknown[]) => mocks.patchGalleryItemCaches(...args),
}));

vi.mock('@features/models', () => ({
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  getModelsSnapshot: () => ({
    models: [
      { base: 'sd-1', key: 'sd-1-model', name: 'SD 1', type: 'main' },
      { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' },
    ],
  }),
  useModelsSelector: (
    selector: (snapshot: {
      models: Array<{ base: 'sd-1' | 'flux'; key: string; name: string; type: 'main' }>;
    }) => unknown
  ) =>
    selector({
      models: [
        { base: 'sd-1', key: 'sd-1-model', name: 'SD 1', type: 'main' },
        { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' },
      ],
    }),
}));

vi.mock('@workbench/useOpenWorkbenchWidget', () => ({
  useOpenWorkbenchWidget: () => mocks.openWorkbenchWidget,
}));

vi.mock('@workbench/settings/store', () => ({
  useWorkbenchPreferenceSelector: (selector: (value: typeof preferences) => boolean) => selector(preferences),
}));

vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: (...args: unknown[]) => mocks.downloadBlob(...args),
}));

vi.mock('@workbench/canvas-operations/api', async () => {
  // The notice mapper is pure; the rest of the canvas-operations surface stays doubled.
  const { getCanvasImportNotice } = await import('@workbench/canvas-operations/canvasImportNotice');
  return {
    createCanvasFromImages: (...args: unknown[]) => mocks.createCanvasFromImages(...args),
    getCanvasEngine: vi.fn(),
    getCanvasImportNotice,
    importGalleryImagesToCanvas: vi.fn(),
  };
});

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    canvas: { apply: vi.fn() },
    gallery: {
      patchItems: (...args: unknown[]) => mocks.galleryPatchItems(...args),
      removeItems: (...args: unknown[]) => mocks.galleryRemoveItems(...args),
      selectImage: vi.fn(),
      selectItem: (...args: unknown[]) => mocks.gallerySelectItem(...args),
      setItemMultiSelection: (...args: unknown[]) => mocks.gallerySetItemMultiSelection(...args),
      setCompareImage: vi.fn(),
    },
    generation: { patchSettings: vi.fn() },
    projects: { create: (...args: unknown[]) => mocks.createProject(...args) },
    notifications: {
      add: (...args: unknown[]) => mocks.notificationsAdd(...args),
      reportError: (...args: unknown[]) => mocks.reportError(...args),
    },
    widgets: {
      patchValues: (...args: unknown[]) => mocks.galleryWidgetsPatchValues(...args),
    },
  }),
  useWorkbenchQueries: () => ({
    getProject: vi.fn(),
    getSnapshot: (...args: unknown[]) => mocks.getSnapshot(...args),
    isActiveProject: vi.fn(() => true),
  }),
  // Only the Video panel's reference capacity is selected; no test here exercises it.
  useWorkbenchSelector: () => false,
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === 'widgets.gallery.uncategorized') {
        return 'Uncategorized';
      }
      if (key === 'widgets.gallery.itemActions.move.success') {
        return `Moved to ${String(values?.board)}`;
      }
      if (key === 'widgets.gallery.itemActions.loadWorkflow.loadedLabel') {
        return `${key}:${String(values?.name)}`;
      }

      return key;
    },
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<ImageActions>();
interface ItemActionContext {
  getItemSelectionPage?: (item: GalleryItem) => number;
  filterIdentity: string;
  items: GalleryItem[];
  loadOrderedRefs(): Promise<GalleryItemRef[]>;
  selectedItemKey: GalleryItemKey | null;
}
let currentItemActionContext: ItemActionContext | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** A `Project`-shaped stub with just enough of `widgetInstances` for the real
 *  `getProjectWidgetValues` (unmocked) to read `gallery`/`upscale` values. */
const makeMockProject = (
  id: string,
  galleryValues: Record<string, unknown> = {},
  upscaleValues: Record<string, unknown> = {}
) => ({
  id,
  widgetInstances: {
    gallery: {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'gallery',
      state: { id: 'gallery' as const, label: 'gallery', values: galleryValues, version: 1 as const },
      typeId: 'gallery' as const,
    },
    upscale: {
      createdAt: '2026-01-01T00:00:00.000Z',
      id: 'upscale',
      state: { id: 'upscale' as const, label: 'upscale', values: upscaleValues, version: 1 as const },
      typeId: 'upscale' as const,
    },
  },
});

const Probe = ({ modelKey = 'sd-1-model', ref }: { modelKey?: string; ref: Ref<ImageActions> }) => {
  const actions = useImageActions({
    boards: [
      {
        archived: false,
        assetCount: 0,
        assetVideoCount: 0,
        id: 'none',
        imageCount: 0,
        kind: 'uncategorized',
        name: '',
        projectId: null,
        videoCount: 0,
      },
    ],
    generateValues: { modelKey },
    getItemActionContext: () => currentItemActionContext,
    onImagesDeleted: mocks.onImagesDeleted,
    projectId: 'project-1',
    requestDeletionConfirmation: mocks.requestDeletionConfirmation,
  });

  useImperativeHandle(ref, () => actions, [actions]);

  return null;
};

beforeEach(() => {
  vi.clearAllMocks();
  preferences.confirmImageDeletion = false;
  mocks.requestDeletionConfirmation.mockImplementation(
    (_itemRefs: readonly GalleryItemRef[], executeDeletion: () => Promise<void>) => executeDeletion()
  );
  mocks.invalidateGallery.mockResolvedValue(undefined);
  mocks.getSnapshot.mockImplementation(() => {
    const project = makeMockProject('project-1');

    return { activeProject: project, projects: [project] };
  });
  accountLifecycle.activate('user-a');
  currentItemActionContext = null;
});

// Seed capabilities after account activation and before mounting so reset and teardown stay inside act.
seedArchitectureCapabilities();

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <Probe ref={actionsRef} />
      </QueryClientProvider>
    );
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('new canvas from images', () => {
  const galleryImage = (imageName: string): GalleryImage => ({
    boardId: 'none',
    height: 512,
    imageCategory: 'general',
    imageName,
    imageUrl: `/${imageName}`,
    queuedAt: '2026-06-15T00:00:00Z',
    sourceQueueItemId: 'queue-item',
    starred: false,
    thumbnailUrl: `/thumb-${imageName}`,
    width: 512,
  });

  it('reports the new canvas and opens it in the center once the images land', async () => {
    mocks.createCanvasFromImages.mockResolvedValueOnce({
      failedImageNames: [],
      layerIds: ['layer-1', 'layer-2'],
      projectId: 'project-2',
      status: 'imported',
    });

    await act(() => actionsRef.current!.createCanvasFromImages([galleryImage('a.png'), galleryImage('b.png')]));

    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      title: 'widgets.canvas.import.newCanvasSuccess',
    });
    expect(mocks.openWorkbenchWidget).toHaveBeenCalledWith('canvas', {
      preferredRegions: ['center'],
      requireCenterView: true,
    });
  });

  it('surfaces the import notice and leaves the layout alone when the images could not land', async () => {
    mocks.createCanvasFromImages.mockResolvedValueOnce({ projectId: 'project-2', status: 'stale-project' });

    await act(() => actionsRef.current!.createCanvasFromImages([galleryImage('a.png')]));

    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'error',
      title: 'widgets.canvas.import.staleProject',
    });
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
  });
});

describe('reference image availability and the capability table', () => {
  it('re-answers when the table lands without the project or the models changing', async () => {
    await act(() => resetArchitectureCapabilities());
    expect(actionsRef.current!.canUseAsReferenceImage).toBe(false);

    // Nothing else re-renders the host: the same project values and the same model list.
    await act(() => setArchitectureCapabilities(architectureCapabilitiesFixture));
    expect(actionsRef.current!.canUseAsReferenceImage).toBe(true);
  });
});

describe('image recall capability cancellation', () => {
  it('re-derives capabilities from the current generate model without another metadata request', async () => {
    const image: GalleryImage = {
      boardId: 'none',
      height: 512,
      imageCategory: 'general',
      imageName: 'recall.png',
      imageUrl: '/full/recall.png',
      queuedAt: '2026-07-30T00:00:00.000Z',
      sourceQueueItemId: 'queue-recall',
      starred: false,
      thumbnailUrl: '/thumb/recall.png',
      width: 512,
    };

    expect(actionsRef.current?.deriveImageRecallCapabilities(image, { clip_skip: 2 }).clipSkip).toBe(true);

    await act(() => {
      root?.render(
        <QueryClientProvider client={new QueryClient()}>
          <Probe ref={actionsRef} modelKey="flux-model" />
        </QueryClientProvider>
      );
    });

    expect(actionsRef.current?.deriveImageRecallCapabilities(image, { clip_skip: 2 }).clipSkip).toBe(false);
    expect(mocks.imageMetadata).not.toHaveBeenCalled();
  });

  it('combines a caller signal with the account lifecycle signal for metadata transport', async () => {
    let resolveMetadata!: (value: null) => void;
    const metadata = new Promise<null>((resolve) => {
      resolveMetadata = resolve;
    });
    mocks.imageMetadata.mockReturnValueOnce(metadata);
    const controller = new AbortController();
    const image: GalleryImage = {
      boardId: 'none',
      height: 512,
      imageCategory: 'general',
      imageName: 'recall.png',
      imageUrl: '/full/recall.png',
      queuedAt: '2026-07-30T00:00:00.000Z',
      sourceQueueItemId: 'queue-recall',
      starred: false,
      thumbnailUrl: '/thumb/recall.png',
      width: 512,
    };
    const getCapabilities = actionsRef.current?.getImageRecallCapabilities as unknown as (
      image: GalleryImage,
      signal?: AbortSignal
    ) => Promise<unknown>;
    let result!: Promise<unknown>;

    await act(async () => {
      result = getCapabilities(image, controller.signal);
      await Promise.resolve();
    });

    expect(mocks.imageMetadata).toHaveBeenCalledWith('recall.png', expect.any(AbortSignal));
    const transportSignal = mocks.imageMetadata.mock.calls[0]?.[1] as AbortSignal;
    expect(transportSignal).toBeInstanceOf(AbortSignal);
    expect(transportSignal.aborted).toBe(false);

    controller.abort();

    expect(transportSignal.aborted).toBe(true);
    resolveMetadata(null);
    await act(async () => {
      await result;
    });
  });
});

describe('load workflow from image', () => {
  const image: GalleryImage = {
    boardId: 'none',
    height: 512,
    imageCategory: 'general',
    imageName: 'made-by-workflow.png',
    imageUrl: '/full/made-by-workflow.png',
    queuedAt: '2026-07-30T00:00:00.000Z',
    sourceQueueItemId: 'queue-workflow',
    starred: false,
    thumbnailUrl: '/thumb/made-by-workflow.png',
    width: 512,
  };

  it('offers the action only once the image record says a workflow is embedded', async () => {
    mocks.imageMetadata.mockResolvedValue(null);
    mocks.imageResolve.mockResolvedValueOnce({ ...image, hasWorkflow: true });

    await expect(actionsRef.current?.getImageRecallCapabilities(image)).resolves.toMatchObject({ workflow: true });

    mocks.imageResolve.mockResolvedValueOnce({ ...image, hasWorkflow: false });

    await expect(actionsRef.current?.getImageRecallCapabilities(image)).resolves.toMatchObject({ workflow: false });

    // An image that came through a record endpoint already knows; no second round trip.
    mocks.imageResolve.mockClear();
    await expect(
      actionsRef.current?.getImageRecallCapabilities({ ...image, hasWorkflow: true })
    ).resolves.toMatchObject({ workflow: true });
    expect(mocks.imageResolve).not.toHaveBeenCalled();
  });

  it('reports, instead of queueing a load for a later editor, when the workflow widget cannot open', async () => {
    mocks.imageWorkflow.mockResolvedValueOnce({ graph: null, workflow: JSON.stringify({ nodes: [] }) });
    mocks.openWorkbenchWidget.mockReturnValueOnce({ ok: false, reason: 'unavailable' });

    await act(async () => {
      await actionsRef.current?.loadImageWorkflow(image);
    });

    expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', title: 'widgets.gallery.itemActions.loadWorkflow.failed' })
    );
  });

  it('hands the embedded workflow to the editor and opens it', async () => {
    mocks.imageWorkflow.mockResolvedValueOnce({
      graph: null,
      workflow: JSON.stringify({ name: 'Embedded', nodes: [] }),
    });

    await act(async () => {
      await actionsRef.current?.loadImageWorkflow(image);
    });

    expect(mocks.openWorkbenchWidget).toHaveBeenCalledWith('workflow', {
      preferredRegions: ['center'],
      requireCenterView: true,
    });
    expect(mocks.requestWorkflowDocumentLoad).toHaveBeenCalledWith(
      { name: 'Embedded', nodes: [] },
      'widgets.gallery.itemActions.loadWorkflow.loadedLabel:Embedded'
    );
  });

  it('says so, and leaves the editor alone, when the image embeds no workflow', async () => {
    await act(async () => {
      await actionsRef.current?.loadImageWorkflow(image);
    });

    expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', title: 'widgets.gallery.itemActions.loadWorkflow.missing' })
    );
  });
});

describe('partial image mutation outcomes', () => {
  it('moves optimistically, then rolls back and re-applies only backend-confirmed images', async () => {
    mocks.getItemBoardIds.mockReturnValue(new Map([['image:locked.png', 'board-0']]));
    mocks.itemMoveToBoard.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [{ kind: 'image', name: 'locked.png' }],
      succeeded: [{ kind: 'image', name: 'moved.png' }],
    });

    await act(async () => {
      await actionsRef.current?.moveImagesToBoard(['moved.png', 'locked.png'], 'board-1');
    });

    // The whole selection moves before the transport is even asked.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(1, expect.anything(), {
      boardId: 'board-1',
      kind: 'move',
      result: {
        failed: [],
        succeeded: [
          { kind: 'image', name: 'moved.png' },
          { kind: 'image', name: 'locked.png' },
        ],
      },
    });
    expect(mocks.patchGalleryItemCaches.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.itemMoveToBoard.mock.invocationCallOrder[0] ?? 0
    );
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:moved.png', 'image:locked.png'], {
      boardId: 'board-1',
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      boardId: 'board-1',
      kind: 'move',
      result: {
        affectedBoardIds: ['board-1'],
        failed: [{ kind: 'image', name: 'locked.png' }],
        succeeded: [{ kind: 'image', name: 'moved.png' }],
      },
    });
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(2, ['image:locked.png'], { boardId: 'board-0' });
  });

  it('stars optimistically before the request and reverts only backend-rejected images', async () => {
    mocks.itemSetStarred.mockResolvedValue({
      affectedBoardIds: [],
      failed: [{ kind: 'image', name: 'locked.png' }],
      succeeded: [{ kind: 'image', name: 'starred.png' }],
    });

    await act(async () => {
      await actionsRef.current?.setImagesStarred(['starred.png', 'locked.png'], true);
    });

    // The whole selection paints before the transport is even asked.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(1, expect.anything(), {
      kind: 'star',
      result: {
        failed: [],
        succeeded: [
          { kind: 'image', name: 'starred.png' },
          { kind: 'image', name: 'locked.png' },
        ],
      },
      starred: true,
    });
    expect(mocks.patchGalleryItemCaches.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.itemSetStarred.mock.invocationCallOrder[0] ?? 0
    );
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:starred.png', 'image:locked.png'], {
      starred: true,
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      kind: 'star',
      result: { failed: [], succeeded: [{ kind: 'image', name: 'starred.png' }] },
      starred: true,
    });
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(2, ['image:locked.png'], { starred: false });
  });
});

type ExpectedItemActions = {
  deleteItems(refs: Array<{ kind: 'image' | 'video'; name: string }>): Promise<void>;
  downloadItem(item: { fullUrl: string; kind: 'image' | 'video'; name: string }): Promise<void>;
  downloadItems(
    refs: Array<{ kind: 'image' | 'video'; name: string }>,
    loadedItems?: Array<{ fullUrl: string; kind: 'image' | 'video'; name: string }>
  ): Promise<void>;
  moveItemsToBoard(refs: Array<{ kind: 'image' | 'video'; name: string }>, boardId: string): Promise<void>;
  openItemInPreview(item: { fullUrl: string; kind: 'image' | 'video'; name: string }): void;
  setItemsStarred(refs: Array<{ kind: 'image' | 'video'; name: string }>, starred: boolean): Promise<void>;
};

const getItemActions = (): ExpectedItemActions => actionsRef.current as unknown as ExpectedItemActions;

describe('mixed item mutation outcomes', () => {
  it('gates deletion through confirmation when the preference is enabled', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    const result = { affectedBoardIds: ['none'], failed: [], succeeded: refs };
    let executeDeletion: (() => Promise<void>) | null = null;
    preferences.confirmImageDeletion = true;
    mocks.requestDeletionConfirmation.mockImplementation((_itemRefs, execute) => {
      executeDeletion = execute;
      return Promise.resolve();
    });
    mocks.itemDelete.mockResolvedValue(result);

    await act(() => {
      root?.render(
        <QueryClientProvider client={new QueryClient()}>
          <Probe ref={actionsRef} />
        </QueryClientProvider>
      );
    });
    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function));
    expect(mocks.itemDelete).not.toHaveBeenCalled();

    await act(async () => {
      await executeDeletion?.();
    });
    expect(mocks.itemDelete).toHaveBeenCalledWith(refs, expect.any(AbortSignal));
  });

  it('deletes immediately without opening confirmation when the preference is disabled', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.requestDeletionConfirmation).not.toHaveBeenCalled();
    expect(mocks.itemDelete).toHaveBeenCalledWith(refs, expect.any(AbortSignal));
  });

  it('uses the localized Uncategorized label in move notifications', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    mocks.itemMoveToBoard.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'none');
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      title: 'Moved to Uncategorized',
    });
  });

  it('moves mixed refs optimistically and reverts only the failed qualified key', async () => {
    const refs = [
      { kind: 'image' as const, name: 'shared' },
      { kind: 'video' as const, name: 'shared' },
    ];
    const result = {
      affectedBoardIds: ['board-1', 'board-2'],
      failed: [refs[1]],
      succeeded: [refs[0]],
    };
    mocks.getItemBoardIds.mockReturnValue(new Map([['video:shared', 'board-1']]));
    mocks.itemMoveToBoard.mockResolvedValue(result);

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-2');
    });

    expect(mocks.itemMoveToBoard).toHaveBeenCalledWith(refs, 'board-2', expect.any(AbortSignal));
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:shared', 'video:shared'], {
      boardId: 'board-2',
    });
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      boardId: 'board-2',
      kind: 'move',
      result,
    });
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(2, ['video:shared'], { boardId: 'board-1' });
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });

  it('stars same-name media independently and reverts only the failed qualified key', async () => {
    const refs = [
      { kind: 'image' as const, name: 'shared' },
      { kind: 'video' as const, name: 'shared' },
    ];
    mocks.itemSetStarred.mockResolvedValue({
      affectedBoardIds: [],
      failed: [refs[0]],
      succeeded: [refs[1]],
    });

    await act(async () => {
      await getItemActions().setItemsStarred(refs, true);
    });

    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:shared', 'video:shared'], { starred: true });
    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      kind: 'star',
      result: { failed: [], succeeded: [refs[1]] },
      starred: true,
    });
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(2, ['image:shared'], { starred: false });
    expect(mocks.galleryRemoveItems).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });

  it('deletes optimistically, then rolls back and re-removes only confirmed refs on partial failure', async () => {
    const refs = [
      { kind: 'video' as const, name: 'gone.mp4' },
      { kind: 'image' as const, name: 'locked.png' },
    ];
    const result = {
      affectedBoardIds: ['board-1'],
      failed: [refs[1]],
      succeeded: [refs[0]],
    };
    mocks.itemDelete.mockResolvedValue(result);

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    // Everything requested vanishes before the transport is asked.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(1, expect.anything(), {
      kind: 'delete',
      result: { failed: [], succeeded: refs },
    });
    expect(mocks.patchGalleryItemCaches.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.itemDelete.mock.invocationCallOrder[0] ?? 0
    );
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(1, ['video:gone.mp4', 'image:locked.png']);

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), { kind: 'delete', result });
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(2, ['video:gone.mp4']);
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });

  it('attempts operation-level invalidation only once when invalidation itself rejects', async () => {
    const ref = { kind: 'video' as const, name: 'clip.mp4' };
    mocks.itemSetStarred.mockResolvedValue({
      affectedBoardIds: [],
      failed: [],
      succeeded: [ref],
    });
    mocks.invalidateGallery.mockRejectedValue(new Error('invalidation failed'));

    await act(async () => {
      await getItemActions().setItemsStarred([ref], true);
    });

    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });
});

describe('total transport failure rollback', () => {
  const recentImageFixture = {
    height: 512,
    imageName: 'gone.png',
    imageUrl: '/full/gone.png',
    queuedAt: '2026-07-30T00:00:00.000Z',
    sourceQueueItemId: 'queue-gone',
    thumbnailUrl: '/thumb/gone.png',
    width: 512,
  };

  it('restores optimistically removed items when the delete transport fails entirely', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    mocks.itemDelete.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    // The optimistic removal happened up front, before the transport was asked.
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(1, ['image:gone.png']);

    // Transport failures must roll back immediately; trailing invalidation may fail too.
    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();

    // The removal stub leaves widget state unchanged, so CAS restoration does nothing. Real reducer rollback is
    // covered in galleryOptimisticRollback.test.ts.
    expect(mocks.galleryWidgetsPatchValues).not.toHaveBeenCalled();

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
  });

  it('restores the gallery widget store when the optimistic removal actually changed it', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    const before = makeMockProject('project-1', { recentImages: [recentImageFixture] });
    const afterRemoval = makeMockProject('project-1', { recentImages: [] });

    // First call is the pre-removal capture; every call after simulates the
    // store having actually applied the optimistic removal.
    mocks.getSnapshot.mockReturnValueOnce({ activeProject: before, projects: [before] }).mockReturnValue({
      activeProject: afterRemoval,
      projects: [afterRemoval],
    });
    mocks.itemDelete.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.galleryWidgetsPatchValues).toHaveBeenCalledWith(
      'gallery',
      { recentImages: [recentImageFixture] },
      'project-1',
      'system'
    );
  });

  it('does not clobber a gallery widget field something else changed before the rollback runs', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    const before = makeMockProject('project-1', { recentImages: [recentImageFixture] });
    const afterRemoval = makeMockProject('project-1', { recentImages: [] });
    const concurrentlyChanged = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, imageName: 'new-generation.png' }],
    });

    mocks.getSnapshot
      .mockReturnValueOnce({ activeProject: before, projects: [before] }) // capture, pre-removal
      .mockReturnValueOnce({ activeProject: afterRemoval, projects: [afterRemoval] }) // diff, post-removal
      .mockReturnValue({ activeProject: concurrentlyChanged, projects: [concurrentlyChanged] }); // restore-time read
    mocks.itemDelete.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    // A generation completed mid-flight and rewrote recentImages: the CAS
    // check must see that and skip the restore rather than clobbering it.
    expect(mocks.galleryWidgetsPatchValues).not.toHaveBeenCalled();
  });

  it('does not roll back caches or the widget store once a partial success has been confirmed', async () => {
    const refs = [{ kind: 'image' as const, name: 'done.png' }];
    const result = { affectedBoardIds: ['none'], failed: [], succeeded: refs };
    mocks.itemDelete.mockResolvedValue(result);
    mocks.onImagesDeleted.mockImplementation(() => {
      throw new Error('caller-supplied callback exploded');
    });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    // The confirmed deletion cache patch applied before the callback threw.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      kind: 'delete',
      result,
    });
    // The `rollbackCaches` closure returned for the *optimistic* patch (call
    // 1) must never fire once confirmation has begun applying.
    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).not.toHaveBeenCalled();
    expect(mocks.galleryWidgetsPatchValues).not.toHaveBeenCalled();
    // The callback's own throw is still surfaced as an error.
    expect(mocks.reportError).toHaveBeenCalledOnce();
  });

  it('restores the previous board ids when the move transport fails entirely, including items known only to the store', async () => {
    const refs = [
      { kind: 'image' as const, name: 'moved.png' },
      { kind: 'image' as const, name: 'overlay-only.png' },
    ];
    mocks.getItemBoardIds.mockReturnValue(new Map([['image:moved.png', 'board-0']]));
    const beforePaint = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, boardId: 'board-overlay', imageName: 'overlay-only.png' }],
    });
    // Simulate optimistic board state after the initial pre-paint snapshot because patchItems is stubbed.
    const afterPaint = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, boardId: 'board-1', imageName: 'overlay-only.png' }],
    });
    mocks.getSnapshot
      .mockReturnValueOnce({ activeProject: beforePaint, projects: [beforePaint] })
      .mockReturnValue({ activeProject: afterPaint, projects: [afterPaint] });
    mocks.itemMoveToBoard.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-1');
    });

    // The optimistic move happened up front, before the transport was asked.
    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:moved.png', 'image:overlay-only.png'], {
      boardId: 'board-1',
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();

    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:moved.png'], { boardId: 'board-0' });
    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:overlay-only.png'], { boardId: 'board-overlay' });

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
  });

  it('does not clobber a store-known board id a concurrent move already carried onto a different board', async () => {
    const refs = [{ kind: 'image' as const, name: 'overlay-only.png' }];
    mocks.getItemBoardIds.mockReturnValue(new Map());
    const beforeFirstMove = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, boardId: 'board-A', imageName: 'overlay-only.png' }],
    });
    // A second, concurrent `moveItemsToBoard` call already carried the item
    // from this move's target ('board-B') onward to 'board-C' by the time
    // this (first) request's rollback runs.
    const concurrentlyMovedOn = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, boardId: 'board-C', imageName: 'overlay-only.png' }],
    });
    mocks.getSnapshot
      .mockReturnValueOnce({ activeProject: beforeFirstMove, projects: [beforeFirstMove] })
      .mockReturnValue({ activeProject: concurrentlyMovedOn, projects: [concurrentlyMovedOn] });
    mocks.itemMoveToBoard.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-B');
    });

    expect(mocks.galleryPatchItems).not.toHaveBeenCalledWith(['image:overlay-only.png'], { boardId: 'board-A' });
    // Only the initial optimistic paint touched the store.
    expect(mocks.galleryPatchItems).toHaveBeenCalledOnce();
  });

  it('reverts the optimistic star paint per item when the transport fails entirely, not by blanket inversion', async () => {
    const refs = [
      { kind: 'image' as const, name: 'was-starred.png' },
      { kind: 'image' as const, name: 'was-unstarred.png' },
    ];
    // Capture each prior flag: one item was already starred, so blanket inversion would corrupt rollback. Later
    // reads simulate the stubbed optimistic paint.
    mocks.getItemStarred
      .mockReturnValueOnce(
        new Map([
          ['image:was-starred.png', true],
          ['image:was-unstarred.png', false],
        ])
      )
      .mockReturnValue(
        new Map([
          ['image:was-starred.png', true],
          ['image:was-unstarred.png', true],
        ])
      );
    mocks.itemSetStarred.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().setItemsStarred(refs, true);
    });

    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:was-starred.png', 'image:was-unstarred.png'], {
      starred: true,
    });

    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:was-starred.png'], { starred: true });
    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:was-unstarred.png'], { starred: false });
    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
  });

  it('does not clobber a star toggle a concurrent mutation already applied, in either the cache or the store', async () => {
    const refs = [{ kind: 'image' as const, name: 'shared.png' }];
    // Simulate a concurrent toggle replacing this action's painted flag in both cache and store.
    mocks.getItemStarred
      .mockReturnValueOnce(new Map([['image:shared.png', false]]))
      .mockReturnValue(new Map([['image:shared.png', false]]));
    const beforePaint = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, imageName: 'shared.png', starred: false }],
    });
    const concurrentToggle = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, imageName: 'shared.png', starred: false }],
    });
    mocks.getSnapshot
      .mockReturnValueOnce({ activeProject: beforePaint, projects: [beforePaint] })
      .mockReturnValue({ activeProject: concurrentToggle, projects: [concurrentToggle] });
    mocks.itemSetStarred.mockRejectedValue(new Error('network down'));
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().setItemsStarred(refs, true);
    });

    expect(mocks.galleryPatchItems).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
  });
});

const galleryItem = (kind: GalleryItem['kind'], name: string): GalleryItem => {
  const base = {
    boardId: 'board-1',
    category: 'general' as const,
    createdAt: '2026-07-30T00:00:00.000Z',
    fullUrl: `/full/${name}`,
    height: 64,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumb/${name}`,
    width: 64,
  };

  return kind === 'video' ? { ...base, durationSeconds: 4, kind } : { ...base, kind };
};

describe('primary successor after confirmed deletion', () => {
  it('selects the next surviving item in display order — the one that takes the deleted slot', async () => {
    // Selection also keeps deletion out of the leading starred block; core/selection.test.ts covers that rule.
    const before = galleryItem('video', 'before.mp4');
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('image', 'after.png');
    const refs = [before, primary, after].map(({ kind, name }) => ({ kind, name }));
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [before, primary, after],
      loadOrderedRefs: () => Promise.resolve(refs),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: 'primary.png' }],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: 'primary.png' }]);
    });

    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(after, 'project-1');
  });

  it('resolves an unloaded successor by qualified ref', async () => {
    const primary = galleryItem('image', 'primary.png');
    const before = galleryItem('image', 'before.png');
    const unloaded = galleryItem('video', 'unloaded.mp4');
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [before, primary],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: before.name },
          { kind: 'video' as const, name: unloaded.name },
          { kind: 'image' as const, name: primary.name },
        ]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.resolveItem.mockResolvedValue(unloaded);
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    });

    expect(mocks.resolveItem).toHaveBeenCalledWith({ kind: 'video', name: unloaded.name }, expect.any(AbortSignal));
    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(unloaded, 'project-1');
  });

  it('falls back to materialized order and then the nearest successor when names fail', async () => {
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('video', 'after.mp4');
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, after],
      loadOrderedRefs: () => Promise.reject(new Error('names unavailable')),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    });

    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(after, 'project-1');
  });

  it('does not promote a failed deletion and keeps a failed primary selected', async () => {
    const failedBefore = galleryItem('video', 'failed-before.mp4');
    const survivingBefore = galleryItem('image', 'surviving-before.png');
    const primary = galleryItem('image', 'primary.png');
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [survivingBefore, failedBefore, primary],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: survivingBefore.name },
          { kind: 'video' as const, name: failedBefore.name },
          { kind: 'image' as const, name: primary.name },
        ]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [{ kind: 'video', name: failedBefore.name }],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    await act(async () => {
      await getItemActions().deleteItems([
        { kind: 'video', name: failedBefore.name },
        { kind: 'image', name: primary.name },
      ]);
    });

    expect(mocks.gallerySetItemMultiSelection).toHaveBeenCalledWith(
      ['video:failed-before.mp4', 'image:surviving-before.png'],
      survivingBefore,
      'project-1'
    );
    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();

    mocks.gallerySetItemMultiSelection.mockClear();
    mocks.gallerySelectItem.mockClear();
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: [],
      failed: [{ kind: 'image', name: primary.name }],
      succeeded: [],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    });

    expect(mocks.gallerySetItemMultiSelection).not.toHaveBeenCalled();
    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
  });

  it('stamps the successor with the page the host navigates from, when the host provides one', async () => {
    // Preserve the successor's host page rather than stamping the grid's unrelated navigation window.
    const before = galleryItem('video', 'before.mp4');
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('image', 'after.png');
    const refs = [before, primary, after].map(({ kind, name }) => ({ kind, name }));
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      getItemSelectionPage: () => 30,
      items: [before, primary, after],
      loadOrderedRefs: () => Promise.resolve(refs),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: 'primary.png' }],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: 'primary.png' }]);
    });

    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(after, 'project-1', 30, true);
  });

  it('opens an item in Preview at the page the host navigates from', () => {
    const item = galleryItem('image', 'deep.png');

    currentItemActionContext = {
      filterIdentity: 'filter-a',
      getItemSelectionPage: () => 30,
      items: [item],
      loadOrderedRefs: () => Promise.resolve([{ kind: 'image' as const, name: item.name }]),
      selectedItemKey: 'image:deep.png',
    };

    act(() => {
      getItemActions().openItemInPreview(item);
    });

    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(item, 'project-1', 30, true);
  });

  it('carries the host page into a retained multi-selection after a partial failure', async () => {
    const successor = galleryItem('image', 'successor.png');
    const primary = galleryItem('image', 'primary.png');
    const failedImage = galleryItem('image', 'failed.png');
    const requested = [
      { kind: 'image' as const, name: failedImage.name },
      { kind: 'image' as const, name: primary.name },
    ];
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      getItemSelectionPage: () => 30,
      items: [successor, primary, failedImage],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: successor.name },
          { kind: 'image' as const, name: primary.name },
          { kind: 'image' as const, name: failedImage.name },
        ]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [requested[0]],
      succeeded: [requested[1]],
    });

    await act(async () => {
      await getItemActions().deleteItems(requested);
    });

    expect(mocks.gallerySetItemMultiSelection).toHaveBeenCalledWith(
      ['image:failed.png', 'image:successor.png'],
      successor,
      'project-1',
      30
    );
  });

  it('atomically retains failed qualified selections while promoting a surviving successor', async () => {
    const successor = galleryItem('image', 'successor.png');
    const primary = galleryItem('image', 'shared');
    const failedSameNameVideo = galleryItem('video', 'shared');
    const failedImage = galleryItem('image', 'failed.png');
    const requested = [
      { kind: 'image' as const, name: failedImage.name },
      { kind: 'image' as const, name: primary.name },
      { kind: 'video' as const, name: failedSameNameVideo.name },
    ];
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [successor, primary, failedSameNameVideo, failedImage],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: successor.name },
          { kind: 'video' as const, name: failedSameNameVideo.name },
          { kind: 'image' as const, name: primary.name },
          { kind: 'image' as const, name: failedImage.name },
        ]),
      selectedItemKey: 'image:shared',
    };
    const result = {
      affectedBoardIds: ['board-1'],
      failed: [requested[0], requested[2]],
      succeeded: [requested[1]],
    };
    mocks.itemDelete.mockResolvedValue(result);

    await act(async () => {
      await getItemActions().deleteItems(requested);
    });

    expect(mocks.galleryRemoveItems).toHaveBeenCalledWith(['image:shared']);
    expect(mocks.gallerySetItemMultiSelection).toHaveBeenCalledWith(
      ['image:failed.png', 'video:shared', 'image:successor.png'],
      successor,
      'project-1'
    );
    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
  });

  it('rejects an unloaded successor after the filter or selection becomes stale', async () => {
    const primary = galleryItem('image', 'primary.png');
    const successor = galleryItem('video', 'successor.mp4');
    let resolveItem = (_item: GalleryItem): void => undefined;
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: primary.name },
          { kind: 'video' as const, name: successor.name },
        ]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.resolveItem.mockReturnValue(
      new Promise((resolve) => {
        resolveItem = resolve;
      })
    );
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    const deletion = getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    await vi.waitFor(() => expect(mocks.resolveItem).toHaveBeenCalledOnce());
    currentItemActionContext = {
      filterIdentity: 'filter-b',
      items: [],
      loadOrderedRefs: () => Promise.resolve([]),
      selectedItemKey: 'video:other.mp4',
    };
    resolveItem(successor);
    await act(() => deletion);

    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
  });

  it('does not apply confirmed deletion side effects after the account changes while resolving a successor', async () => {
    const primary = galleryItem('image', 'primary.png');
    const successor = galleryItem('video', 'successor.mp4');
    let resolveItem = (_item: GalleryItem): void => undefined;
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'image' as const, name: primary.name },
          { kind: 'video' as const, name: successor.name },
        ]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.resolveItem.mockReturnValue(
      new Promise((resolve) => {
        resolveItem = resolve;
      })
    );
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    const deletion = getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    await vi.waitFor(() => expect(mocks.resolveItem).toHaveBeenCalledOnce());
    // The switch drops the capability table, which the hook now subscribes to.
    await act(() => accountLifecycle.activate('user-b'));
    resolveItem(successor);
    await act(() => deletion);

    // The optimistic removal happened up front, on the account that asked for
    // it; nothing further may apply once the account has changed.
    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledWith(expect.anything(), {
      kind: 'delete',
      result: { failed: [], succeeded: [{ kind: 'image', name: primary.name }] },
    });
    expect(mocks.galleryRemoveItems).toHaveBeenCalledOnce();
    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('does not promote an unrelated item when the captured order does not contain the deleted primary', async () => {
    const primary = galleryItem('image', 'primary.png');
    const unrelated = galleryItem('video', 'unrelated.mp4');
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, unrelated],
      loadOrderedRefs: () => Promise.resolve([{ kind: 'video', name: unrelated.name }]),
      selectedItemKey: 'image:primary.png',
    };
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    });

    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
  });
});

describe('mixed item downloads', () => {
  const image = { fullUrl: '/full/still.png', kind: 'image' as const, name: 'still.png' };
  const firstVideo = { fullUrl: '/full/first.mp4', kind: 'video' as const, name: 'first.mp4' };
  const secondVideo = { fullUrl: '/full/second.mp4', kind: 'video' as const, name: 'second.mp4' };

  it('creates exactly one existing archive for an image-only selection', async () => {
    mocks.downloadArchive.mockResolvedValue({
      blob: new Blob(['archive']),
      fileName: 'images.zip',
    });
    vi.stubGlobal('fetch', vi.fn());

    await act(async () => {
      await getItemActions().downloadItems([{ kind: 'image', name: image.name }], [image]);
    });

    expect(mocks.downloadArchive).toHaveBeenCalledOnce();
    expect(mocks.downloadArchive).toHaveBeenCalledWith({
      imageNames: [image.name],
      signal: expect.any(AbortSignal),
    });
    expect(fetch).not.toHaveBeenCalled();
    expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), 'images.zip');
  });

  it('downloads a single video from its protected full URL with its actual name', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('video-one')));

    await act(async () => {
      await getItemActions().downloadItem(firstVideo);
    });

    expect(fetch).toHaveBeenCalledWith(firstVideo.fullUrl, { signal: expect.any(AbortSignal) });
    expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), firstVideo.name);
  });

  it('downloads a mixed selection archive-first and videos sequentially', async () => {
    const order: string[] = [];
    mocks.downloadArchive.mockImplementation(() => {
      order.push('archive');
      return Promise.resolve({ blob: new Blob(['archive']), fileName: 'images.zip' });
    });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        order.push(url);
        return Promise.resolve(new Response(url));
      })
    );
    mocks.downloadBlob.mockImplementation((_blob: Blob, name: string) => order.push(`save:${name}`));

    await act(async () => {
      await getItemActions().downloadItems(
        [
          { kind: 'video', name: firstVideo.name },
          { kind: 'image', name: image.name },
          { kind: 'video', name: secondVideo.name },
        ],
        [image, firstVideo, secondVideo]
      );
    });

    expect(mocks.downloadArchive).toHaveBeenCalledWith({
      imageNames: [image.name],
      signal: expect.any(AbortSignal),
    });
    expect(order).toEqual([
      'archive',
      'save:images.zip',
      firstVideo.fullUrl,
      `save:${firstVideo.name}`,
      secondVideo.fullUrl,
      `save:${secondVideo.name}`,
    ]);
  });

  it('continues after a failed video and summarizes the partial outcome once', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValueOnce(new Error('first failed')).mockResolvedValueOnce(new Response('video-two'))
    );

    await act(async () => {
      await getItemActions().downloadItems(
        [
          { kind: 'video', name: firstVideo.name },
          { kind: 'video', name: secondVideo.name },
        ],
        [firstVideo, secondVideo]
      );
    });

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(mocks.downloadBlob).toHaveBeenCalledOnce();
    expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), secondVideo.name);
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });

  it('skips an unresolved video and continues with the next video in order', async () => {
    mocks.resolveItem.mockRejectedValueOnce(new Error('missing video')).mockResolvedValueOnce(secondVideo);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('video-two')));

    await act(async () => {
      await getItemActions().downloadItems([
        { kind: 'video', name: firstVideo.name },
        { kind: 'video', name: secondVideo.name },
      ]);
    });

    expect(mocks.resolveItem).toHaveBeenNthCalledWith(
      1,
      { kind: 'video', name: firstVideo.name },
      expect.any(AbortSignal)
    );
    expect(mocks.resolveItem).toHaveBeenNthCalledWith(
      2,
      { kind: 'video', name: secondVideo.name },
      expect.any(AbortSignal)
    );
    expect(fetch).toHaveBeenCalledOnce();
    expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), secondVideo.name);
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });
});

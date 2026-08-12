import type { GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { useImageActions } from './useImageActions';

const mocks = vi.hoisted(() => ({
  addToBoard: vi.fn(),
  deleteImages: vi.fn(),
  downloadArchive: vi.fn(),
  downloadBlob: vi.fn(),
  galleryRemoveItems: vi.fn(),
  galleryPatchItems: vi.fn(),
  getItemBoardIds: vi.fn((..._args: unknown[]) => new Map<string, string>()),
  gallerySelectItem: vi.fn(),
  gallerySetItemMultiSelection: vi.fn(),
  imageMetadata: vi.fn(),
  invalidateGallery: vi.fn(),
  invalidateGalleryItems: vi.fn(),
  itemDelete: vi.fn(),
  itemMoveToBoard: vi.fn(),
  itemSetStarred: vi.fn(),
  notificationsAdd: vi.fn(),
  openWorkbenchWidget: vi.fn(),
  patchGalleryItemCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  removeFromBoard: vi.fn(),
  reportError: vi.fn(),
  requestDeletionConfirmation: vi.fn(),
  resolveItem: vi.fn(),
  setStarred: vi.fn(),
}));
const preferences = vi.hoisted(() => ({ confirmImageDeletion: false }));

vi.mock('@features/gallery', () => ({
  galleryImages: { metadata: (...args: unknown[]) => mocks.imageMetadata(...args) },
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
  toGalleryItemKey: ({ kind, name }: { kind: string; name: string }) => `${kind}:${name}`,
  toGalleryItemRef: ({ kind, name }: { kind: 'image' | 'video'; name: string }) => ({ kind, name }),
}));

vi.mock('@features/gallery/queries', () => ({
  getGalleryItemBoardIdsFromCaches: (...args: unknown[]) => mocks.getItemBoardIds(...args),
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

vi.mock('@workbench/canvas-operations/api', () => ({
  getCanvasEngine: vi.fn(),
  getCanvasImportNotice: vi.fn(),
  importGalleryImagesToCanvas: vi.fn(),
}));

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
    notifications: {
      add: (...args: unknown[]) => mocks.notificationsAdd(...args),
      reportError: (...args: unknown[]) => mocks.reportError(...args),
    },
  }),
  useWorkbenchQueries: () => ({
    getProject: vi.fn(),
    getSnapshot: () => ({ activeProject: { id: 'project-1' }, projects: [] }),
    isActiveProject: vi.fn(() => true),
  }),
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

      return key;
    },
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<ImageActions>();
interface ItemActionContext {
  filterIdentity: string;
  items: GalleryItem[];
  loadOrderedRefs(): Promise<GalleryItemRef[]>;
  selectedItemKey: GalleryItemKey | null;
}
let currentItemActionContext: ItemActionContext | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ modelKey = 'sd-1-model', ref }: { modelKey?: string; ref: Ref<ImageActions> }) => {
  const actions = useImageActions({
    boards: [
      {
        archived: false,
        assetCount: 0,
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
    projectId: 'project-1',
    requestDeletionConfirmation: mocks.requestDeletionConfirmation,
  });

  useImperativeHandle(ref, () => actions, [actions]);

  return null;
};

beforeEach(async () => {
  vi.clearAllMocks();
  preferences.confirmImageDeletion = false;
  mocks.requestDeletionConfirmation.mockImplementation(
    (_itemRefs: readonly GalleryItemRef[], executeDeletion: () => Promise<void>) => executeDeletion()
  );
  mocks.invalidateGallery.mockResolvedValue(undefined);
  accountLifecycle.activate('user-a');
  currentItemActionContext = null;
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

    // Partial failure: the optimistic patch is rolled back wholesale, the
    // confirmed subset re-applied, and the rejected ref returns to the board
    // it was captured on.
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

    // Only the rejected ref flips back once the backend answers.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      kind: 'star',
      result: { failed: [], succeeded: [{ kind: 'image', name: 'locked.png' }] },
      starred: false,
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
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(2, expect.anything(), {
      kind: 'star',
      result: { failed: [], succeeded: [refs[0]] },
      starred: false,
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

    // The failed ref is resurrected by the rollback, then only the confirmed
    // ref is removed again.
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
  it('selects the nearest surviving predecessor from ordered names before a successor', async () => {
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

    expect(mocks.gallerySelectItem).toHaveBeenCalledWith(before, 'project-1');
  });

  it('resolves an unloaded predecessor by qualified ref', async () => {
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('image', 'after.png');
    const unloaded = galleryItem('video', 'unloaded.mp4');
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, after],
      loadOrderedRefs: () =>
        Promise.resolve([
          { kind: 'video' as const, name: unloaded.name },
          { kind: 'image' as const, name: primary.name },
          { kind: 'image' as const, name: after.name },
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
    accountLifecycle.activate('user-b');
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

import type { GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type * as QueueUtilityModule from '@features/queue/utility';
import type { InvocationTemplatesSnapshot } from '@features/workflow/contracts';
import type * as WorkflowReact from '@features/workflow/react';
import type { CreateCanvasFromImagesResult } from '@workbench/canvas-operations/api';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { parseOpenApiToTemplates } from '@features/workflow/react';
import { clearLogs, getLogSnapshot } from '@platform/logging/logger';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES } from './imageRecall';
import { useImageActions } from './useImageActions';

const mocks = vi.hoisted(() => ({
  addToBoard: vi.fn(),
  canAddToCachedGalleryBoard: vi.fn((..._args: unknown[]) => true),
  createCanvasFromImages: vi.fn((..._args: unknown[]): Promise<CreateCanvasFromImagesResult> =>
    Promise.resolve({ projectId: null, status: 'empty' })
  ),
  createProject: vi.fn(),
  deleteImages: vi.fn(),
  downloadArchive: vi.fn(),
  downloadBlob: vi.fn(),
  ensureGalleryBoardDestinations: vi.fn((): Promise<void> | null => null),
  galleryRemoveItems: vi.fn(),
  galleryPatchItems: vi.fn(),
  galleryWidgetsPatchValues: vi.fn(),
  getItemBoardIds: vi.fn((..._args: unknown[]) => new Map<string, string>()),
  getCanvasEngine: vi.fn(),
  getItemStarred: vi.fn((..._args: unknown[]) => new Map<string, boolean>()),
  getProject: vi.fn(),
  getSnapshot: vi.fn(),
  getInvocationTemplatesSnapshot: vi.fn<() => InvocationTemplatesSnapshot>(),
  refreshInvocationTemplates: vi.fn<() => Promise<void>>(),
  gallerySelectItem: vi.fn(),
  gallerySetItemMultiSelection: vi.fn(),
  imageMetadata: vi.fn(),
  imageResolve: vi.fn((..._args: unknown[]): Promise<unknown> => Promise.reject(new Error('no record'))),
  imageResolveMany: vi.fn((..._args: unknown[]) => Promise.resolve([])),
  importGalleryImagesToCanvas: vi.fn(),
  isActiveProject: vi.fn(),
  imageWorkflow: vi.fn(
    (
      ..._args: unknown[]
    ): Promise<{ graph: string | null; resolved_inputs?: string | null; workflow: string | null }> =>
      Promise.resolve({ graph: null, workflow: null })
  ),
  requestWorkflowDocumentLoad: vi.fn(),
  submitUtilityGraph: vi.fn((..._args: unknown[]): Promise<unknown> => Promise.reject(new Error('not submitted'))),
  videoWorkflow: vi.fn((..._args: unknown[]): Promise<{ graph: string | null; workflow: string | null }> =>
    Promise.resolve({ graph: null, workflow: null })
  ),
  invalidateGallery: vi.fn(),
  invalidateGalleryBoardDestinations: vi.fn(),
  invalidateGalleryItems: vi.fn(),
  itemDelete: vi.fn(),
  itemMoveToBoard: vi.fn(),
  itemSetStarred: vi.fn(),
  notificationsAdd: vi.fn(),
  notifyUndoable: vi.fn(),
  onImagesDeleted: vi.fn(),
  openWorkbenchWidget: vi.fn((..._args: unknown[]): { ok: boolean; reason?: string; region?: string } => ({
    ok: true,
    region: 'center',
  })),
  patchGalleryItemCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  removeFromBoard: vi.fn(),
  reportError: vi.fn(),
  revealLayerProperties: vi.fn(),
  sendGalleryImageAsControl: vi.fn(),
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
  galleryVideos: {
    workflow: (...args: unknown[]) => mocks.videoWorkflow(...args),
  },
  legacyGeneratedImageToGalleryItem: (image: { boardId?: string; imageName: string; starred?: boolean }) => ({
    boardId: image.boardId ?? 'none',
    kind: 'image' as const,
    name: image.imageName,
    starred: image.starred ?? false,
  }),
  toGalleryItemKey: ({ kind, name }: { kind: string; name: string }) => `${kind}:${name}`,
  toGalleryItemRef: ({ kind, name }: { kind: 'image' | 'video'; name: string }) => ({ kind, name }),
}));

vi.mock('@features/queue/utility', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueUtilityModule>()),
  submitUtilityGraph: (...args: unknown[]) => mocks.submitUtilityGraph(...args),
}));
vi.mock('@platform/transport/socketHub', () => ({ socketHub: { on: () => () => {} } }));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkflowReact>()),
  getInvocationTemplatesSnapshot: () => mocks.getInvocationTemplatesSnapshot(),
  refreshInvocationTemplates: () => mocks.refreshInvocationTemplates(),
  requestWorkflowDocumentLoad: (...args: unknown[]) => mocks.requestWorkflowDocumentLoad(...args),
}));

vi.mock('@features/workflow/preview', async () => {
  // Exercise the real conversion without mounting the preview UI's unrelated dependencies.
  const { storedGraphToDocument } = await import('@features/workflow/core/graphToDocument');
  return { storedGraphToDocument };
});

vi.mock('@features/gallery/queries', () => ({
  canAddToCachedGalleryBoard: (...args: unknown[]) => mocks.canAddToCachedGalleryBoard(...args),
  ensureGalleryBoardDestinations: () => mocks.ensureGalleryBoardDestinations(),
  fetchGalleryImageMetadata: (_client: unknown, _owner: unknown, imageName: string) => mocks.imageMetadata(imageName),
  fetchGalleryVideoMetadata: () => Promise.resolve(null),
  getGalleryItemBoardIdsFromCaches: (...args: unknown[]) => mocks.getItemBoardIds(...args),
  getGalleryItemStarredFromCaches: (...args: unknown[]) => mocks.getItemStarred(...args),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
  invalidateGalleryBoardDestinations: (...args: unknown[]) => mocks.invalidateGalleryBoardDestinations(...args),
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

vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ undoable: (...args: unknown[]) => mocks.notifyUndoable(...args) }),
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
  const { getCanvasImportNotice, getSendAsControlNotice } =
    await import('@workbench/canvas-operations/canvasImportNotice');
  return {
    createCanvasFromImages: (...args: unknown[]) => mocks.createCanvasFromImages(...args),
    getCanvasEngine: (...args: unknown[]) => mocks.getCanvasEngine(...args),
    getCanvasImportNotice,
    getSendAsControlNotice,
    importGalleryImagesToCanvas: (...args: unknown[]) => mocks.importGalleryImagesToCanvas(...args),
    sendGalleryImageAsControl: (...args: unknown[]) => mocks.sendGalleryImageAsControl(...args),
  };
});

vi.mock('@workbench/widgets/layers/revealLayerProperties', () => ({
  useRevealLayerProperties: () => mocks.revealLayerProperties,
}));

vi.mock('@workbench/WorkbenchContext', () => {
  // The workbench store owns one commands object, so it is the same on every read.
  const commands = {
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
  };

  return {
    useWorkbenchCommands: () => commands,
    useWorkbenchQueries: () => ({
      getProject: (...args: unknown[]) => mocks.getProject(...args),
      getSnapshot: (...args: unknown[]) => mocks.getSnapshot(...args),
      isActiveProject: (...args: unknown[]) => mocks.isActiveProject(...args),
    }),
    // Only the Video panel's reference capacity is selected; no test here exercises it.
    useWorkbenchSelector: () => false,
  };
});

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === 'widgets.gallery.uncategorized') {
        return 'Uncategorized';
      }
      if (key === 'widgets.gallery.itemActions.move.success') {
        return `Moved to ${String(values?.board)}`;
      }
      if (key.startsWith('widgets.gallery.itemActions.undoStale.')) {
        return `${key}:${JSON.stringify(values)}`;
      }
      if (key.startsWith('widgets.gallery.itemActions.loadWorkflow.loadedLabel')) {
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
  canvas: { document: { stacks: { control: [], inpaint_mask: [], raster: [], regional_guidance: [] } } },
  id,
  workflows: { activeWorkflowId: 'wf', entries: [] },
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
        isInbox: false,
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
  mocks.isActiveProject.mockImplementation(() => true);
  accountLifecycle.activate('user-a');
  mocks.refreshInvocationTemplates.mockReset().mockResolvedValue(undefined);
  mocks.getInvocationTemplatesSnapshot.mockReturnValue({
    error: null,
    status: 'loaded',
    templates: parseOpenApiToTemplates({
      components: {
        schemas: {
          IntegerInvocation: {
            class: 'invocation',
            properties: {
              type: { default: 'integer' },
              value: { type: 'integer', default: 0, input: 'any', field_kind: 'input' },
            },
            output: { $ref: '#/components/schemas/IntegerOutput' },
            version: '1.0.0',
          },
          IntegerOutput: {
            properties: { type: { default: 'integer_output' }, value: { type: 'integer', field_kind: 'output' } },
          },
          IterateInvocation: {
            class: 'invocation',
            properties: {
              type: { default: 'iterate' },
              collection: { type: 'array', items: { type: 'integer' }, input: 'any', field_kind: 'input' },
            },
            output: { $ref: '#/components/schemas/IterateOutput' },
            version: '1.0.0',
          },
          IterateOutput: {
            properties: { type: { default: 'iterate_output' }, item: { type: 'integer', field_kind: 'output' } },
          },
          CollectInvocation: {
            class: 'invocation',
            properties: {
              type: { default: 'collect' },
              item: { type: 'integer', input: 'connection', field_kind: 'input' },
            },
            output: { $ref: '#/components/schemas/CollectOutput' },
            version: '1.0.0',
          },
          CollectOutput: {
            properties: {
              type: { default: 'collect_output' },
              collection: { type: 'array', items: { type: 'integer' }, field_kind: 'output' },
            },
          },
        },
      },
    }),
  });
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

describe('canvas destinations', () => {
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
      title: 'widgets.canvas.import.newProjectSuccess',
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

  it('adds to the current canvas without creating a project', async () => {
    mocks.getProject.mockImplementation((id: string) => makeMockProject(id));
    mocks.importGalleryImagesToCanvas.mockResolvedValueOnce({
      failedImageNames: [],
      layerIds: ['layer-1'],
      status: 'imported',
    });

    await act(() => actionsRef.current!.sendToCanvas([galleryImage('a.png')], 'raster'));

    expect(mocks.createProject).not.toHaveBeenCalled();
    expect(mocks.createCanvasFromImages).not.toHaveBeenCalled();
    expect(mocks.importGalleryImagesToCanvas).toHaveBeenCalledWith(
      expect.objectContaining({ destination: 'raster', project: expect.objectContaining({ id: 'project-1' }) })
    );
    expect(mocks.openWorkbenchWidget).toHaveBeenCalledWith('canvas', {
      preferredRegions: ['center'],
      requireCenterView: true,
    });
  });

  it('keeps the destination captured when the import started after another project becomes active', async () => {
    let activeProjectId = 'project-1';
    mocks.isActiveProject.mockImplementation((id: string) => id === activeProjectId);
    mocks.getProject.mockImplementation((id: string) => makeMockProject(id));
    mocks.getSnapshot.mockImplementation(() => ({
      activeProject: makeMockProject(activeProjectId),
      projects: [makeMockProject('project-1'), makeMockProject('project-2')],
    }));
    mocks.importGalleryImagesToCanvas.mockImplementationOnce(async () => {
      // The user switches projects while the images are being prepared.
      activeProjectId = 'project-2';
      await Promise.resolve();
      return { failedImageNames: [], layerIds: ['layer-1'], status: 'imported' };
    });

    await act(() => actionsRef.current!.sendToCanvas([galleryImage('a.png')], 'control-resized'));

    expect(mocks.getCanvasEngine).toHaveBeenCalledWith('project-1');
    expect(mocks.importGalleryImagesToCanvas).toHaveBeenCalledWith(
      expect.objectContaining({ project: expect.objectContaining({ id: 'project-1' }) })
    );
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({ kind: 'success', title: 'widgets.canvas.import.success' });
    // The canvas the images went to is no longer on screen; the one that is must not be brought forward.
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
  });

  it('names the control layer an image filled and shows its Properties', async () => {
    mocks.getProject.mockImplementation((id: string) => makeMockProject(id));
    mocks.sendGalleryImageAsControl.mockResolvedValueOnce({
      layerId: 'control-1',
      layerName: 'Depth 1',
      status: 'filled',
    });

    await act(() => actionsRef.current!.sendImageAsControl(galleryImage('a.png')));

    expect(mocks.sendGalleryImageAsControl).toHaveBeenCalledWith(
      expect.objectContaining({ image: expect.objectContaining({ imageName: 'a.png' }) })
    );
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'success',
      title: 'widgets.canvas.import.controlFilled',
    });
    expect(mocks.revealLayerProperties).toHaveBeenCalledWith('control-1');
    // The menu has no trigger to restore focus to: the focusing open hands it to the Canvas first.
    expect(mocks.openWorkbenchWidget).toHaveBeenCalledWith('canvas', {
      preferredRegions: ['center'],
      requireCenterView: true,
    });
    expect(mocks.openWorkbenchWidget.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.revealLayerProperties.mock.invocationCallOrder[0]!
    );
  });

  it('reports a new control layer as an import, and a refusal without moving the layout', async () => {
    mocks.getProject.mockImplementation((id: string) => makeMockProject(id));
    mocks.sendGalleryImageAsControl.mockResolvedValueOnce({ layerId: 'control-2', status: 'created' });

    await act(() => actionsRef.current!.sendImageAsControl(galleryImage('a.png')));

    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith({
      kind: 'success',
      title: 'widgets.canvas.import.success',
    });
    expect(mocks.revealLayerProperties).toHaveBeenLastCalledWith('control-2');

    mocks.revealLayerProperties.mockClear();
    mocks.openWorkbenchWidget.mockClear();
    mocks.sendGalleryImageAsControl.mockResolvedValueOnce({ status: 'blocked' });
    await act(() => actionsRef.current!.sendImageAsControl(galleryImage('a.png')));

    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith({ kind: 'info', title: 'widgets.canvas.import.blocked' });
    expect(mocks.revealLayerProperties).not.toHaveBeenCalled();
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
  });
});

describe('compare selection handler', () => {
  it('stays the same handler while selection-dependent actions are rebuilt', async () => {
    const before = actionsRef.current!;

    await act(() => {
      root?.render(
        <QueryClientProvider client={new QueryClient()}>
          <Probe modelKey="other-model" ref={actionsRef} />
        </QueryClientProvider>
      );
    });

    // Thumbnail strips hand it to every thumb; a new handler per selection would re-render them all.
    expect(actionsRef.current).not.toBe(before);
    expect(actionsRef.current!.selectForCompare).toBe(before.selectForCompare);
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

describe('image recall capabilities', () => {
  it('derives capabilities from the current generate model', async () => {
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
  });

  it('offers only an embedded workflow while the Generate panel has no values', async () => {
    const image: GalleryImage = {
      boardId: 'none',
      hasWorkflow: true,
      height: 512,
      imageCategory: 'general',
      imageName: 'workflow.png',
      imageUrl: '/full/workflow.png',
      queuedAt: '2026-07-30T00:00:00.000Z',
      sourceQueueItemId: 'queue-workflow',
      starred: false,
      thumbnailUrl: '/thumb/workflow.png',
      width: 512,
    };

    // Without the capability table there are no Generate values to recall into.
    await act(() => resetArchitectureCapabilities());
    expect(actionsRef.current?.deriveImageRecallCapabilities(image, { seed: 7 })).toEqual({
      ...EMPTY_IMAGE_RECALL_CAPABILITIES,
      workflow: true,
    });
    await act(() => setArchitectureCapabilities(architectureCapabilitiesFixture));
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
      graph: 'invalid graph ignored when an authored workflow exists',
      resolved_inputs: '{"nodes":{"seed":{"id":"seed","type":"integer","value":18}},"edges":[]}',
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
    expect(mocks.getInvocationTemplatesSnapshot).not.toHaveBeenCalled();
  });

  it('loads the full source generation graph even when resolved inputs are available', async () => {
    mocks.getInvocationTemplatesSnapshot.mockReturnValueOnce({ error: null, status: 'idle', templates: {} });
    mocks.imageWorkflow.mockResolvedValueOnce({
      graph: JSON.stringify({
        id: 'generated',
        nodes: {
          seed: { id: 'seed', type: 'integer', value: 42 },
          iterator: { id: 'iterator', type: 'iterate', collection: [17, 18] },
          collector: { id: 'collector', type: 'collect' },
        },
        edges: [
          { source: { node_id: 'iterator', field: 'item' }, destination: { node_id: 'collector', field: 'item' } },
        ],
      }),
      resolved_inputs: JSON.stringify({
        id: 'resolved',
        nodes: { seed: { id: 'seed', type: 'integer', value: 18 } },
        edges: [],
      }),
      workflow: null,
    });

    await act(async () => {
      await actionsRef.current?.loadImageWorkflow({ ...image, hasWorkflow: true });
    });

    expect(mocks.notificationsAdd.mock.calls).toEqual([]);
    expect(mocks.requestWorkflowDocumentLoad).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: [
          expect.objectContaining({
            id: 'seed',
            data: expect.objectContaining({
              type: 'integer',
              inputs: { value: expect.objectContaining({ value: 42 }) },
            }),
          }),
          expect.objectContaining({
            id: 'iterator',
            data: expect.objectContaining({
              type: 'iterate',
              inputs: { collection: expect.objectContaining({ value: [17, 18] }) },
            }),
          }),
          expect.objectContaining({ id: 'collector', data: expect.objectContaining({ type: 'collect' }) }),
        ],
        edges: [
          expect.objectContaining({
            source: 'iterator',
            sourceHandle: 'item',
            target: 'collector',
            targetHandle: 'item',
          }),
        ],
      }),
      expect.any(String)
    );
    expect(mocks.openWorkbenchWidget).toHaveBeenCalled();
    expect(mocks.refreshInvocationTemplates).toHaveBeenCalledOnce();
  });

  it.each(['{', JSON.stringify({ nodes: [], edges: [] }), JSON.stringify({ nodes: {}, edges: [] })])(
    'reports an unusable generation graph without opening the editor: %s',
    async (graph) => {
      mocks.imageWorkflow.mockResolvedValueOnce({ graph, workflow: null });

      await act(async () => {
        await actionsRef.current?.loadImageWorkflow(image);
      });

      expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
      expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
      expect(mocks.notificationsAdd).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'error', title: 'widgets.gallery.itemActions.loadWorkflow.failed' })
      );
    }
  );

  it.each(['account', 'project'] as const)(
    'does not load the graph after the %s changes while templates load',
    async (changedOwner) => {
      mocks.imageWorkflow.mockResolvedValueOnce({
        graph: JSON.stringify({ nodes: { seed: { id: 'seed', type: 'integer', value: 42 } }, edges: [] }),
        workflow: null,
      });
      mocks.getInvocationTemplatesSnapshot.mockReturnValueOnce({ status: 'idle', templates: {}, error: null });
      let finishLoading!: () => void;
      mocks.refreshInvocationTemplates.mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishLoading = resolve;
          })
      );

      const pendingLoad = actionsRef.current!.loadImageWorkflow(image);
      await vi.waitFor(() => expect(mocks.refreshInvocationTemplates).toHaveBeenCalled());

      await act(async () => {
        if (changedOwner === 'account') {
          accountLifecycle.activate('user-b');
        } else {
          mocks.isActiveProject.mockReturnValue(false);
        }
        finishLoading();
        await pendingLoad;
      });

      expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
      expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
      expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    }
  );

  it('reports a template load failure instead of opening an empty workflow', async () => {
    mocks.imageWorkflow.mockResolvedValueOnce({
      graph: JSON.stringify({ nodes: { seed: { id: 'seed', type: 'integer', value: 42 } }, edges: [] }),
      workflow: null,
    });
    mocks.getInvocationTemplatesSnapshot.mockReturnValue({ status: 'error', templates: {}, error: 'Offline' });

    await act(async () => {
      await actionsRef.current?.loadImageWorkflow(image);
    });

    expect(mocks.refreshInvocationTemplates).toHaveBeenCalledOnce();
    expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', message: 'Offline' }));
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

describe('generate PBR maps', () => {
  const image: GalleryImage = {
    boardId: 'none',
    height: 512,
    imageCategory: 'general',
    imageName: 'texture.png',
    imageUrl: '/full/texture.png',
    queuedAt: '2026-07-30T00:00:00.000Z',
    sourceQueueItemId: 'backend-gallery',
    starred: false,
    thumbnailUrl: '/thumb/texture.png',
    width: 512,
  };
  const createSettlement = () => {
    let resolve!: () => void;
    let reject!: (error: unknown) => void;
    const settled = new Promise<void>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { reject, resolve, settled };
  };

  it('queues one pbr_maps graph into the project auto-add board, then refreshes the gallery when it completes', async () => {
    const settlement = createSettlement();
    mocks.getSnapshot.mockImplementation(() => {
      const project = makeMockProject('project-1', {
        autoAddBoardId: 'board-materials',
        selectedBoardId: 'board-other',
      });

      return { activeProject: project, projects: [project] };
    });
    mocks.submitUtilityGraph.mockResolvedValueOnce({ itemIds: [41], origin: 'utility:1', settled: settlement.settled });

    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });

    expect(mocks.submitUtilityGraph).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        prepend: true,
        signal: expect.any(AbortSignal),
        graph: expect.objectContaining({
          nodes: {
            pbr_maps: expect.objectContaining({
              board: { board_id: 'board-materials' },
              image: { image_name: 'texture.png' },
              is_intermediate: false,
              type: 'pbr_maps',
            }),
          },
        }),
      })
    );
    expect(mocks.notificationsAdd).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ kind: 'success', title: 'widgets.gallery.itemActions.pbrMaps.queued' })
    );
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();

    await act(async () => {
      settlement.resolve();
      await settlement.settled;
    });

    expect(mocks.invalidateGallery).toHaveBeenCalledTimes(1);
    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'success', title: 'widgets.gallery.itemActions.pbrMaps.completed' })
    );
  });

  it('leaves the maps Uncategorized while the gallery follows no board', async () => {
    mocks.getSnapshot.mockImplementation(() => {
      const project = makeMockProject('project-1', { autoAddBoardId: 'none' });

      return { activeProject: project, projects: [project] };
    });
    mocks.submitUtilityGraph.mockResolvedValueOnce({
      itemIds: [42],
      origin: 'utility:2',
      settled: new Promise(() => {}),
    });

    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });

    const [request] = mocks.submitUtilityGraph.mock.calls[0] as [{ graph: { nodes: Record<string, unknown> } }];
    expect(request.graph.nodes.pbr_maps).not.toHaveProperty('board');
  });

  it('waits for board permissions and leaves maps in the inbox when the destination is refused', async () => {
    mocks.getSnapshot.mockImplementation(() => {
      const project = makeMockProject('project-1', { autoAddBoardId: 'view-only' });

      return { activeProject: project, projects: [project] };
    });
    const listing = createSettlement();
    mocks.ensureGalleryBoardDestinations.mockReturnValueOnce(listing.settled);
    mocks.canAddToCachedGalleryBoard.mockReturnValueOnce(false);
    mocks.submitUtilityGraph.mockResolvedValueOnce({
      itemIds: [48],
      origin: 'utility:8',
      settled: new Promise(() => {}),
    });

    await act(async () => {
      const submission = actionsRef.current?.generatePbrMaps(image);

      expect(mocks.submitUtilityGraph).not.toHaveBeenCalled();
      listing.resolve();
      await submission;
    });

    const [request] = mocks.submitUtilityGraph.mock.calls[0] as [{ graph: { nodes: Record<string, unknown> } }];
    expect(request.graph.nodes.pbr_maps).not.toHaveProperty('board');
    expect(mocks.canAddToCachedGalleryBoard).toHaveBeenCalledWith('view-only', expect.any(QueryClient));
  });

  it('does not submit if the account changes while board permissions load', async () => {
    const listing = createSettlement();
    mocks.ensureGalleryBoardDestinations.mockReturnValueOnce(listing.settled);

    await act(async () => {
      const submission = actionsRef.current?.generatePbrMaps(image);

      accountLifecycle.activate('user-b');
      listing.resolve();
      await submission;
    });

    expect(mocks.submitUtilityGraph).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
  });

  it('reports an enqueue failure and a failed run as logged PBR error notices', async () => {
    const { UtilityQueueError } = await import('@features/queue/utility');
    clearLogs();
    // The runtime wraps the backend's JSON refusal as an ApiError cause and derives the readable message the way
    // `submitUtilityGraph` does; the toast must show that detail, never the JSON.
    const cause = new ApiError(JSON.stringify({ detail: 'queue paused' }), 409);
    mocks.submitUtilityGraph.mockRejectedValueOnce(
      new UtilityQueueError('enqueue', getApiErrorMessage(cause, 'Failed to enqueue utility graph.'), cause)
    );

    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledExactlyOnceWith({
      kind: 'error',
      message: 'queue paused',
      title: 'widgets.gallery.itemActions.pbrMaps.failed',
    });

    const settlement = createSettlement();
    mocks.submitUtilityGraph.mockResolvedValueOnce({ itemIds: [43], origin: 'utility:3', settled: settlement.settled });
    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });
    await act(async () => {
      settlement.reject(new UtilityQueueError('failed', 'No PBR model'));
      await settlement.settled.catch(() => {});
    });

    // A run that fails after it was queued names the action, as an enqueue failure does.
    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith({
      kind: 'error',
      message: 'No PBR model',
      title: 'widgets.gallery.itemActions.pbrMaps.failed',
    });
    expect(mocks.reportError).not.toHaveBeenCalled();
    expect(
      getLogSnapshot()
        .entries.filter((entry) => entry.source.area === 'image-actions')
        .map(({ level, message, namespace }) => ({ level, message, namespace }))
    ).toEqual([
      { level: 'error', message: 'No PBR model', namespace: 'gallery' },
      { level: 'error', message: 'queue paused', namespace: 'gallery' },
    ]);
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('turns a canceled or removed item into an info notice, and says nothing once the watch is aborted', async () => {
    const { UtilityQueueError } = await import('@features/queue/utility');
    const canceled = createSettlement();
    mocks.submitUtilityGraph.mockResolvedValueOnce({ itemIds: [44], origin: 'utility:4', settled: canceled.settled });
    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });
    await act(async () => {
      canceled.reject(new UtilityQueueError('canceled', 'The utility graph was canceled.'));
      await canceled.settled.catch(() => {});
    });

    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith(
      expect.objectContaining({ kind: 'info', title: 'widgets.gallery.itemActions.pbrMaps.canceled' })
    );
    expect(mocks.reportError).not.toHaveBeenCalled();

    // A queue clear deletes a pending item without a terminal event; the watch reports it as removed.
    const removed = createSettlement();
    mocks.submitUtilityGraph.mockResolvedValueOnce({ itemIds: [47], origin: 'utility:7', settled: removed.settled });
    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });
    await act(async () => {
      removed.reject(new UtilityQueueError('removed', 'The utility graph item was removed from the queue.'));
      await removed.settled.catch(() => {});
    });

    expect(mocks.notificationsAdd).toHaveBeenLastCalledWith({
      kind: 'info',
      message: 'widgets.gallery.itemActions.pbrMaps.removedMessage',
      title: 'widgets.gallery.itemActions.pbrMaps.removed',
    });
    // The maps may already be in the gallery if the item completed before it was pruned.
    expect(mocks.invalidateGallery).toHaveBeenCalledTimes(1);
    expect(mocks.reportError).not.toHaveBeenCalled();

    const aborted = createSettlement();
    mocks.notificationsAdd.mockClear();
    mocks.submitUtilityGraph.mockResolvedValueOnce({ itemIds: [45], origin: 'utility:5', settled: aborted.settled });
    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });
    await act(async () => {
      aborted.reject(new UtilityQueueError('aborted', 'The utility graph watch was aborted.'));
      await aborted.settled.catch(() => {});
    });

    expect(mocks.notificationsAdd).toHaveBeenCalledTimes(1);
    expect(mocks.reportError).not.toHaveBeenCalled();
  });

  it('suppresses every notice when the account changes while the graph is being accepted', async () => {
    const settlement = createSettlement();
    mocks.submitUtilityGraph.mockImplementationOnce(() => {
      accountLifecycle.activate('user-b');
      return Promise.resolve({ itemIds: [46], origin: 'utility:6', settled: settlement.settled });
    });

    await act(async () => {
      await actionsRef.current?.generatePbrMaps(image);
    });
    await act(async () => {
      settlement.resolve();
      await settlement.settled;
    });

    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });
});

describe('video workflow loading', () => {
  const video = {
    boardId: 'none',
    category: 'general' as const,
    createdAt: '2026-07-30T00:00:00.000Z',
    durationSeconds: 4,
    fullUrl: '/full/clip.mp4',
    height: 512,
    isIntermediate: false,
    kind: 'video' as const,
    name: 'clip.mp4',
    starred: false,
    thumbnailUrl: '/thumb/clip.mp4',
    width: 512,
  };

  it('hands the embedded workflow to the editor and opens it', async () => {
    mocks.videoWorkflow.mockResolvedValueOnce({
      graph: null,
      workflow: JSON.stringify({ name: 'Embedded', nodes: [] }),
    });

    await act(async () => {
      await actionsRef.current?.loadVideoWorkflow(video);
    });

    expect(mocks.videoWorkflow).toHaveBeenCalledWith('clip.mp4', expect.any(AbortSignal));
    expect(mocks.openWorkbenchWidget).toHaveBeenCalledWith('workflow', {
      preferredRegions: ['center'],
      requireCenterView: true,
    });
    expect(mocks.requestWorkflowDocumentLoad).toHaveBeenCalledWith(
      { name: 'Embedded', nodes: [] },
      'widgets.gallery.itemActions.loadWorkflow.loadedLabelVideo:Embedded'
    );
  });

  it('says so, and leaves the editor alone, when the video embeds no workflow', async () => {
    await act(async () => {
      await actionsRef.current?.loadVideoWorkflow(video);
    });

    expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
    expect(mocks.openWorkbenchWidget).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', title: 'widgets.gallery.itemActions.loadWorkflow.missingVideo' })
    );
  });

  it('reports when the workflow widget cannot open', async () => {
    mocks.videoWorkflow.mockResolvedValueOnce({ graph: null, workflow: JSON.stringify({ nodes: [] }) });
    mocks.openWorkbenchWidget.mockReturnValueOnce({ ok: false, reason: 'unavailable' });

    await act(async () => {
      await actionsRef.current?.loadVideoWorkflow(video);
    });

    expect(mocks.requestWorkflowDocumentLoad).not.toHaveBeenCalled();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', title: 'widgets.gallery.itemActions.loadWorkflow.failedVideo' })
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
  deleteItems(
    refs: Array<{ kind: 'image' | 'video'; name: string }>,
    options?: { returnFocus?: () => HTMLElement | null; targetsOffScreen?: boolean }
  ): Promise<void>;
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
    const returnFocus = () => null;
    await act(async () => {
      await getItemActions().deleteItems(refs, { returnFocus });
    });

    // The dialog learns where focus goes once the deletion has removed the control that asked for it.
    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: true,
      returnFocus,
      usage: [],
    });
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

  it.each([
    [
      'more than one item',
      [
        { kind: 'image' as const, name: 'a.png' },
        { kind: 'video' as const, name: 'b.mp4' },
      ],
      false,
    ],
    ['an item off screen', [{ kind: 'image' as const, name: 'a.png' }], true],
  ])('asks before deleting %s even with routine confirmations off', async (_case, refs, targetsOffScreen) => {
    mocks.requestDeletionConfirmation.mockImplementation(() => Promise.resolve());

    await act(async () => {
      await getItemActions().deleteItems(refs, { targetsOffScreen });
    });

    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: false,
      returnFocus: undefined,
      usage: [],
    });
    expect(mocks.itemDelete).not.toHaveBeenCalled();
  });

  it('still confirms, naming the surfaces, when the project uses an item and confirmation is off', async () => {
    const refs = [
      { kind: 'image' as const, name: 'source.png' },
      { kind: 'image' as const, name: 'unused.png' },
    ];
    let executeDeletion: (() => Promise<void>) | null = null;
    mocks.requestDeletionConfirmation.mockImplementation((_itemRefs, execute) => {
      executeDeletion = execute;
      return Promise.resolve();
    });
    mocks.getProject.mockReturnValue(
      makeMockProject('project-1', {}, { inputImage: { height: 1, image_name: 'source.png', width: 1 } })
    );
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: false,
      returnFocus: undefined,
      usage: [{ names: ['source.png'], surface: 'upscale' }],
    });
    expect(mocks.itemDelete).not.toHaveBeenCalled();

    await act(async () => {
      await executeDeletion?.();
    });
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

  it('reports a move to another board with an Undo that returns each item to the board it came from', async () => {
    const refs = [
      { kind: 'image' as const, name: 'a.png' },
      { kind: 'video' as const, name: 'b.mp4' },
      { kind: 'image' as const, name: 'unknown.png' },
    ];
    mocks.getItemBoardIds.mockReturnValue(
      new Map([
        ['image:a.png', 'board-1'],
        ['video:b.mp4', 'none'],
      ])
    );
    mocks.itemMoveToBoard.mockResolvedValue({ affectedBoardIds: [], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-2');
    });

    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    // The hook's board list does not name board-2.
    expect(mocks.notifyUndoable).toHaveBeenCalledExactlyOnceWith('Moved to widgets.gallery.unknownBoard', {
      onUndo: expect.any(Function),
      projectId: 'project-1',
    });

    mocks.getItemBoardIds.mockReturnValue(new Map());
    const [, { onUndo }] = mocks.notifyUndoable.mock.calls[0] as [string, { onUndo: () => void }];
    await act(async () => {
      onUndo();
      await Promise.resolve();
    });

    // Each back to its own board; one whose board was not known stays where it went.
    expect(mocks.itemMoveToBoard.mock.calls.slice(1).map(([moved, boardId]) => [moved, boardId])).toEqual([
      [[refs[0]], 'board-1'],
      [[refs[1]], 'none'],
    ]);
    // Undoing reports plainly: there is no Undo of the Undo.
    expect(mocks.notifyUndoable).toHaveBeenCalledOnce();
    // The item whose board was not known is reported, not silently skipped.
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'info',
      title: 'widgets.gallery.itemActions.undoStale.partial:{"count":3,"restored":2}',
    });
  });

  it('leaves an item a later move took elsewhere when an earlier move is undone', async () => {
    const refs = [{ kind: 'image' as const, name: 'moved-twice.png' }];
    mocks.itemMoveToBoard.mockResolvedValue({ affectedBoardIds: [], failed: [], succeeded: refs });

    mocks.getItemBoardIds.mockReturnValue(new Map([['image:moved-twice.png', 'board-a']]));
    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-b');
    });
    mocks.getItemBoardIds.mockReturnValue(new Map([['image:moved-twice.png', 'board-b']]));
    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-c');
    });
    mocks.getItemBoardIds.mockReturnValue(new Map([['image:moved-twice.png', 'board-c']]));

    const [[, first], [, second]] = mocks.notifyUndoable.mock.calls as [string, { onUndo: () => void }][];
    await act(async () => {
      first.onUndo();
      await Promise.resolve();
    });

    expect(mocks.itemMoveToBoard).toHaveBeenCalledTimes(2);
    expect(mocks.notificationsAdd).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      title: 'widgets.gallery.itemActions.undoStale.none:{"count":1}',
    });

    // The newer move's Undo still applies.
    await act(async () => {
      second.onUndo();
      await Promise.resolve();
    });
    expect(mocks.itemMoveToBoard).toHaveBeenLastCalledWith(refs, 'board-b', expect.any(AbortSignal));
  });

  it('leaves an item a later unstar changed when an earlier star is undone', async () => {
    const refs = [{ kind: 'image' as const, name: 'toggled.png' }];
    mocks.itemSetStarred.mockResolvedValue({ affectedBoardIds: [], failed: [], succeeded: refs });

    mocks.getItemStarred.mockReturnValue(new Map([['image:toggled.png', false]]));
    await act(async () => {
      await getItemActions().setItemsStarred(refs, true);
    });
    mocks.getItemStarred.mockReturnValue(new Map([['image:toggled.png', true]]));
    await act(async () => {
      await getItemActions().setItemsStarred(refs, false);
    });
    mocks.getItemStarred.mockReturnValue(new Map([['image:toggled.png', false]]));

    const [, { onUndo }] = mocks.notifyUndoable.mock.calls[0] as [string, { onUndo: () => void }];
    await act(async () => {
      onUndo();
      await Promise.resolve();
    });

    expect(mocks.itemSetStarred).toHaveBeenCalledTimes(2);
    expect(mocks.notificationsAdd).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      title: 'widgets.gallery.itemActions.undoStale.none:{"count":1}',
    });
  });

  it('reports a star with an Undo that gives each item back its own flag', async () => {
    const refs = [
      { kind: 'image' as const, name: 'was-starred.png' },
      { kind: 'image' as const, name: 'was-not.png' },
      { kind: 'image' as const, name: 'unknown.png' },
    ];
    mocks.getItemStarred.mockReturnValue(
      new Map([
        ['image:was-starred.png', true],
        ['image:was-not.png', false],
      ])
    );
    mocks.itemSetStarred.mockResolvedValue({ affectedBoardIds: [], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().setItemsStarred(refs, true);
    });

    mocks.getItemStarred.mockReturnValue(new Map());
    const [, { onUndo }] = mocks.notifyUndoable.mock.calls[0] as [string, { onUndo: () => void }];
    await act(async () => {
      onUndo();
      await Promise.resolve();
    });

    // Only the item the star is known to have changed is unstarred again; one whose flag was not known stays.
    expect(mocks.itemSetStarred).toHaveBeenLastCalledWith([refs[1]], false, expect.any(AbortSignal));
    expect(mocks.itemSetStarred).toHaveBeenCalledTimes(2);
    // The item whose flag was not known is reported, not silently skipped.
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'info',
      title: 'widgets.gallery.itemActions.undoStale.partial:{"count":2,"restored":1}',
    });
  });

  it('offers no Undo for a move that partly failed', async () => {
    const refs = [
      { kind: 'image' as const, name: 'a.png' },
      { kind: 'image' as const, name: 'b.png' },
    ];
    mocks.getItemBoardIds.mockReturnValue(new Map([['image:a.png', 'board-1']]));
    mocks.itemMoveToBoard.mockResolvedValue({ affectedBoardIds: [], failed: [refs[1]], succeeded: [refs[0]] });

    await act(async () => {
      await getItemActions().moveItemsToBoard(refs, 'board-2');
    });

    expect(mocks.notifyUndoable).not.toHaveBeenCalled();
    expect(mocks.reportError).toHaveBeenCalledOnce();
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
    // The rollback restored every page's total, so the re-removal lowers them across the listing again.
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      { kind: 'delete', result },
      { totals: 'listing' }
    );
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(2, ['video:gone.mp4']);
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd.mock.calls.length + mocks.reportError.mock.calls.length).toBe(1);
  });

  it('re-removes a fully confirmed deletion only from pages still holding it, without lowering totals again', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    const result = { affectedBoardIds: ['board-1'], failed: [], succeeded: refs };
    mocks.itemDelete.mockResolvedValue(result);

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).not.toHaveBeenCalled();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      { kind: 'delete', result },
      { totals: 'holder' }
    );
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

  it('restores the selection when the server rejects every item instead of failing the request', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    const selected = { kind: 'image', name: 'gone.png' };
    const before = makeMockProject('project-1', {
      selectedImage: selected,
      selectedImageName: 'image:gone.png',
      selectedImageNames: ['image:gone.png'],
    });
    const afterRemoval = makeMockProject('project-1', {
      selectedImage: null,
      selectedImageName: null,
      selectedImageNames: [],
    });

    mocks.getSnapshot.mockReturnValueOnce({ activeProject: before, projects: [before] }).mockReturnValue({
      activeProject: afterRemoval,
      projects: [afterRemoval],
    });
    // A rejected request is reported as a result whose items all failed.
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: [], failed: refs, succeeded: [] });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.galleryWidgetsPatchValues).toHaveBeenCalledWith(
      'gallery',
      { selectedImage: selected, selectedImageName: 'image:gone.png', selectedImageNames: ['image:gone.png'] },
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
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      { kind: 'delete', result },
      { totals: 'holder' }
    );
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

/** Backs the workbench snapshot with a gallery selection that removals and selections change, as the store does. */
const modelGallerySelectionStore = (initial: GalleryItemKey | null) => {
  let selected = initial;
  mocks.getSnapshot.mockImplementation(() => {
    const project = makeMockProject('project-1', { selectedImageName: selected });

    return { activeProject: project, projects: [project] };
  });

  return {
    remove: (itemKeys: GalleryItemKey[]) => {
      if (selected && itemKeys.includes(selected)) {
        selected = null;
      }
    },
    select: (itemKey: GalleryItemKey) => {
      selected = itemKey;
    },
  };
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

  it('selects the successor once the optimistic removal has cleared the host selection', async () => {
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('image', 'after.png');
    const refs = [primary, after].map(({ kind, name }) => ({ kind, name }));
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, after],
      loadOrderedRefs: () => Promise.resolve(refs),
      selectedItemKey: 'image:primary.png',
    };
    const store = modelGallerySelectionStore('image:primary.png');
    mocks.galleryRemoveItems.mockImplementationOnce((itemKeys: GalleryItemKey[]) => {
      store.remove(itemKeys);
      currentItemActionContext = { ...currentItemActionContext!, items: [after], selectedItemKey: null };
    });
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

  it('selects the successor beside a rejected item when the host has re-rendered without the primary', async () => {
    const primary = galleryItem('image', 'primary.png');
    const rejected = galleryItem('image', 'rejected.png');
    const after = galleryItem('image', 'after.png');
    const refs = [primary, rejected, after].map(({ kind, name }) => ({ kind, name }));
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, rejected, after],
      loadOrderedRefs: () => Promise.resolve(refs),
      selectedItemKey: 'image:primary.png',
    };
    const store = modelGallerySelectionStore('image:primary.png');
    mocks.galleryRemoveItems.mockImplementationOnce((itemKeys: GalleryItemKey[]) => {
      store.remove(itemKeys);
      currentItemActionContext = { ...currentItemActionContext!, items: [after], selectedItemKey: null };
    });
    // Restoring the rejected item's removal writes the selection back to the store; the host has not re-rendered.
    mocks.galleryWidgetsPatchValues.mockImplementation((_widgetId: string, values: Record<string, unknown>) => {
      if (typeof values.selectedImageName === 'string') {
        store.select(values.selectedImageName as GalleryItemKey);
      }
    });
    mocks.itemDelete.mockResolvedValue({
      affectedBoardIds: ['board-1'],
      failed: [{ kind: 'image', name: rejected.name }],
      succeeded: [{ kind: 'image', name: primary.name }],
    });

    await act(async () => {
      await getItemActions().deleteItems([
        { kind: 'image', name: primary.name },
        { kind: 'image', name: rejected.name },
      ]);
    });

    expect(mocks.gallerySetItemMultiSelection).toHaveBeenCalledWith(
      ['image:rejected.png', 'image:after.png'],
      after,
      'project-1'
    );
  });

  it('keeps an item selected during the deletion even when the host has not loaded it', async () => {
    const primary = galleryItem('image', 'primary.png');
    const after = galleryItem('image', 'after.png');
    const refs = [primary, after].map(({ kind, name }) => ({ kind, name }));
    currentItemActionContext = {
      filterIdentity: 'filter-a',
      items: [primary, after],
      loadOrderedRefs: () => Promise.resolve(refs),
      selectedItemKey: 'image:primary.png',
    };
    const store = modelGallerySelectionStore('image:primary.png');
    mocks.galleryRemoveItems.mockImplementationOnce((itemKeys: GalleryItemKey[]) => {
      store.remove(itemKeys);
      currentItemActionContext = { ...currentItemActionContext!, items: [after], selectedItemKey: null };
    });
    // While the request is in flight the user selects an item on a page the host has not loaded, so the host
    // still reports no selection.
    mocks.itemDelete.mockImplementation(() => {
      store.select('image:far-away.png');

      return Promise.resolve({
        affectedBoardIds: ['board-1'],
        failed: [],
        succeeded: [{ kind: 'image', name: primary.name }],
      });
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: primary.name }]);
    });

    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
    expect(mocks.gallerySetItemMultiSelection).not.toHaveBeenCalled();
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
      30,
      true
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

import type { GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type * as GalleryMediaUsageModule from '@features/gallery/mediaUsage';
import type { GalleryMediaUsageReport } from '@features/gallery/mediaUsage';
import type * as QueueUtilityModule from '@features/queue/utility';
import type { InvocationTemplatesSnapshot } from '@features/workflow/contracts';
import type * as WorkflowReact from '@features/workflow/react';
import type { CreateCanvasFromImagesResult } from '@workbench/canvas-operations/api';
import type * as SettingsStoreModule from '@workbench/settings/store';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { parseOpenApiToTemplates } from '@features/workflow/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';

import type * as MediaUsageReportModule from './mediaUsageReport';
import type { ImageActions } from './useImageActions';

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
  open: vi.fn((..._args: unknown[]) => ({ changedWorkspace: false, ok: true, region: 'center', workspaceId: 'edit' })),
  patchGalleryItemCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  removeFromBoard: vi.fn(),
  reportError: vi.fn(),
  revealLayerProperties: vi.fn(),
  sendGalleryImageAsControl: vi.fn(),
  requestDeletionConfirmation: vi.fn(),
  resolveItem: vi.fn(),
  resolveMediaUsage: vi.fn<(...args: unknown[]) => Promise<GalleryMediaUsageReport>>(),
  checkGalleryMediaUsage: vi.fn(),
  setStarred: vi.fn(),
}));
const preferences = vi.hoisted(() => ({ confirmImageDeletion: false }));

vi.mock('@features/gallery', () => ({
  GALLERY_REQUEST_MAX_NAMES: 1000,
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

// One test runs the real resolver over this tab's projects, with the server's answer doubled.
vi.mock('./mediaUsageReport', async (importOriginal) => ({
  ...(await importOriginal<typeof MediaUsageReportModule>()),
  resolveMediaUsage: (...args: unknown[]) => mocks.resolveMediaUsage(...args),
}));
vi.mock('@features/gallery/mediaUsage', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryMediaUsageModule>()),
  checkGalleryMediaUsage: (...args: unknown[]) => mocks.checkGalleryMediaUsage(...args),
}));

vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ undoable: (...args: unknown[]) => mocks.notifyUndoable(...args) }),
}));

vi.mock('@workbench/workspace/useWorkspaceOpen', () => ({
  useWorkspaceOpen: () => mocks.open,
}));

vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof SettingsStoreModule>()),
  useWorkbenchPreferenceSelector: (selector: (value: typeof preferences) => boolean) => selector(preferences),
}));

vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: (...args: unknown[]) => mocks.downloadBlob(...args),
}));

vi.mock('@workbench/canvas-operations/api', async () => {
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
      getSnapshot: (...args: unknown[]) => ({ scopedDefaults: {}, ...(mocks.getSnapshot(...args) as object) }),
      isActiveProject: (...args: unknown[]) => mocks.isActiveProject(...args),
    }),
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
      if (
        key.startsWith('widgets.gallery.itemActions.undoStale.') ||
        key.startsWith('widgets.gallery.itemActions.download.')
      ) {
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

const makeMockProject = (
  id: string,
  galleryValues: Record<string, unknown> = {},
  upscaleValues: Record<string, unknown> = {}
) => ({
  canvas: {
    document: { stacks: { control: [], inpaint_mask: [], raster: [], regional_guidance: [] } },
    snapshots: [],
    stagingArea: { pendingImages: [] },
  },
  id,
  name: id,
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

/** Serves `initial` until the optimistic removal applies, then `removed`; `set` moves the store on. */
const serveProjectStates = (
  initial: ReturnType<typeof makeMockProject>,
  removed: ReturnType<typeof makeMockProject>
) => {
  let current = initial;

  mocks.getSnapshot.mockImplementation(() => ({ activeProject: current, projects: [current] }));
  mocks.galleryRemoveItems.mockImplementation(() => {
    current = removed;
  });
  onTestFinished(() => {
    mocks.galleryRemoveItems.mockReset();
  });

  return {
    set: (project: ReturnType<typeof makeMockProject>) => {
      current = project;
    },
  };
};

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

const usageReport = (overrides: Partial<GalleryMediaUsageReport> = {}): GalleryMediaUsageReport => ({
  checkedCount: 1,
  groups: [],
  inUseCount: 0,
  onlyImages: true,
  scope: 'items',
  serverCheck: 'complete',
  ...overrides,
});

beforeEach(() => {
  vi.clearAllMocks();
  preferences.confirmImageDeletion = false;
  mocks.resolveMediaUsage.mockResolvedValue(usageReport());
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

// Seed after account activation and before mounting so reset and teardown stay inside act.
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
    expect(mocks.open).toHaveBeenCalledWith({ surface: 'stage', typeId: 'canvas' });
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
    expect(mocks.open).toHaveBeenCalledWith({ surface: 'stage', typeId: 'canvas' });
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
    expect(mocks.open).not.toHaveBeenCalled();
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

  it('hands the embedded workflow to the editor and opens it', async () => {
    mocks.imageWorkflow.mockResolvedValueOnce({
      graph: 'invalid graph ignored when an authored workflow exists',
      resolved_inputs: '{"nodes":{"seed":{"id":"seed","type":"integer","value":18}},"edges":[]}',
      workflow: JSON.stringify({ name: 'Embedded', nodes: [] }),
    });

    await act(async () => {
      await actionsRef.current?.loadImageWorkflow(image);
    });

    expect(mocks.open).toHaveBeenCalledWith({ surface: 'stage', typeId: 'workflow' });
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
    expect(mocks.open).toHaveBeenCalled();
    expect(mocks.refreshInvocationTemplates).toHaveBeenCalledOnce();
  });

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
      expect(mocks.open).not.toHaveBeenCalled();
      expect(mocks.notificationsAdd).not.toHaveBeenCalled();
    }
  );
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
    expect(mocks.open).toHaveBeenCalledWith({ surface: 'stage', typeId: 'workflow' });
    expect(mocks.requestWorkflowDocumentLoad).toHaveBeenCalledWith(
      { name: 'Embedded', nodes: [] },
      'widgets.gallery.itemActions.loadWorkflow.loadedLabelVideo:Embedded'
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
  it('gates deletion through confirmation when the preference is enabled, checking usage as the dialog opens', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    const result = { affectedBoardIds: ['none'], failed: [], succeeded: refs };
    let executeDeletion: (() => Promise<void>) | null = null;
    const captured: { checkUsage?: (signal: AbortSignal) => Promise<GalleryMediaUsageReport> } = {};
    preferences.confirmImageDeletion = true;
    mocks.requestDeletionConfirmation.mockImplementation((_itemRefs, execute, options) => {
      executeDeletion = execute;
      captured.checkUsage = options.checkUsage;
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

    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: true,
      checkUsage: expect.any(Function),
      returnFocus,
    });
    expect(mocks.resolveMediaUsage).not.toHaveBeenCalled();
    expect(captured.checkUsage).toBeDefined();
    expect(mocks.itemDelete).not.toHaveBeenCalled();

    await act(async () => {
      await executeDeletion?.();
    });
    expect(mocks.itemDelete).toHaveBeenCalledWith(refs, expect.any(AbortSignal));
  });

  it('deletes one visible item without asking once the check finds nothing using it', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    // A click is waiting: a few seconds per stage, not the dialog's longer allowance.
    expect(mocks.resolveMediaUsage).toHaveBeenCalledWith({ kind: 'items', refs }, expect.anything(), {
      projectReadsTimeoutMs: 3000,
      serverTimeoutMs: 3000,
    });
    expect(mocks.requestDeletionConfirmation).not.toHaveBeenCalled();
    expect(mocks.itemDelete).toHaveBeenCalledWith(refs, expect.any(AbortSignal));
  });

  it.each([
    [
      'something uses it',
      usageReport({
        groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'upscale' }] }],
        inUseCount: 1,
      }),
    ],
    ['the check could not reach the server', usageReport({ serverCheck: 'unavailable' })],
  ])('still asks with routine confirmations off when %s, showing the report', async (_case, usage) => {
    const refs = [{ kind: 'image' as const, name: 'source.png' }];
    let executeDeletion: (() => Promise<void>) | null = null;
    mocks.resolveMediaUsage.mockResolvedValue(usage);
    mocks.requestDeletionConfirmation.mockImplementation((_itemRefs, execute) => {
      executeDeletion = execute;
      return Promise.resolve();
    });
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: true,
      checkUsage: expect.any(Function),
      returnFocus: undefined,
      usage,
    });
    expect(mocks.itemDelete).not.toHaveBeenCalled();

    await act(async () => {
      await executeDeletion?.();
    });
    expect(mocks.itemDelete).toHaveBeenCalledWith(refs, expect.any(AbortSignal));
  });

  it('blocks the silent deletion of an item a project in this tab uses but has not saved yet', async () => {
    const { resolveMediaUsage } = await vi.importActual<typeof MediaUsageReportModule>('./mediaUsageReport');
    const refs = [{ kind: 'image' as const, name: 'source.png' }];
    mocks.resolveMediaUsage.mockImplementation(resolveMediaUsage as (...args: unknown[]) => Promise<never>);
    mocks.checkGalleryMediaUsage.mockResolvedValue({
      checkedImageCount: 1,
      checkedVideoCount: 0,
      documents: [],
      documentsTotal: 0,
      inUseImageCount: 0,
      inUseVideoCount: 0,
      otherAccounts: { documentCount: 0, imageCount: 0, queueItemCount: 0, videoCount: 0 },
      queue: { imageCount: 0, imageNames: [], itemCount: 0, videoCount: 0, videoNames: [] },
    });
    mocks.getSnapshot.mockImplementation(() => {
      const project = makeMockProject(
        'project-1',
        {},
        { inputImage: { height: 1, image_name: 'source.png', width: 1 } }
      );

      return { activeProject: project, projects: [project] };
    });
    mocks.requestDeletionConfirmation.mockImplementation(() => Promise.resolve());

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.itemDelete).not.toHaveBeenCalled();
    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(
      refs,
      expect.any(Function),
      expect.objectContaining({
        usage: expect.objectContaining({
          groups: [{ count: 1, kind: 'openProject', surfaces: [{ count: 1, surface: 'upscale' }] }],
          serverCheck: 'complete',
        }),
      })
    );
  });

  it('asks instead when this tab starts using the item while the check runs, and has the dialog check again', async () => {
    const refs = [{ kind: 'image' as const, name: 'source.png' }];
    mocks.resolveMediaUsage.mockImplementation(() => {
      mocks.getSnapshot.mockImplementation(() => {
        const project = makeMockProject(
          'project-1',
          {},
          { inputImage: { height: 1, image_name: 'source.png', width: 1 } }
        );

        return { activeProject: project, projects: [project] };
      });
      return Promise.resolve(usageReport());
    });
    mocks.requestDeletionConfirmation.mockImplementation(() => Promise.resolve());

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.itemDelete).not.toHaveBeenCalled();
    expect(mocks.requestDeletionConfirmation).toHaveBeenCalledWith(refs, expect.any(Function), {
      canSkipFutureConfirmations: true,
      checkUsage: expect.any(Function),
      returnFocus: undefined,
    });
  });

  it('runs one check and one deletion for repeated presses, saying so while a slow check runs', async () => {
    const refs = [{ kind: 'image' as const, name: 'image.png' }];
    let settle: (report: GalleryMediaUsageReport) => void = () => {};
    const createToast = vi.spyOn(toaster, 'create');
    const dismissToast = vi.spyOn(toaster, 'dismiss');
    mocks.resolveMediaUsage.mockImplementation(
      () =>
        new Promise((resolve) => {
          settle = resolve;
        })
    );
    mocks.itemDelete.mockResolvedValue({ affectedBoardIds: ['none'], failed: [], succeeded: refs });

    let first!: Promise<void>;
    await act(() => {
      first = getItemActions().deleteItems(refs);
      void getItemActions().deleteItems(refs);
      void getItemActions().deleteItems([...refs]);
    });
    await vi.waitFor(() => expect(createToast).toHaveBeenCalledWith(expect.objectContaining({ type: 'loading' })));

    await act(async () => {
      settle(usageReport());
      await first;
    });

    expect(mocks.resolveMediaUsage).toHaveBeenCalledOnce();
    expect(mocks.itemDelete).toHaveBeenCalledOnce();
    expect(dismissToast).toHaveBeenCalledWith(createToast.mock.results[0]?.value);
    expect(mocks.reportError).not.toHaveBeenCalled();

    mocks.resolveMediaUsage.mockResolvedValue(usageReport());
    await act(async () => {
      await getItemActions().deleteItems(refs);
    });
    expect(mocks.resolveMediaUsage).toHaveBeenCalledTimes(2);
    createToast.mockRestore();
    dismissToast.mockRestore();
  });

  it('neither deletes nor asks when the account changes during the check', async () => {
    mocks.resolveMediaUsage.mockImplementation(() => {
      accountLifecycle.invalidate();
      accountLifecycle.activate('user-b');
      return Promise.reject(new Error('Account scope is no longer active.'));
    });

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: 'image.png' }]);
    });

    expect(mocks.requestDeletionConfirmation).not.toHaveBeenCalled();
    expect(mocks.itemDelete).not.toHaveBeenCalled();
  });

  it('asks, with the check failed, when the check cannot run at all (its code did not load)', async () => {
    const failure = new Error('Failed to fetch dynamically imported module');
    mocks.resolveMediaUsage.mockRejectedValue(failure);
    mocks.requestDeletionConfirmation.mockImplementation(() => Promise.resolve());

    await act(async () => {
      await getItemActions().deleteItems([{ kind: 'image', name: 'image.png' }]);
    });

    const options = mocks.requestDeletionConfirmation.mock.calls[0]?.[2] as {
      checkUsage: (signal: AbortSignal) => Promise<unknown>;
    };
    await expect(options.checkUsage(new AbortController().signal)).rejects.toBe(failure);
    expect(mocks.itemDelete).not.toHaveBeenCalled();
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
      checkUsage: expect.any(Function),
      returnFocus: undefined,
    });
    expect(mocks.itemDelete).not.toHaveBeenCalled();
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

    expect(mocks.itemMoveToBoard.mock.calls.slice(1).map(([moved, boardId]) => [moved, boardId])).toEqual([
      [[refs[0]], 'board-1'],
      [[refs[1]], 'none'],
    ]);
    expect(mocks.notifyUndoable).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).toHaveBeenCalledWith({
      kind: 'info',
      title: 'widgets.gallery.itemActions.undoStale.partial:{"count":3,"restored":2}',
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

    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(1, expect.anything(), {
      kind: 'delete',
      result: { failed: [], succeeded: refs },
    });
    expect(mocks.patchGalleryItemCaches.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.itemDelete.mock.invocationCallOrder[0] ?? 0
    );
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(1, ['video:gone.mp4', 'image:locked.png'], {
      keepComparisonForSuccessor: true,
      projectId: 'project-1',
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      { kind: 'delete', result },
      { totals: 'listing' }
    );
    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(2, ['video:gone.mp4'], {
      keepComparisonForSuccessor: false,
      projectId: 'project-1',
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

    expect(mocks.galleryRemoveItems).toHaveBeenNthCalledWith(1, ['image:gone.png'], {
      keepComparisonForSuccessor: true,
      projectId: 'project-1',
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();

    // The removal stub leaves widget state unchanged; galleryOptimisticRollback.test.ts covers the real reducer.
    expect(mocks.galleryWidgetsPatchValues).not.toHaveBeenCalled();

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
  });

  it('does not clobber a gallery widget field something else changed before the rollback runs', async () => {
    const refs = [{ kind: 'image' as const, name: 'gone.png' }];
    const before = makeMockProject('project-1', { recentImages: [recentImageFixture] });
    const afterRemoval = makeMockProject('project-1', { recentImages: [] });
    const concurrentlyChanged = makeMockProject('project-1', {
      recentImages: [{ ...recentImageFixture, imageName: 'new-generation.png' }],
    });

    const store = serveProjectStates(before, afterRemoval);
    mocks.itemDelete.mockImplementation(() => {
      store.set(concurrentlyChanged);
      return Promise.reject(new Error('network down'));
    });
    mocks.invalidateGallery.mockRejectedValue(new Error('network down'));

    await act(async () => {
      await getItemActions().deleteItems(refs);
    });

    expect(mocks.galleryWidgetsPatchValues).not.toHaveBeenCalled();
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
    // patchItems is stubbed, so simulate the optimistic board state.
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

    expect(mocks.galleryPatchItems).toHaveBeenNthCalledWith(1, ['image:moved.png', 'image:overlay-only.png'], {
      boardId: 'board-1',
    });

    expect(mocks.patchGalleryItemCaches.mock.results[0]?.value).toHaveBeenCalledOnce();

    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:moved.png'], { boardId: 'board-0' });
    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:overlay-only.png'], { boardId: 'board-overlay' });

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(mocks.notificationsAdd).not.toHaveBeenCalled();
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
    await act(() => accountLifecycle.activate('user-b'));
    resolveItem(successor);
    await act(() => deletion);

    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledOnce();
    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledWith(expect.anything(), {
      kind: 'delete',
      result: { failed: [], succeeded: [{ kind: 'image', name: primary.name }] },
    });
    expect(mocks.galleryRemoveItems).toHaveBeenCalledOnce();
    expect(mocks.gallerySelectItem).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
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

  it('downloads a selection larger than one archive request takes as several archives', async () => {
    const imageNames = Array.from({ length: 2500 }, (_, index) => `image-${index}.png`);
    mocks.downloadArchive.mockImplementation(({ imageNames: archiveNames }: { imageNames: string[] }) =>
      Promise.resolve({ blob: new Blob(['archive']), fileName: `${archiveNames[0]}.zip` })
    );
    vi.stubGlobal('fetch', vi.fn());

    await act(async () => {
      await getItemActions().downloadItems(imageNames.map((name) => ({ kind: 'image', name })));
    });

    expect(mocks.downloadArchive.mock.calls.map(([request]) => request.imageNames)).toEqual([
      imageNames.slice(0, 1000),
      imageNames.slice(1000, 2000),
      imageNames.slice(2000),
    ]);
    expect(mocks.downloadBlob.mock.calls.map(([, name]) => name)).toEqual([
      'image-0.png.zip',
      'image-1000.png.zip',
      'image-2000.png.zip',
    ]);
    expect(mocks.notificationsAdd.mock.calls.map(([notice]) => notice)).toEqual([
      {
        kind: 'info',
        message: 'widgets.gallery.itemActions.download.preparingArchives:{"count":3}',
        title: 'widgets.gallery.preparingDownload',
      },
      { kind: 'success', title: 'widgets.gallery.itemActions.download.successArchives:{"count":3}' },
    ]);
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
});

import type { GalleryImage } from '@features/gallery';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { RasterCompositeExportResult } from '@workbench/canvas-engine/exportRasterComposite';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { uploadCanvasImage } from './backend/canvasImages';

import { createFromBbox, type CreateFromBboxDestination } from './createFromBbox';

const bbox = { height: 512, width: 768, x: 31, y: 47 };
const uploadedName = 'bbox-upload.png';
const defaultBlob = new Blob(['pixels'], { type: 'image/png' });

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const galleryImage: GalleryImage = {
  boardId: 'none',
  height: bbox.height,
  imageCategory: 'other',
  imageName: uploadedName,
  imageUrl: `https://images.example/${uploadedName}`,
  queuedAt: '2026-01-01T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  starred: false,
  thumbnailUrl: `https://images.example/thumb/${uploadedName}`,
  width: bbox.width,
};

const withProject = (): { project: Project; state: WorkbenchState } => {
  const state = createInitialWorkbenchState();
  const current = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
  const document = {
    ...current.canvas.document,
    bbox,
    stacks: stacksFrom([createEmptyPaintLayer('Existing', 'previous')]),
    selectedLayerId: 'previous',
  };
  const project = { ...current, canvas: { ...current.canvas, document } };
  return { project, state: { ...state, projects: [project] } };
};

const createHarness = (
  options: {
    canCommitStructural?: boolean;
    exportResult?: RasterCompositeExportResult;
    resolved?: GalleryImage[];
    staleDocument?: boolean;
  } = {}
) => {
  const { project, state } = withProject();
  const staleState: WorkbenchState = {
    ...state,
    projects: state.projects.map((candidate) =>
      candidate.id === project.id
        ? { ...candidate, canvas: { ...candidate.canvas, document: { ...candidate.canvas.document } } }
        : candidate
    ),
  };
  const getState = () => (options.staleDocument ? staleState : state);
  const order: string[] = [];
  const exportResult = options.exportResult ?? {
    blob: defaultBlob,
    rect: bbox,
    status: 'ok',
  };
  const flushPendingUploads = vi.fn(() => {
    order.push('flush');
    return Promise.resolve();
  });
  const getDocument = vi.fn(() => {
    order.push('document');
    return project.canvas.document as unknown as CanvasDocumentContractV3;
  });
  const exportRasterComposite = vi.fn<CanvasEngine['exports']['exportRasterComposite']>(() => {
    order.push('export');
    return Promise.resolve(exportResult);
  });
  const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
    order.push('upload');
    return Promise.resolve({ height: bbox.height, imageName: uploadedName, width: bbox.width });
  });
  const resolveImages = vi.fn((imageNames: string[]) => {
    order.push('resolve');
    expect(imageNames).toEqual([uploadedName]);
    return Promise.resolve(options.resolved ?? [galleryImage]);
  });
  const commitStructural = vi.fn((label: string, forward: CanvasProjectMutation) => {
    order.push('commit');
    void label;
    void forward;
    return { status: 'committed' as const };
  });
  const engine = {
    document: { captureInsertionAnchor: createTestInsertionAnchorCapture(project.id), getDocument },
    exports: { exportRasterComposite },
    layers: {
      canCommitStructural: vi.fn(() => options.canCommitStructural ?? true),
      commitStructural,
    },
    lifecycle: { flushPendingUploads },
    projectId: project.id,
  } as unknown as CanvasEngine;
  const applyCanvasMutation = vi.fn();

  const run = (destination: CreateFromBboxDestination) =>
    createFromBbox({
      applyCanvasMutation,
      destination,
      engine,
      getProject: (projectId) => getState().projects.find((candidate) => candidate.id === projectId) ?? null,
      isActiveProject: (projectId) => getState().activeProjectId === projectId,
      project,
      resolveImages,
      uploadImage,
    });

  return {
    applyCanvasMutation,
    commitStructural,
    exportRasterComposite,
    order,
    project,
    resolveImages,
    run,
    uploadImage,
  };
};

const getCommittedLayers = (harness: ReturnType<typeof createHarness>): readonly CanvasLayerContract[] => {
  const forward = harness.commitStructural.mock.calls[0]?.[1] as CanvasProjectMutation | undefined;
  if (forward?.type !== 'applyCanvasLayerStackMutation' || !forward.add) {
    throw new Error('Expected add stack mutation');
  }
  return forward.add.flatMap((insertion) => insertion.nodes as CanvasLayerContract[]);
};

describe('createFromBbox', () => {
  beforeEach(() => {
    accountLifecycle.activate('create-from-bbox-test');
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('creates a raster layer at the bbox origin from the uploaded composite', async () => {
    const harness = createHarness();

    const result = await harness.run('raster');

    expect(harness.order).toEqual(['flush', 'document', 'export', 'upload', 'resolve', 'commit']);
    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'rect', rect: bbox });
    expect(harness.uploadImage).toHaveBeenCalledWith(defaultBlob, {
      fileName: 'bbox.png',
      imageCategory: 'other',
      isIntermediate: false,
      signal: expect.any(AbortSignal),
    });
    expect(harness.resolveImages).toHaveBeenCalledWith([uploadedName], expect.any(AbortSignal));

    const layers = getCommittedLayers(harness);
    expect(layers).toHaveLength(1);
    expect(layers[0]).toMatchObject({
      source: { image: { height: bbox.height, imageName: uploadedName, width: bbox.width }, type: 'image' },
      transform: { x: bbox.x, y: bbox.y },
      type: 'raster',
    });
    expect(result).toEqual({ destination: 'raster', layerIds: [layers[0]!.id], status: 'created' });
    expect(harness.applyCanvasMutation).not.toHaveBeenCalled();
  });

  it('creates a control layer sourced from the uploaded composite', async () => {
    const harness = createHarness();

    const result = await harness.run('control');

    const layers = getCommittedLayers(harness);
    expect(layers[0]).toMatchObject({
      source: { image: { imageName: uploadedName }, type: 'image' },
      transform: { x: bbox.x, y: bbox.y },
    });
    expect(result).toMatchObject({ destination: 'control', status: 'created' });
  });

  it('stores the resolved gallery image on a regional reference layer', async () => {
    const harness = createHarness();

    const result = await harness.run('regional-reference');

    const layers = getCommittedLayers(harness);
    const layer = layers[0] as CanvasLayerContract & { referenceImages: { config: { image: unknown } }[] };
    expect(layer.referenceImages[0]?.config.image).toBe(galleryImage);
    expect(result).toMatchObject({ destination: 'regional-reference', status: 'created' });
  });

  it('returns the upload for the global-reference destination without resolving or committing', async () => {
    const harness = createHarness();

    await expect(harness.run('global-reference')).resolves.toEqual({
      image: { height: bbox.height, imageName: uploadedName, width: bbox.width },
      status: 'uploaded',
    });

    expect(harness.resolveImages).not.toHaveBeenCalled();
    expect(harness.commitStructural).not.toHaveBeenCalled();
  });

  it.each(['empty', 'stale', 'not-ready', 'over-budget'] as const)('returns %s without uploading', async (status) => {
    const harness = createHarness({ exportResult: { status } });

    await expect(harness.run('raster')).resolves.toEqual({ status });

    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('aborts with stale-document when the canvas changed during export and upload', async () => {
    const harness = createHarness({ staleDocument: true });

    await expect(harness.run('raster')).resolves.toEqual({ status: 'stale-document' });

    expect(harness.uploadImage).toHaveBeenCalled();
    expect(harness.commitStructural).not.toHaveBeenCalled();
    expect(harness.applyCanvasMutation).not.toHaveBeenCalled();
  });

  it('returns blocked when the engine cannot commit a structural edit', async () => {
    const harness = createHarness({ canCommitStructural: false });

    await expect(harness.run('raster')).resolves.toEqual({ status: 'blocked' });

    expect(harness.commitStructural).not.toHaveBeenCalled();
  });

  it('propagates upload rejection', async () => {
    const harness = createHarness();
    harness.uploadImage.mockRejectedValueOnce(new Error('upload failed'));

    await expect(harness.run('raster')).rejects.toThrow('upload failed');

    expect(harness.resolveImages).not.toHaveBeenCalled();
  });

  it('throws when the uploaded image cannot be resolved', async () => {
    const harness = createHarness({ resolved: [] });

    await expect(harness.run('raster')).rejects.toThrow('could not be resolved');

    expect(harness.commitStructural).not.toHaveBeenCalled();
  });

  it('does not upload account A bbox pixels after account B becomes active', async () => {
    const harness = createHarness();
    const exportResult = deferred<RasterCompositeExportResult>();

    harness.exportRasterComposite.mockReturnValueOnce(exportResult.promise);
    const create = harness.run('raster');

    await vi.waitFor(() => expect(harness.exportRasterComposite).toHaveBeenCalledOnce());
    accountLifecycle.activate('create-from-bbox-test-b');
    exportResult.resolve({ blob: defaultBlob, rect: bbox, status: 'ok' });

    await expect(create).resolves.toEqual({ status: 'stale' });
    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.resolveImages).not.toHaveBeenCalled();
    expect(harness.commitStructural).not.toHaveBeenCalled();
  });
});

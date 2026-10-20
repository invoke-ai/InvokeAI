import type { GalleryBoard, GalleryBoardAction } from '@features/gallery/contracts';
import type { GenerateModelConfig, GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { RasterCompositeExportResult } from '@workbench/canvas-engine/exportRasterComposite';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/projectContracts';

import { galleryKeys } from '@features/gallery/queries';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { uploadCanvasImage } from './backend/canvasImages';

import { getCanvasSaveBoardId, saveCanvasToGallery } from './saveCanvasToGallery';

const model: MainModelConfig = {
  base: 'sd-1',
  hash: 'model-hash',
  key: 'model-key',
  name: 'Test Model',
  type: 'main',
};

const defaultRect: Rect = { height: 456, width: 123, x: -4, y: 8 };
const defaultBlob = new Blob(['pixels'], { type: 'image/png' });

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const createProject = (boardId = 'board-1', generateOverrides: Partial<GenerateWidgetValues> = {}): Project => {
  const project = createInitialWorkbenchState().projects[0]!;
  const generate = getProjectWidgetInstance(project, 'generate');
  const gallery = getProjectWidgetInstance(project, 'gallery');

  if (!generate || !gallery) {
    throw new Error('Expected initial project to contain generate and gallery widgets');
  }

  generate.state.values = {
    ...getDefaultGenerateSettings(model),
    model,
    modelKey: model.key,
    negativePromptEnabled: true,
    negativePrompt: 'avoid blur',
    positivePrompt: 'a canvas prompt',
    seed: 123,
    ...generateOverrides,
  };
  gallery.state.values = { ...gallery.state.values, selectedBoardId: boardId };

  return project;
};

const createDocument = (bbox: Rect): CanvasDocumentContractV3 => ({ bbox }) as unknown as CanvasDocumentContractV3;

const createHarness = (
  options: {
    document?: CanvasDocumentContractV3 | null;
    exportResult?: RasterCompositeExportResult;
    onFlush?: () => void;
  } = {}
) => {
  const order: string[] = [];
  let document = options.document === undefined ? createDocument(defaultRect) : options.document;
  const exportResult = options.exportResult ?? { blob: defaultBlob, rect: defaultRect, status: 'ok' };
  const flushPendingUploads = vi.fn(() => {
    order.push('flush');
    options.onFlush?.();
    return Promise.resolve();
  });
  const getDocument = vi.fn(() => {
    order.push('document');
    return document;
  });
  const exportRasterComposite = vi.fn<CanvasEngine['exports']['exportRasterComposite']>(() => {
    order.push('export');
    return Promise.resolve(exportResult);
  });
  const uploadImage = vi.fn<typeof uploadCanvasImage>(() => {
    order.push('upload');
    return Promise.resolve({ height: defaultRect.height, imageName: 'saved.png', width: defaultRect.width });
  });
  const engine = {
    document: { getDocument },
    exports: { exportRasterComposite },
    lifecycle: { flushPendingUploads },
  } as unknown as CanvasEngine;

  return {
    engine,
    exportRasterComposite,
    flushPendingUploads,
    getDocument,
    order,
    setDocument: (next: CanvasDocumentContractV3 | null) => {
      document = next;
    },
    uploadImage,
  };
};

/** A fresh listing of every readable board, as `ensureGalleryBoardDestinations` leaves it: the boards named are destinations. */
const listGalleryBoardsInCache = (...boardIds: string[]) =>
  queryClient.setQueryData(
    galleryKeys.boards(captureAccountScope(), {
      includeArchived: true,
      includeDateBoards: false,
      orderBy: 'created_at',
      orderDir: 'DESC',
    }),
    boardIds.map((id) => ({
      archived: false,
      assetCount: 0,
      assetVideoCount: 0,
      id,
      imageCount: 0,
      isInbox: false,
      kind: 'board' as const,
      name: id,
      projectId: null,
      videoCount: 0,
    }))
  );

describe('saveCanvasToGallery', () => {
  afterEach(() => {
    queryClient.clear();
  });

  beforeEach(() => {
    accountLifecycle.activate('save-canvas-test');
    listGalleryBoardsInCache('board-1', 'board-physical', 'project-board');
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('flushes before exporting canvas content and uploads with gallery metadata', async () => {
    const harness = createHarness();

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        project: createProject(),
        region: 'canvas',
        uploadImage: harness.uploadImage,
      })
    ).resolves.toEqual({ imageName: 'saved.png', status: 'saved' });

    expect(harness.order).toEqual(['flush', 'document', 'export', 'upload']);
    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'content' });
    expect(harness.uploadImage).toHaveBeenCalledWith(defaultBlob, {
      boardId: 'board-1',
      fileName: 'canvas.png',
      imageCategory: 'general',
      isIntermediate: false,
      metadata: {
        height: 456,
        model: { base: 'sd-1', hash: 'model-hash', key: 'model-key', name: 'Test Model', type: 'main' },
        negative_prompt: 'avoid blur',
        positive_prompt: 'a canvas prompt',
        seed: 123,
        width: 123,
      },
      signal: expect.any(AbortSignal),
    });
  });

  it('exports the post-flush document bbox and uses the bbox filename', async () => {
    const preFlushRect: Rect = { height: 20, width: 10, x: 0, y: 0 };
    const postFlushRect: Rect = { height: 40, width: 30, x: 5, y: 6 };
    let harness: ReturnType<typeof createHarness>;
    harness = createHarness({
      document: createDocument(preFlushRect),
      onFlush: () => harness.setDocument(createDocument(postFlushRect)),
    });

    await saveCanvasToGallery({
      engine: harness.engine,
      project: createProject(),
      region: 'bbox',
      uploadImage: harness.uploadImage,
    });

    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'rect', rect: postFlushRect });
    expect(harness.uploadImage).toHaveBeenCalledWith(defaultBlob, expect.objectContaining({ fileName: 'bbox.png' }));
  });

  it.each([
    ['board-physical', 'board-physical'],
    ['none', undefined],
    ['by_date:2026-07-15', undefined],
  ])('normalizes selected board %s to upload board %s', async (selectedBoardId, expectedBoardId) => {
    const harness = createHarness();

    await saveCanvasToGallery({
      engine: harness.engine,
      project: createProject(selectedBoardId),
      region: 'canvas',
      uploadImage: harness.uploadImage,
    });

    expect(harness.uploadImage.mock.calls[0]?.[1]?.boardId).toBe(expectedBoardId);
  });

  it.each([
    ['no board is selected', undefined],
    ['a date bucket is selected', 'by_date:2026-07-15'],
  ])('saves to the project board when %s', async (_case, selectedBoardId) => {
    const harness = createHarness();
    const project = createProject();
    const gallery = getProjectWidgetInstance(project, 'gallery')!;

    gallery.state.values = { ...gallery.state.values, projectBoardId: 'project-board', selectedBoardId };

    await saveCanvasToGallery({
      engine: harness.engine,
      project,
      region: 'canvas',
      uploadImage: harness.uploadImage,
    });

    expect(harness.uploadImage.mock.calls[0]?.[1]?.boardId).toBe('project-board');
  });

  it('omits disabled negative prompts and unsupported models from metadata', async () => {
    const unsupportedModel: GenerateModelConfig = {
      base: 'unsupported',
      key: 'unsupported-model',
      name: 'Unsupported Model',
      type: 'main',
    };
    const harness = createHarness();

    await saveCanvasToGallery({
      engine: harness.engine,
      project: createProject('none', {
        model: unsupportedModel,
        modelKey: unsupportedModel.key,
        negativePromptEnabled: false,
      }),
      region: 'canvas',
      uploadImage: harness.uploadImage,
    });

    const metadata = harness.uploadImage.mock.calls[0]?.[1]?.metadata;
    expect(metadata).not.toHaveProperty('model');
    expect(metadata).not.toHaveProperty('negative_prompt');
    expect(metadata).toMatchObject({ height: 456, positive_prompt: 'a canvas prompt', seed: 123, width: 123 });
  });

  it.each(['empty', 'stale', 'not-ready'] as const)('returns %s without uploading', async (status) => {
    const harness = createHarness({ exportResult: { status } });

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        project: createProject(),
        region: 'canvas',
        uploadImage: harness.uploadImage,
      })
    ).resolves.toEqual({ status });

    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('returns not-ready without exporting when there is no post-flush document', async () => {
    const harness = createHarness({ document: null });

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        project: createProject(),
        region: 'bbox',
        uploadImage: harness.uploadImage,
      })
    ).resolves.toEqual({ status: 'not-ready' });

    expect(harness.exportRasterComposite).not.toHaveBeenCalled();
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('propagates export rejection without uploading', async () => {
    const harness = createHarness();
    harness.exportRasterComposite.mockRejectedValueOnce(new Error('export failed'));

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        project: createProject(),
        region: 'canvas',
        uploadImage: harness.uploadImage,
      })
    ).rejects.toThrow('export failed');

    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('propagates upload rejection', async () => {
    const harness = createHarness();
    harness.uploadImage.mockRejectedValueOnce(new Error('upload failed'));

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        project: createProject(),
        region: 'canvas',
        uploadImage: harness.uploadImage,
      })
    ).rejects.toThrow('upload failed');
  });

  it('does not upload account A canvas pixels after account B becomes active', async () => {
    const harness = createHarness();
    const exportResult = deferred<RasterCompositeExportResult>();

    harness.exportRasterComposite.mockReturnValueOnce(exportResult.promise);
    const save = saveCanvasToGallery({
      engine: harness.engine,
      project: createProject(),
      region: 'canvas',
      uploadImage: harness.uploadImage,
    });

    await vi.waitFor(() => expect(harness.exportRasterComposite).toHaveBeenCalledOnce());
    accountLifecycle.activate('save-canvas-test-b');
    exportResult.resolve({ blob: defaultBlob, rect: defaultRect, status: 'ok' });

    await expect(save).resolves.toEqual({ status: 'stale' });
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });
});

describe('getCanvasSaveBoardId', () => {
  afterEach(() => {
    queryClient.clear();
  });

  it('saves to the project inbox while the selected board is one the caller may only view', () => {
    accountLifecycle.activate('save-canvas-destination');
    const board = (id: string, actions: GalleryBoardAction[]): GalleryBoard => ({
      access: {
        actions,
        adminOverride: false,
        generalAccess: 'viewer',
        role: actions.includes('rename') ? 'owner' : 'viewer',
      },
      archived: false,
      assetCount: 0,
      assetVideoCount: 0,
      id,
      imageCount: 0,
      isInbox: id === 'inbox',
      kind: 'board',
      name: id,
      projectId: null,
      videoCount: 0,
    });
    queryClient.setQueryData(
      galleryKeys.boards(captureAccountScope(), {
        includeArchived: false,
        includeDateBoards: false,
        orderBy: 'created_at',
        orderDir: 'DESC',
      }),
      [
        board('inbox', ['read', 'add_media', 'rename']),
        board('shared', ['read']),
        board('moodboard', ['read', 'add_media']),
      ]
    );
    const withInbox = (selected: string) => {
      const project = createProject(selected);
      const gallery = getProjectWidgetInstance(project, 'gallery')!;
      gallery.state.values = { ...gallery.state.values, projectBoardId: 'inbox' };
      return project;
    };

    expect(getCanvasSaveBoardId(withInbox('shared'))).toBe('inbox');
    expect(getCanvasSaveBoardId(withInbox('moodboard'))).toBe('moodboard');
  });
});

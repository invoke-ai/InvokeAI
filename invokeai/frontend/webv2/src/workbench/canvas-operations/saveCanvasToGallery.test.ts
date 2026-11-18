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
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
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
    steps: 23,
    cfgScale: 6,
    cfgRescaleMultiplier: 0.1,
    scheduler: 'euler',
    clipSkip: 1,
    seamlessXAxis: false,
    seamlessYAxis: true,
    ...generateOverrides,
  };
  // Following the selected board, so the board under test is where saves go.
  gallery.state.values = { ...gallery.state.values, followSelectedBoard: true, selectedBoardId: boardId };

  return project;
};

const createDocument = (bbox: Rect): CanvasDocumentContractV3 => ({ ...createEmptyCanvasDocument(), bbox });

const createHarness = (
  options: {
    document?: CanvasDocumentContractV3 | null;
    exportResult?: RasterCompositeExportResult;
    onFlush?: () => void;
    project?: Project;
  } = {}
) => {
  const order: string[] = [];
  let document = options.document === undefined ? createDocument(defaultRect) : options.document;
  let project = options.project ?? createProject();
  project = { ...project, canvas: { ...project.canvas, document: document ?? project.canvas.document } };
  const getProject = vi.fn((projectId: string) => (projectId === project.id ? project : null));
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
    getProject,
    order,
    project,
    setDocument: (next: CanvasDocumentContractV3 | null) => {
      document = next;
      if (next) {
        project = { ...project, canvas: { ...project.canvas, document: next } };
      }
    },
    setProject: (next: Project) => {
      project = next;
      document = next.canvas.document;
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
    vi.unstubAllGlobals();
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
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
        uploadImage: harness.uploadImage,
      })
    ).resolves.toEqual({ boardId: 'board-1', imageName: 'saved.png', status: 'saved' });

    expect(harness.order).toEqual(['flush', 'document', 'export', 'document', 'upload']);
    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'content' });
    expect(harness.uploadImage).toHaveBeenCalledWith(defaultBlob, {
      boardId: 'board-1',
      fileName: 'canvas.png',
      imageCategory: 'general',
      isIntermediate: false,
      metadata: {
        metadata_source: 'canvas_save',
        vae: null,
        steps: 23,
        cfg_scale: 6,
        cfg_rescale_multiplier: 0.1,
        scheduler: 'euler',
        clip_skip: 1,
        seamless_x: false,
        seamless_y: true,
        loras: [],
        ref_images: [],
        height: 456,
        model: { base: 'sd-1', hash: 'model-hash', key: 'model-key', name: 'Test Model', type: 'main' },
        negative_prompt: 'avoid blur',
        positive_prompt: 'a canvas prompt',
        canvas_v3_metadata: expect.objectContaining({ version: 1, document: createDocument(defaultRect) }),
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
      getProject: harness.getProject,
      project: harness.project,
      region: 'bbox',
      scopedDefaults: {},
      uploadImage: harness.uploadImage,
    });

    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'rect', rect: postFlushRect });
    expect(harness.uploadImage).toHaveBeenCalledWith(defaultBlob, expect.objectContaining({ fileName: 'bbox.png' }));
  });

  it('captures the post-flush document and settings after a cold board read, keeping the click-time destination', async () => {
    queryClient.clear();
    const boards = deferred<Response>();
    const fetchBoards = vi.fn<typeof fetch>((url) =>
      new URL(String(url)).pathname === '/api/v1/boards/'
        ? boards.promise
        : Promise.resolve(Response.json({ total: 0 }))
    );
    vi.stubGlobal('fetch', fetchBoards);
    const harness = createHarness();
    const flush = deferred<void>();
    harness.flushPendingUploads.mockReturnValueOnce(flush.promise);

    const save = saveCanvasToGallery({
      engine: harness.engine,
      getProject: harness.getProject,
      project: harness.project,
      region: 'bbox',
      scopedDefaults: {},
      uploadImage: harness.uploadImage,
    });

    await vi.waitFor(() => expect(fetchBoards).toHaveBeenCalled());
    expect(harness.flushPendingUploads).not.toHaveBeenCalled();
    const whileListingBoards = structuredClone(harness.project);
    whileListingBoards.canvas.document.bbox = { height: 40, width: 30, x: 5, y: 6 };
    getProjectWidgetInstance(whileListingBoards, 'generate')!.state.values.positivePrompt =
      'edited during board lookup';
    getProjectWidgetInstance(whileListingBoards, 'canvas')!.state.values.denoisingStrength = 0.4;
    getProjectWidgetInstance(whileListingBoards, 'gallery')!.state.values.selectedBoardId = 'board-physical';
    harness.setProject(whileListingBoards);
    boards.resolve(
      Response.json(
        ['board-1', 'board-physical'].map((board_id) => ({
          archived: false,
          asset_count: 0,
          board_id,
          board_name: board_id,
          image_count: 0,
        }))
      )
    );

    await vi.waitFor(() => expect(harness.flushPendingUploads).toHaveBeenCalledOnce());
    const afterUploads = structuredClone(whileListingBoards);
    const capturedBbox = { height: 80, width: 60, x: 15, y: 16 };
    afterUploads.canvas.document.bbox = capturedBbox;
    Object.assign(getProjectWidgetInstance(afterUploads, 'generate')!.state.values, {
      positivePrompt: 'edited while paint was uploading',
      steps: 31,
    });
    Object.assign(getProjectWidgetInstance(afterUploads, 'canvas')!.state.values, {
      denoisingStrength: 0.6,
      maskBlur: 8,
      scaleMethod: 'manual',
      scaledHeight: 768,
      scaledWidth: 1024,
    });
    harness.setProject(afterUploads);
    flush.resolve();

    await expect(save).resolves.toEqual({ boardId: 'board-1', imageName: 'saved.png', status: 'saved' });
    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'rect', rect: capturedBbox });
    expect(harness.uploadImage.mock.calls[0]?.[1]).toMatchObject({
      boardId: 'board-1',
      metadata: {
        positive_prompt: 'edited while paint was uploading',
        steps: 31,
        canvas_v3_metadata: {
          document: { bbox: capturedBbox },
          settings: {
            strength: 0.6,
            compositing: { maskBlur: 8 },
            scaling: { height: 768, method: 'manual', width: 1024 },
          },
        },
      },
    });
  });

  it('captures current settings with the unchanged document after async export', async () => {
    const harness = createHarness();
    const exported = deferred<RasterCompositeExportResult>();
    harness.exportRasterComposite.mockReturnValueOnce(exported.promise);
    const save = saveCanvasToGallery({
      engine: harness.engine,
      getProject: harness.getProject,
      project: harness.project,
      region: 'canvas',
      scopedDefaults: {},
      uploadImage: harness.uploadImage,
    });

    await vi.waitFor(() => expect(harness.exportRasterComposite).toHaveBeenCalledOnce());
    const edited = structuredClone(harness.project);
    edited.canvas.document = harness.project.canvas.document;
    Object.assign(getProjectWidgetInstance(edited, 'generate')!.state.values, {
      positivePrompt: 'edited during export',
      steps: 51,
    });
    getProjectWidgetInstance(edited, 'canvas')!.state.values.denoisingStrength = 0.2;
    harness.setProject(edited);
    exported.resolve({ blob: defaultBlob, rect: defaultRect, status: 'ok' });

    await expect(save).resolves.toMatchObject({ status: 'saved' });
    expect(harness.uploadImage.mock.calls[0]?.[1]?.metadata).toMatchObject({
      positive_prompt: 'edited during export',
      steps: 51,
      canvas_v3_metadata: { document: { bbox: defaultRect }, settings: { strength: 0.2 } },
    });
  });

  it.each(['closed-project', 'mismatched-document', 'expired-account'] as const)(
    'does not upload a %s after pending paint uploads finish',
    async (reason) => {
      const harness = createHarness();
      const flush = deferred<void>();
      harness.flushPendingUploads.mockReturnValueOnce(flush.promise);
      const save = saveCanvasToGallery({
        engine: harness.engine,
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
        uploadImage: harness.uploadImage,
      });

      await vi.waitFor(() => expect(harness.flushPendingUploads).toHaveBeenCalledOnce());
      if (reason === 'closed-project') {
        harness.getProject.mockReturnValue(null);
      } else if (reason === 'mismatched-document') {
        harness.getDocument.mockReturnValue(createDocument({ ...defaultRect, x: 999 }));
      } else {
        accountLifecycle.activate('another-account');
      }
      flush.resolve();

      await expect(save).resolves.toEqual({ status: 'stale' });
      expect(harness.uploadImage).not.toHaveBeenCalled();
    }
  );

  it.each(['edited-document', 'project-document-mismatch', 'closed-project'] as const)(
    'does not upload a %s after async export',
    async (reason) => {
      const harness = createHarness();
      const exported = deferred<RasterCompositeExportResult>();
      harness.exportRasterComposite.mockReturnValueOnce(exported.promise);
      const save = saveCanvasToGallery({
        engine: harness.engine,
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
        uploadImage: harness.uploadImage,
      });

      await vi.waitFor(() => expect(harness.exportRasterComposite).toHaveBeenCalledOnce());
      if (reason === 'edited-document') {
        harness.setDocument(createDocument({ ...defaultRect, x: 999 }));
      } else if (reason === 'project-document-mismatch') {
        const edited = structuredClone(harness.project);
        edited.canvas.document.bbox.x = 999;
        harness.getProject.mockReturnValue(edited);
      } else {
        harness.getProject.mockReturnValue(null);
      }
      exported.resolve({ blob: defaultBlob, rect: defaultRect, status: 'ok' });

      await expect(save).resolves.toEqual({ status: 'stale' });
      expect(harness.uploadImage).not.toHaveBeenCalled();
    }
  );

  it.each([
    ['board-physical', 'board-physical'],
    ['none', undefined],
    ['by_date:2026-07-15', undefined],
  ])('normalizes selected board %s to upload board %s', async (selectedBoardId, expectedBoardId) => {
    const harness = createHarness({ project: createProject(selectedBoardId) });

    await saveCanvasToGallery({
      engine: harness.engine,
      getProject: harness.getProject,
      project: harness.project,
      region: 'canvas',
      scopedDefaults: {},
      uploadImage: harness.uploadImage,
    });

    expect(harness.uploadImage.mock.calls[0]?.[1]?.boardId).toBe(expectedBoardId);
  });

  it.each([
    ['no board is selected', undefined],
    ['a date bucket is selected', 'by_date:2026-07-15'],
  ])('saves to the project board when %s', async (_case, selectedBoardId) => {
    const harness = createHarness();
    const project = harness.project;
    const gallery = getProjectWidgetInstance(project, 'gallery')!;

    gallery.state.values = {
      ...gallery.state.values,
      followSelectedBoard: true,
      projectBoardId: 'project-board',
      selectedBoardId,
    };

    await saveCanvasToGallery({
      engine: harness.engine,
      getProject: harness.getProject,
      project,
      region: 'canvas',
      scopedDefaults: {},
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
    const harness = createHarness({
      project: createProject('none', {
        model: unsupportedModel,
        modelKey: unsupportedModel.key,
        negativePromptEnabled: false,
      }),
    });

    await saveCanvasToGallery({
      engine: harness.engine,
      getProject: harness.getProject,
      project: harness.project,
      region: 'canvas',
      scopedDefaults: {},
      uploadImage: harness.uploadImage,
    });

    const metadata = harness.uploadImage.mock.calls[0]?.[1]?.metadata;
    expect(metadata).not.toHaveProperty('seed');
    expect(metadata).not.toHaveProperty('generation_mode');
    expect(metadata).not.toHaveProperty('model');
    expect(metadata).not.toHaveProperty('negative_prompt');
    expect(metadata).toMatchObject({ height: 456, positive_prompt: 'a canvas prompt', width: 123 });
  });

  it.each(['empty', 'stale', 'not-ready'] as const)('returns %s without uploading', async (status) => {
    const harness = createHarness({ exportResult: { status } });

    await expect(
      saveCanvasToGallery({
        engine: harness.engine,
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
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
        getProject: harness.getProject,
        project: harness.project,
        region: 'bbox',
        scopedDefaults: {},
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
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
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
        getProject: harness.getProject,
        project: harness.project,
        region: 'canvas',
        scopedDefaults: {},
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
      getProject: harness.getProject,
      project: harness.project,
      region: 'canvas',
      scopedDefaults: {},
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

    expect(getCanvasSaveBoardId(withInbox('shared'), {})).toBe('inbox');
    expect(getCanvasSaveBoardId(withInbox('moodboard'), {})).toBe('moodboard');

    // A project that does not pin following the selected board follows the user's default for it.
    const unpinned = withInbox('moodboard');
    const gallery = getProjectWidgetInstance(unpinned, 'gallery')!;
    const { followSelectedBoard: _pin, ...values } = gallery.state.values;
    gallery.state.values = values;

    expect(getCanvasSaveBoardId(unpinned, {})).toBe('inbox');
    expect(getCanvasSaveBoardId(unpinned, { gallery: { followSelectedBoard: true } })).toBe('moodboard');
  });
});

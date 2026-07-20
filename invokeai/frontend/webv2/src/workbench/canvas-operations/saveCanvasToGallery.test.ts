import type { GenerateModelConfig, GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { RasterCompositeExportResult } from '@workbench/canvas-engine/exportRasterComposite';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/projectContracts';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import type { uploadCanvasImage } from './backend/canvasImages';

import { saveCanvasToGallery } from './saveCanvasToGallery';

const model: MainModelConfig = {
  base: 'sd-1',
  hash: 'model-hash',
  key: 'model-key',
  name: 'Test Model',
  type: 'main',
};

const defaultRect: Rect = { height: 456, width: 123, x: -4, y: 8 };
const defaultBlob = new Blob(['pixels'], { type: 'image/png' });

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

const createDocument = (bbox: Rect): CanvasDocumentContractV2 => ({ bbox }) as unknown as CanvasDocumentContractV2;

const createHarness = (
  options: {
    document?: CanvasDocumentContractV2 | null;
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
    setDocument: (next: CanvasDocumentContractV2 | null) => {
      document = next;
    },
    uploadImage,
  };
};

describe('saveCanvasToGallery', () => {
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
});

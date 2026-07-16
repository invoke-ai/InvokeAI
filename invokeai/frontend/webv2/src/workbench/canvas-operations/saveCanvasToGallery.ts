import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/types';

import { isDateBoardId } from '@workbench/gallery/api';
import { isSupportedGenerateModel } from '@workbench/generation/baseGenerationPolicies';
import { toModelIdentifier } from '@workbench/generation/graphBuilder';
import { normalizeGenerateWidgetValues } from '@workbench/generation/settings';
import { getProjectWidgetValues } from '@workbench/widgetState';

import { uploadCanvasImage } from './backend/canvasImages';

export type CanvasGallerySaveRegion = 'canvas' | 'bbox';

export type SaveCanvasToGalleryResult =
  | { status: 'saved'; imageName: string }
  | { status: 'empty' | 'stale' | 'not-ready' };

const buildCanvasSaveMetadata = (project: Project, rect: Rect): Record<string, unknown> => {
  const generateValues = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'));
  const dimensions = { height: rect.height, width: rect.width };

  if (!generateValues) {
    return dimensions;
  }

  return {
    ...dimensions,
    ...(isSupportedGenerateModel(generateValues.model) ? { model: toModelIdentifier(generateValues.model) } : {}),
    ...(generateValues.negativePromptEnabled ? { negative_prompt: generateValues.negativePrompt } : {}),
    positive_prompt: generateValues.positivePrompt,
    seed: generateValues.seed,
  };
};

const getCanvasSaveBoardId = (project: Project): string | undefined => {
  const selectedBoardId = getProjectWidgetValues(project, 'gallery').selectedBoardId;

  return typeof selectedBoardId === 'string' && selectedBoardId !== 'none' && !isDateBoardId(selectedBoardId)
    ? selectedBoardId
    : undefined;
};

export const saveCanvasToGallery = async (options: {
  engine: CanvasEngine;
  region: CanvasGallerySaveRegion;
  project: Project;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<SaveCanvasToGalleryResult> => {
  const { engine, project, region, uploadImage = uploadCanvasImage } = options;
  const boardId = getCanvasSaveBoardId(project);

  await engine.lifecycle.flushPendingUploads();
  const document = engine.document.getDocument();
  if (!document) {
    return { status: 'not-ready' };
  }

  const exported = await engine.exports.exportRasterComposite(
    region === 'canvas' ? { bounds: 'content' } : { bounds: 'rect', rect: document.bbox }
  );
  if (exported.status !== 'ok') {
    return exported;
  }

  const uploaded = await uploadImage(exported.blob, {
    boardId,
    fileName: region === 'canvas' ? 'canvas.png' : 'bbox.png',
    imageCategory: 'general',
    isIntermediate: false,
    metadata: buildCanvasSaveMetadata(project, exported.rect),
  });

  return { imageName: uploaded.imageName, status: 'saved' };
};

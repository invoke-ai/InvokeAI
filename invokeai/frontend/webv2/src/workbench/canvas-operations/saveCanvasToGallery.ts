import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/projectContracts';

import { isGalleryVirtualBoard } from '@features/gallery';
import { toModelIdentifier } from '@features/generation/graph';
import {
  getEffectivePrompts,
  isSupportedGenerateModel,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';

import type { CanvasCompositeExportEngine, CanvasCompositeRegion } from './exportCanvasComposite';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite } from './exportCanvasComposite';

export type CanvasGallerySaveRegion = CanvasCompositeRegion;

export type SaveCanvasToGalleryResult =
  | { status: 'saved'; imageName: string }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' };

const buildCanvasSaveMetadata = (project: Project, rect: Rect): Record<string, unknown> => {
  const generateValues = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'));
  const dimensions = { height: rect.height, width: rect.width };

  if (!generateValues) {
    return dimensions;
  }

  // The merged prompts, matching what a generated image's `core_metadata` records.
  // A saved canvas is recalled through the same path, so the two must agree or
  // recalling from a saved image would restore different text.
  const effectivePrompts = getEffectivePrompts(generateValues);

  return {
    ...dimensions,
    ...(isSupportedGenerateModel(generateValues.model) ? { model: toModelIdentifier(generateValues.model) } : {}),
    ...(generateValues.negativePromptEnabled ? { negative_prompt: effectivePrompts.negativePrompt } : {}),
    positive_prompt: effectivePrompts.positivePrompt,
    seed: generateValues.seed,
  };
};

const getCanvasSaveBoardId = (project: Project): string | undefined => {
  const selectedBoardId = getProjectWidgetValues(project, 'gallery').selectedBoardId;

  return typeof selectedBoardId === 'string' && selectedBoardId !== 'none' && !isGalleryVirtualBoard(selectedBoardId)
    ? selectedBoardId
    : undefined;
};

export const saveCanvasToGallery = async (options: {
  engine: CanvasCompositeExportEngine;
  region: CanvasGallerySaveRegion;
  project: Project;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<SaveCanvasToGalleryResult> => {
  const owner = captureAccountScope();
  const { engine, project, region, uploadImage = uploadCanvasImage } = options;
  const boardId = getCanvasSaveBoardId(project);

  try {
    const exported = await exportCanvasComposite(engine, region);

    assertAccountScopeCurrent(owner);
    if (exported.status !== 'ok') {
      return exported;
    }

    const uploaded = await uploadImage(exported.blob, {
      boardId,
      fileName: region === 'canvas' ? 'canvas.png' : 'bbox.png',
      imageCategory: 'general',
      isIntermediate: false,
      metadata: buildCanvasSaveMetadata(project, exported.rect),
      signal: owner.signal,
    });

    assertAccountScopeCurrent(owner);
    return { imageName: uploaded.imageName, status: 'saved' };
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }

    throw error;
  }
};

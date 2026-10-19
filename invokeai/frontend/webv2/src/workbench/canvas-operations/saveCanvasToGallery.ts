import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/projectContracts';

import { getGalleryAutoAddBoardId } from '@features/gallery/contracts';
import { canAddToCachedGalleryBoard, ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { toModelIdentifier } from '@features/generation/graph';
import {
  getEffectivePrompts,
  isSupportedGenerateModel,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
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

  // Save merged prompts to match generated core_metadata and restore identical text on recall.
  const effectivePrompts = getEffectivePrompts(generateValues);

  return {
    ...dimensions,
    ...(isSupportedGenerateModel(generateValues.model) ? { model: toModelIdentifier(generateValues.model) } : {}),
    ...(generateValues.negativePromptEnabled ? { negative_prompt: effectivePrompts.negativePrompt } : {}),
    positive_prompt: effectivePrompts.positivePrompt,
    seed: generateValues.seed,
  };
};

const toCanvasSaveBoardId = (galleryValues: Record<string, unknown>): string | undefined => {
  const boardId = getGalleryAutoAddBoardId(galleryValues, canAddToCachedGalleryBoard);

  return boardId !== null && boardId !== 'none' ? boardId : undefined;
};

/**
 * Where Canvas puts new gallery images: the project's auto-add board, or no board (Uncategorized), as the cached
 * board listing says. Callers resolve through `resolveCanvasSaveBoardId`, which has the listing read first.
 */
export const getCanvasSaveBoardId = (project: Project): string | undefined =>
  toCanvasSaveBoardId(getProjectWidgetValues(project, 'gallery'));

/**
 * `getCanvasSaveBoardId` from a listing read first, so a board is never trusted unseen nor redirected for want of
 * one. The project's gallery values are read before anything is awaited: a later change to the project cannot
 * redirect a save already under way.
 */
export const resolveCanvasSaveBoardId = async (project: Project): Promise<string | undefined> => {
  const galleryValues = getProjectWidgetValues(project, 'gallery');

  await ensureGalleryBoardDestinations(queryClient);

  return toCanvasSaveBoardId(galleryValues);
};

export const saveCanvasToGallery = async (options: {
  engine: CanvasCompositeExportEngine;
  region: CanvasGallerySaveRegion;
  project: Project;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<SaveCanvasToGalleryResult> => {
  const owner = captureAccountScope();
  const { engine, project, region, uploadImage = uploadCanvasImage } = options;

  try {
    const boardId = await resolveCanvasSaveBoardId(project);
    assertAccountScopeCurrent(owner);
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

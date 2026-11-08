import type { ControlLayerGraphInput } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';
import type {
  AdjustmentRefusal,
  CanvasDocumentContractV3,
  CanvasNodeContract,
  LayerUnreadable,
} from '@workbench/canvas-engine/api';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Project } from '@workbench/projectContracts';
import type { ScopedDefaults } from '@workbench/settings/contracts';

import { getGalleryAutoAddBoardId } from '@features/gallery/contracts';
import { canAddToCachedGalleryBoard, ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { toModelIdentifier } from '@features/generation/graph';
import {
  getEffectivePrompts,
  isSupportedGenerateModel,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';
import { getModelsSnapshot } from '@features/models';
import { queryClient } from '@platform/query/client';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { readCanvasCompositingSettings } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { readCanvasScaling } from '@workbench/widgets/canvas/invoke/canvasScaling';
import { readCanvasDenoisingStrength } from '@workbench/widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from '@workbench/widgetState';

import type { CanvasCompositeExportEngine, CanvasCompositeRegion } from './exportCanvasComposite';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite } from './exportCanvasComposite';

export type CanvasGallerySaveRegion = CanvasCompositeRegion;

export type SaveCanvasToGalleryResult =
  /** `boardId` is where the image went: a board id, or `none` for Uncategorized. */
  | { status: 'saved'; boardId: string; imageName: string }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' }
  | LayerUnreadable
  | AdjustmentRefusal;

/** Image metadata is independent of project persistence: no staging, history, project ids or stored media URLs. */
export const recordCanvasMetadata = (
  document: CanvasDocumentContractV3,
  {
    controls,
    models = [],
    settings = {},
  }: {
    controls?: readonly ControlLayerGraphInput[];
    models?: readonly ModelConfig[];
    settings?: Record<string, unknown>;
  } = {}
): Record<string, unknown> => {
  const controlModels: Record<string, unknown> = Object.create(null);
  const copy = structuredClone(document);
  const controlsById = new Map(controls?.map((control) => [control.id, control]));
  const modelsByKey = new Map(models.map((model) => [model.key, model]));
  const visit = (nodes: CanvasNodeContract[]): void => {
    for (const node of nodes) {
      if (node.type === 'group') {
        visit(node.children);
      } else if (node.type === 'control') {
        const executed = controlsById.get(node.id);
        const model = executed?.model ?? modelsByKey.get(node.adapter.model ?? '');
        if (model) {
          controlModels[node.id] = toModelIdentifier({ ...model });
          node.adapter.model = model.key;
        }
        if (executed) {
          node.adapter.kind = executed.kind;
          node.adapter.controlMode = executed.controlMode;
          node.adapter.beginEndStepPct = [...executed.beginEndStepPct];
          node.adapter.weight = executed.weight;
        }
      } else if (node.type === 'regional_guidance') {
        for (const reference of node.referenceImages) {
          if (reference.config.image) {
            reference.config.image.imageUrl = '';
            reference.config.image.thumbnailUrl = '';
          }
        }
      }
    }
  };
  Object.values(copy.stacks).forEach(visit);
  return { controlModels, document: copy, settings: structuredClone(settings), version: 1 };
};

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
    metadata_source: 'canvas_save',
    steps: generateValues.steps,
    cfg_scale: generateValues.cfgScale,
    cfg_rescale_multiplier: generateValues.cfgRescaleMultiplier,
    scheduler: generateValues.scheduler,
    clip_skip: generateValues.clipSkip,
    seamless_x: generateValues.seamlessXAxis,
    seamless_y: generateValues.seamlessYAxis,
    loras: generateValues.loras.map((lora) => ({
      model: toModelIdentifier(lora.model),
      weight: lora.weight,
      isEnabled: lora.isEnabled,
    })),
    ref_images: structuredClone(generateValues.referenceImages),
    vae: generateValues.vae ? toModelIdentifier(generateValues.vae) : null,
    ...(isSupportedGenerateModel(generateValues.model) ? { model: toModelIdentifier(generateValues.model) } : {}),
    ...(generateValues.negativePromptEnabled ? { negative_prompt: effectivePrompts.negativePrompt } : {}),
    positive_prompt: effectivePrompts.positivePrompt,
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
export const getCanvasSaveBoardId = (project: Project, scopedDefaults: ScopedDefaults): string | undefined =>
  toCanvasSaveBoardId(getResolvedProjectWidgetValues(project, 'gallery', scopedDefaults));

/**
 * `getCanvasSaveBoardId` from a listing read first, so a board is never trusted unseen nor redirected for want of
 * one. The project's gallery values are read before anything is awaited: a later change to the project cannot
 * redirect a save already under way.
 */
export const resolveCanvasSaveBoardId = async (
  project: Project,
  scopedDefaults: ScopedDefaults
): Promise<string | undefined> => {
  const galleryValues = getResolvedProjectWidgetValues(project, 'gallery', scopedDefaults);

  await ensureGalleryBoardDestinations(queryClient);

  return toCanvasSaveBoardId(galleryValues);
};

export const saveCanvasToGallery = async (options: {
  engine: CanvasCompositeExportEngine;
  region: CanvasGallerySaveRegion;
  /** The originating project's current immutable state, read after export to pair metadata with its document. */
  getProject: (projectId: string) => Project | null;
  project: Project;
  /** The user's defaults captured with `project`: the destination follows the Gallery's resolved settings. */
  scopedDefaults: ScopedDefaults;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<SaveCanvasToGalleryResult> => {
  const owner = captureAccountScope();
  const { engine, getProject, project, region, scopedDefaults, uploadImage = uploadCanvasImage } = options;

  try {
    const boardId = await resolveCanvasSaveBoardId(project, scopedDefaults);
    assertAccountScopeCurrent(owner);
    const exported = await exportCanvasComposite(engine, region);

    assertAccountScopeCurrent(owner);
    if (exported.status !== 'ok') {
      return exported;
    }
    const capturedProject = getProject(project.id);
    if (!capturedProject || capturedProject.canvas.document !== exported.document) {
      return { status: 'stale' };
    }
    const canvasValues = getProjectWidgetValues(capturedProject, 'canvas');
    const metadata = {
      ...buildCanvasSaveMetadata(capturedProject, exported.rect),
      canvas_v3_metadata: recordCanvasMetadata(exported.document, {
        models: getModelsSnapshot().models,
        settings: {
          strength: readCanvasDenoisingStrength(canvasValues),
          compositing: readCanvasCompositingSettings(canvasValues),
          scaling: readCanvasScaling(canvasValues),
        },
      }),
    };
    const uploaded = await uploadImage(exported.blob, {
      boardId,
      fileName: region === 'canvas' ? 'canvas.png' : 'bbox.png',
      imageCategory: 'general',
      isIntermediate: false,
      metadata,
      signal: owner.signal,
    });

    assertAccountScopeCurrent(owner);
    return { boardId: boardId ?? 'none', imageName: uploaded.imageName, status: 'saved' };
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }

    throw error;
  }
};

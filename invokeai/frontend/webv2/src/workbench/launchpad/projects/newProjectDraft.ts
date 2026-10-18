import type { GalleryImage } from '@features/gallery';
import type { AspectRatioId, GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { NewProjectRequest } from '@workbench/projects/newProjectRequest';

import {
  ASPECT_RATIO_MAP,
  calculateNewSize,
  clampDimension,
  deriveAspectRatioId,
  getDefaultGenerateSettings,
  getGenerateModelSelectionResult,
  getGenerationDimensions,
  isGenerateModelSelectable,
} from '@features/generation/settings';

/**
 * The New-project dialog's choices, kept apart from the models and capabilities that arrive while it is open: the
 * model and size are resolved against the current catalog at render time, so a late-loading catalog or a model that
 * was uninstalled since the last visit never leaves the form pointing at nothing.
 */

export type NewProjectModel = ModelConfig & GenerateModelConfig;

export interface Dimensions {
  height: number;
  width: number;
}

export interface NewProjectDraft {
  intent: LaunchpadIntentId;
  name: string;
  /** A model the user picked this session; null follows the remembered model, then the first selectable one. */
  modelKey: string | null;
  /** A size the user set this session; null follows the resolved model's defaults. */
  customSize: Dimensions | null;
  /** The chip the size came from; null once a typed size no longer matches one, or while following the defaults. */
  aspectRatioId: NewProjectAspectRatioId | null;
  /** The library workflow a Workflow start opens with; null starts from the blank one. */
  libraryWorkflow: { workflowId: string; name: string } | null;
  /** Whether a Canvas start opens blank or on an image. */
  canvasStart: 'blank' | 'image';
  /** The image a Canvas start opens on; the canvas and the size follow its dimensions while one is chosen. */
  startImage: GalleryImage | null;
}

/** Offered as chips: the everyday ratios, widest first within each orientation. */
export const NEW_PROJECT_ASPECT_RATIOS = [
  '1:1',
  '3:2',
  '2:3',
  '16:9',
  '9:16',
  '4:3',
  '3:4',
] as const satisfies readonly AspectRatioId[];

export type NewProjectAspectRatioId = (typeof NEW_PROJECT_ASPECT_RATIOS)[number];

/** Generate and Canvas projects start from a model and a size; the other starts choose their inputs in the editor. */
export const hasGenerationSettings = (intent: LaunchpadIntentId): boolean =>
  intent === 'generate' || intent === 'canvas';

export const createNewProjectDraft = (intent: LaunchpadIntentId): NewProjectDraft => ({
  aspectRatioId: null,
  customSize: null,
  canvasStart: 'blank',
  intent,
  libraryWorkflow: null,
  modelKey: null,
  name: '',
  startImage: null,
});

/** A chosen start image fixes the size: the canvas is sized to it, so the chips and inputs rest. */
export const getStartImage = (draft: NewProjectDraft): GalleryImage | null =>
  draft.intent === 'canvas' && draft.canvasStart === 'image' ? draft.startImage : null;

/** Switching back to Blank forgets the image, so Image starts empty again next time. */
export const setCanvasStart = (draft: NewProjectDraft, canvasStart: 'blank' | 'image'): NewProjectDraft => ({
  ...draft,
  canvasStart,
  startImage: canvasStart === 'image' ? draft.startImage : null,
});

export const setStartImage = (draft: NewProjectDraft, startImage: GalleryImage | null): NewProjectDraft => ({
  ...draft,
  canvasStart: 'image',
  startImage,
});

/** Installed models a Generate project can start from, by name; the first one is the fallback selection. */
export const listSelectableGenerateModels = (models: readonly ModelConfig[]): NewProjectModel[] =>
  models
    .filter((model): model is NewProjectModel => isGenerateModelSelectable(model))
    .sort((left, right) => left.name.localeCompare(right.name));

/** The model the form shows: the session's pick, else the remembered one, else the first selectable; null when none. */
export const resolveNewProjectModel = (
  draft: NewProjectDraft,
  selectableModels: readonly NewProjectModel[],
  rememberedModelKey: string | null
): NewProjectModel | null =>
  (draft.modelKey === null ? undefined : selectableModels.find((model) => model.key === draft.modelKey)) ??
  (rememberedModelKey === null ? undefined : selectableModels.find((model) => model.key === rememberedModelKey)) ??
  selectableModels[0] ??
  null;

const getModelDefaultSize = (model: NewProjectModel | null): Dimensions => {
  const { height, width } = getDefaultGenerateSettings(model ?? undefined);

  return { height, width };
};

const snapToModel = (size: Dimensions, model: NewProjectModel | null): Dimensions => {
  const { grid } = getGenerationDimensions(model ?? undefined);

  return { height: clampDimension(size.height, grid), width: clampDimension(size.width, grid) };
};

/**
 * The size the form shows: a chosen start image's dimensions on the model's grid (what the canvas bbox sync derives),
 * else the session's custom size, else the resolved model's defaults.
 */
export const resolveNewProjectSize = (draft: NewProjectDraft, model: NewProjectModel | null): Dimensions => {
  const startImage = getStartImage(draft);

  if (startImage) {
    return snapToModel({ height: startImage.height, width: startImage.width }, model);
  }

  return draft.customSize ?? getModelDefaultSize(model);
};

/**
 * A size from a chip follows the new model (its budget at the same ratio); a typed size stays, re-snapped to the new
 * grid; a default size follows the new model's defaults.
 */
export const selectModel = (draft: NewProjectDraft, model: NewProjectModel): NewProjectDraft => {
  const next = { ...draft, modelKey: model.key };

  if (draft.aspectRatioId !== null && draft.customSize) {
    return selectAspectRatio(next, draft.aspectRatioId, model);
  }

  return { ...next, customSize: draft.customSize ? snapToModel(draft.customSize, model) : null };
};

/** The offered chip matching a size exactly; snapped chip sizes rarely do, so chip picks are remembered instead. */
const getOfferedAspectRatio = (size: Dimensions): NewProjectAspectRatioId | null => {
  const ratioId = deriveAspectRatioId(size.width, size.height);

  return NEW_PROJECT_ASPECT_RATIOS.find((candidate) => candidate === ratioId) ?? null;
};

/** Reshapes the model's optimal pixel budget to the ratio; unlike the editor's picker, which keeps the current area. */
export const selectAspectRatio = (
  draft: NewProjectDraft,
  ratioId: NewProjectAspectRatioId,
  model: NewProjectModel | null
): NewProjectDraft => {
  const { grid, optimal } = getGenerationDimensions(model ?? undefined);

  return {
    ...draft,
    aspectRatioId: ratioId,
    customSize: calculateNewSize(ASPECT_RATIO_MAP[ratioId].ratio, optimal * optimal, grid),
  };
};

/** Typed values snap to the model's grid within the dimension bounds; a non-number leaves the draft alone. */
export const commitDimension = (
  draft: NewProjectDraft,
  model: NewProjectModel | null,
  key: keyof Dimensions,
  value: number
): NewProjectDraft => {
  if (!Number.isFinite(value)) {
    return draft;
  }

  const { grid } = getGenerationDimensions(model ?? undefined);
  const customSize = { ...resolveNewProjectSize(draft, model), [key]: clampDimension(value, grid) };

  return { ...draft, aspectRatioId: getOfferedAspectRatio(customSize), customSize };
};

export const swapDimensions = (draft: NewProjectDraft, model: NewProjectModel | null): NewProjectDraft => {
  const { height, width } = resolveNewProjectSize(draft, model);
  const customSize = { height: width, width: height };
  const inverseId = draft.aspectRatioId === null ? null : ASPECT_RATIO_MAP[draft.aspectRatioId].inverseId;

  return {
    ...draft,
    aspectRatioId:
      NEW_PROJECT_ASPECT_RATIOS.find((candidate) => candidate === inverseId) ?? getOfferedAspectRatio(customSize),
    customSize,
  };
};

/** The chip to show active: the one picked, else the one a default, typed or image size matches exactly. */
export const getActiveAspectRatio = (draft: NewProjectDraft, size: Dimensions): NewProjectAspectRatioId | null =>
  getStartImage(draft) === null ? (draft.aspectRatioId ?? getOfferedAspectRatio(size)) : getOfferedAspectRatio(size);

/**
 * Generate and Canvas starts carry the chosen model's defaults with the chosen size over them, with the dependent
 * component selections (VAE, encoders) resolved against the catalog the way picking the model in the editor would.
 */
export const buildNewProjectRequest = ({
  draft,
  model,
  models,
}: {
  draft: NewProjectDraft;
  model: NewProjectModel | null;
  models: readonly ModelConfig[];
}): NewProjectRequest => {
  const name = draft.name.trim();
  const libraryWorkflow = draft.intent === 'workflow' ? draft.libraryWorkflow : null;
  const startImage = getStartImage(draft);

  if (!hasGenerationSettings(draft.intent) || model === null) {
    return { generateSettings: null, intent: draft.intent, libraryWorkflow, name: name || null, startImage };
  }

  const { height, width } = resolveNewProjectSize(draft, model);
  const { settings } = getGenerateModelSelectionResult({
    currentValues: getDefaultGenerateSettings(model),
    model,
    models,
  });

  // An image start takes its ratio from the image's exact dimensions (what the bbox sync derives from the canvas),
  // whatever chip was picked before it; the snapped size is what generation renders at.
  const ratioSource = startImage ? { height: startImage.height, width: startImage.width } : { height, width };
  const aspectRatioId: AspectRatioId =
    startImage === null
      ? (draft.aspectRatioId ?? deriveAspectRatioId(width, height))
      : deriveAspectRatioId(ratioSource.width, ratioSource.height);

  return {
    generateSettings: {
      ...settings,
      aspectRatioId,
      aspectRatioIsLocked: false,
      aspectRatioValue:
        aspectRatioId === 'Free'
          ? ratioSource.height > 0
            ? ratioSource.width / ratioSource.height
            : 1
          : ASPECT_RATIO_MAP[aspectRatioId].ratio,
      height,
      width,
    },
    intent: draft.intent,
    libraryWorkflow,
    name: name || null,
    startImage,
  };
};

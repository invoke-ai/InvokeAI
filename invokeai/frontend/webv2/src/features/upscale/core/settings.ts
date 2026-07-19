import type {
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaePrecision,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { sanitizeBatchCount } from '@features/generation/batch';
import {
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  isVaeCompatibleWithGenerateModel,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_POSITIVE_PROMPT_HEIGHT_PX,
  SEED_MAX,
} from '@features/generation/settings';

import type { SpandrelModelConfig, TileControlNetModelConfig, UpscaleWidgetValues } from './types';

export const UPSCALE_SCALE_SLIDER_MIN = 2;
export const UPSCALE_SCALE_SLIDER_MAX = 8;
export const UPSCALE_SCALE_MIN = 1;
export const UPSCALE_SCALE_MAX = 16;
export const UPSCALE_CREATIVITY_MIN = -10;
export const UPSCALE_CREATIVITY_MAX = 10;
export const UPSCALE_STRUCTURE_MIN = -10;
export const UPSCALE_STRUCTURE_MAX = 10;
export const UPSCALE_TILE_SIZE_MIN = 512;
export const UPSCALE_TILE_SIZE_MAX = 1536;
export const UPSCALE_TILE_OVERLAP_MIN = 16;
export const UPSCALE_TILE_OVERLAP_MAX = 512;

export const UPSCALE_PRESETS = {
  conservative: { creativity: -5, structure: 5 },
  balanced: { creativity: 0, structure: 0 },
  creative: { creativity: 5, structure: -2 },
  artistic: { creativity: 8, structure: -5 },
} as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const normalizePromptHeight = (value: unknown, min: number, max: number, fallback: number): number =>
  isFiniteNumber(value) ? Math.min(max, Math.max(min, value)) : fallback;

export const isSupportedUpscaleMainModel = (value: unknown): value is MainModelConfig =>
  isMainModelConfig(value) && (value.base === 'sd-1' || value.base === 'sdxl');

export const isSpandrelModelConfig = (value: unknown): value is SpandrelModelConfig =>
  isModelIdentifierConfig(value) && value.type === 'spandrel_image_to_image';

export const isTileControlNetModelConfig = (value: unknown): value is TileControlNetModelConfig =>
  isModelIdentifierConfig(value) && value.type === 'controlnet';

export const isTileControlNetCandidate = (
  value: unknown,
  model: Pick<MainModelConfig, 'base'> | null
): value is TileControlNetModelConfig => {
  if (!model || !isTileControlNetModelConfig(value) || value.base !== model.base) {
    return false;
  }

  const name = value.name.toLowerCase();

  return name.includes('tile') || name.includes('union');
};

export const isUpscaleImage = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  value.image_name.length > 0 &&
  isFiniteNumber(value.width) &&
  value.width > 0 &&
  isFiniteNumber(value.height) &&
  value.height > 0;

const isVaePrecision = (value: unknown): value is VaePrecision => value === 'fp16' || value === 'fp32';

const normalizeLoras = (value: unknown): GenerateLora[] =>
  Array.isArray(value)
    ? value.filter(
        (item): item is GenerateLora =>
          isRecord(item) &&
          isLoraModelConfig(item.model) &&
          typeof item.isEnabled === 'boolean' &&
          isFiniteNumber(item.weight)
      )
    : [];

export const createDefaultUpscaleWidgetValues = (models: readonly ModelConfig[] = []): UpscaleWidgetValues => {
  const model = (models.find(isSupportedUpscaleMainModel) as MainModelConfig | undefined) ?? null;

  return {
    batchCount: 1,
    cfgScale: 2,
    clipSkip: 0,
    creativity: 0,
    inputImage: null,
    loras: [],
    model,
    negativePrompt: '',
    negativePromptEnabled: true,
    negativePromptHeightPx: DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
    positivePrompt: '',
    positivePromptHeightPx: DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
    scale: 4,
    scheduler: 'kdpm_2',
    seed: 0,
    shouldRandomizeSeed: true,
    steps: 30,
    structure: 0,
    tileControlnetModel: models.find((candidate) => isTileControlNetCandidate(candidate, model)) ?? null,
    tileOverlap: 128,
    tileSize: 1024,
    upscaleModel: (models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null,
    vae: null,
    vaePrecision: 'fp32',
  };
};

/**
 * Heals older/partial persisted values without silently clamping invalid user
 * input. Range checks remain invocation validation so actionable errors can be
 * shown instead of changing a saved project behind the user's back.
 */
export const normalizeUpscaleWidgetValues = (value: unknown): UpscaleWidgetValues | null => {
  if (!isRecord(value)) {
    return null;
  }

  const defaults = createDefaultUpscaleWidgetValues();

  return {
    batchCount: sanitizeBatchCount(value.batchCount),
    cfgScale: isFiniteNumber(value.cfgScale) ? value.cfgScale : defaults.cfgScale,
    clipSkip: isFiniteNumber(value.clipSkip) ? value.clipSkip : defaults.clipSkip,
    creativity: isFiniteNumber(value.creativity) ? value.creativity : defaults.creativity,
    inputImage:
      value.inputImage === null || value.inputImage === undefined
        ? null
        : isUpscaleImage(value.inputImage)
          ? value.inputImage
          : null,
    loras: normalizeLoras(value.loras),
    model: isMainModelConfig(value.model) ? value.model : null,
    negativePrompt: typeof value.negativePrompt === 'string' ? value.negativePrompt : defaults.negativePrompt,
    negativePromptEnabled:
      typeof value.negativePromptEnabled === 'boolean' ? value.negativePromptEnabled : defaults.negativePromptEnabled,
    negativePromptHeightPx: normalizePromptHeight(
      value.negativePromptHeightPx,
      MIN_NEGATIVE_PROMPT_HEIGHT_PX,
      MAX_NEGATIVE_PROMPT_HEIGHT_PX,
      defaults.negativePromptHeightPx
    ),
    positivePrompt: typeof value.positivePrompt === 'string' ? value.positivePrompt : defaults.positivePrompt,
    positivePromptHeightPx: normalizePromptHeight(
      value.positivePromptHeightPx,
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      defaults.positivePromptHeightPx
    ),
    scale: isFiniteNumber(value.scale) ? value.scale : defaults.scale,
    scheduler: typeof value.scheduler === 'string' ? value.scheduler : defaults.scheduler,
    seed: isFiniteNumber(value.seed) ? value.seed : defaults.seed,
    shouldRandomizeSeed:
      typeof value.shouldRandomizeSeed === 'boolean' ? value.shouldRandomizeSeed : defaults.shouldRandomizeSeed,
    steps: isFiniteNumber(value.steps) ? value.steps : defaults.steps,
    structure: isFiniteNumber(value.structure) ? value.structure : defaults.structure,
    tileControlnetModel: isTileControlNetModelConfig(value.tileControlnetModel) ? value.tileControlnetModel : null,
    tileOverlap: isFiniteNumber(value.tileOverlap) ? value.tileOverlap : defaults.tileOverlap,
    tileSize: isFiniteNumber(value.tileSize) ? value.tileSize : defaults.tileSize,
    upscaleModel: isSpandrelModelConfig(value.upscaleModel) ? value.upscaleModel : null,
    vae: isVaeModelConfig(value.vae) ? value.vae : null,
    vaePrecision: isVaePrecision(value.vaePrecision) ? value.vaePrecision : defaults.vaePrecision,
  };
};

export const syncUpscaleWidgetValuesWithModels = (
  values: UpscaleWidgetValues,
  models: readonly ModelConfig[]
): UpscaleWidgetValues => {
  const modelsByKey = new Map(models.map((model) => [model.key, model]));
  const storedMain = values.model ? modelsByKey.get(values.model.key) : undefined;
  const model: MainModelConfig | null = isSupportedUpscaleMainModel(storedMain)
    ? storedMain
    : ((models.find(isSupportedUpscaleMainModel) as MainModelConfig | undefined) ?? null);
  const storedSpandrel = values.upscaleModel ? modelsByKey.get(values.upscaleModel.key) : undefined;
  const upscaleModel: SpandrelModelConfig | null = isSpandrelModelConfig(storedSpandrel)
    ? storedSpandrel
    : ((models.find(isSpandrelModelConfig) as SpandrelModelConfig | undefined) ?? null);
  const storedControlNet = values.tileControlnetModel ? modelsByKey.get(values.tileControlnetModel.key) : undefined;
  const tileControlnetModel = isTileControlNetCandidate(storedControlNet, model)
    ? storedControlNet
    : (models.find((candidate) => isTileControlNetCandidate(candidate, model)) ?? null);
  const storedVae = values.vae ? modelsByKey.get(values.vae.key) : undefined;
  const vae =
    storedVae && isVaeModelConfig(storedVae) && model && isVaeCompatibleWithGenerateModel(model, storedVae)
      ? storedVae
      : null;
  const loras = model
    ? values.loras.flatMap((lora) => {
        const installed = modelsByKey.get(lora.model.key);

        return installed && isLoraModelConfig(installed) && isLoraCompatibleWithModel(installed, model)
          ? [{ ...lora, model: installed }]
          : [];
      })
    : [];

  if (
    values.model === model &&
    values.upscaleModel === upscaleModel &&
    values.tileControlnetModel === tileControlnetModel &&
    values.vae === vae &&
    loras.length === values.loras.length &&
    loras.every((lora, index) => lora.model === values.loras[index]?.model)
  ) {
    return values;
  }

  return { ...values, loras, model, tileControlnetModel, upscaleModel, vae };
};

const addRangeReason = (reasons: string[], label: string, value: number, min: number, max: number): void => {
  if (!Number.isFinite(value) || value < min || value > max) {
    reasons.push(`${label} must be between ${min} and ${max}.`);
  }
};

export const getUpscaleValidationReasons = (values: UpscaleWidgetValues, models?: readonly ModelConfig[]): string[] => {
  const reasons: string[] = [];

  if (!values.inputImage) {
    reasons.push('Upscale needs an input image. Upload one or send one from Gallery.');
  }
  if (!values.upscaleModel) {
    reasons.push('Upscale needs a Spandrel image-to-image model.');
  }
  if (!values.model) {
    reasons.push('Upscale needs an SD1.5 or SDXL main model.');
  } else if (!isSupportedUpscaleMainModel(values.model)) {
    reasons.push('Upscale supports only SD1.5 and SDXL main models.');
  }
  if (!values.tileControlnetModel) {
    reasons.push('Upscale needs a Tile or Union ControlNet compatible with the main model.');
  } else if (!isTileControlNetCandidate(values.tileControlnetModel, values.model)) {
    reasons.push('The Tile ControlNet must match the main model base and be a Tile or Union model.');
  }

  addRangeReason(reasons, 'Scale', values.scale, UPSCALE_SCALE_MIN, UPSCALE_SCALE_MAX);
  addRangeReason(reasons, 'Creativity', values.creativity, UPSCALE_CREATIVITY_MIN, UPSCALE_CREATIVITY_MAX);
  addRangeReason(reasons, 'Structure', values.structure, UPSCALE_STRUCTURE_MIN, UPSCALE_STRUCTURE_MAX);
  addRangeReason(reasons, 'Tile size', values.tileSize, UPSCALE_TILE_SIZE_MIN, UPSCALE_TILE_SIZE_MAX);
  addRangeReason(reasons, 'Tile overlap', values.tileOverlap, UPSCALE_TILE_OVERLAP_MIN, UPSCALE_TILE_OVERLAP_MAX);
  addRangeReason(reasons, 'Steps', values.steps, 1, 1000);
  addRangeReason(reasons, 'CFG scale', values.cfgScale, 0, 100);
  addRangeReason(reasons, 'Seed', values.seed, 0, SEED_MAX);

  if (models) {
    const installedByKey = new Map(models.map((model) => [model.key, model]));
    const required: ModelIdentifierConfig[] = [
      ...(values.model ? [values.model] : []),
      ...(values.upscaleModel ? [values.upscaleModel] : []),
      ...(values.tileControlnetModel ? [values.tileControlnetModel] : []),
    ];

    for (const selected of required) {
      const installed = installedByKey.get(selected.key);

      if (!installed) {
        reasons.push(`${selected.name} is no longer installed.`);
      }
    }

    const installedMain = values.model ? installedByKey.get(values.model.key) : undefined;
    const installedSpandrel = values.upscaleModel ? installedByKey.get(values.upscaleModel.key) : undefined;
    const installedControlNet = values.tileControlnetModel
      ? installedByKey.get(values.tileControlnetModel.key)
      : undefined;

    if (values.model && installedMain && !isSupportedUpscaleMainModel(installedMain)) {
      reasons.push(`${values.model.name} is not an installed SD1.5 or SDXL main model.`);
    }
    if (values.upscaleModel && installedSpandrel && !isSpandrelModelConfig(installedSpandrel)) {
      reasons.push(`${values.upscaleModel.name} is not an installed Spandrel model.`);
    }
    if (
      values.tileControlnetModel &&
      installedControlNet &&
      !isTileControlNetCandidate(installedControlNet, values.model)
    ) {
      reasons.push(`${values.tileControlnetModel.name} is not an installed compatible Tile or Union ControlNet.`);
    }
    if (values.vae) {
      const installedVae = installedByKey.get(values.vae.key);

      if (!installedVae) {
        reasons.push(`${values.vae.name} is no longer installed.`);
      } else if (
        !values.model ||
        !isVaeModelConfig(installedVae) ||
        !isVaeCompatibleWithGenerateModel(values.model, installedVae)
      ) {
        reasons.push(`${values.vae.name} is not compatible with ${values.model?.name ?? 'the main model'}.`);
      }
    }
    for (const lora of values.loras) {
      const installedLora = installedByKey.get(lora.model.key);

      if (!installedLora) {
        reasons.push(`${lora.model.name} is no longer installed.`);
      } else if (
        !isLoraModelConfig(installedLora) ||
        (values.model && !isLoraCompatibleWithModel(installedLora, values.model))
      ) {
        reasons.push(`${lora.model.name} is not compatible with ${values.model?.name ?? 'the main model'}.`);
      }
    }
  }

  return reasons;
};

export const getUpscaleOutputDimensions = (
  image: Pick<ImageWithDims, 'width' | 'height'>,
  scale: number
): { width: number; height: number } => ({
  height: Math.floor((image.height * scale) / 8) * 8,
  width: Math.floor((image.width * scale) / 8) * 8,
});

export const clearDeletedUpscaleInput = (
  values: UpscaleWidgetValues,
  deletedImageNames: ReadonlySet<string>
): UpscaleWidgetValues =>
  values.inputImage && deletedImageNames.has(values.inputImage.image_name) ? { ...values, inputImage: null } : values;

export const resolveUpscaleSeed = (values: UpscaleWidgetValues): number =>
  values.shouldRandomizeSeed ? Math.floor(Math.random() * SEED_MAX) : values.seed;

export const cloneUpscaleWidgetValues = (values: UpscaleWidgetValues): UpscaleWidgetValues => structuredClone(values);

import type {
  AspectRatioId,
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
  VaePrecision,
} from './types';

import { sanitizeBatchCount } from './batch';
import { isVaeCompatibleWithGenerateModel } from './componentCompatibility';

/** Preset ratios, with the preset to switch to when dimensions are swapped. */
export const ASPECT_RATIO_MAP: Record<Exclude<AspectRatioId, 'Free'>, { ratio: number; inverseId: AspectRatioId }> = {
  '8:1': { inverseId: '1:8', ratio: 8 / 1 },
  '4:1': { inverseId: '1:4', ratio: 4 / 1 },
  '21:9': { inverseId: '9:21', ratio: 21 / 9 },
  '16:9': { inverseId: '9:16', ratio: 16 / 9 },
  '3:2': { inverseId: '2:3', ratio: 3 / 2 },
  '5:4': { inverseId: '4:5', ratio: 5 / 4 },
  '4:3': { inverseId: '3:4', ratio: 4 / 3 },
  '1:1': { inverseId: '1:1', ratio: 1 },
  '3:4': { inverseId: '4:3', ratio: 3 / 4 },
  '4:5': { inverseId: '5:4', ratio: 4 / 5 },
  '2:3': { inverseId: '3:2', ratio: 2 / 3 },
  '9:16': { inverseId: '16:9', ratio: 9 / 16 },
  '1:4': { inverseId: '4:1', ratio: 1 / 4 },
  '9:21': { inverseId: '21:9', ratio: 9 / 21 },
  '1:8': { inverseId: '8:1', ratio: 1 / 8 },
};

export const ASPECT_RATIO_OPTIONS: AspectRatioId[] = [
  'Free',
  '21:9',
  '16:9',
  '3:2',
  '4:3',
  '1:1',
  '3:4',
  '2:3',
  '9:16',
  '9:21',
];

export const isAspectRatioId = (value: unknown): value is AspectRatioId =>
  value === 'Free' || (typeof value === 'string' && value in ASPECT_RATIO_MAP);

export const DIMENSION_GRID = 8;
export const MIN_DIMENSION = 64;
export const MAX_DIMENSION = 4096;

export const SEED_MAX = 4_294_967_295;
export const MIN_NEGATIVE_PROMPT_HEIGHT_PX = 56;
export const MAX_NEGATIVE_PROMPT_HEIGHT_PX = 240;
export const DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX = 56;
export const MIN_POSITIVE_PROMPT_HEIGHT_PX = 96;
export const MAX_POSITIVE_PROMPT_HEIGHT_PX = 360;
export const DEFAULT_POSITIVE_PROMPT_HEIGHT_PX = 96;

export const DEFAULT_LORA_WEIGHT_CONFIG = {
  coarseStep: 0.05,
  initial: 0.75,
  numberInputMax: 10,
  numberInputMin: -10,
  sliderMax: 2,
  sliderMin: -1,
};

export const clampDimension = (value: number, grid = DIMENSION_GRID): number =>
  Math.min(MAX_DIMENSION, Math.max(MIN_DIMENSION, Math.round(value / grid) * grid));

/** Fit a width/height of the given ratio into approximately the given pixel area, snapped to the grid. */
export const calculateNewSize = (
  ratio: number,
  area: number,
  grid = DIMENSION_GRID
): { width: number; height: number } => {
  const exactWidth = Math.sqrt(area * ratio);

  return { height: clampDimension(exactWidth / ratio, grid), width: clampDimension(exactWidth, grid) };
};

/** Best-matching preset for existing dimensions, falling back to Free. */
export const deriveAspectRatioId = (width: number, height: number): AspectRatioId => {
  if (height <= 0) {
    return 'Free';
  }

  const ratio = width / height;

  for (const [id, entry] of Object.entries(ASPECT_RATIO_MAP)) {
    if (Math.abs(entry.ratio - ratio) < 0.0001) {
      return id as AspectRatioId;
    }
  }

  return 'Free';
};

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const hasFiniteNumber = (record: Record<string, unknown>, key: string): boolean =>
  typeof record[key] === 'number' && Number.isFinite(record[key]);

const getClampedNumber = (record: Record<string, unknown>, key: string, min: number, max: number, fallback: number) =>
  hasFiniteNumber(record, key) ? Math.min(Math.max(record[key] as number, min), max) : fallback;

export const isMainModelConfig = (value: unknown): value is MainModelConfig =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.name === 'string' &&
  typeof value.base === 'string' &&
  value.type === 'main';

export const isExternalImageGeneratorModelConfig = (value: unknown): value is GenerateModelConfig =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.name === 'string' &&
  value.base === 'external' &&
  value.type === 'external_image_generator';

export const isGenerateModelConfig = (value: unknown): value is GenerateModelConfig =>
  isMainModelConfig(value) || isExternalImageGeneratorModelConfig(value);

export const isModelIdentifierConfig = (value: unknown): value is ComponentModelConfig =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.name === 'string' &&
  typeof value.base === 'string' &&
  typeof value.type === 'string';

export const isVaeModelConfig = (value: unknown): value is VaeModelConfig =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.name === 'string' &&
  typeof value.base === 'string' &&
  value.type === 'vae';

export const isLoraModelConfig = (value: unknown): value is LoraModelConfig =>
  isRecord(value) &&
  typeof value.key === 'string' &&
  typeof value.name === 'string' &&
  typeof value.base === 'string' &&
  value.type === 'lora';

const getModelIdentifierOrNull = (value: unknown): ComponentModelConfig | null =>
  isModelIdentifierConfig(value) ? value : null;

const getMainModelOrNull = (value: unknown): MainModelConfig | null => (isMainModelConfig(value) ? value : null);

const isVaePrecision = (value: unknown): value is VaePrecision => value === 'fp16' || value === 'fp32';

const isGenerateLora = (value: unknown): value is GenerateLora =>
  isRecord(value) &&
  isLoraModelConfig(value.model) &&
  hasFiniteNumber(value, 'weight') &&
  typeof value.isEnabled === 'boolean';

export const DEFAULT_REFERENCE_IMAGE_LIMIT = 5;

const REFERENCE_IMAGE_CONFIG_TYPES = new Set([
  'external_reference_image',
  'flux2_reference_image',
  'flux_kontext_reference_image',
  'flux_redux',
  'ip_adapter',
  'qwen_image_reference_image',
]);

const isClipVisionModel = (value: unknown): value is 'ViT-H' | 'ViT-G' | 'ViT-L' =>
  value === 'ViT-H' || value === 'ViT-G' || value === 'ViT-L';

const isIPAdapterMethod = (
  value: unknown
): value is 'full' | 'style' | 'composition' | 'style_strong' | 'style_precise' =>
  value === 'full' ||
  value === 'style' ||
  value === 'composition' ||
  value === 'style_strong' ||
  value === 'style_precise';

const isFluxReduxImageInfluence = (value: unknown): value is 'lowest' | 'low' | 'medium' | 'high' | 'highest' =>
  value === 'lowest' || value === 'low' || value === 'medium' || value === 'high' || value === 'highest';

export const isReferenceImageAsset = (value: unknown): value is GenerateReferenceImageAsset =>
  isRecord(value) &&
  typeof value.imageName === 'string' &&
  typeof value.imageUrl === 'string' &&
  typeof value.thumbnailUrl === 'string' &&
  typeof value.queuedAt === 'string' &&
  typeof value.sourceQueueItemId === 'string' &&
  hasFiniteNumber(value, 'width') &&
  hasFiniteNumber(value, 'height');

const cloneReferenceImageAsset = (image: GenerateReferenceImageAsset): GenerateReferenceImageAsset => ({ ...image });

const normalizeReferenceImageAsset = (value: unknown): GenerateReferenceImageAsset | null =>
  value === null ? null : isReferenceImageAsset(value) ? cloneReferenceImageAsset(value) : null;

const normalizeReferenceImageConfig = (value: unknown): GenerateReferenceImageConfig | null => {
  if (!isRecord(value) || typeof value.type !== 'string' || !REFERENCE_IMAGE_CONFIG_TYPES.has(value.type)) {
    return null;
  }

  const image = normalizeReferenceImageAsset(value.image);

  if (value.image !== null && image === null) {
    return null;
  }

  switch (value.type) {
    case 'external_reference_image':
    case 'flux2_reference_image':
    case 'qwen_image_reference_image':
      return { image, type: value.type };
    case 'flux_kontext_reference_image':
      return { image, model: getMainModelOrNull(value.model), type: value.type };
    case 'flux_redux':
      return {
        image,
        imageInfluence: isFluxReduxImageInfluence(value.imageInfluence) ? value.imageInfluence : 'highest',
        model: getModelIdentifierOrNull(value.model),
        type: value.type,
      };
    case 'ip_adapter':
      return {
        beginEndStepPct:
          Array.isArray(value.beginEndStepPct) &&
          value.beginEndStepPct.length === 2 &&
          value.beginEndStepPct.every((item) => typeof item === 'number' && Number.isFinite(item))
            ? [value.beginEndStepPct[0], value.beginEndStepPct[1]]
            : [0, 1],
        clipVisionModel: isClipVisionModel(value.clipVisionModel) ? value.clipVisionModel : 'ViT-H',
        image,
        method: isIPAdapterMethod(value.method) ? value.method : 'full',
        model: getModelIdentifierOrNull(value.model),
        type: value.type,
        weight: hasFiniteNumber(value, 'weight') ? (value.weight as number) : 1,
      };
  }

  return null;
};

export const normalizeReferenceImages = (value: unknown): GenerateReferenceImage[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const images: GenerateReferenceImage[] = [];

  for (const item of value) {
    if (!isRecord(item) || typeof item.id !== 'string') {
      continue;
    }

    const config = normalizeReferenceImageConfig(item.config);

    if (!config) {
      continue;
    }

    images.push({ config, id: item.id, isEnabled: item.isEnabled !== false });
  }

  return images;
};

const cloneReferenceImageConfig = (config: GenerateReferenceImageConfig): GenerateReferenceImageConfig => {
  switch (config.type) {
    case 'external_reference_image':
    case 'flux2_reference_image':
    case 'qwen_image_reference_image':
      return { ...config, image: config.image ? cloneReferenceImageAsset(config.image) : null };
    case 'flux_kontext_reference_image':
      return {
        ...config,
        image: config.image ? cloneReferenceImageAsset(config.image) : null,
        model: config.model ? { ...config.model } : null,
      };
    case 'flux_redux':
    case 'ip_adapter':
      return {
        ...config,
        image: config.image ? cloneReferenceImageAsset(config.image) : null,
        model: config.model ? { ...config.model } : null,
      };
  }
};

export const cloneReferenceImages = (
  referenceImages: readonly GenerateReferenceImage[] = []
): GenerateReferenceImage[] =>
  referenceImages.map((referenceImage) => ({
    ...referenceImage,
    config: cloneReferenceImageConfig(referenceImage.config),
  }));

export const getDefaultLoraWeight = (model: LoraModelConfig): number => {
  const weight = model.default_settings?.weight;

  return Number.isFinite(weight) && weight !== null && weight !== undefined
    ? weight
    : DEFAULT_LORA_WEIGHT_CONFIG.initial;
};

const areModelSnapshotsEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }

  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
};

export const syncGenerateLorasWithModels = (loras: GenerateLora[], models: readonly unknown[]): GenerateLora[] => {
  const modelsByKey = new Map<string, LoraModelConfig>();

  for (const model of models) {
    if (isLoraModelConfig(model)) {
      modelsByKey.set(model.key, model);
    }
  }

  let didSync = false;
  const syncedLoras = loras.map((lora) => {
    const currentModel = modelsByKey.get(lora.model.key);

    if (!currentModel || areModelSnapshotsEqual(currentModel, lora.model)) {
      return lora;
    }

    didSync = true;
    return { ...lora, model: currentModel };
  });

  return didSync ? syncedLoras : loras;
};

const syncModelIdentifierWithModels = <T extends ComponentModelConfig | null>(
  value: T,
  modelsByKey: Map<string, unknown>
): T => {
  if (!value) {
    return value;
  }

  const currentModel = modelsByKey.get(value.key);

  return isModelIdentifierConfig(currentModel) && !areModelSnapshotsEqual(currentModel, value)
    ? (currentModel as T)
    : value;
};

export const getModelDefaultVae = (
  model: GenerateModelConfig,
  vaeModels: readonly VaeModelConfig[]
): VaeModelConfig | null => {
  const defaultVaeKey = (model.default_settings as { vae?: string | null } | null | undefined)?.vae;

  return defaultVaeKey
    ? (vaeModels.find((vae) => vae.key === defaultVaeKey && isVaeCompatibleWithGenerateModel(model, vae)) ?? null)
    : null;
};

export const hasModelDefaultVae = (model: GenerateModelConfig): boolean =>
  typeof (model.default_settings as { vae?: unknown } | null | undefined)?.vae === 'string';

export const cloneGenerateWidgetValues = (
  values: GenerateWidgetValues
): GenerateWidgetValues & Record<string, unknown> => ({
  ...values,
  clipEmbedModel: values.clipEmbedModel ? { ...values.clipEmbedModel } : null,
  clipGEmbedModel: values.clipGEmbedModel ? { ...values.clipGEmbedModel } : null,
  clipLEmbedModel: values.clipLEmbedModel ? { ...values.clipLEmbedModel } : null,
  componentSourceModel: values.componentSourceModel ? { ...values.componentSourceModel } : null,
  loras: values.loras.map((lora) => ({ ...lora, model: { ...lora.model } })),
  model: { ...values.model },
  qwen3EncoderModel: values.qwen3EncoderModel ? { ...values.qwen3EncoderModel } : null,
  qwenVLEncoderModel: values.qwenVLEncoderModel ? { ...values.qwenVLEncoderModel } : null,
  referenceImages: cloneReferenceImages(values.referenceImages),
  t5EncoderModel: values.t5EncoderModel ? { ...values.t5EncoderModel } : null,
  vae: values.vae ? { ...values.vae } : null,
});

const syncReferenceImageConfigWithModels = (
  config: GenerateReferenceImageConfig,
  modelsByKey: Map<string, unknown>
): GenerateReferenceImageConfig => {
  switch (config.type) {
    case 'flux_kontext_reference_image': {
      const model = syncModelIdentifierWithModels(config.model, modelsByKey);

      return model === config.model ? config : { ...config, model: isMainModelConfig(model) ? model : config.model };
    }
    case 'flux_redux':
    case 'ip_adapter': {
      const model = syncModelIdentifierWithModels(config.model, modelsByKey);

      return model === config.model ? config : { ...config, model };
    }
    case 'external_reference_image':
    case 'flux2_reference_image':
    case 'qwen_image_reference_image':
      return config;
  }
};

export const syncReferenceImagesWithModels = (
  referenceImages: GenerateReferenceImage[],
  models: readonly unknown[]
): GenerateReferenceImage[] => {
  const modelsByKey = new Map<string, unknown>();

  for (const model of models) {
    if (isModelIdentifierConfig(model)) {
      modelsByKey.set(model.key, model);
    }
  }

  let didSync = false;
  const synced = referenceImages.map((referenceImage) => {
    const config = syncReferenceImageConfigWithModels(referenceImage.config, modelsByKey);

    if (config === referenceImage.config) {
      return referenceImage;
    }

    didSync = true;
    return { ...referenceImage, config };
  });

  return didSync ? synced : referenceImages;
};

export const syncGenerateWidgetValuesWithModels = (
  values: GenerateWidgetValues,
  models: readonly unknown[]
): GenerateWidgetValues => {
  const modelsByKey = new Map<string, unknown>();

  for (const model of models) {
    if (isModelIdentifierConfig(model)) {
      modelsByKey.set(model.key, model);
    }
  }

  const currentModel = modelsByKey.get(values.model.key);
  const model =
    isGenerateModelConfig(currentModel) && !areModelSnapshotsEqual(currentModel, values.model)
      ? currentModel
      : values.model;
  const vae = syncModelIdentifierWithModels(values.vae, modelsByKey);
  const componentSourceModel = syncModelIdentifierWithModels(values.componentSourceModel, modelsByKey);
  const nextValues: GenerateWidgetValues = {
    ...values,
    clipEmbedModel: syncModelIdentifierWithModels(values.clipEmbedModel, modelsByKey),
    clipGEmbedModel: syncModelIdentifierWithModels(values.clipGEmbedModel, modelsByKey),
    clipLEmbedModel: syncModelIdentifierWithModels(values.clipLEmbedModel, modelsByKey),
    componentSourceModel: isMainModelConfig(componentSourceModel) ? componentSourceModel : values.componentSourceModel,
    loras: syncGenerateLorasWithModels(values.loras, models),
    model,
    modelKey: model.key,
    qwen3EncoderModel: syncModelIdentifierWithModels(values.qwen3EncoderModel, modelsByKey),
    qwenVLEncoderModel: syncModelIdentifierWithModels(values.qwenVLEncoderModel, modelsByKey),
    referenceImages: syncReferenceImagesWithModels(values.referenceImages, models),
    t5EncoderModel: syncModelIdentifierWithModels(values.t5EncoderModel, modelsByKey),
    vae: isVaeModelConfig(vae) ? vae : values.vae,
  };

  return nextValues.model === values.model &&
    nextValues.loras === values.loras &&
    nextValues.vae === values.vae &&
    nextValues.t5EncoderModel === values.t5EncoderModel &&
    nextValues.clipEmbedModel === values.clipEmbedModel &&
    nextValues.clipLEmbedModel === values.clipLEmbedModel &&
    nextValues.clipGEmbedModel === values.clipGEmbedModel &&
    nextValues.qwen3EncoderModel === values.qwen3EncoderModel &&
    nextValues.qwenVLEncoderModel === values.qwenVLEncoderModel &&
    nextValues.referenceImages === values.referenceImages &&
    nextValues.componentSourceModel === values.componentSourceModel &&
    nextValues.modelKey === values.modelKey
    ? values
    : nextValues;
};

export const isLoraCompatibleWithModel = (
  loraModel: Pick<LoraModelConfig, 'base' | 'variant'>,
  mainModel: Pick<GenerateModelConfig, 'base' | 'variant'>
): boolean => {
  if (mainModel.base === 'flux2') {
    if (loraModel.base !== 'flux2' && loraModel.base !== 'flux') {
      return false;
    }

    return (
      loraModel.base === 'flux' || !mainModel.variant || !loraModel.variant || loraModel.variant === mainModel.variant
    );
  }

  if (loraModel.base !== mainModel.base) {
    return false;
  }

  return true;
};

/**
 * Validate stored widget values and fill in fields that older persisted
 * projects predate. The core fields must be present and well-typed; newer
 * fields fall back to their defaults so old projects keep their prompts and
 * dimensions instead of being reset wholesale.
 */
export const normalizeGenerateSettings = (values: unknown): GenerateSettings | null => {
  if (!isRecord(values)) {
    return null;
  }

  const isCoreShapeValid =
    typeof values.modelKey === 'string' &&
    typeof values.positivePrompt === 'string' &&
    typeof values.negativePrompt === 'string' &&
    typeof values.scheduler === 'string' &&
    typeof values.shouldRandomizeSeed === 'boolean' &&
    ['width', 'height', 'steps', 'cfgScale', 'cfgRescaleMultiplier', 'seed'].every((key) =>
      hasFiniteNumber(values, key)
    );

  if (!isCoreShapeValid) {
    return null;
  }

  const width = values.width as number;
  const height = values.height as number;

  return {
    aspectRatioId: isAspectRatioId(values.aspectRatioId) ? values.aspectRatioId : deriveAspectRatioId(width, height),
    aspectRatioIsLocked: typeof values.aspectRatioIsLocked === 'boolean' ? values.aspectRatioIsLocked : false,
    aspectRatioValue:
      hasFiniteNumber(values, 'aspectRatioValue') && (values.aspectRatioValue as number) > 0
        ? (values.aspectRatioValue as number)
        : height > 0
          ? width / height
          : 1,
    batchCount: sanitizeBatchCount(values.batchCount),
    cfgRescaleMultiplier: values.cfgRescaleMultiplier as number,
    cfgScale: values.cfgScale as number,
    clipSkip: hasFiniteNumber(values, 'clipSkip') ? (values.clipSkip as number) : 0,
    colorCompensation: typeof values.colorCompensation === 'boolean' ? values.colorCompensation : false,
    height,
    loras: Array.isArray(values.loras) ? values.loras.filter(isGenerateLora) : [],
    modelKey: values.modelKey as string,
    negativePromptEnabled: typeof values.negativePromptEnabled === 'boolean' ? values.negativePromptEnabled : true,
    negativePrompt: values.negativePrompt as string,
    negativePromptHeightPx: getClampedNumber(
      values,
      'negativePromptHeightPx',
      MIN_NEGATIVE_PROMPT_HEIGHT_PX,
      MAX_NEGATIVE_PROMPT_HEIGHT_PX,
      DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX
    ),
    positivePrompt: values.positivePrompt as string,
    positivePromptHeightPx: getClampedNumber(
      values,
      'positivePromptHeightPx',
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      DEFAULT_POSITIVE_PROMPT_HEIGHT_PX
    ),
    referenceImages: normalizeReferenceImages(values.referenceImages),
    scheduler: values.scheduler as string,
    seamlessXAxis: typeof values.seamlessXAxis === 'boolean' ? values.seamlessXAxis : false,
    seamlessYAxis: typeof values.seamlessYAxis === 'boolean' ? values.seamlessYAxis : false,
    seed: values.seed as number,
    shouldRandomizeSeed: values.shouldRandomizeSeed as boolean,
    steps: values.steps as number,
    vae: isVaeModelConfig(values.vae) ? values.vae : null,
    vaePrecision: isVaePrecision(values.vaePrecision) ? values.vaePrecision : 'fp32',
    t5EncoderModel: getModelIdentifierOrNull(values.t5EncoderModel),
    clipEmbedModel: getModelIdentifierOrNull(values.clipEmbedModel),
    clipLEmbedModel: getModelIdentifierOrNull(values.clipLEmbedModel),
    clipGEmbedModel: getModelIdentifierOrNull(values.clipGEmbedModel),
    qwen3EncoderModel: getModelIdentifierOrNull(values.qwen3EncoderModel),
    qwenVLEncoderModel: getModelIdentifierOrNull(values.qwenVLEncoderModel),
    componentSourceModel: getMainModelOrNull(values.componentSourceModel),
    width,
  };
};

export const normalizeGenerateWidgetValues = (values: unknown): GenerateWidgetValues | null => {
  const settings = normalizeGenerateSettings(values);

  if (!settings || !isRecord(values) || !isGenerateModelConfig(values.model)) {
    return null;
  }

  return { ...settings, model: values.model };
};

export const isGenerateSettings = (values: unknown): values is GenerateSettings => {
  const normalized = normalizeGenerateSettings(values);

  if (!normalized || !isRecord(values)) {
    return false;
  }

  // Strict only over the keys normalize would have to invent.
  return (
    isAspectRatioId(values.aspectRatioId) &&
    typeof values.aspectRatioIsLocked === 'boolean' &&
    hasFiniteNumber(values, 'aspectRatioValue') &&
    hasFiniteNumber(values, 'clipSkip') &&
    typeof values.colorCompensation === 'boolean' &&
    typeof values.negativePromptEnabled === 'boolean' &&
    hasFiniteNumber(values, 'negativePromptHeightPx') &&
    hasFiniteNumber(values, 'positivePromptHeightPx') &&
    Array.isArray(values.loras) &&
    values.loras.every(isGenerateLora) &&
    Array.isArray(values.referenceImages) &&
    normalizeReferenceImages(values.referenceImages).length === values.referenceImages.length &&
    typeof values.seamlessXAxis === 'boolean' &&
    typeof values.seamlessYAxis === 'boolean' &&
    isVaePrecision(values.vaePrecision) &&
    (values.vae === null || isVaeModelConfig(values.vae)) &&
    (values.t5EncoderModel === null || isModelIdentifierConfig(values.t5EncoderModel)) &&
    (values.clipEmbedModel === null || isModelIdentifierConfig(values.clipEmbedModel)) &&
    (values.clipLEmbedModel === null || isModelIdentifierConfig(values.clipLEmbedModel)) &&
    (values.clipGEmbedModel === null || isModelIdentifierConfig(values.clipGEmbedModel)) &&
    (values.qwen3EncoderModel === null || isModelIdentifierConfig(values.qwen3EncoderModel)) &&
    (values.qwenVLEncoderModel === null || isModelIdentifierConfig(values.qwenVLEncoderModel)) &&
    (values.componentSourceModel === null || isMainModelConfig(values.componentSourceModel))
  );
};

export const isGenerateWidgetValues = (values: unknown): values is GenerateWidgetValues =>
  isGenerateSettings(values) && isGenerateModelConfig((values as unknown as Record<string, unknown>).model);

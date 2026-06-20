import type {
  AspectRatioId,
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
  VaePrecision,
} from './types';

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
  t5EncoderModel: values.t5EncoderModel ? { ...values.t5EncoderModel } : null,
  vae: values.vae ? { ...values.vae } : null,
});

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
    ['batchCount', 'width', 'height', 'steps', 'cfgScale', 'cfgRescaleMultiplier', 'seed'].every((key) =>
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
    batchCount: values.batchCount as number,
    cfgRescaleMultiplier: values.cfgRescaleMultiplier as number,
    cfgScale: values.cfgScale as number,
    clipSkip: hasFiniteNumber(values, 'clipSkip') ? (values.clipSkip as number) : 0,
    height,
    loras: Array.isArray(values.loras) ? values.loras.filter(isGenerateLora) : [],
    modelKey: values.modelKey as string,
    negativePrompt: values.negativePrompt as string,
    positivePrompt: values.positivePrompt as string,
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
    Array.isArray(values.loras) &&
    values.loras.every(isGenerateLora) &&
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

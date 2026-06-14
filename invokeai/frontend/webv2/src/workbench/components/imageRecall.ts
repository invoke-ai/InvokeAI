import { getSettingsWithModelDefaults } from '../generation/graph';
import { CLIP_SKIP_MAX, clampDimension, deriveAspectRatioId, isKnownScheduler, SEED_MAX } from '../generation/settings';
import type { GenerateWidgetValues, MainModelConfig, SupportedGenerateBase, VaeModelConfig } from '../generation/types';
import type { GalleryImage } from '../gallery/api';

export type ImageRecallKind = 'all' | 'remix' | 'prompts' | 'seed' | 'dimensions' | 'clipSkip';

export interface ImageRecallCapabilities {
  all: boolean;
  remix: boolean;
  prompts: boolean;
  seed: boolean;
  dimensions: boolean;
  clipSkip: boolean;
}

export const EMPTY_IMAGE_RECALL_CAPABILITIES: ImageRecallCapabilities = {
  all: false,
  clipSkip: false,
  dimensions: false,
  prompts: false,
  remix: false,
  seed: false,
};

type RecalledField =
  | 'model'
  | 'vae'
  | 'prompts'
  | 'seed'
  | 'size'
  | 'steps'
  | 'cfg'
  | 'scheduler'
  | 'seamless'
  | 'clipSkip';

export interface ImageRecallResult {
  fields: RecalledField[];
  values: GenerateWidgetValues;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const getRecord = (value: unknown, key: string): Record<string, unknown> | null => {
  if (!isRecord(value)) {
    return null;
  }

  const child = value[key];

  return isRecord(child) ? child : null;
};

const getString = (metadata: unknown, key: string): string | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

  return typeof value === 'string' ? value : null;
};

const getNullableString = (metadata: unknown, key: string): string | null | undefined => {
  if (!isRecord(metadata)) {
    return undefined;
  }

  const value = metadata[key];

  if (value === null) {
    return null;
  }

  return typeof value === 'string' ? value : undefined;
};

const getBoolean = (metadata: unknown, key: string): boolean | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

  return typeof value === 'boolean' ? value : null;
};

const getNumber = (metadata: unknown, key: string): number | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const getInteger = (metadata: unknown, key: string): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && Number.isInteger(value) ? value : null;
};

const getSeed = (metadata: unknown): number | null => {
  const seed = getInteger(metadata, 'seed');

  return seed !== null && seed >= 0 && seed <= SEED_MAX ? seed : null;
};

const getDimension = (metadata: unknown, key: 'height' | 'width'): number | null => {
  const dimension = getNumber(metadata, key);

  return dimension !== null && dimension >= 64 ? clampDimension(dimension) : null;
};

const getImageDimension = (image: GalleryImage, key: 'height' | 'width'): number | null => {
  const dimension = image[key];

  return Number.isFinite(dimension) && dimension >= 64 ? clampDimension(dimension) : null;
};

const getSteps = (metadata: unknown): number | null => {
  const steps = getInteger(metadata, 'steps');

  return steps !== null && steps >= 1 ? steps : null;
};

const getCfgScale = (metadata: unknown): number | null => {
  const cfgScale = getNumber(metadata, 'cfg_scale');

  return cfgScale !== null && cfgScale >= 1 ? cfgScale : null;
};

const getCfgRescaleMultiplier = (metadata: unknown): number | null => {
  const cfgRescaleMultiplier = getNumber(metadata, 'cfg_rescale_multiplier');

  return cfgRescaleMultiplier !== null && cfgRescaleMultiplier >= 0 && cfgRescaleMultiplier < 1
    ? cfgRescaleMultiplier
    : null;
};

const getScheduler = (metadata: unknown): string | null => {
  const scheduler = getString(metadata, 'scheduler');

  return scheduler !== null && isKnownScheduler(scheduler) ? scheduler : null;
};

const getClipSkip = (metadata: unknown): number | null => {
  const clipSkip = getInteger(metadata, 'clip_skip');

  return clipSkip !== null && clipSkip >= 0 ? clipSkip : null;
};

const supportsClipSkip = (base: string): base is Exclude<SupportedGenerateBase, 'sdxl'> =>
  base === 'sd-1' || base === 'sd-2';

const getClipSkipMax = (model: Pick<MainModelConfig, 'base'>): number | null =>
  supportsClipSkip(model.base) ? CLIP_SKIP_MAX[model.base] : null;

const getImageSize = (image: GalleryImage): Pick<GenerateWidgetValues, 'height' | 'width'> | null => {
  const width = getImageDimension(image, 'width');
  const height = getImageDimension(image, 'height');

  return width !== null && height !== null ? { height, width } : null;
};

const getMetadataSize = (metadata: unknown): Partial<Pick<GenerateWidgetValues, 'height' | 'width'>> => {
  const width = getDimension(metadata, 'width');
  const height = getDimension(metadata, 'height');

  return {
    ...(width !== null ? { width } : {}),
    ...(height !== null ? { height } : {}),
  };
};

const getMetadataModelKey = (metadata: unknown, key: 'model' | 'vae'): string | null => {
  const model = getRecord(metadata, key);

  return typeof model?.key === 'string' ? model.key : null;
};

const getSupportedMetadataModel = (metadata: unknown, supportedModels: MainModelConfig[]): MainModelConfig | null => {
  const modelKey = getMetadataModelKey(metadata, 'model');

  return modelKey ? (supportedModels.find((model) => model.key === modelKey) ?? null) : null;
};

const getMetadataVae = (
  metadata: unknown,
  model: MainModelConfig,
  vaeModels: VaeModelConfig[]
): VaeModelConfig | null | undefined => {
  if (!isRecord(metadata) || !('vae' in metadata)) {
    return undefined;
  }

  if (metadata.vae === null) {
    return null;
  }

  const vaeKey = getMetadataModelKey(metadata, 'vae');

  if (!vaeKey) {
    return undefined;
  }

  return vaeModels.find((vae) => vae.key === vaeKey && vae.base === model.base) ?? undefined;
};

const getDefaultVae = (model: MainModelConfig, vaeModels: VaeModelConfig[]): VaeModelConfig | null => {
  const defaultVaeKey = model.default_settings?.vae;

  return defaultVaeKey ? (vaeModels.find((vae) => vae.key === defaultVaeKey && vae.base === model.base) ?? null) : null;
};

const getPromptPatch = (
  metadata: unknown
): Partial<Pick<GenerateWidgetValues, 'negativePrompt' | 'positivePrompt'>> => {
  const positivePrompt = getString(metadata, 'positive_prompt');
  const negativePrompt = getNullableString(metadata, 'negative_prompt');

  return {
    ...(positivePrompt !== null ? { positivePrompt } : {}),
    ...(negativePrompt !== undefined ? { negativePrompt: negativePrompt ?? '' } : {}),
  };
};

const hasPrompt = (metadata: unknown): boolean => Object.keys(getPromptPatch(metadata)).length > 0;

const hasMetadataSize = (metadata: unknown): boolean => Object.keys(getMetadataSize(metadata)).length > 0;

const hasGenerationSettings = (metadata: unknown): boolean =>
  getSteps(metadata) !== null ||
  getCfgScale(metadata) !== null ||
  getCfgRescaleMultiplier(metadata) !== null ||
  getScheduler(metadata) !== null ||
  getBoolean(metadata, 'seamless_x') !== null ||
  getBoolean(metadata, 'seamless_y') !== null;

const withDimensions = (
  values: GenerateWidgetValues,
  size: Partial<Pick<GenerateWidgetValues, 'height' | 'width'>>
): GenerateWidgetValues => {
  const width = size.width ?? values.width;
  const height = size.height ?? values.height;

  return {
    ...values,
    ...size,
    aspectRatioId: deriveAspectRatioId(width, height),
    aspectRatioValue: height > 0 ? width / height : 1,
  };
};

const getSupportedClipSkip = (metadata: unknown, model: MainModelConfig): number | null => {
  const clipSkip = getClipSkip(metadata);
  const clipSkipMax = getClipSkipMax(model);

  return clipSkip !== null && clipSkipMax !== null ? Math.min(clipSkipMax, clipSkip) : null;
};

export const getImageRecallCapabilities = ({
  currentValues,
  image,
  metadata,
  supportedModels,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  metadata: unknown;
  supportedModels: MainModelConfig[];
  vaeModels: VaeModelConfig[];
}): ImageRecallCapabilities => {
  const supportedMetadataModel = getSupportedMetadataModel(metadata, supportedModels);
  const clipSkipModel = supportedMetadataModel ?? currentValues.model;
  const hasVae = getMetadataVae(metadata, clipSkipModel, vaeModels) !== undefined;
  const hasClipSkip = getSupportedClipSkip(metadata, clipSkipModel) !== null;
  const hasSeed = getSeed(metadata) !== null;
  const hasPrompts = hasPrompt(metadata);
  const hasSize = hasMetadataSize(metadata);
  const hasSettings = hasGenerationSettings(metadata);
  const hasModel = supportedMetadataModel !== null;
  const hasAnyMetadata = hasModel || hasVae || hasPrompts || hasSeed || hasSize || hasSettings || hasClipSkip;
  const hasNonSeedMetadata = hasModel || hasVae || hasPrompts || hasSize || hasSettings || hasClipSkip;

  return {
    all: hasAnyMetadata,
    clipSkip: getSupportedClipSkip(metadata, currentValues.model) !== null,
    dimensions: getImageSize(image) !== null,
    prompts: hasPrompts,
    remix: hasNonSeedMetadata,
    seed: hasSeed,
  };
};

export const buildImageRecallSettings = ({
  currentValues,
  image,
  kind,
  metadata,
  supportedModels,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  kind: ImageRecallKind;
  metadata: unknown;
  supportedModels: MainModelConfig[];
  vaeModels: VaeModelConfig[];
}): ImageRecallResult | null => {
  const fields: RecalledField[] = [];
  let values: GenerateWidgetValues = {
    ...currentValues,
    model: { ...currentValues.model },
    vae: currentValues.vae ? { ...currentValues.vae } : null,
  };

  if (kind === 'all' || kind === 'remix') {
    const model = getSupportedMetadataModel(metadata, supportedModels);

    if (model) {
      values = { ...getSettingsWithModelDefaults(values, model), model, vae: getDefaultVae(model, vaeModels) };
      fields.push('model');
    }

    const vae = getMetadataVae(metadata, values.model, vaeModels);

    if (vae !== undefined) {
      values = { ...values, vae };
      fields.push('vae');
    }
  }

  if (kind === 'all' || kind === 'remix' || kind === 'prompts') {
    const promptPatch = getPromptPatch(metadata);

    if (Object.keys(promptPatch).length > 0) {
      values = { ...values, ...promptPatch };
      fields.push('prompts');
    }
  }

  if (kind === 'all' || kind === 'seed') {
    const seed = getSeed(metadata);

    if (seed !== null) {
      values = { ...values, seed, shouldRandomizeSeed: false };
      fields.push('seed');
    }
  }

  if (kind === 'dimensions') {
    const size = getImageSize(image);

    if (size) {
      values = withDimensions(values, size);
      fields.push('size');
    }
  } else if (kind === 'all' || kind === 'remix') {
    const size = getMetadataSize(metadata);

    if (Object.keys(size).length > 0) {
      values = withDimensions(values, size);
      fields.push('size');
    }
  }

  if (kind === 'all' || kind === 'remix') {
    const steps = getSteps(metadata);
    const cfgScale = getCfgScale(metadata);
    const cfgRescaleMultiplier = getCfgRescaleMultiplier(metadata);
    const scheduler = getScheduler(metadata);
    const seamlessXAxis = getBoolean(metadata, 'seamless_x');
    const seamlessYAxis = getBoolean(metadata, 'seamless_y');

    if (steps !== null) {
      values = { ...values, steps };
      fields.push('steps');
    }

    if (cfgScale !== null || cfgRescaleMultiplier !== null) {
      values = {
        ...values,
        ...(cfgScale !== null ? { cfgScale } : {}),
        ...(cfgRescaleMultiplier !== null ? { cfgRescaleMultiplier } : {}),
      };
      fields.push('cfg');
    }

    if (scheduler !== null) {
      values = { ...values, scheduler };
      fields.push('scheduler');
    }

    if (seamlessXAxis !== null || seamlessYAxis !== null) {
      values = {
        ...values,
        ...(seamlessXAxis !== null ? { seamlessXAxis } : {}),
        ...(seamlessYAxis !== null ? { seamlessYAxis } : {}),
      };
      fields.push('seamless');
    }
  }

  if (kind === 'all' || kind === 'remix' || kind === 'clipSkip') {
    const clipSkip = getSupportedClipSkip(metadata, values.model);

    if (clipSkip !== null) {
      values = { ...values, clipSkip };
      fields.push('clipSkip');
    }
  }

  return fields.length > 0 ? { fields, values } : null;
};

export const getImageRecallTitle = (kind: ImageRecallKind): string => {
  switch (kind) {
    case 'all':
      return 'Recalled image metadata';
    case 'remix':
      return 'Recalled remix settings';
    case 'prompts':
      return 'Recalled prompts';
    case 'seed':
      return 'Recalled seed';
    case 'dimensions':
      return 'Recalled image size';
    case 'clipSkip':
      return 'Recalled CLIP skip';
  }
};

export const getImageRecallMessage = (fields: RecalledField[]): string =>
  `${fields.length} field${fields.length === 1 ? '' : 's'} applied to Generate.`;

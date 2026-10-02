import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

import {
  clampDimension,
  cloneGenerateWidgetValues,
  deriveAspectRatioId,
  getCompatibleReferenceImages,
  getDimensionGrid,
  getGenerateModelSelectionResult,
  getGenerationUiPolicy,
  isComponentCompatibleWithModel,
  getModelDefaultVae,
  getSettingsWithModelDefaults,
  hasArchitectureCapabilities,
  hasModelDefaultVae,
  isKnownScheduler,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeCompatibleWithGenerateModel,
  isValidKrea2RebalanceWeights,
  MAX_HIDIFFUSION_RATIO,
  MIN_HIDIFFUSION_T1_RATIO,
  normalizeReferenceImages,
} from '@features/generation/settings';
import { SEED_MAX } from '@platform/core/seed';

export type ImageRecallKind = 'all' | 'remix' | 'prompts' | 'seed' | 'dimensions' | 'clipSkip';

export interface ImageRecallCapabilities {
  all: boolean;
  remix: boolean;
  prompts: boolean;
  seed: boolean;
  dimensions: boolean;
  clipSkip: boolean;
  /** The image embeds the workflow that made it, so it can be loaded into the editor. */
  workflow: boolean;
}

export const EMPTY_IMAGE_RECALL_CAPABILITIES: ImageRecallCapabilities = {
  all: false,
  clipSkip: false,
  dimensions: false,
  prompts: false,
  remix: false,
  seed: false,
  workflow: false,
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
  | 'hiDiffusion'
  | 'clipSkip'
  | 'components'
  | 'referenceImages'
  | 'krea2Rebalance';

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

export const getBoolean = (metadata: unknown, key: string): boolean | null => {
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

export const getSeed = (metadata: unknown): number | null => {
  const seed = getInteger(metadata, 'seed');

  return seed !== null && seed >= 0 && seed <= SEED_MAX ? seed : null;
};

const getDimension = (metadata: unknown, key: 'height' | 'width', grid: number): number | null => {
  const dimension = getNumber(metadata, key);

  return dimension !== null && dimension >= 64 ? clampDimension(dimension, grid) : null;
};

const getImageDimension = (image: GalleryImage, key: 'height' | 'width', grid: number): number | null => {
  const dimension = image[key];

  return Number.isFinite(dimension) && dimension >= 64 ? clampDimension(dimension, grid) : null;
};

export const getSteps = (metadata: unknown): number | null => {
  const steps = getInteger(metadata, 'steps');

  return steps !== null && steps >= 1 ? steps : null;
};

export const getCfgScale = (metadata: unknown): number | null => {
  const cfgScale = getNumber(metadata, 'cfg_scale');

  return cfgScale !== null && cfgScale >= 1 ? cfgScale : null;
};

export const getCfgRescaleMultiplier = (metadata: unknown): number | null => {
  const cfgRescaleMultiplier = getNumber(metadata, 'cfg_rescale_multiplier');

  return cfgRescaleMultiplier !== null && cfgRescaleMultiplier >= 0 && cfgRescaleMultiplier < 1
    ? cfgRescaleMultiplier
    : null;
};

export const getScheduler = (metadata: unknown): string | null => {
  const scheduler = getString(metadata, 'scheduler');

  return scheduler !== null && isKnownScheduler(scheduler) ? scheduler : null;
};

const getClipSkip = (metadata: unknown): number | null => {
  const clipSkip = getInteger(metadata, 'clip_skip');

  return clipSkip !== null && clipSkip >= 0 ? clipSkip : null;
};

const getClipSkipMax = (model: GenerateModelConfig): number | null =>
  getGenerationUiPolicy(model, { cfgScale: 1 }).clipSkipMax;

const getHiDiffusionPatch = (
  metadata: unknown,
  model: GenerateModelConfig
): Partial<
  Pick<
    GenerateWidgetValues,
    | 'hiDiffusionEnabled'
    | 'hiDiffusionRauNetEnabled'
    | 'hiDiffusionT1Ratio'
    | 'hiDiffusionT2Ratio'
    | 'hiDiffusionWindowAttentionEnabled'
  >
> => {
  if (!getGenerationUiPolicy(model, { cfgScale: 1 }).hiDiffusionVisible) {
    return {};
  }

  const enabled = getBoolean(metadata, 'hidiffusion');
  const rauNetEnabled = getBoolean(metadata, 'hidiffusion_raunet');
  const windowAttentionEnabled = getBoolean(metadata, 'hidiffusion_window_attn');
  const t1Ratio = getNumber(metadata, 'hidiffusion_t1_ratio');
  const t2Ratio = getNumber(metadata, 'hidiffusion_t2_ratio');

  return {
    ...(enabled !== null ? { hiDiffusionEnabled: enabled } : {}),
    ...(rauNetEnabled !== null ? { hiDiffusionRauNetEnabled: rauNetEnabled } : {}),
    ...(windowAttentionEnabled !== null ? { hiDiffusionWindowAttentionEnabled: windowAttentionEnabled } : {}),
    ...(t1Ratio !== null && t1Ratio >= MIN_HIDIFFUSION_T1_RATIO && t1Ratio <= MAX_HIDIFFUSION_RATIO
      ? { hiDiffusionT1Ratio: t1Ratio }
      : {}),
    ...(t2Ratio !== null && t2Ratio >= 0 && t2Ratio <= MAX_HIDIFFUSION_RATIO ? { hiDiffusionT2Ratio: t2Ratio } : {}),
  };
};

const getImageSize = (
  image: GalleryImage,
  model: GenerateModelConfig
): Pick<GenerateWidgetValues, 'height' | 'width'> | null => {
  // Decline dimension recall without architecture policy; persisting fallback grid 8 can leave sizes the model
  // rejects.
  const grid = getDimensionGrid(model.base, model.variant);

  if (grid === null) {
    return null;
  }

  const width = getImageDimension(image, 'width', grid);
  const height = getImageDimension(image, 'height', grid);

  return width !== null && height !== null ? { height, width } : null;
};

export const getMetadataSize = (
  metadata: unknown,
  model: GenerateModelConfig
): Partial<Pick<GenerateWidgetValues, 'height' | 'width'>> => {
  // Return no size and omit its field label when architecture policy is unavailable.
  const grid = getDimensionGrid(model.base, model.variant);

  if (grid === null) {
    return {};
  }

  const width = getDimension(metadata, 'width', grid);
  const height = getDimension(metadata, 'height', grid);

  return {
    ...(width !== null ? { width } : {}),
    ...(height !== null ? { height } : {}),
  };
};

const getMetadataModelKey = (metadata: unknown, key: 'model' | 'vae'): string | null => {
  const model = getRecord(metadata, key);

  return typeof model?.key === 'string' ? model.key : null;
};

const getSupportedMetadataModel = (
  metadata: unknown,
  supportedModels: GenerateModelConfig[]
): GenerateModelConfig | null => {
  const modelKey = getMetadataModelKey(metadata, 'model');

  return modelKey ? (supportedModels.find((model) => model.key === modelKey) ?? null) : null;
};

const getModelFromMetadata = <T extends ComponentModelConfig>(
  metadata: unknown,
  key: string,
  models: readonly ComponentModelConfig[],
  guard: (model: unknown) => model is T
): T | null | undefined => {
  if (!isRecord(metadata) || !(key in metadata)) {
    return undefined;
  }

  if (metadata[key] === null) {
    return null;
  }

  const modelKey = getMetadataModelKey(metadata, key as 'model' | 'vae');

  if (!modelKey) {
    return undefined;
  }

  const model = models.find((candidate) => candidate.key === modelKey);

  return guard(model) ? model : undefined;
};

const getMetadataComponent = (
  metadata: unknown,
  key: string,
  models: readonly ComponentModelConfig[]
): ComponentModelConfig | null | undefined => getModelFromMetadata(metadata, key, models, isModelIdentifierConfig);

const getMetadataMainModel = (
  metadata: unknown,
  key: string,
  models: readonly ComponentModelConfig[]
): MainModelConfig | null | undefined => getModelFromMetadata(metadata, key, models, isMainModelConfig);

const getMetadataVae = (
  metadata: unknown,
  model: GenerateModelConfig,
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

  const vae = vaeModels.find((vae) => vae.key === vaeKey);

  return vae && isVaeCompatibleWithGenerateModel(model, vae) ? vae : undefined;
};

const getPromptPatch = (
  metadata: unknown
): Partial<
  Pick<GenerateWidgetValues, 'negativePrompt' | 'positivePrompt' | 'promptTemplate' | 'promptTemplateViewMode'>
> => {
  const positivePrompt = getString(metadata, 'positive_prompt');
  const negativePrompt = getNullableString(metadata, 'negative_prompt');
  // Image metadata records the prompt the model was given, so it already has any
  // template baked in. Clearing the active template stops it wrapping the text a
  // second time on the next Invoke.
  const clearTemplate = positivePrompt !== null || negativePrompt !== undefined;

  return {
    ...(positivePrompt !== null ? { positivePrompt } : {}),
    ...(negativePrompt !== undefined ? { negativePrompt: negativePrompt ?? '' } : {}),
    ...(clearTemplate ? { promptTemplate: null, promptTemplateViewMode: false } : {}),
  };
};

const hasPrompt = (metadata: unknown): boolean => Object.keys(getPromptPatch(metadata)).length > 0;

/** Seed templates from merged metadata prompts: these are the words the model received. */
export const getMetadataPrompts = (metadata: unknown): { negativePrompt: string; positivePrompt: string } => ({
  negativePrompt: getNullableString(metadata, 'negative_prompt') ?? '',
  positivePrompt: getString(metadata, 'positive_prompt') ?? '',
});

const hasMetadataSize = (metadata: unknown, model: GenerateModelConfig): boolean =>
  Object.keys(getMetadataSize(metadata, model)).length > 0;

const hasGenerationSettings = (metadata: unknown): boolean =>
  getSteps(metadata) !== null ||
  getCfgScale(metadata) !== null ||
  getCfgRescaleMultiplier(metadata) !== null ||
  getScheduler(metadata) !== null ||
  getBoolean(metadata, 'seamless_x') !== null ||
  getBoolean(metadata, 'seamless_y') !== null;

/** Gate Krea-2 rebalance on model base and validate metadata before forwarding weights verbatim to the node parser. */
const getMetadataKrea2Rebalance = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<GenerateWidgetValues> => {
  if (model?.base !== 'krea-2') {
    return {};
  }

  const enabled = getBoolean(metadata, 'krea2_rebalance_enabled');
  const multiplier = getNumber(metadata, 'krea2_rebalance_multiplier');
  const weights = getString(metadata, 'krea2_rebalance_weights');

  return {
    ...(enabled === null ? {} : { krea2RebalanceEnabled: enabled }),
    ...(multiplier === null ? {} : { krea2RebalanceMultiplier: multiplier }),
    ...(weights !== null && isValidKrea2RebalanceWeights(weights) ? { krea2RebalanceWeights: weights } : {}),
  };
};

const hasKrea2Rebalance = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataKrea2Rebalance(metadata, model)).length > 0;

type RecalledComponentSetting = keyof Pick<
  GenerateWidgetValues,
  | 'clipEmbedModel'
  | 'componentSourceModel'
  | 'gemma2EncoderModel'
  | 'ideogram4UnconditionalModel'
  | 'mistralEncoderModel'
  | 'pidDecoderModel'
  | 'qwen3EncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwenVLEncoderModel'
  | 't5EncoderModel'
  | 'wanLowNoiseModel'
  | 'wanT5EncoderModel'
>;

interface RecalledComponent {
  metadataKey: string;
  setting: RecalledComponentSetting;
  /** The setting holds a main model (a component source or second transformer), not an encoder. */
  isMainModel?: true;
}

/**
 * The component keys the generation graphs record; a key missing here is silently dropped by Remix. The first
 * recorded key wins for a shared setting.
 */
const RECALLED_COMPONENTS: readonly RecalledComponent[] = [
  { isMainModel: true, metadataKey: 'qwen3_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'qwen_image_component_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'wan_component_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'ideogram4_unconditional_model', setting: 'ideogram4UnconditionalModel' },
  { isMainModel: true, metadataKey: 'wan_transformer_low_noise', setting: 'wanLowNoiseModel' },
  { metadataKey: 'clip_embed_model', setting: 'clipEmbedModel' },
  { metadataKey: 'gemma2_encoder', setting: 'gemma2EncoderModel' },
  { metadataKey: 'mistral_encoder', setting: 'mistralEncoderModel' },
  { metadataKey: 'pid_decoder', setting: 'pidDecoderModel' },
  { metadataKey: 'qwen3_encoder', setting: 'qwen3EncoderModel' },
  { metadataKey: 'qwen3_vl_encoder', setting: 'qwen3VLEncoderModel' },
  { metadataKey: 'qwen_image_qwen_vl_encoder', setting: 'qwenVLEncoderModel' },
  { metadataKey: 't5_encoder', setting: 't5EncoderModel' },
  { metadataKey: 'wan_t5_encoder_model', setting: 'wanT5EncoderModel' },
];

const getRecalledComponent = (
  metadata: unknown,
  component: RecalledComponent,
  models: readonly ComponentModelConfig[]
): ComponentModelConfig | null | undefined =>
  component.isMainModel
    ? getMetadataMainModel(metadata, component.metadataKey, models)
    : getMetadataComponent(metadata, component.metadataKey, models);

/** Recorded components that resolve to installed models; `null` clears a slot the image left empty. */
const getComponentPatch = (
  metadata: unknown,
  models: readonly ComponentModelConfig[]
): Partial<Record<RecalledComponentSetting, ComponentModelConfig | null>> => {
  const patch: Partial<Record<RecalledComponentSetting, ComponentModelConfig | null>> = {};

  for (const component of RECALLED_COMPONENTS) {
    const recalled = getRecalledComponent(metadata, component, models);

    if (recalled !== undefined && !(component.setting in patch)) {
      patch[component.setting] = recalled;
    }
  }

  return patch;
};

const hasComponentModels = (metadata: unknown, models: readonly ComponentModelConfig[]): boolean =>
  Object.keys(getComponentPatch(metadata, models)).length > 0;

export const getMetadataReferenceImages = (metadata: unknown) => {
  if (!isRecord(metadata)) {
    return [];
  }

  const direct = normalizeReferenceImages(metadata.ref_images).filter(
    (referenceImage) => referenceImage.config.image !== null
  );
  if (direct.length > 0) {
    return direct;
  }

  const canvasMetadata = getRecord(metadata, 'canvas_v2_metadata');
  const referenceImages = getRecord(canvasMetadata, 'referenceImages');
  const legacyEntities = referenceImages?.entities;
  if (!Array.isArray(legacyEntities)) {
    return [];
  }

  return normalizeReferenceImages(
    legacyEntities.map((entry) =>
      isRecord(entry) && isRecord(entry.ipAdapter)
        ? { config: entry.ipAdapter, id: entry.id, isEnabled: entry.isEnabled }
        : entry
    )
  ).filter((referenceImage) => referenceImage.config.image !== null);
};

export const withDimensions = (
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

export const getSupportedClipSkip = (metadata: unknown, model: GenerateModelConfig): number | null => {
  const clipSkip = getClipSkip(metadata);
  const clipSkipMax = getClipSkipMax(model);

  return clipSkip !== null && clipSkipMax !== null ? Math.min(clipSkipMax, clipSkip) : null;
};

/**
 * Require served policy for Recall All, Remix, and CLIP skip before persisting model defaults or compatibility
 * choices. Prompts/seed are intrinsic; dimension recall gates its own grid.
 */
export const isImageRecallKindAvailable = (kind: ImageRecallKind): boolean =>
  kind === 'prompts' || kind === 'seed' || kind === 'dimensions' || hasArchitectureCapabilities();

export const getImageRecallCapabilities = ({
  currentValues,
  image,
  metadata,
  models,
  supportedModels,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  metadata: unknown;
  supportedModels: GenerateModelConfig[];
  models: ComponentModelConfig[];
  vaeModels: VaeModelConfig[];
}): ImageRecallCapabilities => {
  const supportedMetadataModel = getSupportedMetadataModel(metadata, supportedModels);
  const clipSkipModel = supportedMetadataModel ?? currentValues.model;
  const hasVae = getMetadataVae(metadata, clipSkipModel, vaeModels) !== undefined;
  const hasClipSkip = getSupportedClipSkip(metadata, clipSkipModel) !== null;
  const hasHiDiffusion = Object.keys(getHiDiffusionPatch(metadata, clipSkipModel)).length > 0;
  const hasSeed = getSeed(metadata) !== null;
  const hasPrompts = hasPrompt(metadata);
  const hasSize = hasMetadataSize(metadata, currentValues.model);
  const hasSettings = hasGenerationSettings(metadata);
  const hasComponents = hasComponentModels(metadata, models);
  const hasReferenceImages = getMetadataReferenceImages(metadata).length > 0;
  const hasModel = supportedMetadataModel !== null;
  const hasRebalance = hasKrea2Rebalance(metadata, clipSkipModel);
  const hasAnyMetadata =
    hasModel ||
    hasVae ||
    hasPrompts ||
    hasSeed ||
    hasSize ||
    hasSettings ||
    hasHiDiffusion ||
    hasClipSkip ||
    hasComponents ||
    hasReferenceImages ||
    hasRebalance;
  const hasNonSeedMetadata =
    hasModel ||
    hasVae ||
    hasPrompts ||
    hasSize ||
    hasSettings ||
    hasHiDiffusion ||
    hasClipSkip ||
    hasComponents ||
    hasReferenceImages ||
    hasRebalance;

  return {
    all: hasAnyMetadata && isImageRecallKindAvailable('all'),
    clipSkip: getSupportedClipSkip(metadata, currentValues.model) !== null && isImageRecallKindAvailable('clipSkip'),
    dimensions: getImageSize(image, currentValues.model) !== null,
    prompts: hasPrompts,
    remix: hasNonSeedMetadata && isImageRecallKindAvailable('remix'),
    seed: hasSeed,
    workflow: image.hasWorkflow === true,
  };
};

export const buildImageRecallSettings = ({
  currentValues,
  image,
  kind,
  metadata,
  supportedModels,
  models,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  kind: ImageRecallKind;
  metadata: unknown;
  supportedModels: GenerateModelConfig[];
  models: ComponentModelConfig[];
  vaeModels: VaeModelConfig[];
}): ImageRecallResult | null => {
  if (!isImageRecallKindAvailable(kind)) {
    return null;
  }

  const fields: RecalledField[] = [];
  let values: GenerateWidgetValues = cloneGenerateWidgetValues(currentValues);

  if (kind === 'all' || kind === 'remix') {
    const model = getSupportedMetadataModel(metadata, supportedModels);

    if (model) {
      const valuesWithModelDefaults = getSettingsWithModelDefaults(values, model);
      values = {
        ...valuesWithModelDefaults,
        model,
        ...(hasModelDefaultVae(model) ? { vae: getModelDefaultVae(model, vaeModels) } : {}),
      };
      fields.push('model');
    }

    const componentPatch = getComponentPatch(metadata, models);
    const recordedSettings = Object.keys(componentPatch) as RecalledComponentSetting[];
    const effectiveModel = values.model;
    // A recorded component that does not fit the effective model is skipped, so the current pick stays.
    const fittingSettings = recordedSettings.filter((setting) => {
      const component = componentPatch[setting];

      return !component || isComponentCompatibleWithModel(effectiveModel, values, setting, component);
    });

    values = { ...values, ...Object.fromEntries(fittingSettings.map((setting) => [setting, componentPatch[setting]])) };

    const rebalancePatch = getMetadataKrea2Rebalance(metadata, values.model);

    if (Object.keys(rebalancePatch).length > 0) {
      values = { ...values, ...rebalancePatch };
      fields.push('krea2Rebalance');
    }

    const vae = getMetadataVae(metadata, values.model, vaeModels);

    if (vae !== undefined) {
      values = { ...values, vae };
      fields.push('vae');
    }

    if (model || fittingSettings.length > 0) {
      // Same transition as picking the model: clears picks left over from the previous model and fills an
      // automatic FLUX.2 component source.
      const { settings } = getGenerateModelSelectionResult({ currentValues: values, model: values.model, models });
      values = { ...values, ...settings };
    }

    // Judged after the transition, which can complete or undo a recorded pick.
    if (recordedSettings.some((setting) => values[setting]?.key !== currentValues[setting]?.key)) {
      fields.push('components');
    }

    // Recalled reference images must fit the effective model — when the
    // metadata model was not recalled (e.g. uninstalled), `values.model` is
    // still the current one and incompatible configs are re-targeted or dropped.
    const referenceImages = getCompatibleReferenceImages(getMetadataReferenceImages(metadata), values.model, models);

    if (referenceImages.length > 0) {
      values = { ...values, referenceImages };
      fields.push('referenceImages');
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
      values = { ...values, seed, seedMode: 'fixed' };
      fields.push('seed');
    }
  }

  if (kind === 'dimensions') {
    const size = getImageSize(image, values.model);

    if (size) {
      values = withDimensions(values, size);
      fields.push('size');
    }
  } else if (kind === 'all' || kind === 'remix') {
    const size = getMetadataSize(metadata, values.model);

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

    const hiDiffusionPatch = getHiDiffusionPatch(metadata, values.model);

    if (Object.keys(hiDiffusionPatch).length > 0) {
      values = { ...values, ...hiDiffusionPatch };
      fields.push('hiDiffusion');
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

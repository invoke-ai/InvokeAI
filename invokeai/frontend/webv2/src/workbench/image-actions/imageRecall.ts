import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { TFunction } from 'i18next';

import {
  clampAnimaInpaintAdapterWeight,
  clampDimension,
  clampPidSteps,
  cloneGenerateWidgetValues,
  isIdeogram4SamplerPreset,
  deriveAspectRatioId,
  getCompatibleReferenceImages,
  getComponentSlotsSuppliedByRecord,
  getDimensionGrid,
  getExternalProviderOptionKeys,
  getExternalResolutionPresets,
  getGenerateModelSelectionResult,
  getGenerationUiPolicy,
  isComponentCompatibleWithModel,
  getSettingsWithModelDefaults,
  isKnownScheduler,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeCompatibleWithGenerateModel,
  isGeminiThinkingLevel,
  isOpenAIImageBackground,
  isOpenAIImageQuality,
  isOpenAIInputFidelity,
  isPidMode,
  GEMINI_TEMPERATURE_MAX,
  GEMINI_TEMPERATURE_MIN,
  isRefinerModel,
  isFluxDypePreset,
  isQwenImageEncoderQuantization,
  isValidKrea2RebalanceWeights,
  isValidMossImageSkipLayers,
  MAX_FLUX_DYPE_EXPONENT,
  MAX_FLUX_DYPE_SCALE,
  MAX_HIDIFFUSION_RATIO,
  MAX_KREA2_SEED_VARIANCE_STRENGTH,
  MAX_REFINER_AESTHETIC_SCORE,
  MAX_REFINER_START,
  MAX_TIMESTEP_SHIFT,
  MAX_ZIMAGE_SEED_VARIANCE_STRENGTH,
  MIN_HIDIFFUSION_T1_RATIO,
  MIN_REFINER_AESTHETIC_SCORE,
  MIN_REFINER_CFG_SCALE,
  MIN_REFINER_START,
  MIN_REFINER_STEPS,
  MIN_TIMESTEP_SHIFT,
  MIN_ZIMAGE_SEED_VARIANCE_RANDOMIZE_PERCENT,
  normalizeMossImageSkipLayers,
  normalizeReferenceImages,
  takesReferenceImages,
  getRecallDefaultVae,
  resolveModelDefaultVae,
} from '@features/generation/settings';
import { hasArchitectureCapabilities } from '@features/models/architectures';
import { SEED_MAX } from '@platform/core/seed';

import {
  mergeOpposingWanLoraTargets,
  readMetadataLoraTarget,
  resolveRecordedModel,
  toRecordedModelRef,
  type RecordedModelPolicy,
} from './recordedModel';

const hasCanvasRecall = (metadata: unknown): boolean =>
  isRecord(metadata) &&
  ('canvas_v3_metadata' in metadata ||
    'canvas_v2_metadata' in metadata ||
    'control_layers' in metadata ||
    'controlnets' in metadata ||
    't2iAdapters' in metadata ||
    typeof (metadata.strength ?? metadata.denoise_strength) === 'number');

export type ImageRecallKind = 'all' | 'remix' | 'prompts' | 'seed' | 'dimensions' | 'clipSkip';

export interface ImageRecallCapabilities {
  all: boolean;
  remix: boolean;
  prompts: boolean;
  seed: boolean;
  dimensions: boolean;
  clipSkip: boolean;
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
  | 'loras'
  | 'referenceImages'
  | 'krea2Rebalance'
  | 'externalProvider'
  | 'refiner'
  | 'krea2SeedVariance'
  | 'mossImageSkipLayers'
  | 'modelTuning'
  | 'animaInpaintAdapterWeight';

export type ImageRecallSkipReason =
  | 'ambiguous'
  | 'duplicate'
  | 'incompatible'
  | 'invalid'
  | 'modelUnavailable'
  | 'unresolved';

/** `name` is the best label the record or catalog offers. */
export interface ImageRecallSkip {
  name: string | null;
  reason: ImageRecallSkipReason;
  subject: 'concept' | 'refiner';
}

export interface ImageRecallResult {
  fields: RecalledField[];
  skipped: ImageRecallSkip[];
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

const getNumberOrFirstScheduleValue = (metadata: unknown, key: string): number | null => {
  const scalar = getNumber(metadata, key);

  if (scalar !== null) {
    return scalar;
  }

  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

  return Array.isArray(value) &&
    value.length > 0 &&
    value.every((entry) => typeof entry === 'number' && Number.isFinite(entry))
    ? (value[0] as number)
    : null;
};

/** Distilled-guidance families record their slider as `guidance`, and older records carry no `cfg_scale`; a scheduled
 * `cfg_scale` list recalls its first step. */
export const getCfgScale = (metadata: unknown, model?: GenerateModelConfig): number | null => {
  const { guidanceLabel, guidanceMin } = getGenerationUiPolicy(model);
  const cfgScale =
    guidanceLabel === 'Guidance'
      ? (getNumberOrFirstScheduleValue(metadata, 'guidance') ?? getNumberOrFirstScheduleValue(metadata, 'cfg_scale'))
      : getNumberOrFirstScheduleValue(metadata, 'cfg_scale');

  return cfgScale !== null && cfgScale >= guidanceMin ? cfgScale : null;
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

const getClipSkipMax = (model: GenerateModelConfig): number | null => getGenerationUiPolicy(model).clipSkipMax;

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
  if (!getGenerationUiPolicy(model).hiDiffusionVisible || !isRecord(metadata)) {
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
    ...(getBoolean(metadata, 'hidiffusion_auto_ratios') === true ||
    (getBoolean(metadata, 'hidiffusion_auto_ratios') === null && t1Ratio === null && t2Ratio === null)
      ? getBoolean(metadata, 'hidiffusion_auto_ratios') === true ||
        enabled !== null ||
        metadata.hidiffusion_t1_ratio === null ||
        metadata.hidiffusion_t2_ratio === null
        ? { hiDiffusionT1Ratio: null, hiDiffusionT2Ratio: null }
        : {}
      : {
          hiDiffusionT1Ratio:
            t1Ratio !== null && t1Ratio >= MIN_HIDIFFUSION_T1_RATIO && t1Ratio <= MAX_HIDIFFUSION_RATIO ? t1Ratio : 0,
          hiDiffusionT2Ratio: t2Ratio !== null && t2Ratio >= 0 && t2Ratio <= MAX_HIDIFFUSION_RATIO ? t2Ratio : 0,
        }),
  };
};

const getImageSize = (
  image: GalleryImage,
  model: GenerateModelConfig
): Pick<GenerateWidgetValues, 'height' | 'width'> | null => {
  // Decline dimension recall without architecture policy; persisting fallback grid 8 can leave sizes the model rejects.
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

/** Image recall restores only the exact weights that ran, and only a model the record names unambiguously. */
const IMAGE_RECORDED_MODEL_POLICY: RecordedModelPolicy = { ambiguous: 'refuse', hashMiss: 'refuse' };

const resolveMetadataModel = <T extends ComponentModelConfig>(value: unknown, models: readonly T[]): T | null => {
  const ref = toRecordedModelRef(value);
  const resolved = ref ? resolveRecordedModel(ref, models, IMAGE_RECORDED_MODEL_POLICY) : null;

  return resolved === 'ambiguous' ? null : resolved;
};

const getSupportedMetadataModel = (
  metadata: unknown,
  supportedModels: GenerateModelConfig[]
): GenerateModelConfig | null => resolveMetadataModel(getRecord(metadata, 'model'), supportedModels);

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

  const model = resolveMetadataModel(metadata[key], models);

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

/** The key a record names its VAE under, whether or not the model resolves; older records use a base's own key. */
const getRecordedVaeKey = (metadata: unknown, model: GenerateModelConfig): string | null => {
  if (!isRecord(metadata)) {
    return null;
  }
  const legacyKey = ({ 'qwen-image': 'qwen_image_vae', wan: 'wan_vae_model' } as Record<string, string>)[model.base];

  return 'vae' in metadata ? 'vae' : legacyKey && legacyKey in metadata ? legacyKey : null;
};

const getMetadataVae = (
  metadata: unknown,
  model: GenerateModelConfig,
  vaeModels: VaeModelConfig[]
): VaeModelConfig | null | undefined => {
  const key = getRecordedVaeKey(metadata, model);

  if (!key || !isRecord(metadata)) {
    return undefined;
  }

  if (metadata[key] === null) {
    return null;
  }

  const vae = resolveMetadataModel(metadata[key], vaeModels);

  return vae && isVaeCompatibleWithGenerateModel(model, vae) ? vae : undefined;
};

const getMetadataVaePrecision = (metadata: unknown): 'fp16' | 'fp32' | null => {
  const precision = getString(metadata, 'vae_precision');

  return precision === 'fp16' || precision === 'fp32' ? precision : null;
};

/** The decode settings a record restores onto `model`: only those its panel shows. Recall and its actions both ask. */
const getMetadataVaeSettings = (
  metadata: unknown,
  model: GenerateModelConfig
): Partial<Pick<GenerateWidgetValues, 'colorCompensation' | 'vaePrecision'>> => {
  const policy = getGenerationUiPolicy(model);
  const vaePrecision = getMetadataVaePrecision(metadata);
  const colorCompensation = getBoolean(metadata, 'color_compensation');

  return {
    ...(vaePrecision !== null && policy.vaePrecisionVisible ? { vaePrecision } : {}),
    ...(colorCompensation !== null && policy.colorCompensationVisible ? { colorCompensation } : {}),
  };
};

const getPromptPatch = (
  metadata: unknown
): Partial<
  Pick<GenerateWidgetValues, 'negativePrompt' | 'positivePrompt' | 'promptTemplate' | 'promptTemplateViewMode'>
> => {
  const positivePrompt = getString(metadata, 'positive_prompt');
  const negativePrompt = getNullableString(metadata, 'negative_prompt');
  // The recorded prompt already has any template baked in; clearing the active template stops a second wrap.
  const clearTemplate = positivePrompt !== null || negativePrompt !== undefined;

  return {
    ...(positivePrompt !== null ? { positivePrompt } : {}),
    ...(negativePrompt !== undefined ? { negativePrompt: negativePrompt ?? '' } : {}),
    ...(clearTemplate ? { promptTemplate: null, promptTemplateViewMode: false } : {}),
  };
};

const hasPrompt = (metadata: unknown): boolean => Object.keys(getPromptPatch(metadata)).length > 0;

export const getMetadataPrompts = (metadata: unknown): { negativePrompt: string; positivePrompt: string } => ({
  negativePrompt: getNullableString(metadata, 'negative_prompt') ?? '',
  positivePrompt: getString(metadata, 'positive_prompt') ?? '',
});

const hasMetadataSize = (metadata: unknown, model: GenerateModelConfig): boolean =>
  Object.keys(getMetadataSize(metadata, model)).length > 0;

const hasGenerationSettings = (metadata: unknown, model: GenerateModelConfig): boolean =>
  getSteps(metadata) !== null ||
  getCfgScale(metadata, model) !== null ||
  getCfgRescaleMultiplier(metadata) !== null ||
  getScheduler(metadata) !== null ||
  getBoolean(metadata, 'seamless_x') !== null ||
  getBoolean(metadata, 'seamless_y') !== null;

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

/** Bounds match the enhancer's own controls; a value outside them keeps the current setting. */
const getMetadataKrea2SeedVariance = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<GenerateWidgetValues> => {
  if (model?.base !== 'krea-2') {
    return {};
  }

  const enabled = getBoolean(metadata, 'krea2_seed_variance_enabled');
  const strength = getNumber(metadata, 'krea2_seed_variance_strength');
  const randomizePercent = getNumber(metadata, 'krea2_seed_variance_randomize_percent');

  return {
    ...(enabled === null ? {} : { krea2SeedVarianceEnabled: enabled }),
    ...(strength !== null && strength >= 0 && strength <= MAX_KREA2_SEED_VARIANCE_STRENGTH
      ? { krea2SeedVarianceStrength: strength }
      : {}),
    ...(randomizePercent !== null && randomizePercent >= 0 && randomizePercent <= 100
      ? { krea2SeedVarianceRandomizePercent: randomizePercent }
      : {}),
  };
};

const hasKrea2SeedVariance = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataKrea2SeedVariance(metadata, model)).length > 0;

/** An empty list is a recorded choice; a list naming a block MOSS-Image lacks keeps the current setting. */
const getMetadataMossImageSkipLayers = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<GenerateWidgetValues> => {
  if (model?.base !== 'moss-image' || !isRecord(metadata)) {
    return {};
  }

  const layers = normalizeMossImageSkipLayers(metadata.moss_image_uncond_skip_layers);

  return layers !== null && isValidMossImageSkipLayers(layers) ? { mossImageUncondSkipLayers: layers } : {};
};

const hasMossImageSkipLayers = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataMossImageSkipLayers(metadata, model)).length > 0;

/** On a stamped generation record a missing list or switch means none or the default; canvas saves record neither. */
const isGenerationRecord = (metadata: unknown): boolean => getString(metadata, 'generation_mode') !== null;

/**
 * A graph run's record, which names the components its loaders used: a generation record, or a Refine upscale, which
 * writes no `generation_mode` but always `creativity` (a Model only upscale copies its source's keys instead). A Canvas
 * save records the panel's model and VAE but no components.
 */
const recordsGraphRun = (metadata: unknown): boolean =>
  isGenerationRecord(metadata) || getNumber(metadata, 'creativity') !== null;

const getBoundedNumber = (metadata: unknown, key: string, min: number, max: number): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && value >= min && value <= max ? value : null;
};

/** `'auto'` (webv2) or `null` (legacy) when the node derived it, a number if overridden; else undefined. */
const getRecordedTimestepShift = (metadata: unknown, key: string): number | null | undefined => {
  if (!isRecord(metadata) || !(key in metadata)) {
    return undefined;
  }

  if (metadata[key] === null || metadata[key] === 'auto') {
    return null;
  }

  return getBoundedNumber(metadata, key, MIN_TIMESTEP_SHIFT, MAX_TIMESTEP_SHIFT) ?? undefined;
};

type RecordedOrDefault = <T>(key: string, recorded: T | undefined, fallback: T) => T | undefined;

const getFamilyTuning = (
  metadata: Record<string, unknown>,
  model: GenerateModelConfig,
  recordedOrDefault: RecordedOrDefault
): Partial<GenerateWidgetValues> => {
  if (model.base === 'flux') {
    const preset = recordedOrDefault(
      'dype_preset',
      isFluxDypePreset(metadata.dype_preset) ? metadata.dype_preset : undefined,
      'off'
    );
    const scale = getBoundedNumber(metadata, 'dype_scale', 0, MAX_FLUX_DYPE_SCALE);
    const exponent = getBoundedNumber(metadata, 'dype_exponent', 0, MAX_FLUX_DYPE_EXPONENT);

    return {
      ...(preset === undefined ? {} : { fluxDypePreset: preset }),
      ...(scale === null ? {} : { fluxDypeScale: scale }),
      ...(exponent === null ? {} : { fluxDypeExponent: exponent }),
    };
  }

  if (model.base === 'z-image') {
    const shift = recordedOrDefault('z_image_shift', getRecordedTimestepShift(metadata, 'z_image_shift'), null);
    const enabled = recordedOrDefault(
      'z_image_seed_variance_enabled',
      getBoolean(metadata, 'z_image_seed_variance_enabled') ?? undefined,
      false
    );
    const strength = getBoundedNumber(metadata, 'z_image_seed_variance_strength', 0, MAX_ZIMAGE_SEED_VARIANCE_STRENGTH);
    const randomizePercent = getBoundedNumber(
      metadata,
      'z_image_seed_variance_randomize_percent',
      MIN_ZIMAGE_SEED_VARIANCE_RANDOMIZE_PERCENT,
      100
    );

    return {
      ...(shift === undefined ? {} : { zImageShift: shift }),
      ...(enabled === undefined ? {} : { zImageSeedVarianceEnabled: enabled }),
      ...(strength === null ? {} : { zImageSeedVarianceStrength: strength }),
      ...(randomizePercent === null ? {} : { zImageSeedVarianceRandomizePercent: randomizePercent }),
    };
  }

  if (model.base === 'qwen-image') {
    const shift = recordedOrDefault('qwen_image_shift', getRecordedTimestepShift(metadata, 'qwen_image_shift'), null);
    const quantization = recordedOrDefault(
      'qwen_image_quantization',
      isQwenImageEncoderQuantization(metadata.qwen_image_quantization) ? metadata.qwen_image_quantization : undefined,
      'none'
    );

    return {
      ...(shift === undefined ? {} : { qwenImageShift: shift }),
      ...(quantization === undefined ? {} : { qwenImageEncoderQuantization: quantization }),
    };
  }

  if (model.base === 'wan') {
    const raw = metadata.wan_guidance_scale_low_noise;
    const parsed = raw === null || (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) ? raw : undefined;
    const guidance = recordedOrDefault('wan_guidance_scale_low_noise', parsed, null);
    return guidance === undefined ? {} : { wanGuidanceScaleLowNoise: guidance };
  }
  if (model.base === 'ideogram-4') {
    const preset = isIdeogram4SamplerPreset(metadata.ideogram4_sampler_preset)
      ? metadata.ideogram4_sampler_preset
      : undefined;
    const steps = getBoundedNumber(metadata, 'ideogram4_steps', 2, 100);
    const overrides = {
      ideogram4Steps: recordedOrDefault(
        'ideogram4_steps',
        metadata.ideogram4_steps === null ? null : steps !== null && Number.isInteger(steps) ? steps : undefined,
        null
      ),
      ideogram4GuidanceScale: recordedOrDefault(
        'ideogram4_guidance_scale',
        metadata.ideogram4_guidance_scale === null
          ? null
          : (getBoundedNumber(metadata, 'ideogram4_guidance_scale', 1, 20) ?? undefined),
        null
      ),
      ideogram4Mu: recordedOrDefault(
        'ideogram4_mu',
        metadata.ideogram4_mu === null ? null : (getBoundedNumber(metadata, 'ideogram4_mu', -4, 4) ?? undefined),
        null
      ),
    };
    const palette = metadata.ideogram4_color_palette;
    return {
      ...(preset === undefined ? {} : { ideogram4SamplerPreset: preset }),
      ...Object.fromEntries(Object.entries(overrides).filter(([, value]) => value !== undefined)),
      ...(Array.isArray(palette) && palette.length <= 256 && palette.every((term) => typeof term === 'string')
        ? { ideogram4ColorPalette: palette }
        : {}),
    };
  }

  if (model.base === 'krea-2') {
    // Recorded only when the switch was sent; a record without it leaves the user's setting.
    const referenceKvCache = getBoolean(metadata, 'krea2_reference_kv_cache');

    return referenceKvCache === null ? {} : { krea2ReferenceKvCache: referenceKvCache };
  }

  return {};
};

/** Graphs record PiD keys only while it decodes, so a same-family record without the mode used the VAE. */
const getPidTuning = (
  metadata: Record<string, unknown>,
  model: GenerateModelConfig,
  recordedOrDefault: RecordedOrDefault
): Partial<Pick<GenerateWidgetValues, 'pidMode' | 'pidSteps'>> => {
  if (!getGenerationUiPolicy(model).pidVisible) {
    return {};
  }

  const mode = recordedOrDefault('pid_mode', isPidMode(metadata.pid_mode) ? metadata.pid_mode : undefined, 'off');
  const steps = getNumber(metadata, 'pid_steps');

  return {
    ...(mode === undefined ? {} : { pidMode: mode }),
    ...(steps === null ? {} : { pidSteps: clampPidSteps(steps) }),
  };
};

/**
 * Gated on the model recall settles on. A same-family generation record that leaves a switch out ran the node's
 * default, so it is restored; values that only matter while a switch is on stay when missing.
 */
const getMetadataModelTuning = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<GenerateWidgetValues> => {
  if (!model || model.type !== 'main' || !isRecord(metadata)) {
    return {};
  }

  const recordsFamily = isGenerationRecord(metadata) && getRecord(metadata, 'model')?.base === model.base;
  const recordedOrDefault: RecordedOrDefault = (key, recorded, fallback) =>
    recorded !== undefined ? recorded : recordsFamily && !(key in metadata) ? fallback : undefined;

  return {
    ...getFamilyTuning(metadata, model, recordedOrDefault),
    ...getPidTuning(metadata, model, recordedOrDefault),
  };
};

const hasModelTuning = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataModelTuning(metadata, model)).length > 0;

const ANIMA_INPAINT_ADAPTER_MODES: readonly string[] = ['anima_inpaint', 'anima_outpaint'];

/** Provider nodes record these keys, so a value applies only to the same provider. */
const getMetadataExternalProvider = (
  metadata: unknown,
  model: GenerateModelConfig | undefined,
  isCanvasFrame: boolean
): Partial<GenerateWidgetValues> => {
  const patch: Partial<GenerateWidgetValues> = {};
  const imageSize = getString(metadata, 'image_size');

  if (imageSize !== null) {
    const width = getNumber(metadata, 'width');
    const height = getNumber(metadata, 'height');
    const ratio = getString(metadata, 'aspect_ratio');
    const named = getExternalResolutionPresets(model).filter(
      (preset) => preset.image_size === imageSize && (ratio === null || preset.aspect_ratio === ratio)
    );
    const exact = named.filter(
      (preset) => (width === null || preset.width === width) && (height === null || preset.height === height)
    );
    // A Canvas output records its frame, not the provider size: pick the named size at the ratio nearest the frame.
    const ratioDistance = (preset: { width: number; height: number }) =>
      width !== null && height !== null && width > 0 && height > 0
        ? Math.abs(Math.log(preset.width / preset.height / (width / height)))
        : Number.NaN;
    const nearest =
      !isCanvasFrame && exact.length === 0 && named.length > 0 && !Number.isNaN(ratioDistance(named[0]!))
        ? named.reduce((best, preset) => (ratioDistance(preset) < ratioDistance(best) ? preset : best))
        : undefined;
    const preset = exact.length === 1 ? exact[0] : nearest;

    // A name alone can describe many aspect ratios; restore only a size the record pins down.
    if (preset) {
      patch.externalImageSize = preset.image_size;
      patch.width = preset.width;
      patch.height = preset.height;
      patch.aspectRatioId = deriveAspectRatioId(preset.width, preset.height);
      patch.aspectRatioValue = preset.width / preset.height;
    }
  }

  // Provider outputs omit fields left at default; the provider stamp separates those from unrelated metadata.
  const usesProviderDefaults = getString(metadata, 'external_provider') === model?.provider_id;
  const isDefault = (key: string): boolean =>
    isRecord(metadata) && (metadata[key] === null || (usesProviderDefaults && !(key in metadata)));

  for (const key of getExternalProviderOptionKeys(model)) {
    switch (key) {
      case 'openaiQuality': {
        const value = getString(metadata, 'openai_quality');

        if (isOpenAIImageQuality(value)) {
          patch.openaiQuality = value;
        }
        break;
      }
      case 'openaiBackground': {
        const value = getString(metadata, 'openai_background');

        if (isOpenAIImageBackground(value)) {
          patch.openaiBackground = value;
        }
        break;
      }
      case 'openaiInputFidelity': {
        const value = getString(metadata, 'openai_input_fidelity');

        if (isOpenAIInputFidelity(value)) {
          patch.openaiInputFidelity = value;
        } else if (isDefault('openai_input_fidelity')) {
          patch.openaiInputFidelity = null;
        }
        break;
      }
      case 'geminiTemperature': {
        const value = getNumber(metadata, 'gemini_temperature');

        if (value !== null && value >= GEMINI_TEMPERATURE_MIN && value <= GEMINI_TEMPERATURE_MAX) {
          patch.geminiTemperature = value;
        } else if (isDefault('gemini_temperature')) {
          patch.geminiTemperature = null;
        }
        break;
      }
      case 'geminiThinkingLevel': {
        const value = getString(metadata, 'gemini_thinking_level');

        if (isGeminiThinkingLevel(value)) {
          patch.geminiThinkingLevel = value;
        } else if (isDefault('gemini_thinking_level')) {
          patch.geminiThinkingLevel = null;
        }
        break;
      }
      case 'seedreamWatermark': {
        const value = getBoolean(metadata, 'seedream_watermark');

        if (value !== null) {
          patch.seedreamWatermark = value;
        }
        break;
      }
      case 'seedreamOptimizePrompt': {
        const value = getBoolean(metadata, 'seedream_optimize_prompt');

        if (value !== null) {
          patch.seedreamOptimizePrompt = value;
        }
        break;
      }
    }
  }

  return patch;
};

const hasExternalProvider = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataExternalProvider(metadata, model, false)).length > 0;

/**
 * A recorded adapter means it ran. `core_metadata` drops `null` on write, so an inpaint/outpaint record without the key
 * ran without one. An out-of-range weight is clamped: applying it would block every inpaint.
 */
const getMetadataAnimaInpaintAdapter = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<Pick<GenerateWidgetValues, 'animaInpaintAdapterEnabled' | 'animaInpaintAdapterWeight'>> => {
  if (model?.base !== 'anima' || !isRecord(metadata)) {
    return {};
  }

  const weight = getNumber(metadata, 'anima_inpaint_adapter_weight');
  const mode = getString(metadata, 'generation_mode');
  const enabled =
    'anima_inpaint_adapter' in metadata
      ? true
      : mode !== null && ANIMA_INPAINT_ADAPTER_MODES.includes(mode)
        ? false
        : null;

  return {
    ...(enabled === null ? {} : { animaInpaintAdapterEnabled: enabled }),
    ...(weight === null ? {} : { animaInpaintAdapterWeight: clampAnimaInpaintAdapterWeight(weight) }),
  };
};

const hasAnimaInpaintAdapterSettings = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataAnimaInpaintAdapter(metadata, model)).length > 0;

/** A recorded model means the upsampler ran; `core_metadata` drops `null`, so a FLUX.3 record without it did not. */
const getMetadataFlux3PromptUpsampler = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<Pick<GenerateWidgetValues, 'flux3PromptUpsamplerEnabled'>> => {
  if (model?.base !== 'flux3' || !isRecord(metadata)) {
    return {};
  }

  if ('flux3_prompt_upsampler' in metadata) {
    return { flux3PromptUpsamplerEnabled: true };
  }

  return getString(metadata, 'generation_mode')?.startsWith('flux3_') ? { flux3PromptUpsamplerEnabled: false } : {};
};

type RecalledComponentSetting = keyof Pick<
  GenerateWidgetValues,
  | 'animaInpaintAdapterModel'
  | 'clipEmbedModel'
  | 'clipGEmbedModel'
  | 'clipLEmbedModel'
  | 'componentSourceModel'
  | 'flux3PromptUpsamplerModel'
  | 'gemma2EncoderModel'
  | 'ideogram4UnconditionalModel'
  | 'mistralEncoderModel'
  | 'pidDecoderModel'
  | 'qwen3EncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwen3VLVisionTowerModel'
  | 'qwen35EncoderModel'
  | 'qwenVLEncoderModel'
  | 't5EncoderModel'
  | 'wanLowNoiseModel'
  | 'wanT5EncoderModel'
>;

interface RecalledComponent {
  metadataKey: string;
  setting: RecalledComponentSetting;
  isMainModel?: true;
}

/** Remix silently drops a key missing here. The first recorded key wins. */
const RECALLED_COMPONENTS: readonly RecalledComponent[] = [
  { isMainModel: true, metadataKey: 'ideogram4_unconditional_model', setting: 'ideogram4UnconditionalModel' },
  { isMainModel: true, metadataKey: 'wan_transformer_low_noise', setting: 'wanLowNoiseModel' },
  { metadataKey: 'anima_inpaint_adapter', setting: 'animaInpaintAdapterModel' },
  { metadataKey: 'clip_embed_model', setting: 'clipEmbedModel' },
  { metadataKey: 'clip_g_model', setting: 'clipGEmbedModel' },
  { metadataKey: 'clip_l_model', setting: 'clipLEmbedModel' },
  { metadataKey: 'flux3_prompt_upsampler', setting: 'flux3PromptUpsamplerModel' },
  { metadataKey: 'gemma2_encoder', setting: 'gemma2EncoderModel' },
  { metadataKey: 'mistral_encoder', setting: 'mistralEncoderModel' },
  { metadataKey: 'pid_decoder', setting: 'pidDecoderModel' },
  { metadataKey: 'qwen3_encoder', setting: 'qwen3EncoderModel' },
  { metadataKey: 'qwen3_vl_encoder', setting: 'qwen3VLEncoderModel' },
  { metadataKey: 'qwen3_vl_vision_tower', setting: 'qwen3VLVisionTowerModel' },
  { metadataKey: 'qwen3_5_encoder', setting: 'qwen35EncoderModel' },
  { metadataKey: 'qwen_image_qwen_vl_encoder', setting: 'qwenVLEncoderModel' },
  { metadataKey: 't5_encoder', setting: 't5EncoderModel' },
  { metadataKey: 'wan_t5_encoder_model', setting: 'wanT5EncoderModel' },
  // After the encoders: the model selection keeps a stored FLUX.2 source only beside the encoder it leaves to a
  // standalone pick, and each entry is judged beside the ones before it.
  { isMainModel: true, metadataKey: 'mistral_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'qwen3_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'qwen_image_component_source', setting: 'componentSourceModel' },
  { isMainModel: true, metadataKey: 'wan_component_source', setting: 'componentSourceModel' },
];

/**
 * Settings no model supplies while empty that graph-run records name whenever the base's loader used them: a record
 * naming the model but not them used none, so a panel pick would change the rerun. FLUX.2 is left to the model
 * transition, which drops a stored source wherever its graph would read none; dev records before 7.0 never named
 * `mistral_source`, so its absence proves nothing there. The Vision Tower is sent only beside a GGUF encoder that
 * takes one, so a record without it ran without.
 */
const SETTINGS_RECORDED_WHEN_LOADED: Partial<Record<string, readonly RecalledComponentSetting[]>> = {
  'krea-2': ['qwen3VLVisionTowerModel'],
  'qwen-image': ['componentSourceModel'],
  'qwen-image-2-1': ['qwen3VLVisionTowerModel'],
  wan: ['componentSourceModel', 'wanLowNoiseModel'],
  'z-image': ['componentSourceModel'],
};

/**
 * Writers that passed a component to their loader without recording it, so its absence from their records proves
 * nothing. A record names the writer through `app_version`; one without it is the current writer's.
 */
const UNRECORDING_WRITERS: readonly {
  base: string;
  /** The first version that recorded the settings. */
  recordedFrom: string;
  settings: readonly RecalledComponentSetting[];
}[] = [
  // v6.10.0rc1 sent `qwen3_source_model` to the loader but never wrote `qwen3_source`.
  { base: 'z-image', recordedFrom: '6.10.0rc2', settings: ['componentSourceModel'] },
  // The legacy frontend passed standalone T5 and CLIP encoders to the loader and wrote none of them; webv2 recorded
  // them from 7.0.0-beta.2.
  { base: 'sd-3', recordedFrom: '7.0.0-beta.2', settings: ['clipGEmbedModel', 'clipLEmbedModel', 't5EncoderModel'] },
];

const PRE_RELEASE_RANK: Record<string, number> = { a: 0, alpha: 0, b: 1, beta: 1, rc: 2 };

/** Orders `major.minor.patch` with an optional `a`/`alpha`/`b`/`beta`/`rc` number; a final release sorts last. */
const parseAppVersion = (version: string | null): number[] | null => {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:[.-]?(alpha|beta|rc|a|b)[.-]?(\d+))?/.exec(version ?? '');

  if (!match) {
    return null;
  }

  const [, major, minor, patch, stage, number] = match;

  return [+major!, +minor!, +patch!, stage ? PRE_RELEASE_RANK[stage]! : 3, +(number ?? 0)];
};

const isWriterBefore = (metadata: unknown, version: string): boolean => {
  const written = parseAppVersion(getString(metadata, 'app_version'));
  const first = parseAppVersion(version)!;
  const differing = written?.findIndex((part, index) => part !== first[index]) ?? -1;

  return differing >= 0 && written![differing]! < first[differing]!;
};

/** Whether a record's writer may have used `setting` of a `base` model without recording it. */
const mayHaveOmittedSetting = (metadata: unknown, base: string, setting: RecalledComponentSetting): boolean =>
  UNRECORDING_WRITERS.some(
    (writer) =>
      writer.base === base && writer.settings.includes(setting) && isWriterBefore(metadata, writer.recordedFrom)
  );

const getRecalledComponent = (
  metadata: unknown,
  component: RecalledComponent,
  models: readonly ComponentModelConfig[]
): ComponentModelConfig | null | undefined =>
  component.isMainModel
    ? getMetadataMainModel(metadata, component.metadataKey, models)
    : getMetadataComponent(metadata, component.metadataKey, models);

/** `null` clears a slot the image left empty. */
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

type RefinerPatch = Partial<
  Pick<
    GenerateWidgetValues,
    | 'refinerModel'
    | 'refinerSteps'
    | 'refinerCfgScale'
    | 'refinerScheduler'
    | 'refinerPositiveAestheticScore'
    | 'refinerNegativeAestheticScore'
    | 'refinerStart'
  >
>;

const getNumberInRange = (metadata: unknown, key: string, min: number, max: number): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && value >= min && value <= max ? value : null;
};

interface RefinerRecall {
  patch: RefinerPatch;
  skipped: ImageRecallSkip | null;
}

const isInstalledRefiner = (model: ComponentModelConfig): model is MainModelConfig =>
  isMainModelConfig(model) && isRefinerModel(model);

/** Records name a refiner only when one ran, so a stamped record without one clears the slot. */
const getRefinerRecall = (
  metadata: unknown,
  model: GenerateModelConfig,
  models: readonly ComponentModelConfig[],
  sourceModel: GenerateModelConfig | null
): RefinerRecall => {
  if (!getGenerationUiPolicy(model).refinerVisible || !isRecord(metadata)) {
    return { patch: {}, skipped: null };
  }

  const patch: RefinerPatch = {};
  const ref = toRecordedModelRef(metadata.refiner_model);

  if (getRecord(metadata, 'model') && sourceModel?.key !== model.key) {
    const recordsRefiner = Object.entries(metadata).some(
      ([key, value]) => key.startsWith('refiner_') && value !== null
    );

    return {
      patch: {},
      skipped: recordsRefiner
        ? { name: ref?.name ?? ref?.key ?? null, reason: 'modelUnavailable', subject: 'refiner' }
        : null,
    };
  }

  if ('refiner_model' in metadata) {
    const refiner = ref
      ? resolveRecordedModel(ref, models.filter(isInstalledRefiner), IMAGE_RECORDED_MODEL_POLICY)
      : null;

    if (metadata.refiner_model === null) {
      patch.refinerModel = null;
    } else if (refiner === null || refiner === 'ambiguous') {
      return {
        patch: {},
        skipped: {
          name: ref?.name ?? ref?.key ?? null,
          reason: !ref ? 'invalid' : refiner === 'ambiguous' ? 'ambiguous' : 'unresolved',
          subject: 'refiner',
        },
      };
    } else {
      patch.refinerModel = refiner;
    }
  } else if (isGenerationRecord(metadata)) {
    patch.refinerModel = null;
  }

  const steps = getInteger(metadata, 'refiner_steps');
  const cfgScale = getNumber(metadata, 'refiner_cfg_scale');
  const scheduler = getString(metadata, 'refiner_scheduler');
  const start = getNumberInRange(metadata, 'refiner_start', MIN_REFINER_START, MAX_REFINER_START);
  const positiveScore = getNumberInRange(
    metadata,
    'refiner_positive_aesthetic_score',
    MIN_REFINER_AESTHETIC_SCORE,
    MAX_REFINER_AESTHETIC_SCORE
  );
  const negativeScore = getNumberInRange(
    metadata,
    'refiner_negative_aesthetic_score',
    MIN_REFINER_AESTHETIC_SCORE,
    MAX_REFINER_AESTHETIC_SCORE
  );

  return {
    patch: {
      ...patch,
      ...(steps !== null && steps >= MIN_REFINER_STEPS ? { refinerSteps: steps } : {}),
      ...(cfgScale !== null && cfgScale >= MIN_REFINER_CFG_SCALE ? { refinerCfgScale: cfgScale } : {}),
      ...(scheduler !== null && isKnownScheduler(scheduler) ? { refinerScheduler: scheduler } : {}),
      ...(start !== null ? { refinerStart: start } : {}),
      ...(positiveScore !== null ? { refinerPositiveAestheticScore: positiveScore } : {}),
      ...(negativeScore !== null ? { refinerNegativeAestheticScore: negativeScore } : {}),
    },
    skipped: null,
  };
};

interface ConceptRecall {
  loras: GenerateLora[] | null;
  skipped: ImageRecallSkip[];
}

/** The metadata lists only LoRAs that ran, so every restored concept is enabled at its recorded weight. */
const getMetadataConcepts = (
  metadata: unknown,
  models: readonly ComponentModelConfig[],
  model: GenerateModelConfig,
  sourceModel: GenerateModelConfig | null
): ConceptRecall => {
  const recorded = isRecord(metadata) ? metadata.loras : undefined;

  // Matching keys alone are not enough: persisted settings can name a model that is no longer installed.
  if (getRecord(metadata, 'model') && sourceModel?.key !== model.key) {
    // Without the model the concepts ran on, the current set stays.
    const entries = Array.isArray(recorded) ? recorded : [];

    return {
      loras: null,
      skipped: entries.map((entry) => ({
        name: toRecordedModelRef(getRecord(entry, 'model') ?? getRecord(entry, 'lora'))?.name ?? null,
        reason: 'modelUnavailable',
        subject: 'concept',
      })),
    };
  }

  if (recorded === undefined || recorded === null) {
    return { loras: isGenerationRecord(metadata) ? [] : null, skipped: [] };
  }

  if (!Array.isArray(recorded)) {
    return { loras: null, skipped: [{ name: null, reason: 'invalid', subject: 'concept' }] };
  }

  const loraModels = models.filter(isLoraModelConfig);
  const loras: GenerateLora[] = [];
  const skipped: ImageRecallSkip[] = [];

  for (const entry of recorded) {
    // Records from before LoRA identifiers were `ModelIdentifierField`s name the LoRA under `lora`.
    const ref = toRecordedModelRef(getRecord(entry, 'model') ?? getRecord(entry, 'lora'));
    const weight = getNumber(entry, 'weight');
    const recordedName = ref?.name ?? null;

    if (!ref || weight === null) {
      skipped.push({ name: recordedName, reason: 'invalid', subject: 'concept' });
      continue;
    }

    const installed = resolveRecordedModel(ref, loraModels, IMAGE_RECORDED_MODEL_POLICY);

    if (installed === null || installed === 'ambiguous') {
      skipped.push({
        name: recordedName,
        reason: installed === 'ambiguous' ? 'ambiguous' : 'unresolved',
        subject: 'concept',
      });
    } else if (!isLoraCompatibleWithModel(installed, model)) {
      skipped.push({ name: installed.name, reason: 'incompatible', subject: 'concept' });
    } else {
      const targetFields = isRecord(entry) ? readMetadataLoraTarget(entry, weight) : {};
      const entryEnabled =
        getString(metadata, 'metadata_source') === 'canvas_save' ? (getBoolean(entry, 'isEnabled') ?? true) : true;
      const existingIndex = loras.findIndex((lora) => lora.model.key === installed.key);

      if (existingIndex !== -1) {
        const merged = mergeOpposingWanLoraTargets(loras[existingIndex]!, {
          target: targetFields.target,
          weight,
        });

        if (merged) {
          loras[existingIndex] = { ...merged, isEnabled: merged.isEnabled || entryEnabled };
        } else {
          skipped.push({ name: installed.name, reason: 'duplicate', subject: 'concept' });
        }
      } else {
        // Kept exactly: graphs and model defaults accept weights beyond the panel's typed range, which bounds only edits.
        loras.push({
          isEnabled: entryEnabled,
          model: installed,
          weight,
          ...targetFields,
        });
      }
    }
  }

  return { loras, skipped };
};

/** Legacy-UI Krea-2 generation graphs used a style reference without recording `ref_images`. */
const LEGACY_UNRECORDED_REFERENCE_BASE = 'krea-2';

const boundReferenceEntries = (entries: unknown[]): unknown[] =>
  entries.length > 10000 ? [...entries.slice(0, 10000), null] : entries;

export const getLegacyCanvasReferenceEntries = (metadata: unknown): unknown[] | null => {
  const references = getRecord(metadata, 'canvas_v2_metadata')?.referenceImages;
  const entries = Array.isArray(references) ? references : isRecord(references) ? references.entities : null;
  return Array.isArray(entries) ? entries : null;
};

const getMetadataReferenceEntries = (metadata: unknown): unknown[] | null => {
  if (!isRecord(metadata)) {
    return null;
  }
  if (
    Array.isArray(metadata.ref_images) &&
    (metadata.ref_images.length === 0 ||
      normalizeReferenceImages(metadata.ref_images.slice(0, 10000)).some((ref) => ref.config.image))
  ) {
    return boundReferenceEntries(metadata.ref_images);
  }
  const legacyEntities = getLegacyCanvasReferenceEntries(metadata);
  let rejectedHistoricalReferences = false;
  if (Array.isArray(legacyEntities) && legacyEntities.length > 0) {
    const entries = boundReferenceEntries(legacyEntities).map((entry) =>
      isRecord(entry) && isRecord(entry.ipAdapter)
        ? { config: entry.ipAdapter, id: entry.id, isEnabled: entry.isEnabled }
        : entry
    );
    if (normalizeReferenceImages(entries).some((ref) => ref.config.image)) {
      return entries;
    }
    rejectedHistoricalReferences = true;
  }
  if (!('ipAdapters' in metadata)) {
    return rejectedHistoricalReferences ? [null] : null;
  }
  if (!Array.isArray(metadata.ipAdapters) || metadata.ipAdapters.length === 0) {
    return [null];
  }
  return boundReferenceEntries(metadata.ipAdapters).map((entry, index) => {
    if (!isRecord(entry)) {
      return null;
    }
    // The modern slot holds one image. Splitting a multi-image embedding changes what the adapter computes.
    const image = Array.isArray(entry.image) && entry.image.length === 1 ? entry.image[0] : entry.image;
    const model = toRecordedModelRef(entry.ip_adapter_model);
    const begin = entry.begin_step_percent ?? 0,
      end = entry.end_step_percent ?? 1;
    if (
      !isRecord(image) ||
      typeof image.image_name !== 'string' ||
      !image.image_name ||
      !model ||
      typeof entry.weight !== 'number' ||
      !Number.isFinite(entry.weight) ||
      entry.weight < -1 ||
      entry.weight > 2 ||
      typeof begin !== 'number' ||
      typeof end !== 'number' ||
      !Number.isFinite(begin) ||
      !Number.isFinite(end) ||
      begin < 0 ||
      end > 1 ||
      begin > end ||
      !['full', 'style', 'composition', 'style_strong', 'style_precise'].includes(String(entry.method)) ||
      !['ViT-L', 'ViT-H', 'ViT-G'].includes(String(entry.clip_vision_model))
    ) {
      return null;
    }
    return {
      legacyIPAdapter: true,
      id: `legacy-ip-adapter-${index}`,
      isEnabled: true,
      config: {
        type: 'ip_adapter',
        image,
        weight: entry.weight,
        beginEndStepPct: [begin, end],
        method: entry.method,
        clipVisionModel: entry.clip_vision_model,
        model: {
          key: model.key ?? '',
          hash: model.hash ?? undefined,
          name: model.name ?? '',
          base: model.base ?? '',
          type: model.type ?? 'ip_adapter',
        },
      },
    };
  });
};

/** Name-only v4 assets wait for their actual Gallery dimensions. */
export const readMetadataReferenceImages = (
  metadata: unknown,
  assets?: ReadonlyMap<string, Pick<GalleryImage, 'width' | 'height'>>,
  models?: readonly ComponentModelConfig[],
  mainModel?: GenerateModelConfig
): { absent: boolean; images: GenerateReferenceImage[] | null; names: string[]; partial: boolean } => {
  const entries = getMetadataReferenceEntries(metadata);
  if (entries === null) {
    // `absent`: the record has no reference entry at all. An unreadable `ref_images` (v6.3–6.10 nested Kontext lists)
    // or a v4.2 object-form `control_layers`, whose global IP-Adapter layers are not read here, still lists some.
    const absent = isRecord(metadata) && !('ref_images' in metadata) && !isRecord(metadata.control_layers);
    return { absent, images: null, names: [], partial: false };
  }
  const names = new Set<string>();
  const ipAdapterModels = models?.filter((model) => model.type === 'ip_adapter');
  const hydrated = entries.map((entry) => {
    if (!isRecord(entry) || !isRecord(entry.config)) {
      return entry;
    }
    const rawEntry = entry;
    let config = entry.config;
    if (config.type === 'ip_adapter' && Array.isArray(config.weight)) {
      return null;
    }
    if (entry.legacyIPAdapter && models) {
      const ref = toRecordedModelRef(config.model);
      const installed = ref
        ? resolveRecordedModel(ref, ipAdapterModels ?? [], { ...IMAGE_RECORDED_MODEL_POLICY, hashMiss: 'matchName' })
        : null;
      if (
        !installed ||
        installed === 'ambiguous' ||
        (ref?.base && ref.base !== installed.base) ||
        (ref?.type && ref.type !== installed.type) ||
        (mainModel && installed.base !== mainModel.base)
      ) {
        return null;
      }
      config = { ...config, model: installed };
      entry = { ...entry, config };
    }
    const asset = config.image;
    const effective =
      isRecord(asset) && isRecord(asset.crop)
        ? asset.crop.image
        : isRecord(asset) && isRecord(asset.original)
          ? asset.original.image
          : asset;
    const name = isRecord(effective) ? (effective.image_name ?? effective.imageName) : null;
    if (typeof name === 'string' && name.length > 0) {
      names.add(name);
      const resolved = assets?.get(name);
      if (resolved && isRecord(asset) && !('original' in asset)) {
        return {
          ...rawEntry,
          config: { ...config, image: { image_name: name, width: resolved.width, height: resolved.height } },
        };
      }
    }
    return entry;
  });
  const images = normalizeReferenceImages(hydrated).filter((ref) => ref.config.image !== null);
  return {
    absent: false,
    images: images.length > 0 || entries.length === 0 ? images : null,
    names: [...names],
    partial: images.length < entries.length,
  };
};

export const getMetadataReferenceImages = (metadata: unknown): GenerateReferenceImage[] | null =>
  readMetadataReferenceImages(metadata).images;

/** Clear an adapter whose model is not installed: a stale key passes validation and fails only on the server. */
const resolveReferenceImageModels = (
  referenceImages: GenerateReferenceImage[],
  models: readonly ComponentModelConfig[]
): GenerateReferenceImage[] =>
  referenceImages.map((referenceImage) => {
    const { config } = referenceImage;

    if ((config.type !== 'ip_adapter' && config.type !== 'flux_redux') || !config.model) {
      return referenceImage;
    }

    const recordedType = config.model.type;
    const ref = toRecordedModelRef(config.model);
    const installed = ref
      ? resolveRecordedModel(
          ref,
          models.filter((model) => model.type === recordedType),
          { ...IMAGE_RECORDED_MODEL_POLICY, hashMiss: 'matchName' }
        )
      : null;

    return { ...referenceImage, config: { ...config, model: installed === 'ambiguous' ? null : installed } };
  });

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

/** Require served policy before persisting model defaults; prompts and seed are intrinsic. */
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
  const hasSettings = hasGenerationSettings(metadata, clipSkipModel);
  const hasVaeSettings = Object.keys(getMetadataVaeSettings(metadata, clipSkipModel)).length > 0;
  const hasComponents = hasComponentModels(metadata, models);
  const referenceRecord = readMetadataReferenceImages(metadata);
  const recordedReferenceImages = referenceRecord.images;
  const hasReferenceImages =
    referenceRecord.names.length > 0 ||
    (recordedReferenceImages !== null &&
      (recordedReferenceImages.length > 0 || currentValues.referenceImages.length > 0));
  const hasModel = supportedMetadataModel !== null;
  const hasProviderOptions = hasExternalProvider(metadata, clipSkipModel);
  const hasRebalance = hasKrea2Rebalance(metadata, clipSkipModel) || hasKrea2SeedVariance(metadata, clipSkipModel);
  const hasSkipLayers = hasMossImageSkipLayers(metadata, clipSkipModel);
  const hasTuning = hasModelTuning(metadata, clipSkipModel);
  const hasRefiner =
    Object.keys(getRefinerRecall(metadata, clipSkipModel, models, supportedMetadataModel).patch).length > 0;
  const hasSwitchSettings =
    hasAnimaInpaintAdapterSettings(metadata, clipSkipModel) ||
    Object.keys(getMetadataFlux3PromptUpsampler(metadata, clipSkipModel)).length > 0;
  const hasConcepts =
    (getMetadataConcepts(metadata, models, clipSkipModel, supportedMetadataModel).loras?.length ?? 0) > 0;
  const hasAnyMetadata =
    hasModel ||
    hasVae ||
    hasVaeSettings ||
    hasPrompts ||
    hasSeed ||
    hasSize ||
    hasSettings ||
    hasHiDiffusion ||
    hasClipSkip ||
    hasComponents ||
    hasReferenceImages ||
    hasConcepts ||
    hasRebalance ||
    hasProviderOptions ||
    hasRefiner ||
    hasSkipLayers ||
    hasTuning ||
    hasSwitchSettings ||
    hasCanvasRecall(metadata);
  const hasNonSeedMetadata =
    hasModel ||
    hasVae ||
    hasVaeSettings ||
    hasPrompts ||
    hasSize ||
    hasSettings ||
    hasHiDiffusion ||
    hasClipSkip ||
    hasComponents ||
    hasReferenceImages ||
    hasConcepts ||
    hasRebalance ||
    hasProviderOptions ||
    hasRefiner ||
    hasSkipLayers ||
    hasTuning ||
    hasSwitchSettings ||
    hasCanvasRecall(metadata);

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
  isCanvasFrame = false,
  kind,
  metadata,
  referenceAssets,
  supportedModels,
  models,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  /** The Generate size is the Canvas frame, which recall sets as recorded. */
  isCanvasFrame?: boolean;
  kind: ImageRecallKind;
  metadata: unknown;
  referenceAssets?: ReadonlyMap<string, Pick<GalleryImage, 'width' | 'height'>>;
  supportedModels: GenerateModelConfig[];
  models: ComponentModelConfig[];
  vaeModels: VaeModelConfig[];
}): ImageRecallResult | null => {
  if (!isImageRecallKindAvailable(kind)) {
    return null;
  }

  const fields: RecalledField[] = [];
  const skipped: ImageRecallSkip[] = [];
  let values: GenerateWidgetValues = cloneGenerateWidgetValues(currentValues);
  let restoredProviderSize = false;
  let referenceRecall: ReturnType<typeof readMetadataReferenceImages> | null = null;

  if (kind === 'all' || kind === 'remix') {
    const model = getSupportedMetadataModel(metadata, supportedModels);

    if (model) {
      const valuesWithModelDefaults = getSettingsWithModelDefaults(values, model);
      const vae = getRecallDefaultVae(resolveModelDefaultVae(model, valuesWithModelDefaults, vaeModels));
      values = { ...valuesWithModelDefaults, model, ...(vae === undefined ? {} : { vae }) };
      fields.push('model');
    }

    const componentPatch = getComponentPatch(metadata, models);
    const recordedSettings = Object.keys(componentPatch) as RecalledComponentSetting[];
    const effectiveModel = values.model;
    // Skip components that do not fit, judged beside those already restored (a Vision Tower depends on a GGUF encoder).
    const fittingSettings: RecalledComponentSetting[] = [];

    for (const setting of recordedSettings) {
      const component = componentPatch[setting];

      // A Diffusers main records itself as its own source, which the builders fall back to without a pick. Clearing
      // the slot reproduces that, where a source left in the panel would supply the components instead.
      if (setting === 'componentSourceModel' && component?.key === effectiveModel.key) {
        values = { ...values, componentSourceModel: null };
        fittingSettings.push(setting);
        continue;
      }

      if (!component || isComponentCompatibleWithModel(effectiveModel, values, setting, component)) {
        values = { ...values, [setting]: component };
        fittingSettings.push(setting);
      }
    }

    // A slot the record names stays with the record, resolved or not: an uninstalled component is not one the source
    // supplied, so the panel's own pick stays.
    const recordsComponent = (setting: RecalledComponentSetting): boolean =>
      RECALLED_COMPONENTS.some(
        (component) => component.setting === setting && isRecord(metadata) && component.metadataKey in metadata
      );
    // A record of a graph run left out the slots its main or its source filled. Standalone picks left in the panel would
    // take those slots over, so clear them to let the same supplier fill them again. Only a recalled main establishes
    // who that was: a record that names none says nothing about the panel's model, and one that records no components
    // (a Canvas save) says nothing about them. The encoders judged are the ones that survive recall: a recorded one that
    // is gone leaves the panel's, and with it the question whether the source still supplies the VAE.
    const judgedModel = model && recordsGraphRun(metadata) ? model : undefined;
    const { componentSourceModel } = componentPatch;
    // A writer that may have used a source without recording it leaves the question open, like an unresolved one.
    const recordedSource = !recordsComponent('componentSourceModel')
      ? judgedModel && mayHaveOmittedSetting(metadata, judgedModel.base, 'componentSourceModel')
        ? ('unresolved' as const)
        : null
      : !fittingSettings.includes('componentSourceModel')
        ? ('unresolved' as const)
        : isMainModelConfig(componentSourceModel)
          ? componentSourceModel
          : null;
    // The model selection below drops a pick the model does not accept, so only an accepted one counts here.
    const survivingEncoder = (setting: 'mistralEncoderModel' | 'qwen3EncoderModel') => {
      const encoder = values[setting];

      return encoder && isComponentCompatibleWithModel(effectiveModel, values, setting, encoder) ? encoder : null;
    };
    const survivingEncoders = {
      mistralEncoderModel: survivingEncoder('mistralEncoderModel'),
      qwen3EncoderModel: survivingEncoder('qwen3EncoderModel'),
    };
    const suppliedByRecord = judgedModel
      ? getComponentSlotsSuppliedByRecord(judgedModel, recordedSource, survivingEncoders)
      : [];
    const isProvenUnused = (setting: RecalledComponentSetting): boolean =>
      !recordsComponent(setting) && !(judgedModel && mayHaveOmittedSetting(metadata, judgedModel.base, setting));
    const clearedSettings = [
      ...suppliedByRecord.filter(
        (key): key is RecalledComponentSetting & typeof key =>
          key !== 'vae' && isProvenUnused(key as RecalledComponentSetting)
      ),
      ...((judgedModel && SETTINGS_RECORDED_WHEN_LOADED[judgedModel.base]) ?? []).filter(isProvenUnused),
    ];

    for (const setting of clearedSettings) {
      values = { ...values, [setting]: null };
    }

    const rebalancePatch = getMetadataKrea2Rebalance(metadata, values.model);

    if (Object.keys(rebalancePatch).length > 0) {
      values = { ...values, ...rebalancePatch };
      fields.push('krea2Rebalance');
    }

    const providerPatch = getMetadataExternalProvider(metadata, values.model, isCanvasFrame);

    if (Object.keys(providerPatch).length > 0) {
      values = { ...values, ...providerPatch };
      fields.push('externalProvider');
      restoredProviderSize = providerPatch.width !== undefined;
    }

    const seedVariancePatch = getMetadataKrea2SeedVariance(metadata, values.model);

    if (Object.keys(seedVariancePatch).length > 0) {
      values = { ...values, ...seedVariancePatch };
      fields.push('krea2SeedVariance');
    }

    const skipLayersPatch = getMetadataMossImageSkipLayers(metadata, values.model);

    if (Object.keys(skipLayersPatch).length > 0) {
      values = { ...values, ...skipLayersPatch };
      fields.push('mossImageSkipLayers');
    }

    const tuningPatch = getMetadataModelTuning(metadata, values.model);

    // A same-family record always restores its switches; count the field only when one changes.
    if (Object.entries(tuningPatch).some(([key, value]) => values[key as keyof GenerateWidgetValues] !== value)) {
      values = { ...values, ...tuningPatch };
      fields.push('modelTuning');
    }

    const adapterPatch = getMetadataAnimaInpaintAdapter(metadata, values.model);
    values = { ...values, ...adapterPatch, ...getMetadataFlux3PromptUpsampler(metadata, values.model) };

    if (adapterPatch.animaInpaintAdapterWeight !== undefined) {
      fields.push('animaInpaintAdapterWeight');
    }

    const vae = getMetadataVae(metadata, values.model, vaeModels);

    if (vae !== undefined) {
      values = { ...values, vae };
      fields.push('vae');
    } else if (suppliedByRecord.includes('vae') && getRecordedVaeKey(metadata, values.model) === null && values.vae) {
      values = { ...values, vae: null };

      if (currentValues.vae) {
        fields.push('vae');
      }
    }

    if (model || fittingSettings.length > 0) {
      const { settings } = getGenerateModelSelectionResult({ currentValues: values, model: values.model, models });
      values = { ...values, ...settings };
    }

    // After the transition, which resets what the model's panel does not show. Older records carry none.
    const vaeSettings = getMetadataVaeSettings(metadata, values.model);

    if (Object.keys(vaeSettings).length > 0) {
      values = { ...values, ...vaeSettings };
      if (!fields.includes('vae')) {
        fields.push('vae');
      }
    }

    // Judged after the transition, which can complete or undo a recorded pick. The adapter switch is a component
    // change too: a record can turn it off without touching the pick.
    if (
      [...recordedSettings, ...clearedSettings].some(
        (setting) => values[setting]?.key !== currentValues[setting]?.key
      ) ||
      values.animaInpaintAdapterEnabled !== currentValues.animaInpaintAdapterEnabled ||
      values.flux3PromptUpsamplerEnabled !== currentValues.flux3PromptUpsamplerEnabled
    ) {
      fields.push('components');
    }

    const concepts = getMetadataConcepts(metadata, models, values.model, model);

    skipped.push(...concepts.skipped);

    if (concepts.loras && (concepts.loras.length > 0 || currentValues.loras.length > 0)) {
      values = { ...values, loras: concepts.loras };
      fields.push('loras');
    }

    // References must fit the effective model, which is still the current one when the recorded model was not recalled.
    referenceRecall = readMetadataReferenceImages(metadata, referenceAssets, models, values.model);
    // v6+ generation graphs omitted `ref_images` only when no references were used, so such a record without a
    // reference entry had none. v5 Canvas-staged records dropped the references they used along with
    // `canvas_v2_metadata`, so older records prove nothing. A model that takes no references keeps them, switched off.
    const recordedReferenceImages =
      referenceRecall.images ??
      (referenceRecall.absent &&
      model &&
      isGenerationRecord(metadata) &&
      Number.parseInt(getString(metadata, 'app_version') ?? '', 10) >= 6 &&
      takesReferenceImages(values.model) &&
      values.model.base !== LEGACY_UNRECORDED_REFERENCE_BASE
        ? []
        : null);
    // Only the base decides; an encoder without a vision tower is flagged by validation, not recalled away.
    const referenceImages = getCompatibleReferenceImages(
      resolveReferenceImageModels(recordedReferenceImages ?? [], models),
      values.model,
      models
    );

    // No recorded references clears the current ones; references that do not fit leave them.
    if (referenceImages.length > 0 || (recordedReferenceImages?.length === 0 && values.referenceImages.length > 0)) {
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
  } else if (restoredProviderSize) {
    fields.push('size');
  } else if (kind === 'all' || kind === 'remix') {
    const size = getMetadataSize(metadata, values.model);

    if (Object.keys(size).length > 0) {
      values = withDimensions(values, size);
      fields.push('size');
    }
  }

  if (kind === 'all' || kind === 'remix') {
    const steps = getSteps(metadata);
    const cfgScale = getCfgScale(metadata, values.model);
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

    const refinerRecall = getRefinerRecall(
      metadata,
      values.model,
      models,
      getSupportedMetadataModel(metadata, supportedModels)
    );
    const refinerPatch = refinerRecall.patch;
    const refinerChanged = (Object.keys(refinerPatch) as (keyof RefinerPatch)[]).some((key) =>
      key === 'refinerModel'
        ? refinerPatch.refinerModel?.key !== values.refinerModel?.key
        : refinerPatch[key] !== values[key]
    );

    if (refinerChanged) {
      values = { ...values, ...refinerPatch };
      fields.push('refiner');
    }
    if (refinerRecall.skipped) {
      skipped.push(refinerRecall.skipped);
    }
  }

  if (kind === 'all' || kind === 'remix' || kind === 'clipSkip') {
    const clipSkip = getSupportedClipSkip(metadata, values.model);

    if (clipSkip !== null) {
      values = { ...values, clipSkip };
      fields.push('clipSkip');
    }
  }

  return fields.length > 0 ||
    skipped.length > 0 ||
    ((kind === 'all' || kind === 'remix') &&
      (hasCanvasRecall(metadata) || referenceRecall?.partial || (referenceRecall?.names.length ?? 0) > 0))
    ? { fields, skipped, values }
    : null;
};

export const getImageRecallTitle = (kind: ImageRecallKind, t: TFunction): string =>
  t(`widgets.gallery.itemActions.recall.imageTitles.${kind}`);

export const getImageRecallMessage = (fields: readonly RecalledField[], t: TFunction): string =>
  t('widgets.gallery.itemActions.recall.fieldsApplied', { count: fields.length });

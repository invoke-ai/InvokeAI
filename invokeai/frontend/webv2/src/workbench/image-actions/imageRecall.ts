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

import {
  clampAnimaInpaintAdapterWeight,
  clampDimension,
  cloneGenerateWidgetValues,
  deriveAspectRatioId,
  getCompatibleReferenceImages,
  getDimensionGrid,
  getExternalProviderOptionKeys,
  getExternalResolutionPresets,
  getGenerateModelSelectionResult,
  getGenerationUiPolicy,
  isComponentCompatibleWithModel,
  getModelDefaultVae,
  getSettingsWithModelDefaults,
  hasArchitectureCapabilities,
  hasModelDefaultVae,
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
  GEMINI_TEMPERATURE_MAX,
  GEMINI_TEMPERATURE_MIN,
  isRefinerModel,
  isFluxDypePreset,
  isQwenImageEncoderQuantization,
  isValidKrea2RebalanceWeights,
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
  | 'loras'
  | 'referenceImages'
  | 'krea2Rebalance'
  | 'externalProvider'
  | 'refiner'
  | 'krea2SeedVariance'
  | 'modelTuning'
  | 'animaInpaintAdapterWeight';

export type ImageRecallSkipReason =
  | 'ambiguous'
  | 'duplicate'
  | 'incompatible'
  | 'invalid'
  | 'modelUnavailable'
  | 'unresolved';

/** A recorded model recall could not restore; `name` is the best label the record or catalog offers. */
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

/**
 * The value the model's single guidance slider holds. Distilled-guidance families record it as `guidance` and
 * their older records carry no `cfg_scale` at all, so for them the dedicated key wins.
 */
export const getCfgScale = (metadata: unknown, model?: GenerateModelConfig): number | null => {
  const { guidanceLabel, guidanceMin } = getGenerationUiPolicy(model);
  const cfgScale =
    guidanceLabel === 'Guidance'
      ? (getNumber(metadata, 'guidance') ?? getNumber(metadata, 'cfg_scale'))
      : getNumber(metadata, 'cfg_scale');

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
  if (!getGenerationUiPolicy(model).hiDiffusionVisible) {
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

/** The installed model a recorded reference names, or null when none (or more than one) does. */
const resolveMetadataModel = <T extends ComponentModelConfig>(value: unknown, models: readonly T[]): T | null => {
  const ref = toRecordedModelRef(value);
  const resolved = ref ? resolveRecordedModel(ref, models) : null;

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

  const vae = resolveMetadataModel(metadata.vae, vaeModels);

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

const hasGenerationSettings = (metadata: unknown, model: GenerateModelConfig): boolean =>
  getSteps(metadata) !== null ||
  getCfgScale(metadata, model) !== null ||
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

/**
 * Generation graphs stamp `generation_mode` and record what ran, so on such a record a missing list or switch means
 * none or the node default. Other records that name a model, such as canvas saves, never record concepts or tuning.
 */
const isGenerationRecord = (metadata: unknown): boolean => getString(metadata, 'generation_mode') !== null;

const getBoundedNumber = (metadata: unknown, key: string, min: number, max: number): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && value >= min && value <= max ? value : null;
};

/**
 * A recorded shift: `'auto'` (webv2) or `null` (legacy) for one the node derived, a number for an override.
 * Undefined when the key is missing or holds something else.
 */
const getRecordedTimestepShift = (metadata: unknown, key: string): number | null | undefined => {
  if (!isRecord(metadata) || !(key in metadata)) {
    return undefined;
  }

  if (metadata[key] === null || metadata[key] === 'auto') {
    return null;
  }

  return getBoundedNumber(metadata, key, MIN_TIMESTEP_SHIFT, MAX_TIMESTEP_SHIFT) ?? undefined;
};

/**
 * Per-family tuning the graphs record under their own keys. Gated on the model recall settles on, since each
 * setting only reaches that family's denoise or encoder nodes.
 *
 * A generation record of that same family that leaves a switch out ran the node's default: webv2 recorded no tuning
 * before the panel offered it. Restoring the default there means a preset turned on since does not survive recall.
 * Values that only matter while a switch is on (DyPE scale, variance strength) stay as they are when missing.
 */
const getMetadataModelTuning = (
  metadata: unknown,
  model: GenerateModelConfig | undefined
): Partial<GenerateWidgetValues> => {
  if (!model || model.type !== 'main' || !isRecord(metadata)) {
    return {};
  }

  const recordsFamily = isGenerationRecord(metadata) && getRecord(metadata, 'model')?.base === model.base;
  const recordedOrDefault = <T>(key: string, recorded: T | undefined, fallback: T): T | undefined =>
    recorded !== undefined ? recorded : recordsFamily && !(key in metadata) ? fallback : undefined;

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

  if (model.base === 'krea-2') {
    // Recorded only when the switch was sent: with edit references and a LoRA. A record without it says nothing
    // about the switch, so the user's setting stays.
    const referenceKvCache = getBoolean(metadata, 'krea2_reference_kv_cache');

    return referenceKvCache === null ? {} : { krea2ReferenceKvCache: referenceKvCache };
  }

  return {};
};

const hasModelTuning = (metadata: unknown, model: GenerateModelConfig | undefined): boolean =>
  Object.keys(getMetadataModelTuning(metadata, model)).length > 0;

/** The canvas inpaint and outpaint graphs record the adapter key whenever the adapter ran. */
const ANIMA_INPAINT_ADAPTER_MODES: readonly string[] = ['anima_inpaint', 'anima_outpaint'];

/**
 * Provider options and the named size, gated on what the effective model's provider accepts: the keys are recorded
 * by the provider nodes (`openai_quality`, `gemini_temperature`, ...), so a value only applies to the same provider.
 */
const getMetadataExternalProvider = (
  metadata: unknown,
  model: GenerateModelConfig | undefined,
  isCanvasFrame: boolean
): Partial<GenerateWidgetValues> => {
  // Local models advertise no provider options or presets, so every lookup below answers nothing for them.
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
    // A Canvas output records its frame, not the provider size: the Canvas asked for the named size at the ratio
    // nearest the frame, so Generate recall picks that one. A Canvas route keeps the recorded frame instead, and the
    // Canvas derives the same named size from it.
    const ratioDistance = (preset: { width: number; height: number }) =>
      width !== null && height !== null && width > 0 && height > 0
        ? Math.abs(Math.log(preset.width / preset.height / (width / height)))
        : Number.NaN;
    const nearest =
      !isCanvasFrame && exact.length === 0 && named.length > 0 && !Number.isNaN(ratioDistance(named[0]!))
        ? named.reduce((best, preset) => (ratioDistance(preset) < ratioDistance(best) ? preset : best))
        : undefined;
    const preset = exact.length === 1 ? exact[0] : nearest;

    // A name alone can describe many aspect ratios. Restore only a size the record pins down, with its dimensions.
    if (preset) {
      patch.externalImageSize = preset.image_size;
      patch.width = preset.width;
      patch.height = preset.height;
      patch.aspectRatioId = deriveAspectRatioId(preset.width, preset.height);
      patch.aspectRatioValue = preset.width / preset.height;
    }
  }

  // Provider outputs omit nullable fields when their default was used. The provider stamp distinguishes those
  // omissions from partial or unrelated metadata, where the current choice should remain untouched.
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
 * The adapter switch and weight; only an Anima image can carry them. A recorded adapter means it ran, so the switch
 * turns on. `core_metadata` drops `null` on write, so an inpaint or outpaint record without the key ran without one:
 * the switch turns off and the current pick stays for the next time it is wanted. Other records say nothing about
 * it. An out-of-range weight is clamped rather than applied, which would block every inpaint until the user edits it.
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

type RecalledComponentSetting = keyof Pick<
  GenerateWidgetValues,
  | 'animaInpaintAdapterModel'
  | 'clipEmbedModel'
  | 'componentSourceModel'
  | 'gemma2EncoderModel'
  | 'ideogram4UnconditionalModel'
  | 'mistralEncoderModel'
  | 'pidDecoderModel'
  | 'qwen3EncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwen35EncoderModel'
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
  { metadataKey: 'anima_inpaint_adapter', setting: 'animaInpaintAdapterModel' },
  { metadataKey: 'clip_embed_model', setting: 'clipEmbedModel' },
  { metadataKey: 'gemma2_encoder', setting: 'gemma2EncoderModel' },
  { metadataKey: 'mistral_encoder', setting: 'mistralEncoderModel' },
  { metadataKey: 'pid_decoder', setting: 'pidDecoderModel' },
  { metadataKey: 'qwen3_encoder', setting: 'qwen3EncoderModel' },
  { metadataKey: 'qwen3_vl_encoder', setting: 'qwen3VLEncoderModel' },
  { metadataKey: 'qwen3_5_encoder', setting: 'qwen35EncoderModel' },
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

/**
 * A recorded `ModelIdentifierField` — `{ key, hash, name, base, type }` — with every field optional on read. Records
 * from before identifier fields named the model as `{ model_name, base_model, model_type }`.
 */
interface RecordedModelRef {
  base: string | null;
  hash: string | null;
  key: string | null;
  name: string | null;
  type: string | null;
}

const toRecordedModelRef = (value: unknown): RecordedModelRef | null => {
  if (!isRecord(value)) {
    return null;
  }

  const ref = {
    base: getString(value, 'base') ?? getString(value, 'base_model'),
    hash: getString(value, 'hash'),
    key: getString(value, 'key'),
    name: getString(value, 'name') ?? getString(value, 'model_name'),
    type: getString(value, 'type') ?? getString(value, 'model_type'),
  };

  return ref.key || ref.hash || (ref.name && ref.base) ? ref : null;
};

/**
 * Resolve the install-local key first, then a portable identifier: the content hash when one was recorded, otherwise
 * name and base (and type, when recorded). A portable identifier counts only when exactly one installed model matches
 * it.
 *
 * By default a recorded hash never falls back to a same-named file, so the main model, VAE, components and concepts
 * restore only the exact weights that ran. `hashMiss: 'matchName'` lets a hash that matches nothing fall through to
 * name and base, as legacy web does for reference-image adapters after a reinstall changes their hash.
 */
const resolveRecordedModel = <T extends ComponentModelConfig>(
  ref: RecordedModelRef,
  models: readonly T[],
  hashMiss: 'refuse' | 'matchName' = 'refuse'
): T | 'ambiguous' | null => {
  const byKey = ref.key ? models.find((model) => model.key === ref.key) : undefined;

  if (byKey) {
    return byKey;
  }

  const byHash = ref.hash ? models.filter((model) => model.hash === ref.hash) : [];
  const matches =
    byHash.length > 0 || (ref.hash && hashMiss === 'refuse')
      ? byHash
      : ref.name && ref.base
        ? models.filter(
            (model) => model.name === ref.name && model.base === ref.base && (!ref.type || model.type === ref.type)
          )
        : [];

  return matches.length > 1 ? 'ambiguous' : (matches[0] ?? null);
};

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
  /** The recorded refiner could not be restored; its current model and tuning stay together. */
  skipped: ImageRecallSkip | null;
}

const isInstalledRefiner = (model: ComponentModelConfig): model is MainModelConfig =>
  isMainModelConfig(model) && isRefinerModel(model);

/**
 * The refiner an SDXL image ran with, resolved against installed refiner models, plus its tuning. Generation records
 * only name a refiner when one ran, so a stamped record without one clears the slot, as a missing `loras` list means
 * no concepts.
 */
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

  // Refiner presence and tuning describe the recorded main model, including records that ran without a refiner.
  if (getRecord(metadata, 'model') && sourceModel?.key !== model.key) {
    // As with concepts, only a refiner the record names is reported; a record without one keeps the slot quietly.
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
    const refiner = ref ? resolveRecordedModel(ref, models.filter(isInstalledRefiner)) : null;

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
  /** The concept list to apply, or null when the metadata says nothing reliable about concepts. */
  loras: GenerateLora[] | null;
  skipped: ImageRecallSkip[];
}

/**
 * Concepts the image was generated with, resolved against installed LoRAs that fit `model`, the model recall settles
 * on. `sourceModel` is the recorded main model as resolved from the installed catalog. The metadata lists only the
 * LoRAs that ran, so every restored concept is enabled at its recorded weight.
 */
const getMetadataConcepts = (
  metadata: unknown,
  models: readonly ComponentModelConfig[],
  model: GenerateModelConfig,
  sourceModel: GenerateModelConfig | null
): ConceptRecall => {
  const recorded = isRecord(metadata) ? metadata.loras : undefined;

  // Matching keys alone are not enough: persisted settings can still name a model that is no longer installed.
  if (getRecord(metadata, 'model') && sourceModel?.key !== model.key) {
    // Concepts belong to the model they ran on. Without it the current set stays, rather than being judged against
    // a model recall did not choose.
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

    const installed = resolveRecordedModel(ref, loraModels);

    if (installed === null || installed === 'ambiguous') {
      skipped.push({
        name: recordedName,
        reason: installed === 'ambiguous' ? 'ambiguous' : 'unresolved',
        subject: 'concept',
      });
    } else if (!isLoraCompatibleWithModel(installed, model)) {
      skipped.push({ name: installed.name, reason: 'incompatible', subject: 'concept' });
    } else if (loras.some((lora) => lora.model.key === installed.key)) {
      skipped.push({ name: installed.name, reason: 'duplicate', subject: 'concept' });
    } else {
      // Kept exactly: graphs and model defaults accept weights beyond the panel's typed range, which bounds only edits.
      loras.push({ isEnabled: true, model: installed, weight });
    }
  }

  return { loras, skipped };
};

/**
 * The references the image was generated with, or null when the record says nothing about them. An explicitly
 * empty modern list means it used none; only a list with no usable entry falls back to the pre-v6 canvas slot.
 */
export const getMetadataReferenceImages = (metadata: unknown): GenerateReferenceImage[] | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  if (Array.isArray(metadata.ref_images) && metadata.ref_images.length === 0) {
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
    return null;
  }

  const legacy = normalizeReferenceImages(
    legacyEntities.map((entry) =>
      isRecord(entry) && isRecord(entry.ipAdapter)
        ? { config: entry.ipAdapter, id: entry.id, isEnabled: entry.isEnabled }
        : entry
    )
  ).filter((referenceImage) => referenceImage.config.image !== null);

  return legacy.length > 0 ? legacy : null;
};

/**
 * Point each recorded adapter at the installed model it names, or clear it so Generate asks for one: a reinstall gives
 * the adapter a new key, and a stale key would pass validation and fail only on the server.
 */
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
          'matchName'
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
  const hasSettings = hasGenerationSettings(metadata, clipSkipModel);
  const hasComponents = hasComponentModels(metadata, models);
  const recordedReferenceImages = getMetadataReferenceImages(metadata);
  const hasReferenceImages =
    recordedReferenceImages !== null &&
    (recordedReferenceImages.length > 0 || currentValues.referenceImages.length > 0);
  const hasModel = supportedMetadataModel !== null;
  const hasProviderOptions = hasExternalProvider(metadata, clipSkipModel);
  const hasRebalance = hasKrea2Rebalance(metadata, clipSkipModel) || hasKrea2SeedVariance(metadata, clipSkipModel);
  const hasTuning = hasModelTuning(metadata, clipSkipModel);
  const hasRefiner =
    Object.keys(getRefinerRecall(metadata, clipSkipModel, models, supportedMetadataModel).patch).length > 0;
  const hasAdapterSettings = hasAnimaInpaintAdapterSettings(metadata, clipSkipModel);
  const hasConcepts =
    (getMetadataConcepts(metadata, models, clipSkipModel, supportedMetadataModel).loras?.length ?? 0) > 0;
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
    hasConcepts ||
    hasRebalance ||
    hasProviderOptions ||
    hasRefiner ||
    hasTuning ||
    hasAdapterSettings;
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
    hasConcepts ||
    hasRebalance ||
    hasProviderOptions ||
    hasRefiner ||
    hasTuning ||
    hasAdapterSettings;

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
  supportedModels,
  models,
  vaeModels,
}: {
  currentValues: GenerateWidgetValues;
  image: GalleryImage;
  /** The Generate size is the Canvas frame (a Canvas route), which recall sets as recorded. */
  isCanvasFrame?: boolean;
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
  const skipped: ImageRecallSkip[] = [];
  let values: GenerateWidgetValues = cloneGenerateWidgetValues(currentValues);
  // A restored provider size is what the provider was asked for; the recorded size may be a Canvas frame instead.
  let restoredProviderSize = false;

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

    const tuningPatch = getMetadataModelTuning(metadata, values.model);

    // A same-family record always restores its switches, so only count the field when one actually changes.
    if (Object.entries(tuningPatch).some(([key, value]) => values[key as keyof GenerateWidgetValues] !== value)) {
      values = { ...values, ...tuningPatch };
      fields.push('modelTuning');
    }

    const adapterPatch = getMetadataAnimaInpaintAdapter(metadata, values.model);
    values = { ...values, ...adapterPatch };

    if (adapterPatch.animaInpaintAdapterWeight !== undefined) {
      fields.push('animaInpaintAdapterWeight');
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

    // Judged after the transition, which can complete or undo a recorded pick. The adapter switch is a component
    // change too: a record can turn it off without touching the pick.
    if (
      recordedSettings.some((setting) => values[setting]?.key !== currentValues[setting]?.key) ||
      values.animaInpaintAdapterEnabled !== currentValues.animaInpaintAdapterEnabled
    ) {
      fields.push('components');
    }

    // Judged against the model the transition settled on; the recorded concepts replace the list it left.
    const concepts = getMetadataConcepts(metadata, models, values.model, model);

    skipped.push(...concepts.skipped);

    if (concepts.loras && (concepts.loras.length > 0 || currentValues.loras.length > 0)) {
      values = { ...values, loras: concepts.loras };
      fields.push('loras');
    }

    // Recalled reference images must fit the effective model — when the
    // metadata model was not recalled (e.g. uninstalled), `values.model` is
    // still the current one and incompatible configs are re-targeted or dropped.
    const recordedReferenceImages = getMetadataReferenceImages(metadata);
    const referenceImages = getCompatibleReferenceImages(
      resolveReferenceImageModels(recordedReferenceImages ?? [], models),
      values.model,
      models
    );

    // A record of no references clears the current ones; a record whose references do not fit leaves them.
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

    // Judged against the model the recall settled on: a non-SDXL main has no refiner to restore.
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

  return fields.length > 0 || skipped.length > 0 ? { fields, skipped, values } : null;
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

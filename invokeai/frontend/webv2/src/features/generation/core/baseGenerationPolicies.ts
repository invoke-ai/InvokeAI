import type {
  GenerationModelCatalogItem as ModelConfig,
  GenerationModelTaxonomyType as ModelTaxonomyType,
  PromptHistoryItem,
} from '@features/generation/core/contracts';
import type { BaseGenerationConfig, GuidanceLabel } from '@features/generation/core/generationConfig';

import {
  getArchitectureFeatures,
  getArchitectureGenerationConfig,
  hasArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  isSupportedGenerateBase,
  SUPPORTED_GENERATE_BASES,
  type SupportedGenerateBase,
} from '@features/generation/core/supportedBases';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { SEED_MAX } from '@platform/core/seed';

import type {
  ExpandPromptSuggestion,
  ExternalGenerationMode,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  MainModelConfig,
  PidMode,
} from './types';

import {
  getCompatibleDiffusersComponentSource,
  isBundledMainForBase,
  isAnimaInpaintAdapter,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isClipVariant,
  isDiffusersMainForBase,
  isFlux2DiffusersSourceForModel,
  isErnieImageMistralEncoder,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isQwen3Vl8bEncoder,
  isIdeogram4UnconditionalBranch,
  isMossImageQwen35Encoder,
  isQwen3Vl4bEncoder,
  isSelfContainedSDNQFlux1Pipeline,
  isSelfContainedSDNQPipeline,
  isVaeAcceptedByBase,
  isZImageQwen3Encoder,
  type GenerateComponentFilter,
} from './componentCompatibility';
import { MAX_DIMENSION, MIN_DIMENSION } from './dimensionLimits';
import { DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS } from './dynamicPrompts';
import {
  getExternalAspectRatioSizes,
  getExternalReferenceImages,
  getExternalResolutionPresets,
  isSeedreamBatchModel,
  resolveExternalCanvasRequestSize,
  resolveExternalImageSize,
  SEEDREAM_BATCH_MAX_TOTAL_IMAGES,
} from './externalProvider';
import { createUntranslatedGenerationIssue, type GenerationIssue, type GenerationIssueFix } from './generationIssues';
import {
  DEFAULT_PID_STEPS,
  getIsPidActive,
  getIsPidSupportedBase,
  getPidDecoderBaseForMainBase,
  getPidDimensionOverrides,
} from './pid';
import {
  DEFAULT_REFINER_CFG_SCALE,
  DEFAULT_REFINER_NEGATIVE_AESTHETIC_SCORE,
  DEFAULT_REFINER_POSITIVE_AESTHETIC_SCORE,
  DEFAULT_REFINER_SCHEDULER,
  DEFAULT_REFINER_START,
  DEFAULT_REFINER_STEPS,
  getRefinerValidationReasons,
  getActiveRefiner,
  isRefinerSupported,
} from './sdxlRefiner';
import {
  clampDimension,
  DEFAULT_ANIMA_INPAINT_ADAPTER_WEIGHT,
  DEFAULT_HIDIFFUSION_T1_RATIO,
  DEFAULT_HIDIFFUSION_T2_RATIO,
  DEFAULT_FLUX_DYPE_EXPONENT,
  DEFAULT_FLUX_DYPE_SCALE,
  DEFAULT_IDEOGRAM4_SAMPLER_PRESET,
  DEFAULT_KREA2_REBALANCE_MULTIPLIER,
  DEFAULT_KREA2_REBALANCE_WEIGHTS,
  DEFAULT_KREA2_SEED_VARIANCE_RANDOMIZE_PERCENT,
  DEFAULT_KREA2_SEED_VARIANCE_STRENGTH,
  DEFAULT_KREA2_STYLE_STRENGTH,
  DEFAULT_MOSS_IMAGE_UNCOND_SKIP_LAYERS,
  DEFAULT_REFERENCE_IMAGE_LIMIT,
  DEFAULT_ZIMAGE_SEED_VARIANCE_RANDOMIZE_PERCENT,
  DEFAULT_ZIMAGE_SEED_VARIANCE_STRENGTH,
  deriveAspectRatioId,
  IDEOGRAM4_GUIDANCE_MAX,
  IDEOGRAM4_GUIDANCE_MIN,
  IDEOGRAM4_MU_MAX,
  IDEOGRAM4_MU_MIN,
  IDEOGRAM4_STEPS_MAX,
  IDEOGRAM4_STEPS_MIN,
  isAspectRatioLocked,
  isGenerateModelConfig,
  isGenerateSettings,
  isLoraCompatibleWithModel,
  isValidKrea2RebalanceWeights,
  isValidMossImageSkipLayers,
  KREA2_REBALANCE_WEIGHT_COUNT,
  MAX_MOSS_IMAGE_SKIP_LAYER,
  normalizeGenerateSettings,
} from './settings';

// Models owns display identity; graph.ts owns graph topology.

export interface SchedulerOption {
  value: string;
  label: string;
}

export type { BaseGenerationConfig, GuidanceLabel } from '@features/generation/core/generationConfig';

type GenerateDefaultSettings =
  | {
      scheduler?: string | null;
      steps?: number | null;
      cfg_scale?: number | null;
      guidance?: number | null;
      cfg_rescale_multiplier?: number | null;
      width?: number | null;
      height?: number | null;
      vae?: string | null;
      vae_precision?: string | null;
    }
  | null
  | undefined;

export const SCHEDULER_OPTIONS: SchedulerOption[] = [
  { label: 'DDIM', value: 'ddim' },
  { label: 'DDPM', value: 'ddpm' },
  { label: 'DEIS', value: 'deis' },
  { label: 'DEIS Karras', value: 'deis_k' },
  { label: 'DPM++ 2S', value: 'dpmpp_2s' },
  { label: 'DPM++ 2S Karras', value: 'dpmpp_2s_k' },
  { label: 'DPM++ 2M', value: 'dpmpp_2m' },
  { label: 'DPM++ 2M Karras', value: 'dpmpp_2m_k' },
  { label: 'DPM++ 2M SDE', value: 'dpmpp_2m_sde' },
  { label: 'DPM++ 2M SDE Karras', value: 'dpmpp_2m_sde_k' },
  { label: 'DPM++ 3M', value: 'dpmpp_3m' },
  { label: 'DPM++ 3M Karras', value: 'dpmpp_3m_k' },
  { label: 'DPM++ SDE', value: 'dpmpp_sde' },
  { label: 'DPM++ SDE Karras', value: 'dpmpp_sde_k' },
  { label: 'ER-SDE', value: 'er_sde' },
  { label: 'Euler', value: 'euler' },
  { label: 'Euler Karras', value: 'euler_k' },
  { label: 'Euler Ancestral', value: 'euler_a' },
  { label: 'Heun', value: 'heun' },
  { label: 'Heun Karras', value: 'heun_k' },
  { label: 'KDPM 2', value: 'kdpm_2' },
  { label: 'KDPM 2 Karras', value: 'kdpm_2_k' },
  { label: 'KDPM 2 Ancestral', value: 'kdpm_2_a' },
  { label: 'KDPM 2 Ancestral Karras', value: 'kdpm_2_a_k' },
  { label: 'LCM', value: 'lcm' },
  { label: 'LMS', value: 'lms' },
  { label: 'LMS Karras', value: 'lms_k' },
  { label: 'PNDM', value: 'pndm' },
  { label: 'TCD', value: 'tcd' },
  { label: 'UniPC', value: 'unipc' },
  { label: 'UniPC Karras', value: 'unipc_k' },
];

export const FLOW_SCHEDULER_OPTIONS: SchedulerOption[] = [
  { label: 'Euler', value: 'euler' },
  { label: 'Heun (2nd order)', value: 'heun' },
  { label: 'LCM', value: 'lcm' },
];

export const FLOW_SCHEDULER_OPTIONS_WITHOUT_LCM: SchedulerOption[] = [
  { label: 'Euler', value: 'euler' },
  { label: 'Heun (2nd order)', value: 'heun' },
];

export const ANIMA_SCHEDULER_OPTIONS: SchedulerOption[] = [
  { label: 'Euler', value: 'euler' },
  { label: 'Heun (2nd order)', value: 'heun' },
  { label: 'DPM++ 2M', value: 'dpmpp_2m' },
  { label: 'DPM++ 2M SDE', value: 'dpmpp_2m_sde' },
  { label: 'ER-SDE', value: 'er_sde' },
  { label: 'LCM', value: 'lcm' },
];

const KNOWN_SCHEDULERS = new Set(SCHEDULER_OPTIONS.map((option) => option.value));
const FLOW_SCHEDULERS = new Set(FLOW_SCHEDULER_OPTIONS.map((option) => option.value));
const FLOW_SCHEDULERS_WITHOUT_LCM = new Set(FLOW_SCHEDULER_OPTIONS_WITHOUT_LCM.map((option) => option.value));
const ANIMA_SCHEDULERS = new Set(ANIMA_SCHEDULER_OPTIONS.map((option) => option.value));

export const isKnownScheduler = (value: string): boolean => KNOWN_SCHEDULERS.has(value);

export { isSupportedGenerateBase, SUPPORTED_GENERATE_BASES, type SupportedGenerateBase };

const FALLBACK_GENERATION_CONFIG: BaseGenerationConfig = {
  dimensions: { grid: 8, optimalSide: 1024 },
  defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
  schedulerSet: 'standard',
  schedulerAppliesToGraph: false,
  guidanceLabel: 'CFG',
  // Use permissive display bounds while generation is blocked so loading cannot clamp persisted values.
  guidance: { min: 1, max: null },
  negativePrompt: { visible: true, usage: 'never' },
  ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
};

/** Variant-first display fallbacks prevent crashes but do not establish invocation eligibility. */
const getBaseGenerationConfig = (
  model: (Pick<GenerateModelConfig, 'base' | 'type'> & { variant?: unknown }) | undefined
): BaseGenerationConfig => {
  if (!model || model.type === 'external_image_generator') {
    return FALLBACK_GENERATION_CONFIG;
  }

  return getArchitectureGenerationConfig(model.base, model.variant) ?? FALLBACK_GENERATION_CONFIG;
};

/** Pixel-exact: hosted providers bucket requests to their own size tables, which are not grid-aligned. */
const EXTERNAL_DIMENSION_GRID = 1;

/** Use variant-aware grids; return null for undescribed local architectures. External providers have no grid. */
export const getDimensionGrid = (base: string, variant?: unknown): number | null =>
  base === 'external' ? EXTERNAL_DIMENSION_GRID : (getArchitectureFeatures(base, variant)?.dimension_grid ?? null);

/** A served architecture row does not imply frontend graph support. */
export const isArchitectureDescribed = (model: Pick<GenerateModelConfig, 'base' | 'type'>): boolean =>
  model.type === 'external_image_generator' || getArchitectureFeatures(model.base) !== undefined;

const getNumber = (value: number | null | undefined, fallback: number): number =>
  Number.isFinite(value) && value !== null && value !== undefined ? value : fallback;

/** Hold a guidance value to what the architecture's denoise node accepts. */
const clampGuidance = (value: number, guidance: BaseGenerationConfig['guidance']): number =>
  Math.min(guidance.max ?? Number.POSITIVE_INFINITY, Math.max(guidance.min, value));

/** Native PiD uses model grid × 4 and a 2048 target optimum. */
export const getGenerationDimensions = (
  // Use the variant-specific optimum, or the base row when no variant is supplied.
  model: (Pick<GenerateModelConfig, 'base' | 'type'> & { variant?: unknown }) | undefined,
  pidMode: PidMode = 'off'
) => {
  const config = getBaseGenerationConfig(model);
  const { grid, optimalSide } = getPidDimensionOverrides(
    pidMode,
    model?.base,
    model?.type === 'external_image_generator' ? EXTERNAL_DIMENSION_GRID : config.dimensions.grid,
    config.dimensions.optimalSide
  );

  return {
    grid,
    min: MIN_DIMENSION,
    max: MAX_DIMENSION,
    optimal: optimalSide,
  };
};

/** guidanceLabel selects between guidance and cfg_scale defaults. */
const getRecordGuidanceValue = (
  defaults: GenerateDefaultSettings | undefined,
  guidanceLabel: GuidanceLabel
): number | null | undefined =>
  guidanceLabel === 'Guidance'
    ? (defaults?.guidance ?? defaults?.cfg_scale)
    : (defaults?.cfg_scale ?? defaults?.guidance);

export const getGenerationDefaults = (model: GenerateModelConfig | undefined) => {
  const config = getBaseGenerationConfig(model);
  const defaults = model?.default_settings as GenerateDefaultSettings;

  return {
    cfgRescaleMultiplier: getNumber(defaults?.cfg_rescale_multiplier, 0),
    // Clamp editable model defaults so reset cannot restore blocked values.
    cfgScale: clampGuidance(
      getNumber(getRecordGuidanceValue(defaults, config.guidanceLabel), config.defaults.cfgScale),
      config.guidance
    ),
    scheduler: defaults?.scheduler ?? config.defaults.scheduler,
    steps: Math.max(1, Math.round(getNumber(defaults?.steps, config.defaults.steps))),
    vaePrecision: defaults?.vae_precision === 'fp16' ? ('fp16' as const) : ('fp32' as const),
  };
};

export const getSchedulerOptions = (
  model:
    | (Pick<GenerateModelConfig, 'base' | 'type' | 'variant'> & Partial<Pick<MainModelConfig, 'format'>>)
    | undefined,
  currentValue?: string
): readonly SchedulerOption[] => {
  const config = getBaseGenerationConfig(model);
  const options = (() => {
    switch (config.schedulerSet) {
      case 'flow':
        return FLOW_SCHEDULER_OPTIONS;
      case 'flow-no-lcm':
        return FLOW_SCHEDULER_OPTIONS_WITHOUT_LCM;
      case 'anima':
        return ANIMA_SCHEDULER_OPTIONS;
      case 'standard':
        return SCHEDULER_OPTIONS;
    }
  })();

  // Keep persisted/metadata scheduler values visible until the user picks a supported option.
  return currentValue && !options.some((option) => option.value === currentValue)
    ? [{ label: currentValue, value: currentValue }, ...options]
    : options;
};

export const coerceSchedulerForGraph = (
  model:
    | (Pick<GenerateModelConfig, 'base' | 'type' | 'variant'> & Partial<Pick<MainModelConfig, 'format'>>)
    | undefined,
  scheduler: string
): string => {
  const config = getBaseGenerationConfig(model);

  // Some graph builders ignore scheduler entirely; coerce to stable metadata/defaults instead of leaking stale UI state.
  if (!config.schedulerAppliesToGraph) {
    return config.defaults.scheduler;
  }

  switch (config.schedulerSet) {
    case 'flow':
      return FLOW_SCHEDULERS.has(scheduler) ? scheduler : 'euler';
    case 'flow-no-lcm':
      return FLOW_SCHEDULERS_WITHOUT_LCM.has(scheduler) ? scheduler : 'euler';
    case 'anima':
      return ANIMA_SCHEDULERS.has(scheduler) ? scheduler : 'euler';
    case 'standard':
      return KNOWN_SCHEDULERS.has(scheduler) ? scheduler : config.defaults.scheduler;
  }
};

export const getPromptPolicy = (
  model: GenerateModelConfig | undefined,
  settings: Pick<GenerateSettings, 'cfgScale' | 'negativePromptEnabled'>
) => {
  if (model?.type === 'external_image_generator') {
    const supportsNegativePrompt = model.capabilities?.supports_negative_prompt === true;

    return {
      negativeVisible: supportsNegativePrompt,
      negativeUsedInGraph: settings.negativePromptEnabled && supportsNegativePrompt,
    };
  }

  const config = getBaseGenerationConfig(model);
  // Visibility controls the textarea; usage controls whether the graph wires negative conditioning.
  const negativeUsedInGraph =
    settings.negativePromptEnabled &&
    (config.negativePrompt.usage === 'always' ||
      (config.negativePrompt.usage === 'cfg-gated' && settings.cfgScale > 1));

  return {
    negativeVisible: config.negativePrompt.visible,
    negativeUsedInGraph,
    ...(config.negativePrompt.usage === 'cfg-gated'
      ? { negativeHelpText: 'Used only when CFG is greater than 1.' }
      : {}),
  };
};

const HF_REPO_ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

/**
 * Whether an installed model's source is the Hugging Face repo `repoId`: the repo itself (also `repo::subfolder` or a
 * huggingface.co URL), or a local folder named like the repo, as an in-place install of a downloaded copy is.
 * Case-insensitive, as Hugging Face repo ids are.
 */
export const isModelFromRepo = (source: string | null | undefined, repoId: string): boolean => {
  if (!source) {
    return false;
  }
  const wanted = repoId.toLowerCase();
  const withoutSubfolder = source.split('::')[0]!.trim();
  const hfPath = /^https?:\/\/(?:www\.)?huggingface\.co\/([^?#]+)/i.exec(withoutSubfolder)?.[1];
  const candidate = (hfPath ? hfPath.split('/').slice(0, 2).join('/') : withoutSubfolder).replace(/[/\\]+$/, '');
  if (HF_REPO_ID.test(candidate)) {
    return candidate.toLowerCase() === wanted;
  }
  const folder = candidate.split(/[/\\]/).pop() ?? '';
  return folder.toLowerCase() === wanted.split('/').pop();
};

/** A family's released prompt enhancer, which Expand Prompt preselects for a text-only rewrite. */
interface ImagePromptEnhancer {
  /** Starter source of the enhancer model; an installed text LLM with this source is preselected. */
  modelSource: string;
  /** The starter model's name, for the hint shown when it is not installed. */
  modelName: string;
  /** Seeded public system prompt. */
  systemPromptId: string;
}

const IMAGE_PROMPT_ENHANCERS: Partial<Record<string, ImagePromptEnhancer>> = {
  // The OpenMOSS prompt-enhancer skill writes a layout draft the `moss_image_layout_prompt` node compiles.
  'moss-image': {
    modelName: 'Gemma 4 E4B (Prompt Enhancer)',
    modelSource: 'google/gemma-4-E4B-it',
    systemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0011',
  },
};

/** What Expand Prompt starts from for this image family, or null when the family ships no enhancer. */
export const getImageExpandPromptSuggestion = (
  model: GenerateModelConfig | undefined
): ExpandPromptSuggestion | null => {
  const enhancer = model?.type === 'main' ? IMAGE_PROMPT_ENHANCERS[model.base] : undefined;
  return enhancer
    ? {
        image: null,
        imageSystemPromptId: null,
        modelName: enhancer.modelName,
        modelSource: enhancer.modelSource,
        systemPromptId: enhancer.systemPromptId,
      }
    : null;
};

export const getGenerationUiPolicy = (model: GenerateModelConfig | undefined) => {
  const config = getBaseGenerationConfig(model);
  const seedVisible = model?.type === 'external_image_generator' ? model.capabilities?.supports_seed === true : true;

  return {
    guidanceLabel: config.guidanceLabel,
    guidanceMin: config.guidance.min,
    guidanceMax: config.guidance.max,
    schedulerVisible: config.schedulerAppliesToGraph,
    clipSkipMax: config.ui.clipSkipMax ?? null,
    cfgRescaleVisible: config.ui.cfgRescale,
    colorCompensationVisible: config.ui.colorCompensation,
    hiDiffusionVisible: model?.type === 'main' && (model.base === 'sd-1' || model.base === 'sdxl'),
    seamlessVisible: config.ui.seamless,
    sdVaeVisible: config.ui.sdVaeOverride,
    vaePrecisionVisible: config.ui.vaePrecision,
    seedVisible,
    // Show PiD controls on supported bases even while PiD is disabled.
    pidVisible: getIsPidSupportedBase(model?.base),
    refinerVisible: isRefinerSupported(model),
    // Both Qwen-Image text encoders take the quantization choice; no other family's encoder does.
    encoderQuantizationVisible: model?.type === 'main' && model.base === 'qwen-image',
    krea2ReferenceKvCacheVisible: model?.type === 'main' && model.base === 'krea-2',
  };
};

export const isSupportedGenerateModel = <T extends { base: string; type: string }>(
  model: T
): model is T & GenerateModelConfig =>
  (model.type === 'main' && isSupportedGenerateBase(model.base)) ||
  (model.type === 'external_image_generator' && model.base === 'external');

/**
 * Exclude unconditional Ideogram branches from selection but retain lookup support. Wan's other expert can span
 * the full schedule.
 */
export const isGenerateModelSelectable = <T extends ModelConfig>(model: T): model is T & GenerateModelConfig =>
  isSupportedGenerateModel(model) && !isIdeogram4UnconditionalBranch(model);

/** Pure prompt-history recall patch shared by keyboard and palette entry points. */
export const getPromptHistoryRecallPatch = ({
  item,
  models,
  values,
}: {
  item: PromptHistoryItem;
  models: readonly ModelConfig[] | undefined;
  values: Record<string, unknown>;
}): Record<string, unknown> | null => {
  const settings = normalizeGenerateSettings(values);

  if (!settings) {
    return null;
  }

  const selectedModel = models?.filter(isSupportedGenerateModel).find((model) => model.key === settings.modelKey);
  const promptPolicy = getPromptPolicy(selectedModel, settings);
  const patch: Record<string, unknown> = { positivePrompt: item.positivePrompt };

  if (promptPolicy.negativeVisible) {
    patch.negativePrompt = item.negativePrompt ?? '';
    patch.negativePromptEnabled = item.negativePrompt ? true : settings.negativePromptEnabled;
  }

  return patch;
};

export const EXTERNAL_PROVIDER_NODE_TYPES: Record<string, string> = {
  alibabacloud: 'alibabacloud_image_generation',
  atlascloud: 'atlascloud_image_generation',
  gemini: 'gemini_image_generation',
  openai: 'openai_image_generation',
  seedream: 'seedream_image_generation',
};

export const getExternalProviderNodeType = (providerId: unknown): string | null =>
  typeof providerId === 'string' ? (EXTERNAL_PROVIDER_NODE_TYPES[providerId] ?? null) : null;

export const getDefaultGenerateSettings = (model?: GenerateModelConfig): GenerateSettings => {
  const defaults = getGenerationDefaults(model);
  const dimensions = getGenerationDimensions(model);
  const defaultSettings = model?.default_settings as GenerateDefaultSettings;
  const width = clampDimension(getNumber(defaultSettings?.width, dimensions.optimal), dimensions.grid);
  const height = clampDimension(getNumber(defaultSettings?.height, dimensions.optimal), dimensions.grid);

  return {
    aspectRatioId: deriveAspectRatioId(width, height),
    aspectRatioIsLocked: false,
    aspectRatioValue: height > 0 ? width / height : 1,
    batchCount: 1,
    cfgRescaleMultiplier: defaults.cfgRescaleMultiplier,
    cfgScale: defaults.cfgScale,
    clipSkip: 0,
    colorCompensation: false,
    hiDiffusionEnabled: false,
    hiDiffusionRauNetEnabled: true,
    hiDiffusionWindowAttentionEnabled: true,
    hiDiffusionT1Ratio: DEFAULT_HIDIFFUSION_T1_RATIO,
    hiDiffusionT2Ratio: DEFAULT_HIDIFFUSION_T2_RATIO,
    dynamicPromptsCombinatorial: true,
    dynamicPromptsEnabled: true,
    dynamicPromptsMaxPrompts: DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS,
    dynamicPromptsSampleSeed: 0,
    dynamicPromptsResample: true,
    dynamicPromptsSeedBehaviour: 'per-iteration',
    clipEmbedModel: null,
    clipGEmbedModel: null,
    clipLEmbedModel: null,
    componentSourceModel: null,
    height,
    loras: [],
    mistralEncoderModel: null,
    modelKey: model?.key ?? '',
    negativePromptEnabled: true,
    negativePrompt: '',
    negativePromptHeightPx: 56,
    positivePrompt: '',
    positivePromptHeightPx: 96,
    expandPromptModelKey: null,
    imageToPromptModelKey: null,
    promptTemplate: null,
    promptTemplateViewMode: false,
    qwen3EncoderModel: null,
    qwenVLEncoderModel: null,
    qwen3VLEncoderModel: null,
    qwen35EncoderModel: null,
    animaInpaintAdapterModel: null,
    animaInpaintAdapterEnabled: true,
    animaInpaintAdapterWeight: DEFAULT_ANIMA_INPAINT_ADAPTER_WEIGHT,
    wanT5EncoderModel: null,
    ideogram4UnconditionalModel: null,
    wanLowNoiseModel: null,
    wanGuidanceScaleLowNoise: null,
    ideogram4SamplerPreset: DEFAULT_IDEOGRAM4_SAMPLER_PRESET,
    ideogram4Steps: null,
    ideogram4GuidanceScale: null,
    ideogram4Mu: null,
    ideogram4ColorPalette: [],
    krea2RebalanceEnabled: false,
    krea2RebalanceMultiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    krea2RebalanceWeights: DEFAULT_KREA2_REBALANCE_WEIGHTS,
    krea2SeedVarianceEnabled: false,
    krea2SeedVarianceStrength: DEFAULT_KREA2_SEED_VARIANCE_STRENGTH,
    krea2SeedVarianceRandomizePercent: DEFAULT_KREA2_SEED_VARIANCE_RANDOMIZE_PERCENT,
    krea2ReferenceKvCache: false,
    fluxDypePreset: 'off',
    fluxDypeScale: DEFAULT_FLUX_DYPE_SCALE,
    fluxDypeExponent: DEFAULT_FLUX_DYPE_EXPONENT,
    zImageShift: null,
    zImageSeedVarianceEnabled: false,
    zImageSeedVarianceStrength: DEFAULT_ZIMAGE_SEED_VARIANCE_STRENGTH,
    zImageSeedVarianceRandomizePercent: DEFAULT_ZIMAGE_SEED_VARIANCE_RANDOMIZE_PERCENT,
    qwenImageShift: null,
    qwenImageEncoderQuantization: 'none',
    mossImageUncondSkipLayers: [...DEFAULT_MOSS_IMAGE_UNCOND_SKIP_LAYERS],
    pidMode: 'off',
    pidDecoderModel: null,
    gemma2EncoderModel: null,
    pidSteps: DEFAULT_PID_STEPS,
    externalImageSize: null,
    openaiQuality: 'auto',
    openaiBackground: 'auto',
    openaiInputFidelity: null,
    geminiTemperature: null,
    geminiThinkingLevel: null,
    seedreamWatermark: false,
    seedreamOptimizePrompt: false,
    referenceImages: [],
    refinerModel: null,
    refinerSteps: DEFAULT_REFINER_STEPS,
    refinerCfgScale: DEFAULT_REFINER_CFG_SCALE,
    refinerScheduler: DEFAULT_REFINER_SCHEDULER,
    refinerPositiveAestheticScore: DEFAULT_REFINER_POSITIVE_AESTHETIC_SCORE,
    refinerNegativeAestheticScore: DEFAULT_REFINER_NEGATIVE_AESTHETIC_SCORE,
    refinerStart: DEFAULT_REFINER_START,
    scheduler: defaults.scheduler,
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: Math.floor(Math.random() * SEED_MAX),
    seedMode: 'random',
    steps: defaults.steps,
    t5EncoderModel: null,
    vae: null,
    vaePrecision: defaults.vaePrecision,
    width,
  };
};

export const getSettingsWithModelDefaults = (
  settings: GenerateSettings,
  model: GenerateModelConfig
): GenerateSettings => {
  const modelDefaults = getDefaultGenerateSettings(model);

  return {
    ...settings,
    aspectRatioId: modelDefaults.aspectRatioId,
    aspectRatioIsLocked: false,
    aspectRatioValue: modelDefaults.aspectRatioValue,
    cfgRescaleMultiplier: modelDefaults.cfgRescaleMultiplier,
    cfgScale: modelDefaults.cfgScale,
    colorCompensation: modelDefaults.colorCompensation,
    height: modelDefaults.height,
    // Concepts stay as they are: the model skips the ones it cannot load, and a compatible model runs them again.
    modelKey: model.key,
    scheduler: modelDefaults.scheduler,
    steps: modelDefaults.steps,
    vaePrecision: modelDefaults.vaePrecision,
    width: modelDefaults.width,
  };
};

export type GenerateComponentValueKey =
  | 't5EncoderModel'
  | 'clipEmbedModel'
  | 'clipLEmbedModel'
  | 'clipGEmbedModel'
  | 'mistralEncoderModel'
  | 'qwen3EncoderModel'
  | 'qwenVLEncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwen35EncoderModel'
  | 'animaInpaintAdapterModel'
  | 'wanT5EncoderModel'
  | 'wanLowNoiseModel'
  | 'ideogram4UnconditionalModel'
  | 'componentSourceModel'
  | 'pidDecoderModel'
  | 'gemma2EncoderModel'
  | 'vae';

/** The settings component slot rules read beyond the selected components themselves. */
export type ComponentPolicySettings = Pick<GenerateSettings, 'componentSourceModel' | 'pidMode'>;

export interface ComponentPolicyContext {
  model: GenerateModelConfig;
  settings: ComponentPolicySettings;
  selectedComponents: Pick<GenerateSettings, GenerateComponentValueKey>;
}

export interface ComponentSlotPolicy {
  key: GenerateComponentValueKey;
  label: string;
  modelTypes: readonly ModelTaxonomyType[];
  valueKind: 'component' | 'vae' | 'main';
  helpText?: string;
  placeholder?: string;
  defaultOpen?: boolean;
  filter?: (candidate: ModelConfig, ctx: ComponentPolicyContext) => boolean;
  required?: (ctx: ComponentPolicyContext) => boolean;
  missingMessage?: string;
  /**
   * Reported whenever the selected value fails `filter`, required or not, so a saved selection the graph cannot use
   * blocks Invoke by name instead of being dropped in favour of a component source.
   */
  incompatibleMessage?: string;
}

export interface ComponentSectionPolicy {
  defaultOpen: boolean;
  slots: readonly ComponentSlotPolicy[];
  /** Family rules beyond the slots; each slot reports its own through `getComponentSlotIssue`. */
  rules?: (ctx: ComponentPolicyContext) => string[];
}

/**
 * What a model switch does to one choice the user made. `turnedOff` keeps it and the new model just does not run it,
 * so switching back runs it again; `adjusted` moves a value into the new model's range; `removed` clears a selection
 * and `reset` returns a value to its default, both lost for good once the switch is kept.
 */
export type GenerateModelSwitchChangeKind = 'adjusted' | 'removed' | 'reset' | 'turnedOff';

export type GenerateModelSwitchSubject =
  | GenerateComponentValueKey
  | 'cfgRescale'
  | 'clipSkip'
  | 'colorCompensation'
  | 'concepts'
  | 'guidance'
  | 'hiDiffusion'
  | 'referenceImages'
  | 'seamless'
  | 'vaePrecision';

export interface GenerateModelSwitchChange {
  kind: GenerateModelSwitchChangeKind;
  subject: GenerateModelSwitchSubject;
  /** How many items of a list (concepts, reference images) it affects. */
  count?: number;
}

export interface GenerateModelSelectionResult {
  settings: GenerateSettings;
  /** Choices the switch changes, which callers confirm or report. */
  changes: readonly GenerateModelSwitchChange[];
  /** The size moved to the model's grid or provider sizes. Nothing is lost and it stays on screen, so it is reported. */
  dimensionsAdjusted: boolean;
}

/** A switch that removes or resets something cannot be taken back by switching back, so it is confirmed first. */
export const isLossyModelSwitch = (result: Pick<GenerateModelSelectionResult, 'changes'>): boolean =>
  result.changes.some((change) => change.kind === 'removed' || change.kind === 'reset');

const TYPE_CLIP_EMBED: ModelTaxonomyType[] = ['clip_embed'];
const TYPE_MAIN: ModelTaxonomyType[] = ['main'];
const TYPE_MISTRAL: ModelTaxonomyType[] = ['mistral_encoder'];
const TYPE_QWEN3: ModelTaxonomyType[] = ['qwen3_encoder'];
const TYPE_QWEN_VL: ModelTaxonomyType[] = ['qwen_vl_encoder'];
const TYPE_QWEN3_VL: ModelTaxonomyType[] = ['qwen3_vl_encoder'];
const TYPE_QWEN35: ModelTaxonomyType[] = ['qwen3_5_encoder'];
const TYPE_WAN_T5: ModelTaxonomyType[] = ['wan_t5_encoder'];
const TYPE_CONTROLNET: ModelTaxonomyType[] = ['controlnet'];
const TYPE_PID_DECODER: ModelTaxonomyType[] = ['pid_decoder'];
const TYPE_GEMMA2: ModelTaxonomyType[] = ['gemma2_encoder'];
const TYPE_T5: ModelTaxonomyType[] = ['t5_encoder'];
const TYPE_VAE: ModelTaxonomyType[] = ['vae'];

const wrapFilter = (filter: GenerateComponentFilter) => (candidate: ModelConfig) => filter(candidate);

const slot = (policy: ComponentSlotPolicy): ComponentSlotPolicy => policy;

const componentSourceSlot = (
  filter: (candidate: ModelConfig, ctx: ComponentPolicyContext) => boolean,
  helpText: string
): ComponentSlotPolicy =>
  slot({
    key: 'componentSourceModel',
    label: 'Component source',
    modelTypes: TYPE_MAIN,
    valueKind: 'main',
    helpText,
    filter,
  });

const t5EncoderSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 't5EncoderModel',
    label: 'T5 Encoder',
    modelTypes: TYPE_T5,
    valueKind: 'component',
    helpText,
    filter: (candidate) => candidate.type === 't5_encoder',
  });

const clipEmbedSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'clipEmbedModel',
    label: 'CLIP Embed',
    modelTypes: TYPE_CLIP_EMBED,
    valueKind: 'component',
    helpText,
    // FLUX.1 encodes with CLIP-L; a CLIP-G fails its text encoder.
    filter: wrapFilter(isClipVariant('large')),
  });

const qwenVlEncoderSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'qwenVLEncoderModel',
    label: 'Qwen VL Encoder',
    modelTypes: TYPE_QWEN_VL,
    valueKind: 'component',
    helpText,
    filter: (candidate) => candidate.type === 'qwen_vl_encoder',
  });

const qwen3VlEncoderSlot = (helpText: string, filter: GenerateComponentFilter): ComponentSlotPolicy =>
  slot({
    key: 'qwen3VLEncoderModel',
    label: 'Qwen3-VL Encoder',
    modelTypes: TYPE_QWEN3_VL,
    valueKind: 'component',
    helpText,
    filter,
  });

const qwen35EncoderSlot = (helpText: string, filter: GenerateComponentFilter): ComponentSlotPolicy =>
  slot({
    key: 'qwen35EncoderModel',
    label: 'Qwen3.5 Encoder',
    modelTypes: TYPE_QWEN35,
    valueKind: 'component',
    helpText,
    filter: wrapFilter(filter),
  });

/** Optional; only the canvas inpaint/outpaint graphs read it, so it is never required. */
const animaInpaintAdapterSlot = (): ComponentSlotPolicy =>
  slot({
    key: 'animaInpaintAdapterModel',
    label: 'Inpaint Adapter',
    modelTypes: TYPE_CONTROLNET,
    valueKind: 'component',
    helpText: 'Optional 4-channel ControlNet-LLLite adapter, used only when inpainting or outpainting on the Canvas.',
    placeholder: 'None',
    filter: wrapFilter(isAnimaInpaintAdapter),
  });

/** Expose PiD slots before enabling it; Z-Image uses FLUX's decoder. */
const pidDecoderSlot = (): ComponentSlotPolicy =>
  slot({
    key: 'pidDecoderModel',
    label: 'PiD Decoder',
    modelTypes: TYPE_PID_DECODER,
    valueKind: 'component',
    helpText: 'Required while PiD is on. Must match the main model’s base.',
    filter: (candidate, ctx) =>
      candidate.type === 'pid_decoder' && candidate.base === getPidDecoderBaseForMainBase(ctx.model.base),
    required: (ctx) => getIsPidActive(ctx.settings.pidMode, ctx.model.base),
    missingMessage: 'Generate needs a PiD decoder while PiD is on.',
  });

const gemma2EncoderSlot = (): ComponentSlotPolicy =>
  slot({
    key: 'gemma2EncoderModel',
    label: 'Gemma-2 Encoder',
    modelTypes: TYPE_GEMMA2,
    valueKind: 'component',
    helpText: 'Required while PiD is on. Shared by every PiD decoder.',
    filter: (candidate) => candidate.type === 'gemma2_encoder',
    required: (ctx) => getIsPidActive(ctx.settings.pidMode, ctx.model.base),
    missingMessage: 'Generate needs a Gemma-2 encoder while PiD is on.',
  });

/** The PiD slots, for splicing into a PiD-capable base's component list. */
const pidSlots = (): ComponentSlotPolicy[] => [pidDecoderSlot(), gemma2EncoderSlot()];

const wanT5EncoderSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'wanT5EncoderModel',
    label: 'Wan T5 Encoder',
    modelTypes: TYPE_WAN_T5,
    valueKind: 'component',
    helpText,
    filter: (candidate) => candidate.type === 'wan_t5_encoder',
  });

/** The low-noise expert is optional; the selected expert can span the full schedule. */
const wanLowNoiseSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'wanLowNoiseModel',
    label: 'Low-noise expert',
    modelTypes: TYPE_MAIN,
    valueKind: 'main',
    helpText,
    filter: (candidate) => candidate.type === 'main' && candidate.base === 'wan',
  });

/** Standalone Ideogram models require the second branch. */
const ideogram4UnconditionalSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'ideogram4UnconditionalModel',
    label: 'Transformer (Unconditional)',
    modelTypes: TYPE_MAIN,
    valueKind: 'main',
    helpText,
    filter: isIdeogram4UnconditionalBranch,
  });

const qwen3EncoderSlot = (helpText: string, filter?: GenerateComponentFilter): ComponentSlotPolicy =>
  slot({
    key: 'qwen3EncoderModel',
    label: 'Qwen3 Encoder',
    modelTypes: TYPE_QWEN3,
    valueKind: 'component',
    helpText,
    filter: filter
      ? (candidate) => candidate.type === 'qwen3_encoder' && filter(candidate)
      : (candidate) => candidate.type === 'qwen3_encoder',
  });

const mistralEncoderSlot = (
  helpText: string,
  filter: GenerateComponentFilter = isFlux2MistralEncoder
): ComponentSlotPolicy =>
  slot({
    key: 'mistralEncoderModel',
    label: 'Mistral Encoder',
    modelTypes: TYPE_MISTRAL,
    valueKind: 'component',
    helpText,
    filter: wrapFilter(filter),
  });

const clipVariantSlot = (
  key: 'clipLEmbedModel' | 'clipGEmbedModel',
  label: string,
  variant: string,
  helpText: string
) =>
  slot({
    key,
    label,
    modelTypes: TYPE_CLIP_EMBED,
    valueKind: 'component',
    helpText,
    filter: wrapFilter(isClipVariant(variant)),
  });

const vaeSlot = (helpText: string, filter: GenerateComponentFilter): ComponentSlotPolicy =>
  slot({ key: 'vae', label: 'VAE', modelTypes: TYPE_VAE, valueKind: 'vae', helpText, filter: wrapFilter(filter) });

const hasCompatibleComponentSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  Boolean(getCompatibleDiffusersComponentSource(ctx.model, ctx.settings.componentSourceModel));

type ComponentSourceCandidate = {
  base: string;
  key: string;
  name: string;
  type: string;
  format?: string;
  variant?: unknown;
};

const isFlux2DiffusersSource = (source: ComponentSourceCandidate | null | undefined): source is MainModelConfig =>
  Boolean(source && isBundledMainForBase('flux2')(source));

const hasFlux2DiffusersVaeSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledMainForBase('flux2')(ctx.model) || isFlux2DiffusersSource(ctx.settings.componentSourceModel));

const hasFlux2DiffusersEncoderSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledMainForBase('flux2')(ctx.model) ||
    Boolean(
      ctx.settings.componentSourceModel && isFlux2DiffusersSourceForModel(ctx.model)(ctx.settings.componentSourceModel)
    ));

export const getFlux2DiffusersComponentSource = (
  model: MainModelConfig,
  settings: GenerateSettings
): MainModelConfig | undefined => {
  const source = settings.componentSourceModel;

  if (!isFlux2DiffusersSource(source)) {
    return undefined;
  }

  const hasStandaloneEncoder = model.variant === 'dev' ? settings.mistralEncoderModel : settings.qwen3EncoderModel;

  if (hasStandaloneEncoder) {
    return source;
  }

  return isFlux2DiffusersSourceForModel(model)(source) ? source : undefined;
};

export const getAutoFlux2ComponentSourceModel = (
  model: GenerateModelConfig | undefined,
  settings: Pick<GenerateSettings, 'mistralEncoderModel' | 'qwen3EncoderModel' | 'vae'>,
  models: readonly ComponentSourceCandidate[]
): MainModelConfig | null | undefined => {
  if (!model || model.type === 'external_image_generator' || model.base !== 'flux2') {
    return undefined;
  }

  const encoderModel = model.variant === 'dev' ? settings.mistralEncoderModel : settings.qwen3EncoderModel;

  if (isBundledMainForBase('flux2')(model) || (encoderModel && settings.vae)) {
    return null;
  }

  const diffusersModels = models.filter(isFlux2DiffusersSource);
  const variantMatch = diffusersModels.find(isFlux2DiffusersSourceForModel(model));

  if (!encoderModel && settings.vae) {
    return variantMatch ?? null;
  }

  return variantMatch ?? diffusersModels[0] ?? null;
};

// Split/quantized families can satisfy required encoder/VAE slots from a bundled Diffusers source.
const isBundledOrDiffusersSourceSatisfied = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledMainForBase(ctx.model.base)(ctx.model) || hasCompatibleComponentSource(ctx));

/**
 * What blocks Invoke on one slot, or null. Shared by the picker, invocation readiness and graph preflight so picker
 * requirements cannot drift.
 */
export const getComponentSlotIssue = (slotPolicy: ComponentSlotPolicy, ctx: ComponentPolicyContext): string | null => {
  const value = ctx.selectedComponents[slotPolicy.key];
  const isValid = value && (!slotPolicy.filter || slotPolicy.filter(value as ModelConfig, ctx));

  if (value && !isValid && slotPolicy.incompatibleMessage) {
    return slotPolicy.incompatibleMessage;
  }

  if (!slotPolicy.required?.(ctx)) {
    return null;
  }

  return isValid ? null : (slotPolicy.missingMessage ?? `Generate needs a ${slotPolicy.label} for this model.`);
};

const createPolicy = (defaultOpen: boolean, slots: readonly ComponentSlotPolicy[]): ComponentSectionPolicy => ({
  defaultOpen,
  slots,
});

const EMPTY_COMPONENT_POLICY: ComponentSectionPolicy = createPolicy(false, []);

const getBaseComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  _settings: ComponentPolicySettings
): ComponentSectionPolicy => {
  if (!model || model.type === 'external_image_generator') {
    return EMPTY_COMPONENT_POLICY;
  }

  // The graph builder sends a VAE by `isVaeCompatibleWithGenerateModel`, which reads the same row.
  const isAcceptedVae = isVaeAcceptedByBase(model.base, model.variant);

  switch (model.base) {
    case 'flux':
      return {
        ...createPolicy(!isSelfContainedSDNQFlux1Pipeline(model), [
          {
            ...t5EncoderSlot('Required for FLUX.1 models.'),
            required: (ctx) => !isSelfContainedSDNQFlux1Pipeline(ctx.model),
            missingMessage: 'Generate needs a T5 Encoder for FLUX models.',
          },
          {
            ...clipEmbedSlot('Required for FLUX.1 models.'),
            required: (ctx) => !isSelfContainedSDNQFlux1Pipeline(ctx.model),
            missingMessage: 'Generate needs a CLIP Embed model for FLUX models.',
          },
          {
            ...vaeSlot('Required for FLUX.1 models.', isAcceptedVae),
            required: (ctx) => !isSelfContainedSDNQFlux1Pipeline(ctx.model),
            missingMessage: 'Generate needs a VAE for FLUX models.',
          },
        ]),
        rules: (ctx) =>
          ctx.model.type !== 'external_image_generator' && ctx.model.variant === 'dev_fill'
            ? ['FLUX Fill models do not support text-to-image generation.']
            : [],
      };
    case 'flux2': {
      const encoderSlot: ComponentSlotPolicy =
        model.variant === 'dev'
          ? {
              ...mistralEncoderSlot(
                'Optional override; otherwise a compatible installed FLUX.2 [dev] Diffusers model is used.'
              ),
              required: (ctx) => !hasFlux2DiffusersEncoderSource(ctx),
              missingMessage: 'Generate needs a Mistral Encoder for non-Diffusers FLUX.2 [dev] models.',
            }
          : {
              ...qwen3EncoderSlot(
                'Optional override; otherwise a compatible installed FLUX.2 Diffusers model is used.'
              ),
              filter: (candidate, ctx) =>
                ctx.model.type !== 'external_image_generator' && isFlux2Qwen3EncoderForModel(ctx.model)(candidate),
              required: (ctx) => !hasFlux2DiffusersEncoderSource(ctx),
              missingMessage: 'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.',
            };

      return createPolicy(!isSelfContainedSDNQPipeline(model) && model.format !== 'diffusers', [
        encoderSlot,
        {
          ...vaeSlot('Optional override; otherwise an installed FLUX.2 Diffusers model is used.', isAcceptedVae),
          required: (ctx) => !hasFlux2DiffusersVaeSource(ctx),
          missingMessage: 'Generate needs a VAE for non-Diffusers FLUX.2 models.',
        },
      ]);
    }
    case 'sd-3':
      return createPolicy(false, [
        t5EncoderSlot('Optional override; the main model is used when omitted.'),
        clipVariantSlot('clipLEmbedModel', 'CLIP L', 'large', 'Optional CLIP-L override.'),
        clipVariantSlot('clipGEmbedModel', 'CLIP G', 'gigantic', 'Optional CLIP-G override.'),
        vaeSlot('Optional VAE override.', isAcceptedVae),
      ]);
    case 'qwen-image':
      return createPolicy(model.format !== 'diffusers', [
        componentSourceSlot(
          (candidate) => isDiffusersMainForBase('qwen-image')(candidate),
          'For non-Diffusers Qwen Image models, select this or provide separate VAE and Qwen VL models.'
        ),
        {
          ...qwenVlEncoderSlot('Optional override, or required with a non-Diffusers model and no component source.'),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Qwen VL Encoder for non-Diffusers Qwen Image models.',
        },
        {
          ...vaeSlot(
            'The same VAE may be installed under the Qwen-Image or Anima base, so both are listed. Optional override, or required with a non-Diffusers model and no component source.',
            isAcceptedVae
          ),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a VAE for non-Diffusers Qwen Image models.',
        },
      ]);
    case 'z-image':
      return createPolicy(!isBundledMainForBase('z-image')(model), [
        componentSourceSlot(
          (candidate) => isBundledMainForBase('z-image')(candidate),
          'Select a bundled Z-Image pipeline to provide VAE and Qwen3 components.'
        ),
        {
          ...qwen3EncoderSlot(
            'Z-Image conditions on the Qwen3 4B encoder, not the 8B one FLUX.2 Klein 9B uses. Required unless a ' +
              'Diffusers component source is available.',
            isZImageQwen3Encoder
          ),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Qwen3 4B Encoder for Z-Image models.',
          incompatibleMessage:
            'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on. Select a Qwen3 4B Encoder or clear it.',
        },
        {
          ...vaeSlot(
            'Z-Image decodes with the FLUX VAE, so FLUX-base VAEs are listed here — they are fully compatible. Required unless a Diffusers component source is available.',
            isAcceptedVae
          ),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a VAE for Z-Image models.',
        },
      ]);
    case 'flux3':
      // A Diffusers FLUX.3 model bundles its VAE and encoder; single-file and GGUF models need both picked.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...vaeSlot(
            'FLUX.3 decodes with the FLUX.2 VAE. Required for non-Diffusers FLUX.3 models; otherwise an optional override.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers FLUX.3 models.',
        },
        {
          ...qwen3VlEncoderSlot(
            'The Qwen3-VL 4B encoder. Required for non-Diffusers FLUX.3 models; otherwise an optional override.',
            isQwen3Vl4bEncoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL Encoder for non-Diffusers FLUX.3 models.',
        },
      ]);
    case 'krea-2':
      // Bundled Krea models supply components; standalone models need a separate encoder and VAE.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...vaeSlot(
            'Krea-2 decodes with the Qwen-Image 16-channel VAE; the same VAE may be installed under the Qwen-Image or Anima base, so both are listed. Required for non-Diffusers Krea-2 models.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers Krea-2 models.',
        },
        {
          ...qwen3VlEncoderSlot('Required for non-Diffusers Krea-2 models.', isQwen3Vl4bEncoder),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL Encoder for non-Diffusers Krea-2 models.',
        },
      ]);
    case 'qwen-image-2-1':
      // A Diffusers pipeline brings its own VAE and encoder; single-file and GGUF transformers need both.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...vaeSlot(
            'Qwen-Image-2.1 decodes with its own 64-channel RGBA VAE. Required for non-Diffusers Qwen-Image-2.1 models.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers Qwen-Image-2.1 models.',
        },
        {
          ...qwen3VlEncoderSlot(
            'Qwen-Image-2.1 conditions on the Qwen3-VL 8B encoder, not the 4B one Krea-2 uses. Required for ' +
              "non-Diffusers Qwen-Image-2.1 models. On a Diffusers model it replaces the model's own encoder, the " +
              'only one that reads reference images.',
            isQwen3Vl8bEncoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL 8B Encoder for non-Diffusers Qwen-Image-2.1 models.',
        },
      ]);
    case 'ernie-image':
      // Bundled ERNIE models supply components; standalone models need explicit components.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...mistralEncoderSlot(
            'ERNIE-Image conditions on Ministral 3B, a different architecture from the Mistral Small 3 encoder FLUX.2 uses. Required for non-Diffusers ERNIE-Image models.',
            isErnieImageMistralEncoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Ministral 3B encoder for non-Diffusers ERNIE-Image models.',
        },
        {
          ...vaeSlot(
            'ERNIE-Image decodes with the FLUX.2 32-channel VAE, which is the base it installs under. Required for non-Diffusers ERNIE-Image models.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers ERNIE-Image models.',
        },
      ]);
    case 'moss-image':
      // A Diffusers pipeline brings its encoder and VAE; a single-file transformer needs both.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...qwen35EncoderSlot(
            'MOSS-Image conditions on the Qwen3.5 9B encoder, not the 4B one Anima-3.8B uses. Required for ' +
              'non-Diffusers MOSS-Image models.',
            isMossImageQwen35Encoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3.5 9B Encoder for non-Diffusers MOSS-Image models.',
        },
        {
          ...vaeSlot(
            'MOSS-Image decodes with the 32-channel FLUX.2 VAE, which is the base it installs under. Required for ' +
              'non-Diffusers MOSS-Image models.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers MOSS-Image models.',
        },
      ]);
    case 'ideogram-4':
      // Standalone Ideogram needs the other branch, encoder, and VAE.
      return createPolicy(model.format !== 'diffusers', [
        {
          ...ideogram4UnconditionalSlot(
            'The second transformer branch. Required for single-file Ideogram 4 models, which hold ' +
              'only the conditional branch.'
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs the unconditional transformer for single-file Ideogram 4 models.',
        },
        {
          ...qwen3VlEncoderSlot(
            'Ideogram 4 conditions on the Qwen3-VL 8B encoder, not the 4B one Krea-2 uses. Required ' +
              'for non-Diffusers Ideogram 4 models.',
            isQwen3Vl8bEncoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL 8B Encoder for non-Diffusers Ideogram 4 models.',
        },
        {
          ...vaeSlot(
            'Ideogram 4 decodes with the 32-channel FLUX.2 VAE, which is the base it installs under. ' +
              'Required for non-Diffusers Ideogram 4 models.',
            isAcceptedVae
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a VAE for non-Diffusers Ideogram 4 models.',
        },
      ]);
    case 'wan':
      // GGUF Wan needs external component sources.
      return createPolicy(model.format !== 'diffusers', [
        componentSourceSlot(
          (candidate) => isDiffusersMainForBase('wan')(candidate),
          'Select a Diffusers Wan model to provide VAE and text-encoder components.'
        ),
        {
          ...vaeSlot('Required unless a Diffusers component source is available.', isAcceptedVae),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a VAE for Wan models.',
        },
        {
          ...wanT5EncoderSlot('Required unless a Diffusers component source is available.'),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Wan T5 Encoder for Wan models.',
        },
        wanLowNoiseSlot('Optional second A14B expert. Without it the high-noise expert runs the whole schedule.'),
      ]);
    case 'anima':
      return createPolicy(true, [
        {
          ...qwen3EncoderSlot('Required for Anima models.', isAnimaQwen3Encoder),
          required: () => true,
          missingMessage: 'Generate needs a Qwen3 Encoder for Anima models.',
        },
        // Anima-3.8B reads a second encoder through its bundled semantic connector; no other Anima does.
        ...(model.variant === 'anima_qwen35'
          ? [
              {
                ...qwen35EncoderSlot(
                  'Anima-3.8B reads Qwen3.5 4B beside Qwen3 0.6B. Required for this model.',
                  isAnimaQwen35Encoder
                ),
                required: () => true,
                missingMessage: 'Generate needs a Qwen3.5 Encoder for Anima-3.8B.',
              },
            ]
          : []),
        {
          ...vaeSlot(
            'Anima decodes with the 16-channel Wan 2.1 VAE, which may be installed under the Anima, Qwen-Image, or Wan base, so all three are listed. Required for Anima models.',
            isAcceptedVae
          ),
          required: () => true,
          missingMessage: 'Generate needs a VAE for Anima models.',
        },
        animaInpaintAdapterSlot(),
      ]);
    default:
      return EMPTY_COMPONENT_POLICY;
  }
};

/** Add the PiD slots; each base policy keeps its other slots and rules. */
export const getComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  settings: ComponentPolicySettings
): ComponentSectionPolicy => {
  const policy = getBaseComponentSectionPolicy(model, settings);

  if (!model || model.type === 'external_image_generator' || !getIsPidSupportedBase(model.base)) {
    return policy;
  }

  const slots = [...policy.slots, ...pidSlots()];

  return { ...policy, slots };
};

export const getComponentPolicyContext = (
  model: GenerateModelConfig,
  settings: GenerateSettings
): ComponentPolicyContext => ({
  model,
  settings,
  selectedComponents: {
    clipEmbedModel: settings.clipEmbedModel,
    clipGEmbedModel: settings.clipGEmbedModel,
    clipLEmbedModel: settings.clipLEmbedModel,
    componentSourceModel: settings.componentSourceModel,
    mistralEncoderModel: settings.mistralEncoderModel,
    qwen3EncoderModel: settings.qwen3EncoderModel,
    qwenVLEncoderModel: settings.qwenVLEncoderModel,
    qwen3VLEncoderModel: settings.qwen3VLEncoderModel,
    qwen35EncoderModel: settings.qwen35EncoderModel,
    animaInpaintAdapterModel: settings.animaInpaintAdapterModel,
    wanT5EncoderModel: settings.wanT5EncoderModel,
    wanLowNoiseModel: settings.wanLowNoiseModel,
    ideogram4UnconditionalModel: settings.ideogram4UnconditionalModel,
    t5EncoderModel: settings.t5EncoderModel,
    pidDecoderModel: settings.pidDecoderModel,
    gemma2EncoderModel: settings.gemma2EncoderModel,
    vae: settings.vae,
  },
});

/** English names, as `widgets.generate.switchSubjects` words them, for the issue text the locale nests them into. */
const COMPONENT_SETTING_LABELS: Record<GenerateComponentValueKey, string> = {
  clipEmbedModel: 'CLIP Embed',
  clipGEmbedModel: 'CLIP G',
  clipLEmbedModel: 'CLIP L',
  mistralEncoderModel: 'Mistral encoder',
  qwen3EncoderModel: 'Qwen3 encoder',
  qwenVLEncoderModel: 'Qwen VL encoder',
  qwen3VLEncoderModel: 'Qwen3-VL encoder',
  qwen35EncoderModel: 'Qwen3.5 encoder',
  animaInpaintAdapterModel: 'Inpaint adapter',
  wanT5EncoderModel: 'Wan T5 encoder',
  wanLowNoiseModel: 'Low-noise expert',
  ideogram4UnconditionalModel: 'Unconditional transformer',
  t5EncoderModel: 'T5 encoder',
  pidDecoderModel: 'PiD decoder',
  gemma2EncoderModel: 'Gemma-2 encoder',
  vae: 'VAE',
  componentSourceModel: 'Component source',
};

/** Every settings key that holds a component model. */
export const GENERATE_COMPONENT_VALUE_KEYS = Object.keys(COMPONENT_SETTING_LABELS) as GenerateComponentValueKey[];

const isSelectedComponentCompatible = (
  slotPolicy: ComponentSlotPolicy | undefined,
  model: GenerateModelConfig,
  settings: GenerateSettings,
  key: GenerateComponentValueKey
): boolean => {
  const value = settings[key];

  if (!value) {
    return true;
  }

  if (key === 'componentSourceModel' && model.type !== 'external_image_generator' && model.base === 'flux2') {
    return Boolean(getFlux2DiffusersComponentSource(model, settings));
  }

  if (!slotPolicy) {
    return false;
  }

  return !slotPolicy.filter || slotPolicy.filter(value as ModelConfig, getComponentPolicyContext(model, settings));
};

/** Whether `candidate` may fill `key` for `model`, by the rule the model-selection transition clears with. */
export const isComponentCompatibleWithModel = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  key: GenerateComponentValueKey,
  candidate: ModelConfig
): boolean => {
  const nextSettings = { ...settings, [key]: candidate };
  const slotPolicy = getComponentSectionPolicy(model, nextSettings).slots.find((slot) => slot.key === key);

  return isSelectedComponentCompatible(slotPolicy, model, nextSettings, key);
};

type ReferenceModelCandidate = { base: string; key: string; name: string; type: string };

const isFluxKontextModel = (model: GenerateModelConfig | undefined): model is MainModelConfig =>
  Boolean(
    model &&
    model.type !== 'external_image_generator' &&
    model.base === 'flux' &&
    model.name.toLowerCase().includes('kontext')
  );

/**
 * Families whose Generate graph loads LoRAs. The backend installs no LoRAs for the others, so their concept picker
 * could only ever be empty; graph coverage holds the builders to this list.
 */
const LORA_SUPPORTED_BASES: ReadonlySet<string> = new Set<SupportedGenerateBase>([
  'anima',
  'flux',
  'flux2',
  'krea-2',
  'qwen-image',
  'qwen-image-2-1',
  'sd-1',
  'sd-2',
  'sdxl',
  'wan',
  'z-image',
]);

export const isLoraSupported = (model: GenerateModelConfig | undefined): boolean =>
  Boolean(model && model.type !== 'external_image_generator' && LORA_SUPPORTED_BASES.has(model.base));

/** Picker and validation must agree on compatibility, including required variants. */
export const isReferenceImageSupported = (model: GenerateModelConfig | undefined): boolean => {
  if (!model) {
    return false;
  }

  if (model.type === 'external_image_generator') {
    return model.capabilities?.supports_reference_images === true;
  }

  const features = getArchitectureFeatures(model.base);

  if (!features || features.max_reference_images <= 0) {
    return false;
  }

  // The backend declares which reference-image features require a specific variant or format.
  return (
    (features.reference_images_require_variant === null ||
      model.variant === features.reference_images_require_variant) &&
    (features.reference_images_require_format === null || model.format === features.reference_images_require_format)
  );
};

/**
 * The model format the selected base needs for reference images, when the model's format is all that keeps them
 * unavailable, so the UI can say what to switch to instead of hiding the section without a reason.
 */
export const getMissingReferenceImageFormat = (model: GenerateModelConfig | undefined): string | null => {
  if (!model || model.type === 'external_image_generator' || isReferenceImageSupported(model)) {
    return null;
  }

  const features = getArchitectureFeatures(model.base);

  if (
    !features ||
    features.max_reference_images <= 0 ||
    (features.reference_images_require_variant !== null && model.variant !== features.reference_images_require_variant)
  ) {
    return null;
  }

  return features.reference_images_require_format;
};

export const getMaxReferenceImages = (model: GenerateModelConfig | undefined): number => {
  if (!model || !isReferenceImageSupported(model)) {
    return 0;
  }

  if (model.type === 'external_image_generator' && typeof model.capabilities?.max_reference_images === 'number') {
    return Math.max(0, model.capabilities.max_reference_images);
  }

  return getArchitectureFeatures(model.base)?.max_reference_images ?? DEFAULT_REFERENCE_IMAGE_LIMIT;
};

const IP_ADAPTER_BASES: ReadonlySet<string> = new Set(['sd-1', 'sdxl', 'flux', 'anima']);

/**
 * Anima's IP-Adapter node takes a strength alone: it has no method choice and brings its own SigLIP2 encoder, so the
 * UI hides those controls and the graph never sends them.
 */
export const isStrengthOnlyIPAdapterBase = (base: string | undefined): boolean => base === 'anima';

export const createReferenceImageId = (): string =>
  `reference_image_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

/** Edit references switch Krea-2 into edit mode, which excludes style references and regional prompts. */
export const hasEnabledKrea2EditReferences = (referenceImages: readonly GenerateReferenceImage[]): boolean =>
  referenceImages.some(
    (referenceImage) => referenceImage.isEnabled && referenceImage.config.type === 'krea2_edit_reference_image'
  );

/**
 * Whether running Krea-2 edit references have an active LoRA to read them. The base model is no instruction editor:
 * without an edit LoRA it copies the references, and the graph leaves the reference K/V cache, which only such a LoRA
 * reads, at the denoise default. A LoRA counts as the graph counts it: enabled and compatible with the model.
 */
export const getKrea2EditLoraState = (
  model: GenerateModelConfig | undefined,
  settings: Pick<GenerateSettings, 'loras' | 'referenceImages'>
): 'noEditReferences' | 'noLora' | 'ready' => {
  if (model?.base !== 'krea-2' || !hasEnabledKrea2EditReferences(settings.referenceImages)) {
    return 'noEditReferences';
  }

  return settings.loras.some((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, model))
    ? 'ready'
    : 'noLora';
};

/** A Krea-2 style reference at zero strength is bypassed, so it neither runs nor counts against the others. */
const isReferenceImageActive = (referenceImage: GenerateReferenceImage): boolean =>
  referenceImage.isEnabled &&
  !(referenceImage.config.type === 'krea2_reference_image' && referenceImage.config.styleStrength <= 0);

/** The family's one reference kind; Krea-2's choice between style and edit is `getDefaultReferenceImageConfigs`'s. */
const getFamilyDefaultReferenceImageConfig = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  image: GenerateReferenceImageAsset | null
): GenerateReferenceImageConfig => {
  const modelBase: string | undefined = model?.base;

  if (model?.type === 'external_image_generator') {
    return { image, type: 'external_reference_image' };
  }

  // FLUX.3 takes its references exactly as FLUX.2 does: plain images, VAE-encoded by the denoiser.
  if (modelBase === 'flux2' || modelBase === 'flux3') {
    return { image, type: 'flux2_reference_image' };
  }

  if (modelBase === 'qwen-image') {
    return { image, type: 'qwen_image_reference_image' };
  }

  if (modelBase === 'qwen-image-2-1') {
    return { image, type: 'qwen_image_2_1_reference_image' };
  }

  if (isFluxKontextModel(model)) {
    return { image, model, type: 'flux_kontext_reference_image' };
  }

  const adapterModel =
    models.find((candidate) => candidate.type === 'ip_adapter' && candidate.base === modelBase) ?? null;

  return {
    beginEndStepPct: [0, 1],
    clipVisionModel: modelBase === 'flux' ? 'ViT-L' : 'ViT-H',
    image,
    method: 'full',
    model: adapterModel,
    type: 'ip_adapter',
    weight: 1,
  };
};

const getReferenceImageConfigSupported = (
  model: GenerateModelConfig,
  referenceImage: GenerateReferenceImage
): boolean => {
  switch (referenceImage.config.type) {
    case 'external_reference_image':
      return model.type === 'external_image_generator' && model.capabilities?.supports_reference_images === true;
    case 'flux2_reference_image':
      return model.type !== 'external_image_generator' && (model.base === 'flux2' || model.base === 'flux3');
    case 'qwen_image_reference_image':
      return model.type !== 'external_image_generator' && model.base === 'qwen-image' && model.variant === 'edit';
    case 'krea2_edit_reference_image':
    case 'krea2_reference_image':
      return model.type !== 'external_image_generator' && model.base === 'krea-2';
    case 'qwen_image_2_1_reference_image':
      return (
        model.type !== 'external_image_generator' && model.base === 'qwen-image-2-1' && isReferenceImageSupported(model)
      );
    case 'flux_kontext_reference_image':
      return isFluxKontextModel(model);
    case 'flux_redux':
      return model.type !== 'external_image_generator' && model.base === 'flux';
    case 'ip_adapter':
      // `isReferenceImageSupported` applies the backend's variant gate: Anima only on its Qwen3 releases.
      return IP_ADAPTER_BASES.has(model.base) && isReferenceImageSupported(model);
  }
};

export const isReferenceImageCompatibleWithModel = (
  model: GenerateModelConfig,
  referenceImage: GenerateReferenceImage
): boolean => {
  if (!getReferenceImageConfigSupported(model, referenceImage)) {
    return false;
  }

  const config = referenceImage.config;

  if ((config.type === 'ip_adapter' || config.type === 'flux_redux') && config.model) {
    return config.model.base === model.base;
  }

  if (config.type === 'flux_kontext_reference_image' && config.model) {
    return config.model.key === model.key;
  }

  return true;
};

/** A new or re-targeted reference, typed for the model; existing references keep theirs. */
interface ReferenceImageAddition {
  image: GenerateReferenceImageAsset | null;
  isEnabled: boolean;
}

/**
 * Configs for `additions` joining `referenceImages`. A Krea-2 addition is the style reference only when it would be
 * the sole running Krea-2 reference, counting the other additions and itself as switched on; arriving with or beside
 * others, it is an edit reference.
 */
const getDefaultReferenceImageConfigs = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  additions: readonly ReferenceImageAddition[],
  referenceImages: readonly GenerateReferenceImage[]
): GenerateReferenceImageConfig[] => {
  if (model?.type !== 'main' || model.base !== 'krea-2') {
    return additions.map(({ image }) => getFamilyDefaultReferenceImageConfig(model, models, image));
  }

  const runningElsewhere = referenceImages.filter(isReferenceImageActive).length;
  const enabledAdditions = additions.filter((addition) => addition.isEnabled).length;

  return additions.map(({ image, isEnabled }) =>
    runningElsewhere + enabledAdditions + (isEnabled ? 0 : 1) === 1
      ? { image, styleStrength: DEFAULT_KREA2_STYLE_STRENGTH, type: 'krea2_reference_image' }
      : { image, type: 'krea2_edit_reference_image' }
  );
};

/** Within the model's limit; Krea-2 runs either one style reference or a set of edit references, never both. */
const canRunWithEnabledReferenceImages = (
  model: GenerateModelConfig,
  enabled: readonly GenerateReferenceImage[],
  referenceImage: GenerateReferenceImage
): boolean => {
  if (enabled.length >= getMaxReferenceImages(model)) {
    return false;
  }

  if (model.base !== 'krea-2') {
    return true;
  }

  return referenceImage.config.type === 'krea2_reference_image'
    ? enabled.length === 0
    : enabled.every((kept) => kept.config.type !== 'krea2_reference_image');
};

/**
 * The list's one running reference when it is a Krea-2 style reference still at its default strength, which an
 * addition turns into an edit reference.
 */
const getLoneDefaultStyleReference = (
  referenceImages: readonly GenerateReferenceImage[]
): GenerateReferenceImage | null => {
  const active = referenceImages.filter(isReferenceImageActive);
  const [lone] = active;

  return active.length === 1 &&
    lone?.config.type === 'krea2_reference_image' &&
    lone.config.styleStrength === DEFAULT_KREA2_STYLE_STRENGTH
    ? lone
    : null;
};

/**
 * `referenceImages` with `additions` appended as enabled references typed for the model; every way of adding a
 * reference goes through here. A lone first Krea-2 reference starts as the style reference, so an edit set built one
 * image at a time begins with one: the next addition turns it into an edit reference too, unless its strength was
 * changed, which marks it as meant for style. Validation then reports the mix.
 */
export const withAddedReferenceImages = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  referenceImages: GenerateReferenceImage[],
  additions: readonly { id: string; image: GenerateReferenceImageAsset | null }[]
): GenerateReferenceImage[] => {
  if (additions.length === 0) {
    return referenceImages;
  }

  const lone = model?.type === 'main' && model.base === 'krea-2' ? getLoneDefaultStyleReference(referenceImages) : null;
  const kept = lone
    ? referenceImages.map((referenceImage): GenerateReferenceImage =>
        referenceImage === lone
          ? { ...referenceImage, config: { image: referenceImage.config.image, type: 'krea2_edit_reference_image' } }
          : referenceImage
      )
    : referenceImages;
  const configs = getDefaultReferenceImageConfigs(
    model,
    models,
    additions.map(({ image }) => ({ image, isEnabled: true })),
    kept
  );

  return [...kept, ...additions.map(({ id }, index) => ({ config: configs[index]!, id, isEnabled: true }))];
};

/**
 * An SD reference that never had an adapter (model null) passes the base check, so a switch to a strength-only base
 * would keep its unset adapter and its SD weight, which may be negative. Re-target it when that leaves it unusable and
 * re-targeting can help: a weight the node rejects, or no adapter while one is installed. Without an installed adapter
 * the re-targeted config is stable, so a repeated switch changes nothing.
 */
const needsStrengthOnlyRetarget = (
  model: GenerateModelConfig,
  referenceImage: GenerateReferenceImage,
  models: readonly ReferenceModelCandidate[]
): boolean => {
  const { config } = referenceImage;

  if (config.type !== 'ip_adapter' || !isStrengthOnlyIPAdapterBase(model.base)) {
    return false;
  }

  if (!(config.weight >= 0 && config.weight <= 2)) {
    return true;
  }

  return (
    config.model === null &&
    models.some((candidate) => candidate.type === 'ip_adapter' && candidate.base === model.base)
  );
};

/**
 * Drop unsupported references; retarget incompatible ones, and strength-only IP-Adapter references left unusable,
 * while preserving image/enabled state. The re-targeted ones are typed together, beside the others; a strength-only
 * base then keeps their step window, same-base adapter and strength. No changes return the same array.
 */
export const getCompatibleReferenceImages = (
  referenceImages: GenerateReferenceImage[],
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[]
): GenerateReferenceImage[] => {
  if (referenceImages.length === 0) {
    return referenceImages;
  }

  if (!isReferenceImageSupported(model)) {
    return [];
  }

  const isKept = referenceImages.map(
    (referenceImage) =>
      isReferenceImageCompatibleWithModel(model, referenceImage) &&
      !needsStrengthOnlyRetarget(model, referenceImage, models)
  );
  const retargeted = referenceImages.filter((_, index) => !isKept[index]);

  if (retargeted.length === 0) {
    return referenceImages;
  }

  const configs = getDefaultReferenceImageConfigs(
    model,
    models,
    retargeted.map(({ config, isEnabled }) => ({ image: config.image, isEnabled })),
    referenceImages.filter((_, index) => isKept[index])
  );
  let retargetIndex = 0;

  return referenceImages.map((referenceImage, index) => {
    if (isKept[index]) {
      return referenceImage;
    }

    const { config } = referenceImage;
    const defaults = configs[retargetIndex++]!;

    // A strength-only base keeps the user's step window and strength; only the adapter and the hidden SD fields reset.
    if (config.type === 'ip_adapter' && defaults.type === 'ip_adapter' && isStrengthOnlyIPAdapterBase(model.base)) {
      // Clamping a negative SD weight to 0 would switch the reference off, so it takes the default strength instead.
      const weight =
        Number.isFinite(config.weight) && config.weight >= 0 ? Math.min(config.weight, 2) : defaults.weight;
      return {
        ...referenceImage,
        config: {
          ...defaults,
          beginEndStepPct: config.beginEndStepPct,
          model: config.model?.base === model.base ? config.model : defaults.model,
          weight,
        },
      };
    }

    return { ...referenceImage, config: defaults };
  });
};

const getSettingsWithCompatibleModelSelections = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[],
  isCanvasFrame: boolean
): GenerateModelSelectionResult => {
  const nextSettings: GenerateSettings = { ...settings, modelKey: model.key };
  const changes: GenerateModelSwitchChange[] = [];
  const change = (subject: GenerateModelSwitchSubject, kind: GenerateModelSwitchChangeKind, count?: number) =>
    changes.push(count === undefined ? { kind, subject } : { count, kind, subject });
  const dimensions = getGenerationDimensions(model);
  const width = clampDimension(nextSettings.width, dimensions.grid);
  const height = clampDimension(nextSettings.height, dimensions.grid);

  if (width !== nextSettings.width || height !== nextSettings.height) {
    nextSettings.width = width;
    nextSettings.height = height;
    nextSettings.aspectRatioId = deriveAspectRatioId(width, height);
    nextSettings.aspectRatioValue = height > 0 ? width / height : 1;
  }

  if (settings.modelKey !== model.key) {
    const sizes = getExternalResolutionPresets(model);
    const supportedSizes = sizes.length > 0 ? sizes : getExternalAspectRatioSizes(model).map(([, size]) => size);
    const matchesSize = supportedSizes.some((size) => size.width === width && size.height === height);

    // The Canvas frame is kept: the Canvas asks a provider for its own size nearest the frame and restores the result.
    if (!matchesSize && supportedSizes.length > 0 && !isCanvasFrame) {
      // Keep the frame's shape when possible, then its pixel budget within that shape.
      const distance = (size: { width: number; height: number }) => ({
        area: Math.abs(Math.log((size.width * size.height) / (width * height))),
        ratio: Math.abs(Math.log(size.width / size.height / (width / height))),
      });
      // Rounded presets of one nominal ratio differ by ~0.02 in log ratio; distinct ratios by 0.1 or more.
      const RATIO_TOLERANCE = 0.05;
      const selected = supportedSizes.reduce((best, size) => {
        const candidateDistance = distance(size);
        const bestDistance = distance(best);
        const sameRatio = Math.abs(candidateDistance.ratio - bestDistance.ratio) < RATIO_TOLERANCE;

        return (!sameRatio && candidateDistance.ratio < bestDistance.ratio) ||
          (sameRatio && candidateDistance.area < bestDistance.area)
          ? size
          : best;
      });

      nextSettings.width = selected.width;
      nextSettings.height = selected.height;
      nextSettings.aspectRatioId = deriveAspectRatioId(selected.width, selected.height);
      nextSettings.aspectRatioValue = selected.width / selected.height;
    }

    nextSettings.externalImageSize = resolveExternalImageSize(nextSettings, model);
  }

  // Concepts stay listed as they are: the graph loads only those the model can, the panel badges the rest, and a
  // compatible model runs them again. Only the ones that ran before and stop now are reported.
  const previousModel = models.find((candidate) => candidate.key === settings.modelKey);
  const runsConcept = (candidate: GenerateModelConfig | undefined, lora: GenerateLora) =>
    Boolean(candidate && isLoraSupported(candidate) && isLoraCompatibleWithModel(lora.model, candidate));
  const stoppedConcepts = settings.loras.filter(
    (lora) =>
      lora.isEnabled &&
      !runsConcept(model, lora) &&
      (!previousModel || !isGenerateModelConfig(previousModel) || runsConcept(previousModel, lora))
  ).length;

  if (stoppedConcepts > 0) {
    change('concepts', 'turnedOff', stoppedConcepts);
  }

  // Enabled references the new family cannot run together are switched off rather than deleted: Invoke is not left
  // blocked, the first enabled ones keep running, and switching back can turn the rest on again. A model without
  // references keeps them as they are, all switched off, for the next model that takes them.
  const referencesSupported = isReferenceImageSupported(model);
  const retargetedReferenceImages = referencesSupported
    ? getCompatibleReferenceImages(settings.referenceImages, model, models)
    : settings.referenceImages;
  const enabledKept: GenerateReferenceImage[] = [];
  let didSwitchOff = false;
  const keptOrSwitchedOff = retargetedReferenceImages.map((referenceImage) => {
    if (!isReferenceImageActive(referenceImage)) {
      return referenceImage;
    }

    if (referencesSupported && canRunWithEnabledReferenceImages(model, enabledKept, referenceImage)) {
      enabledKept.push(referenceImage);
      return referenceImage;
    }

    didSwitchOff = true;
    return { ...referenceImage, isEnabled: false };
  });
  const compatibleReferenceImages = didSwitchOff ? keptOrSwitchedOff : retargetedReferenceImages;

  if (compatibleReferenceImages !== settings.referenceImages) {
    nextSettings.referenceImages = compatibleReferenceImages;
    // A retargeted reference keeps its image; its adapter and weights take the new family's defaults. Only tuning
    // the user chose is lost: a reference still at the previous family's defaults loses nothing.
    const isPreviousDefault = (referenceImage: GenerateReferenceImage) => {
      if (!previousModel || !isGenerateModelConfig(previousModel)) {
        return false;
      }

      const [config] = getDefaultReferenceImageConfigs(
        previousModel,
        models,
        [{ image: referenceImage.config.image, isEnabled: referenceImage.isEnabled }],
        []
      );

      return config !== undefined && areJsonValuesStructurallyEqual(config, referenceImage.config);
    };
    const retargeted = settings.referenceImages.filter(
      (referenceImage, index) => retargetedReferenceImages[index] !== referenceImage
    );
    const tunedRetargets = retargeted.filter((referenceImage) => !isPreviousDefault(referenceImage)).length;
    const switchedOff = keptOrSwitchedOff.filter(
      (referenceImage, index) => referenceImage.isEnabled !== retargetedReferenceImages[index]?.isEnabled
    ).length;

    if (tunedRetargets > 0) {
      change('referenceImages', 'reset', tunedRetargets);
    }

    if (retargeted.length > tunedRetargets) {
      change('referenceImages', 'adjusted', retargeted.length - tunedRetargets);
    }

    if (switchedOff > 0) {
      change('referenceImages', 'turnedOff', switchedOff);
    }
  }

  const policy = getComponentSectionPolicy(model, nextSettings);
  const slotsByKey = new Map(policy.slots.map((slotPolicy) => [slotPolicy.key, slotPolicy]));
  const uiPolicy = getGenerationUiPolicy(model);

  for (const key of GENERATE_COMPONENT_VALUE_KEYS) {
    if (!nextSettings[key]) {
      continue;
    }

    if (key === 'vae' && !slotsByKey.has(key) && uiPolicy.sdVaeVisible && nextSettings.vae?.base === model.base) {
      continue;
    }

    if (!isSelectedComponentCompatible(slotsByKey.get(key), model, nextSettings, key)) {
      nextSettings[key] = null;
      change(key, 'removed');
    }
  }

  // Clamp and report during selection; validation remains the backstop for other write paths.
  const clampedGuidance = clampGuidance(nextSettings.cfgScale, getBaseGenerationConfig(model).guidance);

  if (clampedGuidance !== nextSettings.cfgScale) {
    nextSettings.cfgScale = clampedGuidance;
    change('guidance', 'adjusted');
  }

  if (!uiPolicy.clipSkipMax && nextSettings.clipSkip !== 0) {
    nextSettings.clipSkip = 0;
    change('clipSkip', 'reset');
  } else if (uiPolicy.clipSkipMax && nextSettings.clipSkip > uiPolicy.clipSkipMax) {
    nextSettings.clipSkip = uiPolicy.clipSkipMax;
    change('clipSkip', 'adjusted');
  }

  if (!uiPolicy.cfgRescaleVisible && nextSettings.cfgRescaleMultiplier !== 0) {
    nextSettings.cfgRescaleMultiplier = 0;
    change('cfgRescale', 'reset');
  }

  if (!uiPolicy.colorCompensationVisible && nextSettings.colorCompensation) {
    nextSettings.colorCompensation = false;
    change('colorCompensation', 'reset');
  }

  if (!uiPolicy.hiDiffusionVisible && nextSettings.hiDiffusionEnabled) {
    nextSettings.hiDiffusionEnabled = false;
    change('hiDiffusion', 'reset');
  }

  // The refiner stays selected: it only runs after an SDXL base (`getActiveRefiner`), and the next SDXL pick uses it.
  if (!uiPolicy.seamlessVisible && (nextSettings.seamlessXAxis || nextSettings.seamlessYAxis)) {
    nextSettings.seamlessXAxis = false;
    nextSettings.seamlessYAxis = false;
    change('seamless', 'reset');
  }

  if (!uiPolicy.vaePrecisionVisible) {
    const defaultVaePrecision = getGenerationDefaults(model).vaePrecision;

    if (nextSettings.vaePrecision !== defaultVaePrecision) {
      // Only a precision the user picked is lost; moving from one family's default to the next is no choice undone.
      const previousDefault =
        previousModel && isGenerateModelConfig(previousModel)
          ? getGenerationDefaults(previousModel).vaePrecision
          : undefined;

      if (nextSettings.vaePrecision !== previousDefault) {
        change('vaePrecision', 'reset');
      }

      nextSettings.vaePrecision = defaultVaePrecision;
    }
  }

  return {
    changes,
    dimensionsAdjusted: nextSettings.width !== settings.width || nextSettings.height !== settings.height,
    settings: nextSettings,
  };
};

/** Normalize, reconcile, and apply component policy before persisting model selection. */
export const getGenerateModelSelectionResult = ({
  currentValues,
  isCanvasFrame = false,
  model,
  models,
}: {
  currentValues: unknown;
  /** The width and height are the Canvas frame (a Canvas source), which the Canvas sync resizes to follow them. */
  isCanvasFrame?: boolean;
  model: GenerateModelConfig;
  models: readonly ModelConfig[];
}): GenerateModelSelectionResult => {
  const currentSettings = isGenerateSettings(currentValues)
    ? currentValues
    : (normalizeGenerateSettings(currentValues) ?? getDefaultGenerateSettings(model));
  const result = getSettingsWithCompatibleModelSelections(currentSettings, model, models, isCanvasFrame);
  const componentSourceModel = getAutoFlux2ComponentSourceModel(model, result.settings, models);

  if (componentSourceModel === undefined || componentSourceModel?.key === result.settings.componentSourceModel?.key) {
    return result;
  }

  return {
    ...result,
    settings: { ...result.settings, componentSourceModel },
  };
};

const hasModelKey = (models: readonly ModelConfig[], key: string, type?: string): boolean =>
  models.some((model) => model.key === key && (!type || model.type === type));

const ISSUE_KEY = 'widgets.generate.issues';

/** A translated issue; `message` must read exactly like the locale entry, which the issue tests check. */
const issue = (
  id: string,
  key: string,
  message: string,
  target: GenerationIssue['target'],
  values?: Record<string, number | string>,
  fix?: GenerationIssueFix
): GenerationIssue => ({
  id,
  message,
  messageKey: `${ISSUE_KEY}.${key}`,
  ...(values ? { messageValues: values } : {}),
  target,
  ...(fix ? { fix } : {}),
});

const MODEL_TARGET: GenerationIssue['target'] = { section: 'model' };
/** The reference images as a group (their count, or a rule over all of them); the section lists it above the cards. */
const REFERENCE_IMAGES_TARGET: GenerationIssue['target'] = { field: 'referenceImages', section: 'guidance' };
/** One reference image: its card shows the issue. */
const referenceImageTarget = (referenceImageId: string): GenerationIssue['target'] => ({
  field: 'referenceImage',
  referenceImageId,
  section: 'guidance',
});

/** Installed-model checks: the main model, its components, concepts and enabled references must all be installed. */
export const getGenerationModelAvailabilityIssues = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  models: readonly ModelConfig[]
): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];

  if (!hasModelKey(models, model.key, model.type)) {
    issues.push(
      issue(
        `model:notInstalled:${model.key}`,
        'modelNotInstalled',
        `${model.name} is not installed. Choose another model, or reinstall it.`,
        MODEL_TARGET,
        { name: model.name }
      )
    );
  }

  for (const key of GENERATE_COMPONENT_VALUE_KEYS) {
    const value = settings[key];

    if (value && !hasModelKey(models, value.key, value.type)) {
      issues.push(
        issue(
          `components:notInstalled:${key}`,
          'componentNotInstalled',
          `${COMPONENT_SETTING_LABELS[key]} "${value.name}" is not installed. Choose another, or clear it.`,
          { section: 'components' },
          // The locale nests its own name for the slot, keyed by `component`.
          { component: key, name: value.name },
          { key, kind: 'clearModel' }
        )
      );
    }
  }

  const refiner = getActiveRefiner(settings, model);
  if (refiner && !hasModelKey(models, refiner.key, 'main')) {
    issues.push(
      issue(
        'refiner:notInstalled',
        'refinerNotInstalled',
        `Refiner "${refiner.name}" is not installed. Choose another, or clear it.`,
        { section: 'refiner' },
        { name: refiner.name },
        { key: 'refinerModel', kind: 'clearModel' }
      )
    );
  }

  // Only concepts the model would load need to be installed; the rest are set aside and skipped by the graph.
  for (const lora of settings.loras) {
    if (
      lora.isEnabled &&
      isLoraSupported(model) &&
      isLoraCompatibleWithModel(lora.model, model) &&
      !hasModelKey(models, lora.model.key, 'lora')
    ) {
      issues.push(
        issue(
          `concepts:notInstalled:${lora.model.key}`,
          'conceptNotInstalled',
          `Concept "${lora.model.name}" is not installed. Remove it, or reinstall it.`,
          { field: 'concepts', section: 'guidance' },
          { name: lora.model.name },
          { kind: 'removeConcept', modelKey: lora.model.key }
        )
      );
    }
  }

  settings.referenceImages.forEach((referenceImage, index) => {
    const config = referenceImage.config;

    if (
      referenceImage.isEnabled &&
      'model' in config &&
      config.model &&
      !hasModelKey(models, config.model.key, config.model.type)
    ) {
      issues.push(
        issue(
          `referenceImages:modelNotInstalled:${referenceImage.id}`,
          'referenceModelNotInstalled',
          `Reference image ${index + 1} uses "${config.model.name}", which is not installed. Turn it off, or choose another model for it.`,
          referenceImageTarget(referenceImage.id),
          { index: index + 1, name: config.model.name },
          { kind: 'turnOffReferenceImages', referenceImageIds: [referenceImage.id] }
        )
      );
    }
  });

  return issues;
};

export const getGenerationModelAvailabilityReasons = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  models: readonly ModelConfig[]
): string[] => getGenerationModelAvailabilityIssues(model, settings, models).map(({ message }) => message);

const getDimensionValidationIssues = (model: GenerateModelConfig, settings: GenerateSettings): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];
  // Native target dimensions must be divisible by model grid × 4.
  const dimensions = getGenerationDimensions(model, settings.pidMode);
  const toGrid = (value: number, round: (value: number) => number = Math.round) =>
    Math.min(
      dimensions.max,
      Math.max(dimensions.min, round((Number.isFinite(value) ? value : 0) / dimensions.grid) * dimensions.grid)
    );
  // A size fix keeps a locked or preset ratio, deriving the height from the width; a free size records its new ratio.
  const isRatioConstrained = isAspectRatioLocked(settings);
  const ratio =
    settings.aspectRatioValue > 0
      ? settings.aspectRatioValue
      : settings.height > 0
        ? settings.width / settings.height
        : 1;
  const sizeFix = (width: number, height: number, round?: (value: number) => number): GenerationIssueFix => {
    const fittedWidth = toGrid(width, round);
    const fittedHeight = isRatioConstrained ? toGrid(fittedWidth / ratio, round) : toGrid(height, round);

    return {
      height: fittedHeight,
      kind: 'setDimensions',
      width: fittedWidth,
      ...(isRatioConstrained ? {} : { aspectRatioValue: fittedWidth / fittedHeight }),
    };
  };
  const fittedSize = sizeFix(settings.width, settings.height);
  const target: GenerationIssue['target'] = { section: 'dimensions' };

  for (const side of ['width', 'height'] as const) {
    const value = settings[side];
    const label = side === 'width' ? 'Width' : 'Height';

    if (!Number.isFinite(value) || value < dimensions.min || value > dimensions.max) {
      issues.push(
        issue(
          `dimensions:${side}Range`,
          `${side}Range`,
          `${label} must be between ${dimensions.min} and ${dimensions.max}.`,
          target,
          { max: dimensions.max, min: dimensions.min },
          fittedSize
        )
      );
    } else if (value % dimensions.grid !== 0) {
      issues.push(
        issue(
          `dimensions:${side}Grid`,
          `${side}Grid`,
          `${label} must be a multiple of ${dimensions.grid}.`,
          target,
          { grid: dimensions.grid },
          fittedSize
        )
      );
    }
  }

  const maxSize = model.type === 'external_image_generator' ? model.capabilities?.max_image_size : null;
  if (maxSize && (settings.width > maxSize.width || settings.height > maxSize.height)) {
    issues.push(
      issue(
        'dimensions:providerMax',
        'providerMaxSize',
        `${model.name} supports at most ${maxSize.width}×${maxSize.height}.`,
        target,
        { height: maxSize.height, name: model.name, width: maxSize.width },
        // Scaled down to fit both bounds, then rounded down so the grid never pushes it back over.
        (() => {
          const scale = Math.min(maxSize.width / settings.width, maxSize.height / settings.height, 1);

          return sizeFix(settings.width * scale, settings.height * scale, Math.floor);
        })()
      )
    );
  }

  return issues;
};

/** Reject unsupported PiD instead of silently using ordinary decode. */
const getPidValidationIssues = (model: GenerateModelConfig, settings: GenerateSettings): GenerationIssue[] => {
  if (settings.pidMode === 'off') {
    return [];
  }

  const target: GenerationIssue['target'] = { section: 'advanced' };

  if (!getIsPidSupportedBase(model.base)) {
    return [
      issue(
        'advanced:pidUnsupported',
        'pidUnsupported',
        `${model.name} does not support PiD. Turn PiD off to generate with it.`,
        target,
        { name: model.name },
        { kind: 'turnOffPid' }
      ),
    ];
  }

  return settings.pidSteps < 1 ? [issue('advanced:pidSteps', 'pidSteps', 'PiD steps must be at least 1.', target)] : [];
};

/** Only the encoder of a Qwen-Image-2.1 Diffusers model keeps the vision tower that reads references. */
export const QWEN_IMAGE_21_STANDALONE_ENCODER_REASON =
  "Reference images need the Qwen-Image-2.1 model's own encoder, the only one with a vision tower. " +
  'Clear the Qwen3-VL Encoder component.';

const getReferenceImageValidationIssues = (
  model: GenerateModelConfig,
  settings: GenerateSettings
): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];
  // Use the shared reference limit, but report unsupported images individually rather than a misleading maximum of
  // zero.
  const maxReferenceImages = getMaxReferenceImages(model);
  const active = settings.referenceImages.filter(isReferenceImageActive);

  if (isReferenceImageSupported(model) && active.length > maxReferenceImages) {
    issues.push(
      issue(
        'referenceImages:limit',
        'referenceImageLimit',
        `${model.name} supports at most ${maxReferenceImages} reference image${maxReferenceImages === 1 ? '' : 's'}. Turn off or remove the extra ones.`,
        REFERENCE_IMAGES_TARGET,
        { count: maxReferenceImages, name: model.name },
        {
          kind: 'turnOffReferenceImages',
          referenceImageIds: active.slice(maxReferenceImages).map((referenceImage) => referenceImage.id),
        }
      )
    );
  }

  settings.referenceImages.forEach((referenceImage, position) => {
    if (!referenceImage.isEnabled) {
      return;
    }

    const index = position + 1;
    const id = referenceImage.id;
    const turnOff: GenerationIssueFix = { kind: 'turnOffReferenceImages', referenceImageIds: [id] };

    if (!getReferenceImageConfigSupported(model, referenceImage)) {
      issues.push(
        issue(
          `referenceImages:unsupported:${id}`,
          'referenceImageUnsupported',
          `${model.name} cannot use reference image ${index}. Turn it off, or remove it.`,
          referenceImageTarget(id),
          { index, name: model.name },
          turnOff
        )
      );
      return;
    }

    if (!referenceImage.config.image) {
      issues.push(
        issue(
          `referenceImages:needsImage:${id}`,
          'referenceImageNeedsImage',
          `Reference image ${index} needs an image.`,
          referenceImageTarget(id),
          { index },
          turnOff
        )
      );
    }

    if (referenceImage.config.type === 'ip_adapter') {
      if (!referenceImage.config.model || referenceImage.config.model.base !== model.base) {
        issues.push(
          issue(
            `referenceImages:needsIpAdapter:${id}`,
            'referenceImageNeedsIpAdapter',
            `Reference image ${index} needs an IP Adapter model for ${model.name}.`,
            referenceImageTarget(id),
            { index, name: model.name },
            turnOff
          )
        );
      }

      // A weight recalled from an SD reference can be negative; `anima_ip_adapter.strength` rejects it at enqueue.
      const { weight } = referenceImage.config;
      if (isStrengthOnlyIPAdapterBase(model.base) && !(weight >= 0 && weight <= 2)) {
        issues.push(
          issue(
            `referenceImages:strength:${id}`,
            'referenceImageStrength',
            `Reference image ${index} strength must be between 0 and 2.`,
            referenceImageTarget(id),
            { index },
            turnOff
          )
        );
      }
    }

    if (
      referenceImage.config.type === 'flux_redux' &&
      (!referenceImage.config.model || referenceImage.config.model.base !== model.base)
    ) {
      issues.push(
        issue(
          `referenceImages:needsRedux:${id}`,
          'referenceImageNeedsRedux',
          `Reference image ${index} needs a FLUX Redux model for ${model.name}.`,
          referenceImageTarget(id),
          { index, name: model.name },
          turnOff
        )
      );
    }

    if (
      referenceImage.config.type === 'flux_kontext_reference_image' &&
      (!referenceImage.config.model || referenceImage.config.model.key !== model.key)
    ) {
      issues.push(
        issue(
          `referenceImages:needsKontext:${id}`,
          'referenceImageNeedsKontext',
          `Reference image ${index} needs the selected FLUX Kontext model.`,
          referenceImageTarget(id),
          { index },
          turnOff
        )
      );
    }
  });

  // The encoder slot stays filled across models (Ideogram 4 and the GGUF builds share it), so a Diffusers model can
  // inherit one that cannot read the references.
  if (
    model.base === 'qwen-image-2-1' &&
    settings.qwen3VLEncoderModel &&
    isQwen3Vl8bEncoder(settings.qwen3VLEncoderModel) &&
    settings.referenceImages.some(
      (referenceImage) => referenceImage.isEnabled && referenceImage.config.type === 'qwen_image_2_1_reference_image'
    )
  ) {
    issues.push(
      createUntranslatedGenerationIssue(QWEN_IMAGE_21_STANDALONE_ENCODER_REASON, REFERENCE_IMAGES_TARGET, {
        key: 'qwen3VLEncoderModel',
        kind: 'clearModel',
      })
    );
  }

  return issues;
};

/** Validate guidance for writes bypassing model selection; return null or one error message. */
export const getGuidanceBoundReason = (
  model: Pick<GenerateModelConfig, 'base' | 'name' | 'type'> & { variant?: unknown },
  cfgScale: number
): string | null => {
  const { guidance, guidanceLabel } = getBaseGenerationConfig(model);

  if (cfgScale < guidance.min) {
    return `${guidanceLabel} must be at least ${guidance.min} for ${model.name}.`;
  }

  if (guidance.max !== null && cfgScale > guidance.max) {
    return `${guidanceLabel} must be at most ${guidance.max} for ${model.name}.`;
  }

  return null;
};

/**
 * Combinations the backend refuses only after queueing. The encoder reads edit references with its vision tower, which
 * a GGUF Qwen3-VL encoder does not carry; a Diffusers main without an encoder override uses its bundled one. A
 * standalone encoder folder may lack the vision weights too, but its config does not say so, so the message names
 * what the encoder needs rather than a format that always has it.
 */
const getKrea2ReferenceValidationReasons = (settings: GenerateSettings): string[] => {
  const styleCount = settings.referenceImages.filter(
    (referenceImage) => isReferenceImageActive(referenceImage) && referenceImage.config.type === 'krea2_reference_image'
  ).length;
  const hasEditReferences = hasEnabledKrea2EditReferences(settings.referenceImages);
  const reasons: string[] = [];

  if (styleCount > 1) {
    reasons.push('Krea-2 uses one style reference at a time. Turn off all but one, or switch all of them to Edit.');
  }

  if (styleCount > 0 && hasEditReferences) {
    reasons.push(
      'Krea-2 cannot combine a style reference with edit references. Switch the style reference to Edit, or turn off one kind.'
    );
  }

  const encoder = settings.qwen3VLEncoderModel;

  if (hasEditReferences && encoder && isQwen3Vl4bEncoder(encoder) && encoder.format === 'gguf_quantized') {
    reasons.push(
      `Krea-2 edit references need a Qwen3-VL encoder with vision weights, which the GGUF encoder "${encoder.name}" does not have. Select a safetensors file or Diffusers folder that includes the vision weights.`
    );
  }

  return reasons;
};

/** Scalar family rules are separate from component-slot validation; their fields show the English inline. */
const getModelFamilyValidationIssues = (model: MainModelConfig, settings: GenerateSettings): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];

  if (model.base === 'krea-2') {
    issues.push(
      ...getKrea2ReferenceValidationReasons(settings).map((reason) =>
        createUntranslatedGenerationIssue(reason, REFERENCE_IMAGES_TARGET)
      )
    );
  }

  // Parse raw rebalance strings before forwarding them to the backend.
  if (
    model.base === 'krea-2' &&
    settings.krea2RebalanceEnabled &&
    !isValidKrea2RebalanceWeights(settings.krea2RebalanceWeights)
  ) {
    issues.push(
      createUntranslatedGenerationIssue(
        `Krea-2 rebalance weights must be ${KREA2_REBALANCE_WEIGHT_COUNT} comma-separated numbers.`,
        { field: 'krea2RebalanceWeights', section: 'render' }
      )
    );
  }

  // FLUX.3 Klein is a step-distilled flow map: it samples without CFG, and its denoise node refuses a scale above 1.
  if (model.base === 'flux3' && model.variant === 'flux3_klein' && settings.cfgScale > 1) {
    issues.push(
      createUntranslatedGenerationIssue(
        'FLUX.3 Klein samples without CFG. Set CFG Scale to 1, or use FLUX.3 Klein Base for CFG.',
        { field: 'cfgScale', section: 'render' }
      )
    );
  }

  // The field commits only lists in range, but a persisted one can still name a block MOSS-Image lacks, which would fail
  // the queue item. Without CFG the negative branch never runs, so the list has no effect and its disabled field cannot
  // be fixed.
  if (
    model.base === 'moss-image' &&
    settings.cfgScale > 1 &&
    !isValidMossImageSkipLayers(settings.mossImageUncondSkipLayers)
  ) {
    issues.push(
      createUntranslatedGenerationIssue(
        `MOSS-Image skip layers must be whole numbers from 0 to ${MAX_MOSS_IMAGE_SKIP_LAYER}.`,
        { field: 'mossImageUncondSkipLayers', section: 'render' }
      )
    );
  }

  // Finite persisted Ideogram overrides still need range validation.
  if (model.base === 'ideogram-4') {
    for (const [label, value, min, max] of [
      ['steps', settings.ideogram4Steps, IDEOGRAM4_STEPS_MIN, IDEOGRAM4_STEPS_MAX],
      ['guidance', settings.ideogram4GuidanceScale, IDEOGRAM4_GUIDANCE_MIN, IDEOGRAM4_GUIDANCE_MAX],
      ['mu', settings.ideogram4Mu, IDEOGRAM4_MU_MIN, IDEOGRAM4_MU_MAX],
    ] as const) {
      // Null is "let the preset decide" and is omitted from the graph entirely.
      if (value !== null && (value < min || value > max)) {
        issues.push(
          createUntranslatedGenerationIssue(`Ideogram 4 ${label} must be between ${min} and ${max}.`, {
            section: 'render',
          })
        );
      }
    }
  }

  return issues;
};

const EXTERNAL_MODE_LABELS: Record<ExternalGenerationMode, string> = {
  img2img: 'image-to-image generation',
  inpaint: 'inpainting',
  txt2img: 'text-to-image generation',
};

/** Why an external model cannot run `mode`, or null when it advertises it. */
export const getExternalModeReason = (model: GenerateModelConfig, mode: ExternalGenerationMode): string | null =>
  model.type === 'external_image_generator' && !(model.capabilities?.modes ?? ['txt2img']).includes(mode)
    ? `${model.name} does not support ${EXTERNAL_MODE_LABELS[mode]}.`
    : null;

/** Reported when Generate has no model it can run with; also used by surfaces that cannot read the values at all. */
export const NO_GENERATE_MODEL_ISSUE: GenerationIssue = issue(
  'model:unsupported',
  'noSupportedModel',
  'Generate needs a supported model before it can be invoked.',
  MODEL_TARGET
);

/** The Components section's issues: one per slot, which the slot shows itself, then the family's other rules. */
const getComponentValidationIssues = (model: GenerateModelConfig, settings: GenerateSettings): GenerationIssue[] => {
  const policy = getComponentSectionPolicy(model, settings);
  const ctx = getComponentPolicyContext(model, settings);

  return [
    ...(policy.rules?.(ctx) ?? []).map((message) =>
      createUntranslatedGenerationIssue(message, { section: 'components' })
    ),
    ...policy.slots.flatMap((slotPolicy) => {
      const message = getComponentSlotIssue(slotPolicy, ctx);

      return message
        ? [createUntranslatedGenerationIssue(message, { field: slotPolicy.key, section: 'components' })]
        : [];
    }),
  ];
};

/** `mode` is the external generation mode being compiled; local models are validated the same way for every mode. */
export const getGenerationValidationIssues = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  mode: ExternalGenerationMode = 'txt2img'
): GenerationIssue[] => {
  // All compilers share this validator; check capabilities before model-specific policy.
  if (!hasArchitectureCapabilities()) {
    // Keep core independent of the load-state store; failures direct callers to retry.
    return [
      issue(
        'model:capabilitiesUnavailable',
        'capabilitiesUnavailable',
        'Model capabilities are not available. Generation is blocked until they load; if this persists, retry from the Generate panel.',
        MODEL_TARGET
      ),
    ];
  }

  if (!isSupportedGenerateModel(model)) {
    return [NO_GENERATE_MODEL_ISSUE];
  }

  // Reject supported bases missing from the loaded table rather than compile fallback policy.
  if (!isArchitectureDescribed(model)) {
    return [
      issue(
        'model:architectureNotDescribed',
        'architectureNotDescribed',
        `The backend does not describe the ${model.base} architecture, so it cannot be generated with.`,
        MODEL_TARGET,
        { base: model.base }
      ),
    ];
  }

  const issues = [
    ...getDimensionValidationIssues(model, settings),
    ...getReferenceImageValidationIssues(model, settings),
    ...getPidValidationIssues(model, settings),
  ];

  if (
    getGenerationUiPolicy(model).hiDiffusionVisible &&
    settings.hiDiffusionEnabled &&
    settings.hiDiffusionRauNetEnabled
  ) {
    const t1 = settings.hiDiffusionT1Ratio;
    const t2 = settings.hiDiffusionT2Ratio;
    if (t1 !== null && t2 !== null && t1 < t2) {
      issues.push(
        createUntranslatedGenerationIssue('HiDiffusion T1 ratio must be greater than or equal to T2.', {
          section: 'advanced',
        })
      );
    }
  }

  if (model.type === 'external_image_generator') {
    const modeReason = getExternalModeReason(model, mode);

    if (modeReason) {
      issues.push(createUntranslatedGenerationIssue(modeReason, MODEL_TARGET));
    }

    if (!getExternalProviderNodeType(model.provider_id)) {
      issues.push(
        createUntranslatedGenerationIssue(
          `No invocation node registered for external provider '${model.provider_id ?? ''}'.`,
          MODEL_TARGET
        )
      );
    }

    // Every request emits one output, and the Canvas edit modes add the composite as one init image.
    if (isSeedreamBatchModel(model)) {
      const totalImages = getExternalReferenceImages(settings, model).length + (mode === 'txt2img' ? 0 : 1) + 1;

      if (totalImages > SEEDREAM_BATCH_MAX_TOTAL_IMAGES) {
        issues.push(
          createUntranslatedGenerationIssue(
            `${model.name} supports at most ${SEEDREAM_BATCH_MAX_TOTAL_IMAGES} images total (reference + init + output), got ${totalImages}.`,
            REFERENCE_IMAGES_TARGET
          )
        );
      }
    }

    return issues;
  }

  issues.push(...getComponentValidationIssues(model, settings));
  const guidanceReason = getGuidanceBoundReason(model, settings.cfgScale);

  if (guidanceReason) {
    issues.push(
      createUntranslatedGenerationIssue(
        guidanceReason,
        { field: 'cfgScale', section: 'render' },
        { kind: 'clampGuidance', value: clampGuidance(settings.cfgScale, getBaseGenerationConfig(model).guidance) }
      )
    );
  }

  issues.push(...getModelFamilyValidationIssues(model, settings));
  issues.push(
    ...getRefinerValidationReasons(model, settings).map((reason) =>
      createUntranslatedGenerationIssue(reason, { section: 'refiner' })
    )
  );

  return issues;
};

/** English reasons for compilers and logs, which throw or list them without a translator. */
export const getGenerationValidationReasons = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  mode: ExternalGenerationMode = 'txt2img'
): string[] => getGenerationValidationIssues(model, settings, mode).map(({ message }) => message);

/**
 * What blocks Invoke for Generate's values, shared by the Invoke route and the Generate form so the two always agree.
 * With a Canvas frame, a hosted provider is checked at the size it will be asked for (its own size nearest the frame),
 * and one without text to image as the edit the Canvas will ask for. `models` adds the installed-model checks; leave it
 * out while the catalog is unread, which would otherwise name every model as missing.
 */
export const getGenerateInvocationIssues = (
  values: GenerateSettings & { model: GenerateModelConfig },
  { canvasFrame, models }: { canvasFrame?: { height: number; width: number }; models?: readonly ModelConfig[] } = {}
): GenerationIssue[] => {
  const { model } = values;
  const providerModes =
    canvasFrame !== undefined && model.type === 'external_image_generator'
      ? (model.capabilities?.modes ?? ['txt2img'])
      : null;

  return [
    ...getGenerationValidationIssues(
      model,
      providerModes !== null && canvasFrame
        ? { ...values, ...resolveExternalCanvasRequestSize(model, canvasFrame) }
        : values,
      providerModes !== null && !providerModes.includes('txt2img') ? 'img2img' : 'txt2img'
    ),
    ...(models ? getGenerationModelAvailabilityIssues(model, values, models) : []),
  ];
};

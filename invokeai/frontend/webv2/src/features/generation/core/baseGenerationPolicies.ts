import type {
  GenerationModelCatalogItem as ModelConfig,
  GenerationModelTaxonomyType as ModelTaxonomyType,
  PromptHistoryItem,
} from '@features/generation/core/contracts';
import type { BaseGenerationConfig, GuidanceLabel } from '@features/generation/core/generationConfig';

import { getArchitectureGenerationConfig } from '@features/generation/core/architectureGenerationConfig';
import {
  isSupportedGenerateBase,
  SUPPORTED_GENERATE_BASES,
  type SupportedGenerateBase,
} from '@features/generation/core/supportedBases';
import { getArchitectureFeatures, hasArchitectureCapabilities } from '@features/models/architectures';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { SEED_MAX } from '@platform/core/seed';

import type {
  ExpandPromptSuggestion,
  ComponentModelConfig,
  ExternalGenerationMode,
  GenerateLora,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  MainModelConfig,
  PidMode,
  VaeModelConfig,
} from './types';

import {
  getCompatibleDiffusersComponentSource,
  isBundledMainForBase,
  isAnimaInpaintAdapter,
  isVisionTextLlm,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isClipVariant,
  isDiffusersMainForBase,
  isBundledFlux2Main,
  isFlux2EncoderSourceForModel,
  isFlux2PipelineForModel,
  KLEIN_TO_QWEN3_VARIANT,
  isErnieImageMistralEncoder,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isGgufQwen3Vl4bEncoder,
  isGgufQwen3Vl8bEncoder,
  isQwen3Vl4bVisionTower,
  isQwen3Vl8bEncoder,
  isQwen3Vl8bVisionTower,
  isIdeogram4UnconditionalBranch,
  isMossImageQwen35Encoder,
  isQwen3Vl4bEncoder,
  doesWanSourceSupplyVae,
  hasWanLowNoiseSlot,
  isSelfContainedSDNQFlux1Pipeline,
  isVaeAcceptedByBase,
  isWanLowNoiseExpertFor,
  isVaeCompatibleWithGenerateModel,
  isZImageQwen3Encoder,
  servesAsType,
  type GenerateComponentFilter,
} from './componentCompatibility';
import { DEFAULT_COMPONENT_SLOTS, type DefaultComponentSlot, getDefaultComponentSlot } from './componentDefaults';
import { MAX_DIMENSION, MIN_DIMENSION } from './dimensionLimits';
import { DEFAULT_DYNAMIC_PROMPTS_SETTINGS } from './dynamicPrompts';
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
  isSameInstalledModel,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  isValidKrea2RebalanceWeights,
  isValidMossImageSkipLayers,
  KREA2_REBALANCE_WEIGHT_COUNT,
  MAX_MOSS_IMAGE_SKIP_LAYER,
  normalizeGenerateSettings,
} from './settings';

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

/** Variant-aware; null for undescribed local architectures. */
export const getDimensionGrid = (base: string, variant?: unknown): number | null =>
  base === 'external' ? EXTERNAL_DIMENSION_GRID : (getArchitectureFeatures(base, variant)?.dimension_grid ?? null);

/** A served architecture row does not imply frontend graph support. */
export const isArchitectureDescribed = (model: Pick<GenerateModelConfig, 'base' | 'type'>): boolean =>
  model.type === 'external_image_generator' || getArchitectureFeatures(model.base) !== undefined;

const getNumber = (value: number | null | undefined, fallback: number): number =>
  Number.isFinite(value) && value !== null && value !== undefined ? value : fallback;

const clampGuidance = (value: number, guidance: BaseGenerationConfig['guidance']): number =>
  Math.min(guidance.max ?? Number.POSITIVE_INFINITY, Math.max(guidance.min, value));

/** Native PiD uses model grid × 4 and a 2048 target optimum. */
export const getGenerationDimensions = (
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

  // Some graph builders ignore scheduler; use the default so stale UI state never leaks into metadata.
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

const HF_REPO_SOURCE = /^([A-Za-z0-9][\w.-]*\/[\w.-]+)(?::[\w.-]+)?$/;

/** The repo id (with `:variant`/`::subfolder`), a huggingface.co URL, or a local folder named like it; any case. */
export const isModelFromRepo = (source: string | null | undefined, repoId: string): boolean => {
  if (!source) {
    return false;
  }
  const wanted = repoId.toLowerCase();
  const withoutSubfolder = source.split('::')[0]!.trim();
  if (/^https?:\/\//i.test(withoutSubfolder)) {
    const repo = /^https?:\/\/(?:www\.)?huggingface\.co\/([^/?#]+\/[^/?#]+)/i.exec(withoutSubfolder)?.[1];
    return repo?.toLowerCase() === wanted;
  }

  const candidate = withoutSubfolder.replace(/[/\\]+$/, '');
  const repo = HF_REPO_SOURCE.exec(candidate)?.[1];
  if (repo !== undefined) {
    return repo.toLowerCase() === wanted;
  }
  const folder = candidate.split(/[/\\]/).pop() ?? '';
  return folder.toLowerCase() === wanted.split('/').pop();
};

/** A Hugging Face file URL or a local path ending in `fileName`; case-insensitive. */
const isModelFromFile = (source: string | null | undefined, fileName: string): boolean => {
  const file = source?.split('::')[0]!.split(/[?#]/)[0]!.split(/[/\\]/).pop();
  return Boolean(file) && file!.toLowerCase() === fileName.toLowerCase();
};

export const isSuggestedExpandPromptModel = (
  source: string | null | undefined,
  suggestion: Pick<ExpandPromptSuggestion, 'modelFileNames' | 'modelSource'>
): boolean =>
  (suggestion.modelSource !== null && isModelFromRepo(source, suggestion.modelSource)) ||
  (suggestion.modelFileNames ?? []).some((fileName) => isModelFromFile(source, fileName));

interface ImagePromptEnhancer {
  /** Starter source of the enhancer model; an installed text LLM with this source is preselected. */
  modelSource: string;
  /** Single-file builds of the same model, which Expand Prompt preselects too. */
  modelFileNames?: readonly string[];
  modelName: string;
  systemPromptId: string;
}

const IMAGE_PROMPT_ENHANCERS: Partial<Record<string, ImagePromptEnhancer>> = {
  // The OpenMOSS prompt-enhancer skill writes a layout draft the `moss_image_layout_prompt` node compiles.
  'moss-image': {
    modelName: 'Gemma 4 E4B (Prompt Enhancer)',
    modelSource: 'google/gemma-4-E4B-it',
    systemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0011',
  },
  // Text-only rewrites use BFL's `t2i` prompt. The bf16 folder and ComfyUI's int8/bf16 single files are one model.
  flux3: {
    modelFileNames: [
      'qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors',
      'qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors',
    ],
    modelName: 'FLUX.3 Prompt Upsampler',
    modelSource: 'black-forest-labs/flux-3-image-klein-pup',
    systemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0012',
  },
};

export const getImageExpandPromptSuggestion = (
  model: GenerateModelConfig | undefined
): ExpandPromptSuggestion | null => {
  const enhancer = model?.type === 'main' ? IMAGE_PROMPT_ENHANCERS[model.base] : undefined;
  return enhancer
    ? {
        image: null,
        imageSystemPromptId: null,
        modelFileNames: enhancer.modelFileNames,
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

/** Unconditional Ideogram branches stay resolvable but unselectable; Wan's other expert can span the schedule. */
export const isGenerateModelSelectable = <T extends ModelConfig>(model: T): model is T & GenerateModelConfig =>
  isSupportedGenerateModel(model) && !isIdeogram4UnconditionalBranch(model);

/**
 * The model a fresh project starts on, or the Generate form falls back to when none is chosen: the first one Generate
 * can select as a main model (so no standalone Ideogram 4 branch) whose files are present, else the first. Where a
 * model is stored or chosen, that one stays.
 */
export const pickFallbackGenerateModel = <T extends ModelConfig>(
  models: readonly T[]
): (T & GenerateModelConfig) | undefined => {
  const selectable = models.filter(isGenerateModelSelectable);

  return selectable.find((model) => !model.is_missing) ?? selectable[0];
};

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
    ...DEFAULT_DYNAMIC_PROMPTS_SETTINGS,
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
    qwen3VLVisionTowerModel: null,
    qwen35EncoderModel: null,
    animaInpaintAdapterModel: null,
    animaInpaintAdapterEnabled: true,
    animaInpaintAdapterWeight: DEFAULT_ANIMA_INPAINT_ADAPTER_WEIGHT,
    flux3PromptUpsamplerEnabled: false,
    flux3PromptUpsamplerModel: null,
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

export type ModelDefaultVae =
  /** Sets none, or the "default" sentinel where the VAE is a required component with nothing bundled. */
  | { kind: 'none' }
  /** Not installed or does not fit; `hasBundledVae` says whether a bundled VAE can stand in. */
  | { kind: 'missing'; hasBundledVae: boolean }
  /** `null` is the bundled VAE, which the "default" sentinel asks for wherever there is one. */
  | { kind: 'apply'; vae: VaeModelConfig | null };

const isVaeComponentRequired = (model: MainModelConfig, settings: GenerateSettings): boolean => {
  const vaeSlotPolicy = getComponentSectionPolicy(model, settings).slots.find((slotPolicy) => slotPolicy.key === 'vae');

  return Boolean(vaeSlotPolicy?.required?.(getComponentPolicyContext(model, settings)));
};

/** "default" is the bundled VAE; requiredness reads settings, since a component source can bundle one. */
export const resolveModelDefaultVae = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  vaeModels: readonly VaeModelConfig[]
): ModelDefaultVae => {
  const key = (model.default_settings as { vae?: unknown } | null | undefined)?.vae;

  if (model.type !== 'main' || typeof key !== 'string') {
    return { kind: 'none' };
  }

  // A graph that loads no VAE pick (a self-contained pipeline's own decodes) takes no default either.
  const vaeSlot = getComponentSectionPolicy(model, settings).slots.find((slotPolicy) => slotPolicy.key === 'vae');

  if (vaeSlot?.neverLoaded?.(getComponentPolicyContext(model, settings))) {
    return { kind: 'none' };
  }

  const hasBundledVae = !isVaeComponentRequired(model, settings);

  if (key === '' || key === 'default') {
    return hasBundledVae ? { kind: 'apply', vae: null } : { kind: 'none' };
  }

  const vae = vaeModels.find(
    (candidate) => candidate.key === key && !candidate.is_missing && isVaeCompatibleWithGenerateModel(model, candidate)
  );

  return vae ? { kind: 'apply', vae } : { kind: 'missing', hasBundledVae };
};

/** Recall: default VAE, else bundled, so bundled-VAE images reproduce; `undefined` keeps a required current VAE. */
export const getRecallDefaultVae = (defaultVae: ModelDefaultVae): VaeModelConfig | null | undefined =>
  defaultVae.kind === 'apply'
    ? defaultVae.vae
    : defaultVae.kind === 'missing' && defaultVae.hasBundledVae
      ? null
      : undefined;

export type GenerateComponentValueKey =
  | 't5EncoderModel'
  | 'clipEmbedModel'
  | 'clipLEmbedModel'
  | 'clipGEmbedModel'
  | 'mistralEncoderModel'
  | 'qwen3EncoderModel'
  | 'qwenVLEncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwen3VLVisionTowerModel'
  | 'qwen35EncoderModel'
  | 'animaInpaintAdapterModel'
  | 'flux3PromptUpsamplerModel'
  | 'wanT5EncoderModel'
  | 'wanLowNoiseModel'
  | 'ideogram4UnconditionalModel'
  | 'componentSourceModel'
  | 'pidDecoderModel'
  | 'gemma2EncoderModel'
  | 'vae';

export type ComponentPolicySettings = Pick<
  GenerateSettings,
  | 'animaInpaintAdapterEnabled'
  | 'componentSourceModel'
  | 'flux3PromptUpsamplerEnabled'
  | 'pidMode'
  | 'qwen3VLEncoderModel'
>;

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
  /** Shown while the slot is empty and not required; a required slot always asks for a component. */
  placeholder?: string;
  defaultOpen?: boolean;
  filter?: (candidate: ModelConfig, ctx: ComponentPolicyContext) => boolean;
  required?: (ctx: ComponentPolicyContext) => boolean;
  /** False while the slot's switch is off: nothing loads a kept selection, so an uninstalled one does not block. */
  inUse?: (ctx: ComponentPolicyContext) => boolean;
  /**
   * The model's graph loads no pick for the slot whatever the switches (a self-contained pipeline takes its bundled
   * components), so a default stored for it would do nothing and the Model Manager does not offer it.
   */
  neverLoaded?: (ctx: ComponentPolicyContext) => boolean;
  missingMessage?: string;
  /** Reported even when optional, so a saved unusable selection blocks Invoke by name instead of silently dropping. */
  incompatibleMessage?: string;
}

export interface ComponentSectionPolicy {
  defaultOpen: boolean;
  slots: readonly ComponentSlotPolicy[];
  /** Family rules beyond the slots; each slot reports its own through `getComponentSlotIssue`. */
  rules?: (ctx: ComponentPolicyContext) => string[];
}

/** `turnedOff` returns on switching back; `removed` and `reset` are lost for good once the switch is kept. */
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
  count?: number;
}

export interface GenerateModelSelectionResult {
  settings: GenerateSettings;
  changes: readonly GenerateModelSwitchChange[];
  /** Nothing is lost and it stays on screen, so it is reported rather than confirmed. */
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
const TYPE_QWEN3_VL_VISION_TOWER: ModelTaxonomyType[] = ['qwen3_vl_vision_tower'];
const TYPE_QWEN35: ModelTaxonomyType[] = ['qwen3_5_encoder'];
const TYPE_WAN_T5: ModelTaxonomyType[] = ['wan_t5_encoder'];
const TYPE_CONTROLNET: ModelTaxonomyType[] = ['controlnet'];
const TYPE_TEXT_LLM: ModelTaxonomyType[] = ['text_llm'];
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

/** Optional; only a GGUF encoder needs it, so the slot appears with such an encoder alone. */
const qwen3VlVisionTowerSlot = (helpText: string, filter: GenerateComponentFilter): ComponentSlotPolicy =>
  slot({
    key: 'qwen3VLVisionTowerModel',
    label: 'Vision Tower',
    modelTypes: TYPE_QWEN3_VL_VISION_TOWER,
    valueKind: 'component',
    helpText,
    placeholder: 'None',
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
    inUse: (ctx) => ctx.settings.animaInpaintAdapterEnabled,
  });

const isFlux3PromptUpsamplerOn = (ctx: ComponentPolicyContext): boolean => ctx.settings.flux3PromptUpsamplerEnabled;

/** Exposed before the switch is on, like PiD's; the node reads images, so the model must see them. */
const flux3PromptUpsamplerSlot = (): ComponentSlotPolicy =>
  slot({
    key: 'flux3PromptUpsamplerModel',
    label: 'Prompt Upsampler',
    modelTypes: TYPE_TEXT_LLM,
    valueKind: 'component',
    placeholder: 'None',
    helpText:
      'The FLUX.3 Prompt Upsampler starter model (bf16 or int8), used while Prompt upsampler is on. Other Text LLMs that read images run, but are not trained for its instructions.',
    filter: (candidate) => isVisionTextLlm(candidate),
    required: isFlux3PromptUpsamplerOn,
    inUse: isFlux3PromptUpsamplerOn,
    missingMessage: 'Generate needs a Prompt Upsampler model while Prompt upsampler is on.',
  });

const isPidOn = (ctx: ComponentPolicyContext): boolean => getIsPidActive(ctx.settings.pidMode, ctx.model.base);

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
    required: isPidOn,
    inUse: isPidOn,
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
    required: isPidOn,
    inUse: isPidOn,
    missingMessage: 'Generate needs a Gemma-2 encoder while PiD is on.',
  });

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
    // The loader pairs a single-file A14B model with a different single-file expert of its variant.
    filter: (candidate, ctx) =>
      ctx.model.type !== 'external_image_generator' && isWanLowNoiseExpertFor(ctx.model)(candidate),
    incompatibleMessage:
      'The selected low-noise expert is not a different single-file model of this variant. Select one or clear it.',
  });

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
      ? (candidate) => servesAsType(candidate, 'qwen3_encoder') && filter(candidate)
      : (candidate) => servesAsType(candidate, 'qwen3_encoder'),
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
  is_missing?: boolean;
  key: string;
  name: string;
  type: string;
  format?: string;
  variant?: unknown;
};

const QWEN3_ENCODER_SIZE_LABELS: Partial<Record<string, string>> = { qwen3_4b: 'Qwen3 4B', qwen3_8b: 'Qwen3 8B' };

const isFlux2SourceFor =
  (model: GenerateModelConfig) =>
  (source: ComponentSourceCandidate | null | undefined): source is MainModelConfig =>
    Boolean(source && isFlux2PipelineForModel(model)(source));

const hasFlux2VaeSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledFlux2Main(ctx.model) || isFlux2SourceFor(ctx.model)(ctx.settings.componentSourceModel));

const hasFlux2EncoderSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledFlux2Main(ctx.model) ||
    Boolean(
      ctx.settings.componentSourceModel && isFlux2EncoderSourceForModel(ctx.model)(ctx.settings.componentSourceModel)
    ));

/** The saved component source the FLUX.2 loader can use: its VAE always, its encoder unless a standalone one is set. */
export const getFlux2ComponentSource = (
  model: MainModelConfig,
  settings: GenerateSettings
): MainModelConfig | undefined => {
  const source = settings.componentSourceModel;

  if (!isFlux2SourceFor(model)(source)) {
    return undefined;
  }

  const hasStandaloneEncoder = model.variant === 'dev' ? settings.mistralEncoderModel : settings.qwen3EncoderModel;

  if (hasStandaloneEncoder) {
    return source;
  }

  return isFlux2EncoderSourceForModel(model)(source) ? source : undefined;
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

  if (isBundledFlux2Main(model) || (encoderModel && settings.vae)) {
    return null;
  }

  // A registered pipeline whose files are gone is listed but cannot be loaded.
  const pipelines = models.filter((candidate) => !candidate.is_missing).filter(isFlux2SourceFor(model));
  const encoderSource = pipelines.find(isFlux2EncoderSourceForModel(model));

  if (!encoderModel && settings.vae) {
    return encoderSource ?? null;
  }

  // A pipeline that cannot give the encoder still gives the VAE; the encoder slot then asks for a standalone one.
  return encoderSource ?? pipelines[0] ?? null;
};

// Split/quantized families can satisfy required encoder/VAE slots from a bundled Diffusers source.
const isBundledOrDiffusersSourceSatisfied = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (isBundledMainForBase(ctx.model.base)(ctx.model) || hasCompatibleComponentSource(ctx));

/** A Diffusers pipeline loads its own components, so the loader reads no component source or second transformer. */
const isDiffusersPipeline = (ctx: ComponentPolicyContext): boolean => ctx.model.format === 'diffusers';

/** Whether a Diffusers component source of a single-file Wan model supplies the VAE: the loader needs its width to fit. */
const hasWanSourceVae = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  hasCompatibleComponentSource(ctx) &&
  doesWanSourceSupplyVae(ctx.model, ctx.settings.componentSourceModel!);

/** Shared by the picker, invocation readiness and graph preflight so their requirements cannot drift. */
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

/** A self-contained SDNQ FLUX.1 pipeline always loads its own encoders and VAE (`buildFluxGraph` sends none). */
const hasBundledFlux1Components = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' && isSelfContainedSDNQFlux1Pipeline(ctx.model);

const createPolicy = (defaultOpen: boolean, slots: readonly ComponentSlotPolicy[]): ComponentSectionPolicy => ({
  defaultOpen,
  slots,
});

const EMPTY_COMPONENT_POLICY: ComponentSectionPolicy = createPolicy(false, []);

const getBaseComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  settings: ComponentPolicySettings
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
            neverLoaded: hasBundledFlux1Components,
            required: (ctx) => !hasBundledFlux1Components(ctx),
            missingMessage: 'Generate needs a T5 Encoder for FLUX models.',
          },
          {
            ...clipEmbedSlot('Required for FLUX.1 models.'),
            neverLoaded: hasBundledFlux1Components,
            required: (ctx) => !hasBundledFlux1Components(ctx),
            missingMessage: 'Generate needs a CLIP Embed model for FLUX models.',
          },
          {
            ...vaeSlot('Required for FLUX.1 models.', isAcceptedVae),
            neverLoaded: hasBundledFlux1Components,
            required: (ctx) => !hasBundledFlux1Components(ctx),
            missingMessage: 'Generate needs a VAE for FLUX models.',
          },
        ]),
        rules: (ctx) =>
          ctx.model.type !== 'external_image_generator' && ctx.model.variant === 'dev_fill'
            ? ['FLUX Fill models do not support text-to-image generation.']
            : [],
      };
    case 'flux2': {
      const isDev = model.variant === 'dev';
      // A Klein of unknown size takes either encoder, so its copy names none.
      const kleinQwen3 =
        typeof model.variant === 'string'
          ? QWEN3_ENCODER_SIZE_LABELS[KLEIN_TO_QWEN3_VARIANT[model.variant] ?? '']
          : undefined;
      const encoderSlot: ComponentSlotPolicy = isDev
        ? {
            ...mistralEncoderSlot(
              'Optional override; otherwise taken from an installed FLUX.2 [dev] Diffusers pipeline.'
            ),
            required: (ctx) => !hasFlux2EncoderSource(ctx),
            missingMessage:
              'Generate needs a Mistral Encoder or a FLUX.2 [dev] Diffusers pipeline for this FLUX.2 [dev] model.',
          }
        : {
            ...qwen3EncoderSlot(
              kleinQwen3
                ? `Optional ${kleinQwen3} override; otherwise taken from an installed FLUX.2 Klein Diffusers or SDNQ pipeline of the same size.`
                : 'Optional override; otherwise taken from an installed FLUX.2 Klein Diffusers or SDNQ pipeline.'
            ),
            filter: (candidate, ctx) =>
              ctx.model.type !== 'external_image_generator' && isFlux2Qwen3EncoderForModel(ctx.model)(candidate),
            required: (ctx) => !hasFlux2EncoderSource(ctx),
            missingMessage: kleinQwen3
              ? `Generate needs a ${kleinQwen3} Encoder or a same-size FLUX.2 Klein Diffusers or SDNQ pipeline for this FLUX.2 Klein model.`
              : 'Generate needs a Qwen3 Encoder or a FLUX.2 Klein Diffusers or SDNQ pipeline for this FLUX.2 Klein model.',
          };

      return createPolicy(!isBundledFlux2Main(model), [
        encoderSlot,
        {
          ...vaeSlot(
            isDev
              ? 'Optional override; otherwise taken from an installed FLUX.2 Diffusers pipeline.'
              : 'Optional override; otherwise taken from an installed FLUX.2 Diffusers or SDNQ pipeline.',
            isAcceptedVae
          ),
          required: (ctx) => !hasFlux2VaeSource(ctx),
          missingMessage: isDev
            ? 'Generate needs a FLUX.2 VAE or a FLUX.2 Diffusers pipeline for this FLUX.2 [dev] model.'
            : 'Generate needs a FLUX.2 VAE or a FLUX.2 Diffusers or SDNQ pipeline for this FLUX.2 Klein model.',
        },
      ]);
    }
    case 'sd-3': {
      // A Diffusers pipeline brings all four components; a GGUF transformer brings none of them.
      const required = (ctx: ComponentPolicyContext) => ctx.model.format !== 'diffusers';
      const isDiffusers = model.format === 'diffusers';

      return createPolicy(!isDiffusers, [
        {
          ...t5EncoderSlot(
            isDiffusers
              ? 'Optional override; the main model is used when omitted.'
              : 'Required: a T5-XXL encoder, such as a T5 starter model or a T5 GGUF.'
          ),
          required,
          missingMessage: 'Generate needs a T5 Encoder for non-Diffusers SD3 models.',
        },
        {
          ...clipVariantSlot(
            'clipLEmbedModel',
            'CLIP L',
            'large',
            isDiffusers ? 'Optional CLIP-L override.' : 'Required: a CLIP-L text encoder.'
          ),
          required,
          missingMessage: 'Generate needs a CLIP L encoder for non-Diffusers SD3 models.',
        },
        {
          ...clipVariantSlot(
            'clipGEmbedModel',
            'CLIP G',
            'gigantic',
            isDiffusers
              ? 'Optional CLIP-G override.'
              : 'Required: a CLIP-G text encoder that includes its text_projection, such as clip_g.safetensors.'
          ),
          required,
          missingMessage: 'Generate needs a CLIP G encoder for non-Diffusers SD3 models.',
        },
        {
          ...vaeSlot(
            isDiffusers
              ? 'Optional VAE override.'
              : 'Required: an SD3 VAE. A single-file VAE is listed here once its name or folder says SD3, or its base is set to SD 3.x.',
            isAcceptedVae
          ),
          required,
          missingMessage: 'Generate needs a VAE for non-Diffusers SD3 models.',
        },
      ]);
    }
    case 'qwen-image':
      return createPolicy(model.format !== 'diffusers', [
        {
          ...componentSourceSlot(
            (candidate) => isDiffusersMainForBase('qwen-image')(candidate),
            'For non-Diffusers Qwen Image models, select this or provide separate VAE and Qwen VL models.'
          ),
          neverLoaded: isDiffusersPipeline,
        },
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
        flux3PromptUpsamplerSlot(),
      ]);
    case 'krea-2':
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
        // Same rule as Qwen-Image-2.1: the loader refuses a tower beside any other encoder. Style references skip it.
        ...(settings.qwen3VLEncoderModel && isGgufQwen3Vl4bEncoder(settings.qwen3VLEncoderModel)
          ? [
              qwen3VlVisionTowerSlot(
                'The mmproj file of the GGUF encoder (mmproj-Qwen3VL-4B-Instruct-F16.gguf, the "Qwen3-VL 4B Vision ' +
                  'Tower" starter model). It lets the GGUF encoder read edit reference images; text-to-image and ' +
                  'style references work without it. Install it from the Model Manager.',
                isQwen3Vl4bVisionTower
              ),
            ]
          : []),
      ]);
    case 'qwen-image-2-1':
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
              "non-Diffusers Qwen-Image-2.1 models. On a Diffusers model it replaces the model's own encoder. Reference " +
              'images are read by its vision tower; a GGUF encoder needs the separate Vision Tower below for that.',
            isQwen3Vl8bEncoder
          ),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL 8B Encoder for non-Diffusers Qwen-Image-2.1 models.',
        },
        // The loader rejects a tower beside any other encoder, so the slot (and with it a stale pick) goes with it.
        ...(settings.qwen3VLEncoderModel && isGgufQwen3Vl8bEncoder(settings.qwen3VLEncoderModel)
          ? [
              qwen3VlVisionTowerSlot(
                'The mmproj file of the GGUF encoder (for example mmproj-Qwen3VL-8B-Instruct-F16.gguf from ' +
                  'Qwen/Qwen3-VL-8B-Instruct-GGUF). It lets the GGUF encoder read reference images; ' +
                  'text-to-image works without it. Install it from the Model Manager.',
                isQwen3Vl8bVisionTower
              ),
            ]
          : []),
      ]);
    case 'ernie-image':
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
      return createPolicy(model.format !== 'diffusers', [
        {
          ...ideogram4UnconditionalSlot(
            'The second transformer branch. Required for single-file Ideogram 4 models, which hold ' +
              'only the conditional branch.'
          ),
          neverLoaded: isDiffusersPipeline,
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
      return createPolicy(model.format !== 'diffusers', [
        {
          ...componentSourceSlot(
            (candidate) => isDiffusersMainForBase('wan')(candidate),
            'Select a Diffusers Wan model to provide VAE and text-encoder components. Its VAE is used only when it ' +
              'is a TI2V-5B pipeline for a TI2V-5B model, or an A14B one for an A14B model.'
          ),
          neverLoaded: isDiffusersPipeline,
        },
        {
          ...vaeSlot('Required unless a matching Diffusers component source is available.', isAcceptedVae),
          required: (ctx) => !(isBundledMainForBase('wan')(ctx.model) || hasWanSourceVae(ctx)),
          missingMessage:
            'Generate needs a VAE for Wan models: a component source supplies one only when it is the same variant family.',
        },
        {
          ...wanT5EncoderSlot('Required unless a Diffusers component source is available.'),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Wan T5 Encoder for Wan models.',
        },
        // Only a single-file A14B model reads it; the loader ignores it for the others, so it is not offered there.
        ...(hasWanLowNoiseSlot(model)
          ? [wanLowNoiseSlot('Optional second A14B expert. Without it the high-noise expert runs the whole schedule.')]
          : []),
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
    qwen3VLVisionTowerModel: settings.qwen3VLVisionTowerModel,
    qwen35EncoderModel: settings.qwen35EncoderModel,
    animaInpaintAdapterModel: settings.animaInpaintAdapterModel,
    flux3PromptUpsamplerModel: settings.flux3PromptUpsamplerModel,
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
  qwen3VLVisionTowerModel: 'Qwen3-VL vision tower',
  qwen35EncoderModel: 'Qwen3.5 encoder',
  animaInpaintAdapterModel: 'Inpaint adapter',
  flux3PromptUpsamplerModel: 'Prompt upsampler',
  wanT5EncoderModel: 'Wan T5 encoder',
  wanLowNoiseModel: 'Low-noise expert',
  ideogram4UnconditionalModel: 'Unconditional transformer',
  t5EncoderModel: 'T5 encoder',
  pidDecoderModel: 'PiD decoder',
  gemma2EncoderModel: 'Gemma-2 encoder',
  vae: 'VAE',
  componentSourceModel: 'Component source',
};

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
    return Boolean(getFlux2ComponentSource(model, settings));
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

export type ModelDefaultComponents = Partial<Pick<GenerateSettings, DefaultComponentSlot['setting']>>;

/**
 * `settings` with each component pick replaced by the catalog's record of it, where it is installed: a copy saved with
 * the project keeps the variant or base the pick had when it was made, which the catalog may have reclassified since.
 */
const withCatalogRecords = (settings: GenerateSettings, models: readonly ModelConfig[]): GenerateSettings => {
  const current: Record<string, unknown> = { ...settings };

  for (const setting of GENERATE_COMPONENT_VALUE_KEYS) {
    const picked = settings[setting];
    const installed = picked ? models.find((candidate) => candidate.key === picked.key && !candidate.is_missing) : null;

    if (installed) {
      current[setting] = installed;
    }
  }

  return current as unknown as GenerateSettings;
};

/** Registered and loadable: a model whose files moved is listed (`is_missing`) but a default cannot take it. */
const isInstalled = (models: readonly ModelConfig[], selected: { key: string }): boolean =>
  models.some((installed) => installed.key === selected.key && !installed.is_missing);

const isSlotOffered = (model: GenerateModelConfig, settings: GenerateSettings, setting: GenerateComponentValueKey) =>
  getComponentSectionPolicy(model, settings).slots.some((slotPolicy) => slotPolicy.key === setting);

/** Whether a default for the slot can take effect: the family offers it and the graph loads what it holds. */
const isSlotLoaded = (model: GenerateModelConfig, settings: GenerateSettings, setting: GenerateComponentValueKey) => {
  const context = getComponentPolicyContext(model, settings);

  return getComponentSectionPolicy(model, settings).slots.some(
    (slotPolicy) => slotPolicy.key === setting && !slotPolicy.neverLoaded?.(context)
  );
};

/**
 * The default components the model stores for slots it offers that Apply cannot take: the model is not installed, or
 * it does not fit the slot (an encoder reclassified to another size). Judged beside the defaults that are taken (a
 * vision tower is offered once its encoder default is).
 */
export const getMissingDefaultComponents = (
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): DefaultComponentSlot['setting'][] => {
  const stored = (model.default_settings as { components?: Record<string, string> | null } | null | undefined)
    ?.components;
  const defaults = getDefaultGenerateSettings(model);
  const taken = getModelDefaultComponents(model, defaults, models);
  const settings = { ...defaults, ...taken };
  const context = getComponentPolicyContext(model, settings);
  const loaded = new Set(
    getComponentSectionPolicy(model, settings)
      .slots.filter((slotPolicy) => !slotPolicy.neverLoaded?.(context))
      .map((slotPolicy) => slotPolicy.key)
  );

  return Object.entries(stored ?? {}).flatMap(([persistenceKey, modelKey]) => {
    const slot = getDefaultComponentSlot(persistenceKey);

    return slot && loaded.has(slot.setting) && taken[slot.setting]?.key !== modelKey ? [slot.setting] : [];
  });
};

/** Whether a generation with `model` loads the VAE it is given, so a default VAE for it can take effect. */
export const isDefaultVaeLoaded = (model: GenerateModelConfig): boolean => {
  const settings = getDefaultGenerateSettings(model);
  const vaeSlot = getComponentSectionPolicy(model, settings).slots.find((slotPolicy) => slotPolicy.key === 'vae');

  return !vaeSlot?.neverLoaded?.(getComponentPolicyContext(model, settings));
};

/** The selections of `before` whose slot is offered there but not in `after`: they stay hidden and unloaded. */
export const getClosedComponentSettings = (
  model: GenerateModelConfig,
  before: GenerateSettings,
  after: GenerateSettings
): GenerateComponentValueKey[] =>
  GENERATE_COMPONENT_VALUE_KEYS.filter(
    (setting) => before[setting] && isSlotOffered(model, before, setting) && !isSlotOffered(model, after, setting)
  );

/**
 * The model's default components that fill a slot of `settings`: one that is empty, or whose selection is not
 * installed or does not fit `model`. A selection that fits stays, unless `replace` (a reset to the model's defaults)
 * takes the default over it. Defaults naming an unknown slot, a model that is not installed or one the slot rejects are
 * skipped. A slot that depends on another (the vision tower follows a GGUF encoder) is judged beside the defaults
 * already taken, whatever order they are stored in, and a selection whose slot a taken default closes is cleared
 * (`null`), so no hidden pick is left to block or leak into a later choice.
 */
export const getModelDefaultComponents = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  models: readonly ModelConfig[],
  { replace = false }: { replace?: boolean } = {}
): ModelDefaultComponents => {
  const stored = (model.default_settings as { components?: Record<string, string> | null } | null | undefined)
    ?.components;
  const patch: Partial<Record<DefaultComponentSlot['setting'], ComponentModelConfig>> = {};
  const fresh = withCatalogRecords(settings, models);
  let next = fresh;
  let pending = Object.entries(stored ?? {}).flatMap(([persistenceKey, modelKey]) => {
    const slot = getDefaultComponentSlot(persistenceKey);
    const candidate = models.find((installed) => installed.key === modelKey && !installed.is_missing);

    return slot && candidate && (slot.isMainModel ? isMainModelConfig(candidate) : isModelIdentifierConfig(candidate))
      ? [{ candidate, setting: slot.setting }]
      : [];
  });
  let progressed = true;

  while (progressed && pending.length > 0) {
    progressed = false;
    pending = pending.filter(({ candidate, setting }) => {
      const picked = next[setting];

      if (
        !replace &&
        picked &&
        isInstalled(models, picked) &&
        isComponentCompatibleWithModel(model, next, setting, picked as ModelConfig)
      ) {
        return false;
      }

      // Only a slot the family offers (the Model Manager lists the same ones); a dependent slot waits for the one it
      // hangs on.
      if (!isSlotLoaded(model, next, setting) || !isComponentCompatibleWithModel(model, next, setting, candidate)) {
        return true;
      }

      patch[setting] = candidate;
      next = { ...next, [setting]: candidate };
      progressed = true;
      return false;
    });
  }

  // A default taken beside a pick that a later default replaced (a tower before the encoder that closes it) is judged
  // again in the final settings, so the order they are stored in cannot change the result.
  for (let settled = false; !settled;) {
    settled = true;

    for (const setting of Object.keys(patch) as DefaultComponentSlot['setting'][]) {
      if (
        !isSlotLoaded(model, next, setting) ||
        !isComponentCompatibleWithModel(model, next, setting, patch[setting]!)
      ) {
        delete patch[setting];
        next = { ...next, [setting]: fresh[setting] };
        settled = false;
      }
    }
  }

  const cleared: ModelDefaultComponents = {};

  for (const setting of getClosedComponentSettings(model, fresh, next)) {
    if (setting !== 'vae' && !(setting in patch)) {
      cleared[setting as DefaultComponentSlot['setting']] = null;
    }
  }

  // A slot that holds a main model only took a main model above.
  return { ...cleared, ...patch } as ModelDefaultComponents;
};

/** A component slot Generate offers `model`, as the Model Manager lists it among the model's default components. */
export interface DefaultComponentSlotPolicy {
  helpText?: string;
  filter: (candidate: ModelConfig) => boolean;
  label: string;
  modelTypes: readonly ModelTaxonomyType[];
  persistenceKey: string;
}

/**
 * The slots Generate offers a model for its family, given the default components it stores: resolved like a model
 * switch resolves them, so a slot that depends on another opens once that one is set. The VAE has its own default.
 */
export const getDefaultComponentSlotPolicies = (
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): DefaultComponentSlotPolicy[] => {
  const defaults = getDefaultGenerateSettings(model);
  const settings = { ...defaults, ...getModelDefaultComponents(model, defaults, models) };
  const context = getComponentPolicyContext(model, settings);

  return getComponentSectionPolicy(model, settings).slots.flatMap((slot) => {
    if (slot.neverLoaded?.(context)) {
      return [];
    }

    const persistenceKey = DEFAULT_COMPONENT_SLOTS.find(({ setting }) => setting === slot.key)?.persistenceKey;

    return persistenceKey
      ? [
          {
            filter: (candidate) =>
              (slot.modelTypes as readonly string[]).some((type) => servesAsType(candidate, type)) &&
              (!slot.filter || slot.filter(candidate, context)),
            helpText: slot.helpText,
            label: slot.label,
            modelTypes: slot.modelTypes,
            persistenceKey,
          },
        ]
      : [];
  });
};

/**
 * The default components `model` stores, less the ones whose slot an edit closed: a dependent slot (the vision tower)
 * goes with the pick it hangs on when that is cleared, or when the pick still installed no longer opens it (a
 * safetensors encoder for a GGUF one). Keys this client has no slot for stay, as do the dependents of a pick that is
 * no longer installed, which the user sees and clears first.
 */
export const pruneClosedDefaultComponents = (
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): Record<string, string> => {
  const stored = (model.default_settings as { components?: Record<string, string> | null } | null | undefined)
    ?.components;
  const components = { ...stored };
  const offered = new Set(getDefaultComponentSlotPolicies(model, models).map((slot) => slot.persistenceKey));

  for (const key of Object.keys(components)) {
    const parent = getDefaultComponentSlot(key)?.dependsOn;
    const parentKey = parent === undefined ? undefined : components[parent];

    if (
      parent !== undefined &&
      (parentKey === undefined || (isInstalled(models, { key: parentKey }) && !offered.has(key)))
    ) {
      delete components[key];
    }
  }

  return components;
};

/** `model` without the stored defaults of the `held` slots, which the caller keeps as they are, empty ones included. */
const withoutDefaultsOf = (
  model: GenerateModelConfig,
  held: readonly (DefaultComponentSlot['setting'] | 'vae')[]
): GenerateModelConfig => {
  if (held.length === 0 || model.type === 'external_image_generator') {
    return model;
  }

  const stored = model.default_settings as { components?: Record<string, string> | null; vae?: string | null } | null;
  const heldKeys = new Set(
    DEFAULT_COMPONENT_SLOTS.filter(({ setting }) => held.includes(setting)).map(({ persistenceKey }) => persistenceKey)
  );
  const components = Object.fromEntries(Object.entries(stored?.components ?? {}).filter(([key]) => !heldKeys.has(key)));

  return {
    ...model,
    default_settings: {
      ...stored,
      components,
      ...(held.includes('vae') ? { vae: null } : {}),
    },
  } as GenerateModelConfig;
};

/**
 * What a model's defaults fill into the slots of `settings` that are empty: its default components, and its default
 * VAE where the model has a VAE picker the VAE fits. A selection that fits stays.
 */
export const getModelDefaultFills = (
  fillModel: GenerateModelConfig,
  settings: GenerateSettings,
  models: readonly ModelConfig[],
  { held = [] }: { held?: readonly (DefaultComponentSlot['setting'] | 'vae')[] } = {}
): Partial<GenerateSettings> => {
  const model = withoutDefaultsOf(fillModel, held);
  const components = getModelDefaultComponents(model, settings, models);
  const current = withCatalogRecords(settings, models);
  const filled = { ...current, ...components };
  const hasVaeSlot = isSlotOffered(model, filled, 'vae');
  const fitsVae = (vae: VaeModelConfig) =>
    hasVaeSlot
      ? isComponentCompatibleWithModel(model, filled, 'vae', vae)
      : getGenerationUiPolicy(model).sdVaeVisible && isVaeCompatibleWithGenerateModel(model, vae);

  // A VAE that is installed and fits stays; one left over from another family (a fallback from an unsupported model)
  // gives way to the default.
  if (current.vae && isInstalled(models, current.vae) && fitsVae(current.vae)) {
    return components;
  }

  const defaultVae = resolveModelDefaultVae(model, filled, models.filter(isVaeModelConfig));

  if (defaultVae.kind !== 'apply') {
    return components;
  }

  // The bundled VAE (`null`) is what an empty slot already uses, and what a pick that is gone gives way to.
  if (defaultVae.vae === null) {
    return settings.vae ? { ...components, vae: null } : components;
  }

  return fitsVae(defaultVae.vae) ? { ...components, vae: defaultVae.vae } : components;
};

/** The IP-Adapter a reference image takes by default for `base`: the first one whose files are present. */
const findDefaultIpAdapter = <T extends ReferenceModelCandidate>(
  models: readonly T[],
  base: string | undefined
): T | null =>
  models.find((candidate) => candidate.type === 'ip_adapter' && candidate.base === base && !candidate.is_missing) ??
  null;

type ReferenceModelCandidate = { base: string; is_missing?: boolean; key: string; name: string; type: string };

// FLUX.1 Kontext starter sources from invokeai/backend/model_manager/starter_models/flux.py.
const FLUX_KONTEXT_REPOS = [
  'black-forest-labs/flux.1-kontext-dev',
  'comfy-org/flux1-kontext-dev_comfyui',
  'unsloth/flux.1-kontext-dev-gguf',
];

const isFluxMainModel = (model: GenerateModelConfig | undefined): model is MainModelConfig =>
  model?.type === 'main' && model.base === 'flux';

const isKnownFluxKontextModel = ({ name, source }: MainModelConfig): boolean =>
  name.toLowerCase().includes('kontext') ||
  (typeof source === 'string' && FLUX_KONTEXT_REPOS.some((repo) => isModelFromRepo(source, repo)));

const isBoundKontextReference = (model: MainModelConfig, config: GenerateReferenceImageConfig): boolean =>
  config.type === 'flux_kontext_reference_image' && config.model !== null && isSameInstalledModel(model, config.model);

/** Bases whose Generate graph loads LoRAs; graph coverage holds the builders to this list. */
const LORA_SUPPORTED_BASES: ReadonlySet<string> = new Set<SupportedGenerateBase>([
  'anima',
  'ernie-image',
  'flux',
  'flux2',
  'flux3',
  'ideogram-4',
  'krea-2',
  'moss-image',
  'qwen-image',
  'qwen-image-2-1',
  'sd-1',
  'sd-2',
  'sd-3',
  'sdxl',
  'wan',
  'z-image',
]);

export const isLoraSupported = (model: GenerateModelConfig | undefined): boolean =>
  Boolean(model && model.type !== 'external_image_generator' && LORA_SUPPORTED_BASES.has(model.base));

export type ReferenceVisionComponents = Partial<
  Pick<GenerateSettings, 'qwen3VLEncoderModel' | 'qwen3VLVisionTowerModel'>
>;

/**
 * A GGUF encoder has no vision tower (llama.cpp's mmproj), so it reads references only beside a selected 8B tower.
 * The 8B filters are Qwen-Image-2.1's, the only base with `reference_images_require_vision_tower`.
 */
const hasReferenceVisionTower = (
  model: Exclude<GenerateModelConfig, { type: 'external_image_generator' }>,
  { qwen3VLEncoderModel: encoder, qwen3VLVisionTowerModel: tower }: ReferenceVisionComponents
): boolean => {
  if (!encoder || !isQwen3Vl8bEncoder(encoder)) {
    return model.format === 'diffusers';
  }

  return isGgufQwen3Vl8bEncoder(encoder) ? Boolean(tower && isQwen3Vl8bVisionTower(tower)) : true;
};

/** Base and variant alone, so an encoder change never deletes references; validation flags them instead. */
export const takesReferenceImages = (model: GenerateModelConfig | undefined): boolean => {
  if (!model) {
    return false;
  }

  if (model.type === 'external_image_generator') {
    return model.capabilities?.supports_reference_images === true;
  }

  const features = getArchitectureFeatures(model.base);

  // The backend declares which reference-image features require a specific variant.
  return (
    Boolean(features) &&
    features!.max_reference_images > 0 &&
    (features!.reference_images_require_variant === null ||
      model.variant === features!.reference_images_require_variant)
  );
};

/** `components` (Qwen3-VL encoder, vision tower) matter only for bases whose references the encoder reads. */
export const isReferenceImageSupported = (
  model: GenerateModelConfig | undefined,
  components: ReferenceVisionComponents = {}
): boolean => {
  if (!model || !takesReferenceImages(model)) {
    return false;
  }

  if (model.type === 'external_image_generator') {
    return true;
  }

  return (
    !getArchitectureFeatures(model.base)?.reference_images_require_vision_tower ||
    hasReferenceVisionTower(model, components)
  );
};

export type ReferenceImageUnavailableReason = 'ggufVisionTower' | 'visionTower';

/** Why references are unavailable although the base takes them; null otherwise. */
export const getReferenceImageUnavailableReason = (
  model: GenerateModelConfig | undefined,
  components: ReferenceVisionComponents = {}
): ReferenceImageUnavailableReason | null => {
  if (!model || model.type === 'external_image_generator' || isReferenceImageSupported(model, components)) {
    return null;
  }

  if (!takesReferenceImages(model)) {
    return null;
  }

  return components.qwen3VLEncoderModel && isGgufQwen3Vl8bEncoder(components.qwen3VLEncoderModel)
    ? 'ggufVisionTower'
    : 'visionTower';
};

const getReferenceImageLimit = (model: GenerateModelConfig | undefined): number => {
  if (!model || !takesReferenceImages(model)) {
    return 0;
  }

  if (model.type === 'external_image_generator' && typeof model.capabilities?.max_reference_images === 'number') {
    return Math.max(0, model.capabilities.max_reference_images);
  }

  return getArchitectureFeatures(model.base)?.max_reference_images ?? DEFAULT_REFERENCE_IMAGE_LIMIT;
};

export type ReferenceImageRefusal = 'full' | 'unsupported' | ReferenceImageUnavailableReason;

export interface ReferenceImageRoom {
  free: number;
  isSupported: boolean;
  limit: number;
  refusal: ReferenceImageRefusal | null;
  unavailableReason: ReferenceImageUnavailableReason | null;
}

export const getReferenceImageRoom = (
  model: GenerateModelConfig | undefined,
  components: ReferenceVisionComponents,
  referenceImages: readonly GenerateReferenceImage[]
): ReferenceImageRoom => {
  if (!isReferenceImageSupported(model, components)) {
    const unavailableReason = getReferenceImageUnavailableReason(model, components);

    return { free: 0, isSupported: false, limit: 0, refusal: unavailableReason ?? 'unsupported', unavailableReason };
  }

  const limit = getReferenceImageLimit(model);
  const free = Math.max(0, limit - referenceImages.length);

  return { free, isSupported: true, limit, refusal: free > 0 ? null : 'full', unavailableReason: null };
};

const IP_ADAPTER_BASES: ReadonlySet<string> = new Set(['sd-1', 'sdxl', 'flux', 'anima']);

/** Anima's IP-Adapter node takes only a strength (own SigLIP2 encoder), so method and encoder are never sent. */
export const isStrengthOnlyIPAdapterBase = (base: string | undefined): boolean => base === 'anima';

export const createReferenceImageId = (): string =>
  `reference_image_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

/** Edit references switch Krea-2 into edit mode, which excludes style references and regional prompts. */
export const hasEnabledKrea2EditReferences = (referenceImages: readonly GenerateReferenceImage[]): boolean =>
  referenceImages.some(
    (referenceImage) => referenceImage.isEnabled && referenceImage.config.type === 'krea2_edit_reference_image'
  );

/** Without an edit LoRA the Krea-2 base just copies the references; a LoRA counts if enabled and compatible. */
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

  const adapterModel = findDefaultIpAdapter(models, modelBase);

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
      // The encoder decides whether they run (see `isReferenceImageSupported`); the type fits the base alone.
      return model.type !== 'external_image_generator' && model.base === 'qwen-image-2-1';
    case 'flux_kontext_reference_image':
      return (
        isFluxMainModel(model) &&
        (isBoundKontextReference(model, referenceImage.config) || isKnownFluxKontextModel(model))
      );
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

interface ReferenceImageAddition {
  image: GenerateReferenceImageAsset | null;
  isEnabled: boolean;
}

/** A Krea-2 addition is the style reference only if it would be the sole running Krea-2 reference. */
const getDefaultReferenceImageConfigs = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  additions: readonly ReferenceImageAddition[],
  referenceImages: readonly GenerateReferenceImage[]
): GenerateReferenceImageConfig[] => {
  if (
    isFluxMainModel(model) &&
    (isKnownFluxKontextModel(model) || referenceImages.some(({ config }) => isBoundKontextReference(model, config)))
  ) {
    return additions.map(({ image }) => ({ image, model, type: 'flux_kontext_reference_image' }));
  }

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
  if (enabled.length >= getReferenceImageLimit(model)) {
    return false;
  }

  if (model.base !== 'krea-2') {
    return true;
  }

  return referenceImage.config.type === 'krea2_reference_image'
    ? enabled.length === 0
    : enabled.every((kept) => kept.config.type !== 'krea2_reference_image');
};

/** The lone running Krea-2 style reference at default strength, which an addition turns into an edit reference. */
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

export interface NewReferenceImage {
  id: string;
  image: GenerateReferenceImageAsset | null;
}

// A lone first Krea-2 reference starts as style; the next addition turns it into edit unless its strength was tuned.
const withAddedReferenceImages = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  referenceImages: GenerateReferenceImage[],
  additions: readonly NewReferenceImage[]
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

/** Re-target an adapter-less SD reference on a strength-only base when its weight is rejected or an adapter exists. */
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

  return config.model === null && findDefaultIpAdapter(models, model.base) !== null;
};

/** Drops unsupported, retargets incompatible ones, keeping image/enabled state. No changes return the same array. */
export const getCompatibleReferenceImages = (
  referenceImages: GenerateReferenceImage[],
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[]
): GenerateReferenceImage[] => {
  if (referenceImages.length === 0) {
    return referenceImages;
  }

  if (!takesReferenceImages(model)) {
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

/**
 * Incompatible references are retargeted; ones the family cannot run together are switched off, not deleted.
 * `retargetedReferenceImages` is the list before the switch-off. No changes return the same array.
 */
export const reconcileReferenceImagesWithModel = (
  referenceImages: GenerateReferenceImage[],
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[]
): { referenceImages: GenerateReferenceImage[]; retargetedReferenceImages: GenerateReferenceImage[] } => {
  const referencesSupported = takesReferenceImages(model);
  const retargetedReferenceImages = referencesSupported
    ? getCompatibleReferenceImages(referenceImages, model, models)
    : referenceImages;
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

  return {
    referenceImages: didSwitchOff ? keptOrSwitchedOff : retargetedReferenceImages,
    retargetedReferenceImages,
  };
};

export type ReferenceImageAddRequest = {
  model: GenerateModelConfig | undefined;
  models: readonly ReferenceModelCandidate[];
  referenceImages: GenerateReferenceImage[];
  additions: readonly NewReferenceImage[];
} & (
  | { mode: 'strict'; components: ReferenceVisionComponents }
  | { mode: 'keep'; recalled?: readonly GenerateReferenceImage[] }
);

export interface ReferenceImageAddResult {
  referenceImages: GenerateReferenceImage[];
  added: number;
  dropped: number;
  limit: number;
  refusal: ReferenceImageRefusal | null;
  unavailableReason: ReferenceImageUnavailableReason | null;
}

/** `strict`: a user add, gated on vision components. `keep`: a recall; validation flags a missing encoder. */
export const addReferenceImages = (request: ReferenceImageAddRequest): ReferenceImageAddResult => {
  const { additions, model, models, referenceImages } = request;

  if (request.mode === 'strict') {
    const { free, isSupported, limit, refusal, unavailableReason } = getReferenceImageRoom(
      model,
      request.components,
      referenceImages
    );

    if (!isSupported) {
      return { added: 0, dropped: 0, limit, referenceImages, refusal, unavailableReason };
    }

    const fitting = additions.slice(0, free);

    return {
      added: fitting.length,
      dropped: additions.length - fitting.length,
      limit,
      referenceImages: withAddedReferenceImages(model, models, referenceImages, fitting),
      refusal,
      unavailableReason: null,
    };
  }

  if (!model || !takesReferenceImages(model)) {
    return { added: 0, dropped: 0, limit: 0, referenceImages, refusal: 'unsupported', unavailableReason: null };
  }

  const limit = getReferenceImageLimit(model);
  const joined = getCompatibleReferenceImages([...referenceImages, ...(request.recalled ?? [])], model, models);
  const fitting = additions.slice(0, Math.max(0, limit - joined.length));
  const placed = withAddedReferenceImages(model, models, joined, fitting).slice(0, limit);

  return {
    added: fitting.length,
    dropped: joined.length + additions.length - placed.length,
    limit,
    referenceImages: placed,
    refusal: null,
    unavailableReason: null,
  };
};

const getSettingsWithCompatibleModelSelections = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[],
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

  // Concepts stay listed; only the ones that ran before and stop now are reported.
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

  const { referenceImages: compatibleReferenceImages, retargetedReferenceImages } = reconcileReferenceImagesWithModel(
    settings.referenceImages,
    model,
    models
  );

  if (compatibleReferenceImages !== settings.referenceImages) {
    nextSettings.referenceImages = compatibleReferenceImages;
    // Only user tuning is lost: a reference still at the previous family's defaults loses nothing.
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
    const switchedOff = compatibleReferenceImages.filter(
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

  // The new model's defaults fill the slots left empty, which the confirmation does not count as a loss. A pick that
  // fits stays, and picking the same model again changes nothing.
  if (settings.modelKey !== model.key) {
    Object.assign(nextSettings, getModelDefaultFills(model, nextSettings, models));
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
  const stored = isGenerateSettings(currentValues) ? currentValues : normalizeGenerateSettings(currentValues);
  const computed = getSettingsWithCompatibleModelSelections(
    stored ?? getDefaultGenerateSettings(model),
    model,
    models,
    isCanvasFrame
  );
  // Nothing chosen before (an initial model): the model's defaults fill in as they do on a switch.
  const result = stored
    ? computed
    : { ...computed, settings: { ...computed.settings, ...getModelDefaultFills(model, computed.settings, models) } };
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

/** Its name, else its position, which `describeGenerationIssue` localizes ("Reference image 2"). */
const referenceNameValues = (
  referenceImage: Pick<GenerateReferenceImage, 'name'>,
  position: number
): Record<string, number | string> => ({
  index: position,
  ...(referenceImage.name === undefined ? {} : { referenceName: referenceImage.name }),
});
const englishReferenceName = (referenceImage: Pick<GenerateReferenceImage, 'name'>, position: number): string =>
  referenceImage.name ?? `Reference image ${position}`;
const REFERENCE_IMAGES_TARGET: GenerationIssue['target'] = { field: 'referenceImages', section: 'guidance' };
const referenceImageTarget = (referenceImageId: string): GenerationIssue['target'] => ({
  field: 'referenceImage',
  referenceImageId,
  section: 'guidance',
});

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

  const componentContext = getComponentPolicyContext(model, settings);
  const slotsByKey = new Map(
    getComponentSectionPolicy(model, settings).slots.map((slotPolicy) => [slotPolicy.key, slotPolicy])
  );

  for (const key of GENERATE_COMPONENT_VALUE_KEYS) {
    const value = settings[key];

    // A selection nothing loads (its switch is off) may stay uninstalled; turning the switch on asks for it again.
    if (!value || slotsByKey.get(key)?.inUse?.(componentContext) === false) {
      continue;
    }

    if (!hasModelKey(models, value.key, value.type)) {
      issues.push(
        issue(
          `components:notInstalled:${key}`,
          'componentNotInstalled',
          `${COMPONENT_SETTING_LABELS[key]} "${value.name}" is not installed. Choose another, or clear it.`,
          { section: 'components' },
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
          `${englishReferenceName(referenceImage, index + 1)} uses "${config.model.name}", which is not installed. Turn it off, or choose another model for it.`,
          referenceImageTarget(referenceImage.id),
          { ...referenceNameValues(referenceImage, index + 1), name: config.model.name },
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

/** References are read by a Diffusers encoder's tower, a non-GGUF Qwen3-VL, or a GGUF encoder's mmproj. */
export const QWEN_IMAGE_21_NO_VISION_TOWER_REASON =
  'Reference images need a Qwen3-VL 8B encoder with a vision tower. Select one under Components, or use a Diffusers Qwen-Image-2.1 model.';

/** The same, for a selected GGUF 8B encoder, which only needs its mmproj file added. */
export const QWEN_IMAGE_21_GGUF_VISION_TOWER_REASON =
  "Reference images need this encoder's Vision Tower (the mmproj file). Select it under Components.";

const VISION_TOWER_CPU_ONLY_REASON =
  'The Vision Tower and the encoder must have the same CPU Only setting. Change it in the Model Manager.';

const KREA2_VISION_TOWER_TARGET = { field: 'qwen3VLVisionTowerModel', section: 'components' } as const;

const KREA2_GGUF_VISION_TOWER_REASON =
  "Edit references need this GGUF encoder's Vision Tower (the mmproj file). Select it under Components.";

const getReferenceImageValidationIssues = (
  model: GenerateModelConfig,
  settings: GenerateSettings
): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];
  // Use the shared limit, but report unsupported images individually rather than a misleading maximum of zero.
  const maxReferenceImages = getReferenceImageLimit(model);
  const active = settings.referenceImages.filter(isReferenceImageActive);

  if (takesReferenceImages(model) && active.length > maxReferenceImages) {
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
          `${model.name} cannot use ${englishReferenceName(referenceImage, index)}. Turn it off, or remove it.`,
          referenceImageTarget(id),
          { ...referenceNameValues(referenceImage, index), name: model.name },
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
          `${englishReferenceName(referenceImage, index)} needs an image.`,
          referenceImageTarget(id),
          referenceNameValues(referenceImage, index),
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
            `${englishReferenceName(referenceImage, index)} needs an IP Adapter model for ${model.name}.`,
            referenceImageTarget(id),
            { ...referenceNameValues(referenceImage, index), name: model.name },
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
            `${englishReferenceName(referenceImage, index)} strength must be between 0 and 2.`,
            referenceImageTarget(id),
            referenceNameValues(referenceImage, index),
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
          `${englishReferenceName(referenceImage, index)} needs a FLUX Redux model for ${model.name}.`,
          referenceImageTarget(id),
          { ...referenceNameValues(referenceImage, index), name: model.name },
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
          `${englishReferenceName(referenceImage, index)} needs the selected FLUX Kontext model.`,
          referenceImageTarget(id),
          referenceNameValues(referenceImage, index),
          turnOff
        )
      );
    }
  });

  // A GGUF Qwen VL Encoder has no vision tower (llama.cpp splits it out), so the prompt node refuses references.
  const qwenVLEncoder = settings.qwenVLEncoderModel;
  if (
    qwenVLEncoder?.format === 'gguf_quantized' &&
    settings.referenceImages.some(
      (referenceImage) =>
        referenceImage.isEnabled &&
        referenceImage.config.type === 'qwen_image_reference_image' &&
        getReferenceImageConfigSupported(model, referenceImage)
    )
  ) {
    issues.push(
      createUntranslatedGenerationIssue(
        `Qwen VL Encoder "${qwenVLEncoder.name}" is a GGUF build with no vision tower, so it cannot read reference ` +
          'images. Select a safetensors or Diffusers Qwen VL Encoder, or turn the reference images off.',
        REFERENCE_IMAGES_TARGET,
        { key: 'qwenVLEncoderModel', kind: 'clearModel' }
      )
    );
  }

  const hasQwen21References =
    model.base === 'qwen-image-2-1' &&
    settings.referenceImages.some(
      (referenceImage) => referenceImage.isEnabled && referenceImage.config.type === 'qwen_image_2_1_reference_image'
    );

  // The shared encoder slot can carry one that cannot read references; clearing helps only with a bundled encoder.
  if (hasQwen21References && !isReferenceImageSupported(model, settings)) {
    issues.push(
      // A GGUF 8B encoder needs its tower; a single-file main must swap the encoder; a Diffusers one clears it.
      createUntranslatedGenerationIssue(
        getReferenceImageUnavailableReason(model, settings) === 'ggufVisionTower'
          ? QWEN_IMAGE_21_GGUF_VISION_TOWER_REASON
          : QWEN_IMAGE_21_NO_VISION_TOWER_REASON,
        settings.qwen3VLEncoderModel && isGgufQwen3Vl8bEncoder(settings.qwen3VLEncoderModel)
          ? { field: 'qwen3VLVisionTowerModel', section: 'components' }
          : model.format === 'diffusers'
            ? REFERENCE_IMAGES_TARGET
            : { field: 'qwen3VLEncoderModel', section: 'components' },
        model.format === 'diffusers' ? { key: 'qwen3VLEncoderModel', kind: 'clearModel' } : undefined
      )
    );
  }

  // The tower runs beside the encoder in one forward pass, so a device split between them cannot read references.
  const encoder = settings.qwen3VLEncoderModel;
  const tower = settings.qwen3VLVisionTowerModel;

  if (
    hasQwen21References &&
    encoder &&
    tower &&
    isGgufQwen3Vl8bEncoder(encoder) &&
    isQwen3Vl8bVisionTower(tower) &&
    (encoder.cpu_only === true) !== (tower.cpu_only === true)
  ) {
    issues.push(
      createUntranslatedGenerationIssue(VISION_TOWER_CPU_ONLY_REASON, {
        field: 'qwen3VLVisionTowerModel',
        section: 'components',
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

/** Combinations the backend refuses only after queueing. */
const getKrea2ReferenceValidationReasons = (
  settings: GenerateSettings
): Array<{ reason: string; target: GenerationIssue['target'] }> => {
  const styleCount = settings.referenceImages.filter(
    (referenceImage) => isReferenceImageActive(referenceImage) && referenceImage.config.type === 'krea2_reference_image'
  ).length;
  const hasEditReferences = hasEnabledKrea2EditReferences(settings.referenceImages);
  const reasons: Array<{ reason: string; target: GenerationIssue['target'] }> = [];

  if (styleCount > 1) {
    reasons.push({
      reason: 'Krea-2 uses one style reference at a time. Turn off all but one, or switch all of them to Edit.',
      target: REFERENCE_IMAGES_TARGET,
    });
  }

  if (styleCount > 0 && hasEditReferences) {
    reasons.push({
      reason:
        'Krea-2 cannot combine a style reference with edit references. Switch the style reference to Edit, or turn off one kind.',
      target: REFERENCE_IMAGES_TARGET,
    });
  }

  const encoder = settings.qwen3VLEncoderModel;
  const tower = settings.qwen3VLVisionTowerModel;

  if (hasEditReferences && encoder && isGgufQwen3Vl4bEncoder(encoder)) {
    if (!tower || !isQwen3Vl4bVisionTower(tower)) {
      reasons.push({ reason: KREA2_GGUF_VISION_TOWER_REASON, target: KREA2_VISION_TOWER_TARGET });
    } else if ((encoder.cpu_only === true) !== (tower.cpu_only === true)) {
      reasons.push({ reason: VISION_TOWER_CPU_ONLY_REASON, target: KREA2_VISION_TOWER_TARGET });
    }
  }

  return reasons;
};

/** Scalar family rules are separate from component-slot validation; their fields show the English inline. */
const getModelFamilyValidationIssues = (model: MainModelConfig, settings: GenerateSettings): GenerationIssue[] => {
  const issues: GenerationIssue[] = [];

  if (model.base === 'krea-2') {
    issues.push(
      ...getKrea2ReferenceValidationReasons(settings).map(({ reason, target }) =>
        createUntranslatedGenerationIssue(reason, target)
      )
    );
  }

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

  // A persisted list can name a block MOSS-Image lacks; without CFG it has no effect and its field cannot be fixed.
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

/** Shared by the Invoke route and the Generate form. Omit `models` while the catalog is unread. */
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

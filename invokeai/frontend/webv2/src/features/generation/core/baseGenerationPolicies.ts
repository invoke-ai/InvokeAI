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
import { SEED_MAX } from '@platform/core/seed';

import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  MainModelConfig,
  PidMode,
  VaePrecision,
} from './types';

import {
  getCompatibleDiffusersComponentSource,
  isBundledMainForBase,
  isAnimaQwen35Encoder,
  isAnimaQwen3Encoder,
  isClipVariant,
  isDiffusersMainForBase,
  isFlux2DiffusersSourceForModel,
  isErnieImageMistralEncoder,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isIdeogram4Qwen3VlEncoder,
  isIdeogram4UnconditionalBranch,
  isKrea2Qwen3VlEncoder,
  isNonAnimaQwen3Encoder,
  isSelfContainedSDNQFlux1Pipeline,
  isSelfContainedSDNQPipeline,
  isVaeAcceptedByBase,
  type GenerateComponentFilter,
} from './componentCompatibility';
import { DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS } from './dynamicPrompts';
import {
  DEFAULT_PID_STEPS,
  getIsPidActive,
  getIsPidSupportedBase,
  getPidDecoderBaseForMainBase,
  getPidDimensionOverrides,
} from './pid';
import {
  clampDimension,
  DEFAULT_HIDIFFUSION_T1_RATIO,
  DEFAULT_HIDIFFUSION_T2_RATIO,
  DEFAULT_IDEOGRAM4_SAMPLER_PRESET,
  DEFAULT_KREA2_REBALANCE_MULTIPLIER,
  DEFAULT_KREA2_REBALANCE_WEIGHTS,
  DEFAULT_KREA2_SEED_VARIANCE_RANDOMIZE_PERCENT,
  DEFAULT_KREA2_SEED_VARIANCE_STRENGTH,
  DEFAULT_REFERENCE_IMAGE_LIMIT,
  deriveAspectRatioId,
  IDEOGRAM4_GUIDANCE_MAX,
  IDEOGRAM4_GUIDANCE_MIN,
  IDEOGRAM4_MU_MAX,
  IDEOGRAM4_MU_MIN,
  IDEOGRAM4_STEPS_MAX,
  IDEOGRAM4_STEPS_MIN,
  isGenerateSettings,
  isLoraCompatibleWithModel,
  isValidKrea2RebalanceWeights,
  isWanLoraTargetingMain,
  KREA2_REBALANCE_WEIGHT_COUNT,
  MAX_DIMENSION,
  MIN_DIMENSION,
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

export interface GenerationModelPolicy {
  isSupported: boolean;
  dimensions: {
    grid: number;
    min: number;
    max: number;
    optimal: number;
  };
  defaults: {
    steps: number;
    cfgScale: number;
    cfgRescaleMultiplier: number;
    scheduler: string;
    vaePrecision: VaePrecision;
  };
  scheduler: {
    options: readonly SchedulerOption[];
    defaultValue: string;
    appliesToGraph: boolean;
    coerceForGraph: (value: string) => string;
  };
  prompt: {
    negativeVisible: boolean;
    negativeUsedInGraph: boolean;
    negativeHelpText?: string;
  };
  ui: {
    guidanceLabel: GuidanceLabel;
    /** The node-enforced floor for the guidance control; `null` max means the node enforces none. */
    guidanceMin: number;
    guidanceMax: number | null;
    schedulerVisible: boolean;
    clipSkipMax: number | null;
    cfgRescaleVisible: boolean;
    colorCompensationVisible: boolean;
    hiDiffusionVisible: boolean;
    seamlessVisible: boolean;
    sdVaeVisible: boolean;
    vaePrecisionVisible: boolean;
    seedVisible: boolean;
  };
}

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

/** Use variant-aware grids; return null for undescribed local architectures. External providers use a generic grid. */
export const getDimensionGrid = (base: string, variant?: unknown): number | null =>
  base === 'external'
    ? FALLBACK_GENERATION_CONFIG.dimensions.grid
    : (getArchitectureFeatures(base, variant)?.dimension_grid ?? null);

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
    config.dimensions.grid,
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

export const getGenerationUiPolicy = (
  model: GenerateModelConfig | undefined,
  _settings: Pick<GenerateSettings, 'cfgScale'>
) => {
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

export const getGenerationModelPolicy = (
  model: GenerateModelConfig | undefined,
  settings: GenerateSettings
): GenerationModelPolicy => {
  const config = getBaseGenerationConfig(model);
  const dimensions = getGenerationDimensions(model);
  const defaults = getGenerationDefaults(model);

  return {
    isSupported: model ? isSupportedGenerateModel(model) : false,
    dimensions,
    defaults,
    scheduler: {
      options: getSchedulerOptions(model, settings.scheduler),
      defaultValue: defaults.scheduler,
      appliesToGraph: config.schedulerAppliesToGraph,
      coerceForGraph: (value: string) => coerceSchedulerForGraph(model, value),
    },
    prompt: getPromptPolicy(model, settings),
    ui: getGenerationUiPolicy(model, settings),
  };
};

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
    dynamicPromptsMaxPrompts: DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS,
    dynamicPromptsSampleSeed: 0,
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
    pidMode: 'off',
    pidDecoderModel: null,
    gemma2EncoderModel: null,
    pidSteps: DEFAULT_PID_STEPS,
    referenceImages: [],
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
    loras: settings.loras.map((lora) =>
      isLoraCompatibleWithModel(lora.model, model) ? lora : { ...lora, isEnabled: false }
    ),
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
  | 'wanT5EncoderModel'
  | 'wanLowNoiseModel'
  | 'ideogram4UnconditionalModel'
  | 'componentSourceModel'
  | 'pidDecoderModel'
  | 'gemma2EncoderModel'
  | 'vae';

export interface ComponentPolicyContext {
  model: GenerateModelConfig;
  settings: GenerateSettings;
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
}

export interface ComponentSectionPolicy {
  defaultOpen: boolean;
  slots: readonly ComponentSlotPolicy[];
  validate: (ctx: ComponentPolicyContext) => string[];
}

export interface GenerateModelSelectionResult {
  settings: GenerateSettings;
  clearedLabels: readonly string[];
}

const TYPE_CLIP_EMBED: ModelTaxonomyType[] = ['clip_embed'];
const TYPE_MAIN: ModelTaxonomyType[] = ['main'];
const TYPE_MISTRAL: ModelTaxonomyType[] = ['mistral_encoder'];
const TYPE_QWEN3: ModelTaxonomyType[] = ['qwen3_encoder'];
const TYPE_QWEN_VL: ModelTaxonomyType[] = ['qwen_vl_encoder'];
const TYPE_QWEN3_VL: ModelTaxonomyType[] = ['qwen3_vl_encoder'];
const TYPE_QWEN35: ModelTaxonomyType[] = ['qwen3_5_encoder'];
const TYPE_WAN_T5: ModelTaxonomyType[] = ['wan_t5_encoder'];
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
    filter: (candidate) => candidate.type === 'clip_embed',
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

const qwen35EncoderSlot = (helpText: string): ComponentSlotPolicy =>
  slot({
    key: 'qwen35EncoderModel',
    label: 'Qwen3.5 Encoder',
    modelTypes: TYPE_QWEN35,
    valueKind: 'component',
    helpText,
    filter: wrapFilter(isAnimaQwen35Encoder),
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

// Slot validation is shared by invocation readiness and graph preflight so picker requirements cannot drift.
const validateSlots = (policy: Pick<ComponentSectionPolicy, 'slots'>, ctx: ComponentPolicyContext): string[] =>
  policy.slots.flatMap((slotPolicy) => {
    if (!slotPolicy.required?.(ctx)) {
      return [];
    }

    const value = ctx.selectedComponents[slotPolicy.key];
    const isValid = value && (!slotPolicy.filter || slotPolicy.filter(value as ModelConfig, ctx));

    return isValid ? [] : [slotPolicy.missingMessage ?? `Generate needs a ${slotPolicy.label} for this model.`];
  });

const createPolicy = (defaultOpen: boolean, slots: readonly ComponentSlotPolicy[]): ComponentSectionPolicy => ({
  defaultOpen,
  slots,
  validate: (ctx) => validateSlots({ slots }, ctx),
});

const EMPTY_COMPONENT_POLICY: ComponentSectionPolicy = createPolicy(false, []);

const getBaseComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  _settings: GenerateSettings
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
        validate: (ctx) => [
          ...(ctx.model.type !== 'external_image_generator' && ctx.model.variant === 'dev_fill'
            ? ['FLUX Fill models do not support text-to-image generation.']
            : []),
          ...validateSlots(getBaseComponentSectionPolicy(ctx.model, ctx.settings), ctx),
        ],
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
          ...qwen3EncoderSlot('Required unless a Diffusers component source is available.', isNonAnimaQwen3Encoder),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Qwen3 Encoder for Z-Image models.',
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
          ...qwen3VlEncoderSlot('Required for non-Diffusers Krea-2 models.', isKrea2Qwen3VlEncoder),
          required: (ctx) => ctx.model.format !== 'diffusers',
          missingMessage: 'Generate needs a Qwen3-VL Encoder for non-Diffusers Krea-2 models.',
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
            isIdeogram4Qwen3VlEncoder
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
                ...qwen35EncoderSlot('Anima-3.8B reads Qwen3.5 4B beside Qwen3 0.6B. Required for this model.'),
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
      ]);
    default:
      return EMPTY_COMPONENT_POLICY;
  }
};

/** Add and validate PiD slots; each base policy retains ownership of its other component validation. */
export const getComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  settings: GenerateSettings
): ComponentSectionPolicy => {
  const policy = getBaseComponentSectionPolicy(model, settings);

  if (!model || model.type === 'external_image_generator' || !getIsPidSupportedBase(model.base)) {
    return policy;
  }

  const slots = [...policy.slots, ...pidSlots()];

  return {
    defaultOpen: policy.defaultOpen,
    slots,
    validate: (ctx) => [...policy.validate(ctx), ...validateSlots({ slots: pidSlots() }, ctx)],
  };
};

const getComponentPolicyContext = (model: GenerateModelConfig, settings: GenerateSettings): ComponentPolicyContext => ({
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
    wanT5EncoderModel: settings.wanT5EncoderModel,
    wanLowNoiseModel: settings.wanLowNoiseModel,
    ideogram4UnconditionalModel: settings.ideogram4UnconditionalModel,
    t5EncoderModel: settings.t5EncoderModel,
    pidDecoderModel: settings.pidDecoderModel,
    gemma2EncoderModel: settings.gemma2EncoderModel,
    vae: settings.vae,
  },
});

const COMPONENT_SETTING_LABELS: Record<GenerateComponentValueKey, string> = {
  clipEmbedModel: 'CLIP Embed',
  clipGEmbedModel: 'CLIP G',
  clipLEmbedModel: 'CLIP L',
  mistralEncoderModel: 'Mistral Encoder',
  qwen3EncoderModel: 'Qwen3 Encoder',
  qwenVLEncoderModel: 'Qwen VL Encoder',
  qwen3VLEncoderModel: 'Qwen3-VL Encoder',
  qwen35EncoderModel: 'Qwen3.5 Encoder',
  wanT5EncoderModel: 'Wan T5 Encoder',
  wanLowNoiseModel: 'Low-noise expert',
  ideogram4UnconditionalModel: 'Transformer (Unconditional)',
  t5EncoderModel: 'T5 Encoder',
  pidDecoderModel: 'PiD Decoder',
  gemma2EncoderModel: 'Gemma-2 Encoder',
  vae: 'VAE',
  componentSourceModel: 'Component source',
};

const addClearedLabel = (labels: string[], label: string) => {
  if (!labels.includes(label)) {
    labels.push(label);
  }
};

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

  // The backend declares which reference-image features require a specific variant.
  return (
    features.reference_images_require_variant === null || model.variant === features.reference_images_require_variant
  );
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

export const createReferenceImageId = (): string =>
  `reference_image_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

export const getDefaultReferenceImageConfig = (
  model: GenerateModelConfig | undefined,
  models: readonly ReferenceModelCandidate[],
  image: GenerateReferenceImageAsset | null = null
): GenerateReferenceImageConfig => {
  const modelBase: string | undefined = model?.base;

  if (model?.type === 'external_image_generator') {
    return { image, type: 'external_reference_image' };
  }

  if (modelBase === 'flux2') {
    return { image, type: 'flux2_reference_image' };
  }

  if (modelBase === 'qwen-image') {
    return { image, type: 'qwen_image_reference_image' };
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
      return model.type !== 'external_image_generator' && model.base === 'flux2';
    case 'qwen_image_reference_image':
      return model.type !== 'external_image_generator' && model.base === 'qwen-image' && model.variant === 'edit';
    case 'flux_kontext_reference_image':
      return isFluxKontextModel(model);
    case 'flux_redux':
      return model.type !== 'external_image_generator' && model.base === 'flux';
    case 'ip_adapter':
      return model.type !== 'external_image_generator' && ['sd-1', 'sdxl', 'flux'].includes(model.base);
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

/**
 * Drop unsupported references; retarget incompatible ones while preserving image/enabled state. No changes return
 * the same array.
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

  let didChange = false;
  const next = referenceImages.map((referenceImage) => {
    if (isReferenceImageCompatibleWithModel(model, referenceImage)) {
      return referenceImage;
    }

    didChange = true;
    return { ...referenceImage, config: getDefaultReferenceImageConfig(model, models, referenceImage.config.image) };
  });

  return didChange ? next : referenceImages;
};

const getSettingsWithCompatibleModelSelections = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[]
): GenerateModelSelectionResult => {
  const nextSettings: GenerateSettings = { ...settings, modelKey: model.key };
  const clearedLabels: string[] = [];
  const compatibleLoras = settings.loras.filter((lora) => isLoraCompatibleWithModel(lora.model, model));
  const dimensions = getGenerationDimensions(model);
  const width = clampDimension(nextSettings.width, dimensions.grid);
  const height = clampDimension(nextSettings.height, dimensions.grid);

  if (width !== nextSettings.width || height !== nextSettings.height) {
    nextSettings.width = width;
    nextSettings.height = height;
    nextSettings.aspectRatioId = deriveAspectRatioId(width, height);
    nextSettings.aspectRatioValue = height > 0 ? width / height : 1;
    addClearedLabel(clearedLabels, 'Dimensions');
  }

  if (compatibleLoras.length !== settings.loras.length) {
    nextSettings.loras = compatibleLoras;
    addClearedLabel(clearedLabels, 'LoRAs');
  }

  const compatibleReferenceImages = getCompatibleReferenceImages(settings.referenceImages, model, models);

  if (compatibleReferenceImages !== settings.referenceImages) {
    nextSettings.referenceImages = compatibleReferenceImages;
    addClearedLabel(clearedLabels, 'Reference Images');
  }

  const policy = getComponentSectionPolicy(model, nextSettings);
  const slotsByKey = new Map(policy.slots.map((slotPolicy) => [slotPolicy.key, slotPolicy]));
  const uiPolicy = getGenerationUiPolicy(model, nextSettings);

  for (const key of Object.keys(COMPONENT_SETTING_LABELS) as GenerateComponentValueKey[]) {
    if (!nextSettings[key]) {
      continue;
    }

    if (key === 'vae' && !slotsByKey.has(key) && uiPolicy.sdVaeVisible && nextSettings.vae?.base === model.base) {
      continue;
    }

    if (!isSelectedComponentCompatible(slotsByKey.get(key), model, nextSettings, key)) {
      nextSettings[key] = null;
      addClearedLabel(clearedLabels, COMPONENT_SETTING_LABELS[key]);
    }
  }

  // Clamp and report during selection; validation remains the backstop for other write paths.
  const clampedGuidance = clampGuidance(nextSettings.cfgScale, getBaseGenerationConfig(model).guidance);

  if (clampedGuidance !== nextSettings.cfgScale) {
    nextSettings.cfgScale = clampedGuidance;
    addClearedLabel(clearedLabels, uiPolicy.guidanceLabel);
  }

  if (!uiPolicy.clipSkipMax && nextSettings.clipSkip !== 0) {
    nextSettings.clipSkip = 0;
    addClearedLabel(clearedLabels, 'CLIP skip');
  } else if (uiPolicy.clipSkipMax && nextSettings.clipSkip > uiPolicy.clipSkipMax) {
    nextSettings.clipSkip = uiPolicy.clipSkipMax;
    addClearedLabel(clearedLabels, 'CLIP skip');
  }

  if (!uiPolicy.cfgRescaleVisible && nextSettings.cfgRescaleMultiplier !== 0) {
    nextSettings.cfgRescaleMultiplier = 0;
    addClearedLabel(clearedLabels, 'CFG rescale');
  }

  if (!uiPolicy.colorCompensationVisible && nextSettings.colorCompensation) {
    nextSettings.colorCompensation = false;
    addClearedLabel(clearedLabels, 'Color compensation');
  }

  if (!uiPolicy.hiDiffusionVisible && nextSettings.hiDiffusionEnabled) {
    nextSettings.hiDiffusionEnabled = false;
    addClearedLabel(clearedLabels, 'HiDiffusion');
  }

  if (!uiPolicy.seamlessVisible && (nextSettings.seamlessXAxis || nextSettings.seamlessYAxis)) {
    nextSettings.seamlessXAxis = false;
    nextSettings.seamlessYAxis = false;
    addClearedLabel(clearedLabels, 'Seamless tiling');
  }

  if (!uiPolicy.vaePrecisionVisible) {
    const defaultVaePrecision = getGenerationDefaults(model).vaePrecision;

    if (nextSettings.vaePrecision !== defaultVaePrecision) {
      nextSettings.vaePrecision = defaultVaePrecision;
      addClearedLabel(clearedLabels, 'VAE precision');
    }
  }

  return { settings: nextSettings, clearedLabels };
};

/** Normalize, reconcile, and apply component policy before persisting model selection. */
export const getGenerateModelSelectionResult = ({
  currentValues,
  model,
  models,
}: {
  currentValues: unknown;
  model: GenerateModelConfig;
  models: readonly ModelConfig[];
}): GenerateModelSelectionResult => {
  const currentSettings = isGenerateSettings(currentValues)
    ? currentValues
    : (normalizeGenerateSettings(currentValues) ?? getDefaultGenerateSettings(model));
  const result = getSettingsWithCompatibleModelSelections(currentSettings, model, models);
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

export const getGenerationModelAvailabilityReasons = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  models: readonly ModelConfig[]
): string[] => {
  const reasons: string[] = [];

  if (!hasModelKey(models, model.key, model.type)) {
    reasons.push(`Selected model "${model.name}" is no longer installed.`);
  }

  for (const key of Object.keys(COMPONENT_SETTING_LABELS) as GenerateComponentValueKey[]) {
    const value = settings[key];

    if (value && !hasModelKey(models, value.key, value.type)) {
      reasons.push(`${COMPONENT_SETTING_LABELS[key]} "${value.name}" is no longer installed.`);
    }
  }

  for (const lora of settings.loras) {
    if (!hasModelKey(models, lora.model.key, 'lora')) {
      reasons.push(`LoRA "${lora.model.name}" is no longer installed.`);
    }
  }

  for (const referenceImage of settings.referenceImages) {
    if (!referenceImage.isEnabled) {
      continue;
    }

    const config = referenceImage.config;

    if ('model' in config && config.model && !hasModelKey(models, config.model.key, config.model.type)) {
      reasons.push(`Reference Image model "${config.model.name}" is no longer installed.`);
    }
  }

  return reasons;
};

const getDimensionValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  const reasons: string[] = [];
  // Native target dimensions must be divisible by model grid × 4.
  const dimensions = getGenerationDimensions(model, settings.pidMode);

  if (!Number.isFinite(settings.width) || settings.width < dimensions.min || settings.width > dimensions.max) {
    reasons.push(`Generate width must be between ${dimensions.min} and ${dimensions.max}.`);
  } else if (settings.width % dimensions.grid !== 0) {
    reasons.push(`Generate width must be a multiple of ${dimensions.grid}.`);
  }

  if (!Number.isFinite(settings.height) || settings.height < dimensions.min || settings.height > dimensions.max) {
    reasons.push(`Generate height must be between ${dimensions.min} and ${dimensions.max}.`);
  } else if (settings.height % dimensions.grid !== 0) {
    reasons.push(`Generate height must be a multiple of ${dimensions.grid}.`);
  }

  return reasons;
};

/** Reject unsupported PiD instead of silently using ordinary decode. */
const getPidValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  if (settings.pidMode === 'off') {
    return [];
  }

  if (!getIsPidSupportedBase(model.base)) {
    return [`PiD is not supported for ${model.name}. Turn PiD off to generate with this model.`];
  }

  const reasons: string[] = [];

  if (settings.pidSteps < 1) {
    reasons.push('PiD steps must be at least 1.');
  }

  return reasons;
};

const getReferenceImageValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  const reasons: string[] = [];
  const enabled = settings.referenceImages.filter((referenceImage) => referenceImage.isEnabled);
  // Use the shared reference limit, but report unsupported images individually rather than a misleading maximum of
  // zero.
  const maxReferenceImages = getMaxReferenceImages(model);

  if (isReferenceImageSupported(model) && enabled.length > maxReferenceImages) {
    reasons.push(`Generate supports at most ${maxReferenceImages} reference images for ${model.name}.`);
  }

  enabled.forEach((referenceImage, index) => {
    const prefix = `Reference Image #${index + 1}`;

    if (!getReferenceImageConfigSupported(model, referenceImage)) {
      reasons.push(`${prefix} is not supported by ${model.base} model.`);
      return;
    }

    if (!referenceImage.config.image) {
      reasons.push(`${prefix} needs an image.`);
    }

    if (referenceImage.config.type === 'ip_adapter') {
      if (!referenceImage.config.model || referenceImage.config.model.base !== model.base) {
        reasons.push(`${prefix} needs a compatible IP Adapter model.`);
      }
    }

    if (referenceImage.config.type === 'flux_redux') {
      if (!referenceImage.config.model || referenceImage.config.model.base !== model.base) {
        reasons.push(`${prefix} needs a compatible FLUX Redux model.`);
      }
    }

    if (referenceImage.config.type === 'flux_kontext_reference_image') {
      if (!referenceImage.config.model || referenceImage.config.model.key !== model.key) {
        reasons.push(`${prefix} needs the selected FLUX Kontext model.`);
      }
    }
  });

  return reasons;
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

/** Scalar/LoRA family rules are separate from component-slot validation. */
const getModelFamilyValidationReasons = (model: MainModelConfig, settings: GenerateSettings): string[] => {
  const reasons: string[] = [];

  // Parse raw rebalance strings before forwarding them to the backend.
  if (
    model.base === 'krea-2' &&
    settings.krea2RebalanceEnabled &&
    !isValidKrea2RebalanceWeights(settings.krea2RebalanceWeights)
  ) {
    reasons.push(`Krea-2 rebalance weights must be ${KREA2_REBALANCE_WEIGHT_COUNT} comma-separated numbers.`);
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
        reasons.push(`Ideogram 4 ${label} must be between ${min} and ${max}.`);
      }
    }
  }

  // Report Wan LoRA family mismatches rather than silently drop selected LoRAs.
  if (model.base === 'wan') {
    for (const lora of settings.loras) {
      if (lora.isEnabled && !isWanLoraTargetingMain(lora.model.variant, model.variant)) {
        reasons.push(`${lora.model.name} targets a different Wan model family than ${model.name}.`);
      }
    }
  }

  return reasons;
};

export const getGenerationValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  // All compilers share this validator; check capabilities before model-specific policy.
  if (!hasArchitectureCapabilities()) {
    // Keep core independent of the load-state store; failures direct callers to retry.
    return [
      'Model capabilities are not available. Generation is blocked until they load; if this persists, retry from the Generate panel.',
    ];
  }

  if (!isSupportedGenerateModel(model)) {
    return ['Generate needs a supported model before it can be invoked.'];
  }

  // Reject supported bases missing from the loaded table rather than compile fallback policy.
  if (!isArchitectureDescribed(model)) {
    return [`The backend does not describe the ${model.base} architecture, so it cannot be generated with.`];
  }

  const reasons = [
    ...getDimensionValidationReasons(model, settings),
    ...getReferenceImageValidationReasons(model, settings),
    ...getPidValidationReasons(model, settings),
  ];

  if (model.type === 'external_image_generator') {
    if (model.capabilities?.modes && !model.capabilities.modes.includes('txt2img')) {
      reasons.push(`${model.name} does not support text-to-image generation.`);
    }

    if (!getExternalProviderNodeType(model.provider_id)) {
      reasons.push(`No invocation node registered for external provider '${model.provider_id ?? ''}'.`);
    }

    return reasons;
  }

  const componentPolicy = getComponentSectionPolicy(model, settings);
  reasons.push(...componentPolicy.validate(getComponentPolicyContext(model, settings)));
  const guidanceReason = getGuidanceBoundReason(model, settings.cfgScale);

  if (guidanceReason) {
    reasons.push(guidanceReason);
  }

  reasons.push(...getModelFamilyValidationReasons(model, settings));

  return reasons;
};

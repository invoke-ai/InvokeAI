import type { KnownModelBase } from '@workbench/models/baseIdentity';
import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';

import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
  GenerateSettings,
  MainModelConfig,
  VaePrecision,
} from './types';

import {
  getCompatibleDiffusersComponentSource,
  isAnimaQwen3Encoder,
  isAnimaVae,
  isClipVariant,
  isDiffusersMainForBase,
  isFlux2DiffusersSourceForModel,
  isFlux2Qwen3EncoderForModel,
  isNonAnimaQwen3Encoder,
  isVaeForBases,
  type GenerateComponentFilter,
} from './componentCompatibility';
import {
  clampDimension,
  DEFAULT_REFERENCE_IMAGE_LIMIT,
  deriveAspectRatioId,
  isLoraCompatibleWithModel,
  MAX_DIMENSION,
  MIN_DIMENSION,
  SEED_MAX,
} from './settings';

// Generation policy registry keyed by model base. Display identity stays in @workbench/models/baseIdentity;
// graph topology stays explicit in graph.ts.

export interface SchedulerOption {
  value: string;
  label: string;
}

export type SchedulerSetId = 'standard' | 'flow' | 'flow-no-lcm' | 'anima';

export type NegativePromptUsage = 'always' | 'cfg-gated' | 'never';

export type GuidanceLabel = 'CFG' | 'Guidance';

export interface BaseGenerationConfig {
  dimensions: {
    grid: number;
    optimalSide: number;
  };
  defaults: {
    steps: number;
    cfgScale: number;
    scheduler: string;
  };
  schedulerSet: SchedulerSetId;
  schedulerAppliesToGraph: boolean;
  guidanceLabel: GuidanceLabel;
  negativePrompt: {
    visible: boolean;
    usage: NegativePromptUsage;
  };
  ui: {
    sdVaeOverride: boolean;
    colorCompensation: boolean;
    vaePrecision: boolean;
    seamless: boolean;
    cfgRescale: boolean;
    clipSkipMax?: number;
  };
}

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

export const BASE_GENERATION = {
  'sd-1': {
    dimensions: { grid: 8, optimalSide: 512 },
    defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'always' },
    ui: {
      sdVaeOverride: true,
      colorCompensation: false,
      vaePrecision: true,
      seamless: true,
      cfgRescale: true,
      clipSkipMax: 12,
    },
  },
  'sd-2': {
    dimensions: { grid: 8, optimalSide: 512 },
    defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'always' },
    ui: {
      sdVaeOverride: true,
      colorCompensation: false,
      vaePrecision: true,
      seamless: true,
      cfgRescale: true,
      clipSkipMax: 24,
    },
  },
  sdxl: {
    dimensions: { grid: 8, optimalSide: 1024 },
    defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'always' },
    ui: { sdVaeOverride: true, colorCompensation: true, vaePrecision: true, seamless: true, cfgRescale: false },
  },
  'sd-3': {
    dimensions: { grid: 16, optimalSide: 1024 },
    defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: false,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'always' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  flux: {
    dimensions: { grid: 16, optimalSide: 1024 },
    defaults: { steps: 4, cfgScale: 4, scheduler: 'euler' },
    schedulerSet: 'flow',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'Guidance',
    negativePrompt: { visible: false, usage: 'never' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  flux2: {
    dimensions: { grid: 16, optimalSide: 1024 },
    defaults: { steps: 4, cfgScale: 1, scheduler: 'euler' },
    schedulerSet: 'flow',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'Guidance',
    negativePrompt: { visible: false, usage: 'never' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  cogview4: {
    dimensions: { grid: 32, optimalSide: 1024 },
    defaults: { steps: 30, cfgScale: 7, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: false,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'always' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  'qwen-image': {
    dimensions: { grid: 16, optimalSide: 1024 },
    defaults: { steps: 40, cfgScale: 4, scheduler: 'euler_a' },
    schedulerSet: 'standard',
    schedulerAppliesToGraph: false,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'cfg-gated' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  'z-image': {
    dimensions: { grid: 16, optimalSide: 1024 },
    defaults: { steps: 8, cfgScale: 1, scheduler: 'euler' },
    schedulerSet: 'flow',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'cfg-gated' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
  anima: {
    dimensions: { grid: 8, optimalSide: 1024 },
    defaults: { steps: 30, cfgScale: 4, scheduler: 'euler' },
    schedulerSet: 'anima',
    schedulerAppliesToGraph: true,
    guidanceLabel: 'CFG',
    negativePrompt: { visible: true, usage: 'cfg-gated' },
    ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
  },
} as const satisfies Partial<Record<KnownModelBase, BaseGenerationConfig>>;

export type SupportedGenerateBase = keyof typeof BASE_GENERATION;

export const SUPPORTED_GENERATE_BASES = Object.keys(BASE_GENERATION) as SupportedGenerateBase[];

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
    schedulerVisible: boolean;
    clipSkipMax: number | null;
    cfgRescaleVisible: boolean;
    colorCompensationVisible: boolean;
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
  negativePrompt: { visible: true, usage: 'never' },
  ui: { sdVaeOverride: false, colorCompensation: false, vaePrecision: false, seamless: false, cfgRescale: false },
};

// Fallbacks keep UI selectors crash-safe; isSupportedGenerateModel() still blocks invocation.
const getBaseGenerationConfig = (
  model: Pick<GenerateModelConfig, 'base' | 'type'> | undefined
): BaseGenerationConfig => {
  if (!model || model.type === 'external_image_generator') {
    return FALLBACK_GENERATION_CONFIG;
  }

  return (BASE_GENERATION as Partial<Record<string, BaseGenerationConfig>>)[model.base] ?? FALLBACK_GENERATION_CONFIG;
};

const getNumber = (value: number | null | undefined, fallback: number): number =>
  Number.isFinite(value) && value !== null && value !== undefined ? value : fallback;

export const getGenerationDimensions = (model: Pick<GenerateModelConfig, 'base' | 'type'> | undefined) => {
  const config = getBaseGenerationConfig(model);

  return {
    grid: config.dimensions.grid,
    min: MIN_DIMENSION,
    max: MAX_DIMENSION,
    optimal: config.dimensions.optimalSide,
  };
};

export const getGenerationDefaults = (model: GenerateModelConfig | undefined) => {
  const config = getBaseGenerationConfig(model);
  const defaults = model?.default_settings as GenerateDefaultSettings;

  return {
    cfgRescaleMultiplier: getNumber(defaults?.cfg_rescale_multiplier, 0),
    cfgScale: getNumber(defaults?.cfg_scale ?? defaults?.guidance, config.defaults.cfgScale),
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
    if (model?.base === 'z-image' && model.variant === 'zbase') {
      return FLOW_SCHEDULER_OPTIONS_WITHOUT_LCM;
    }

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

  if (model?.base === 'z-image' && model.variant === 'zbase') {
    return FLOW_SCHEDULERS_WITHOUT_LCM.has(scheduler) ? scheduler : 'euler';
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
    schedulerVisible: config.schedulerAppliesToGraph,
    clipSkipMax: config.ui.clipSkipMax ?? null,
    cfgRescaleVisible: config.ui.cfgRescale,
    colorCompensationVisible: config.ui.colorCompensation,
    seamlessVisible: config.ui.seamless,
    sdVaeVisible: config.ui.sdVaeOverride,
    vaePrecisionVisible: config.ui.vaePrecision,
    seedVisible,
  };
};

export const isSupportedGenerateModel = <T extends { base: string; type: string }>(
  model: T
): model is T & GenerateModelConfig =>
  (model.type === 'main' && model.base in BASE_GENERATION) ||
  (model.type === 'external_image_generator' && model.base === 'external');

export const isGenerateModelSelectable = <T extends ModelConfig>(model: T): boolean => isSupportedGenerateModel(model);

export const EXTERNAL_PROVIDER_NODE_TYPES: Record<string, string> = {
  alibabacloud: 'alibabacloud_image_generation',
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
    clipEmbedModel: null,
    clipGEmbedModel: null,
    clipLEmbedModel: null,
    componentSourceModel: null,
    height,
    loras: [],
    modelKey: model?.key ?? '',
    negativePromptEnabled: true,
    negativePrompt: '',
    negativePromptHeightPx: 56,
    positivePrompt: '',
    positivePromptHeightPx: 96,
    qwen3EncoderModel: null,
    qwenVLEncoderModel: null,
    referenceImages: [],
    scheduler: defaults.scheduler,
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: Math.floor(Math.random() * SEED_MAX),
    shouldRandomizeSeed: true,
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
  | 'qwen3EncoderModel'
  | 'qwenVLEncoderModel'
  | 'componentSourceModel'
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

export interface GenerateSettingsCompatibilityResult {
  settings: GenerateSettings;
  clearedLabels: string[];
}

const TYPE_CLIP_EMBED: ModelTaxonomyType[] = ['clip_embed'];
const TYPE_MAIN: ModelTaxonomyType[] = ['main'];
const TYPE_QWEN3: ModelTaxonomyType[] = ['qwen3_encoder'];
const TYPE_QWEN_VL: ModelTaxonomyType[] = ['qwen_vl_encoder'];
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
  Boolean(source && source.type === 'main' && source.base === 'flux2' && source.format === 'diffusers');

const hasFlux2DiffusersVaeSource = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (ctx.model.format === 'diffusers' || isFlux2DiffusersSource(ctx.settings.componentSourceModel));

const hasFlux2DiffusersQwen3Source = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (ctx.model.format === 'diffusers' ||
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

  if (settings.qwen3EncoderModel) {
    return source;
  }

  return isFlux2DiffusersSourceForModel(model)(source) ? source : undefined;
};

export const getAutoFlux2ComponentSourceModel = (
  model: GenerateModelConfig | undefined,
  settings: Pick<GenerateSettings, 'qwen3EncoderModel' | 'vae'>,
  models: readonly ComponentSourceCandidate[]
): MainModelConfig | null | undefined => {
  if (!model || model.type === 'external_image_generator' || model.base !== 'flux2') {
    return undefined;
  }

  if (model.format === 'diffusers' || (settings.qwen3EncoderModel && settings.vae)) {
    return null;
  }

  const diffusersModels = models.filter(isFlux2DiffusersSource);
  const variantMatch = diffusersModels.find(isFlux2DiffusersSourceForModel(model));

  if (!settings.qwen3EncoderModel && settings.vae) {
    return variantMatch ?? null;
  }

  return variantMatch ?? diffusersModels[0] ?? null;
};

// Split/quantized families can satisfy required encoder/VAE slots from a bundled Diffusers source.
const isBundledOrDiffusersSourceSatisfied = (ctx: ComponentPolicyContext): boolean =>
  ctx.model.type !== 'external_image_generator' &&
  (ctx.model.format === 'diffusers' || hasCompatibleComponentSource(ctx));

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

export const getComponentSectionPolicy = (
  model: GenerateModelConfig | undefined,
  _settings: GenerateSettings
): ComponentSectionPolicy => {
  if (!model || model.type === 'external_image_generator') {
    return EMPTY_COMPONENT_POLICY;
  }

  switch (model.base) {
    case 'flux':
      return {
        ...createPolicy(false, [
          {
            ...t5EncoderSlot('Required for FLUX.1 models.'),
            required: () => true,
            missingMessage: 'Generate needs a T5 Encoder for FLUX models.',
          },
          {
            ...clipEmbedSlot('Required for FLUX.1 models.'),
            required: () => true,
            missingMessage: 'Generate needs a CLIP Embed model for FLUX models.',
          },
          {
            ...vaeSlot('Required for FLUX.1 models.', isVaeForBases(['flux'])),
            required: () => true,
            missingMessage: 'Generate needs a VAE for FLUX models.',
          },
        ]),
        validate: (ctx) => [
          ...(ctx.model.type !== 'external_image_generator' && ctx.model.variant === 'dev_fill'
            ? ['FLUX Fill models do not support text-to-image generation.']
            : []),
          ...validateSlots(getComponentSectionPolicy(ctx.model, ctx.settings), ctx),
        ],
      };
    case 'flux2':
      return createPolicy(model.format !== 'diffusers', [
        {
          ...qwen3EncoderSlot('Optional override; otherwise a compatible installed FLUX.2 Diffusers model is used.'),
          filter: (candidate, ctx) =>
            ctx.model.type !== 'external_image_generator' && isFlux2Qwen3EncoderForModel(ctx.model)(candidate),
          required: (ctx) => !hasFlux2DiffusersQwen3Source(ctx),
          missingMessage: 'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.',
        },
        {
          ...vaeSlot(
            'Optional override; otherwise an installed FLUX.2 Diffusers model is used.',
            isVaeForBases(['flux2'])
          ),
          required: (ctx) => !hasFlux2DiffusersVaeSource(ctx),
          missingMessage: 'Generate needs a VAE for non-Diffusers FLUX.2 models.',
        },
      ]);
    case 'sd-3':
      return createPolicy(false, [
        t5EncoderSlot('Optional override; the main model is used when omitted.'),
        clipVariantSlot('clipLEmbedModel', 'CLIP L', 'large', 'Optional CLIP-L override.'),
        clipVariantSlot('clipGEmbedModel', 'CLIP G', 'gigantic', 'Optional CLIP-G override.'),
        vaeSlot('Optional VAE override.', isVaeForBases(['sd-3'])),
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
            'Optional override, or required with a non-Diffusers model and no component source.',
            isVaeForBases(['qwen-image'])
          ),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a VAE for non-Diffusers Qwen Image models.',
        },
      ]);
    case 'z-image':
      return createPolicy(model.format !== 'diffusers', [
        componentSourceSlot(
          (candidate) => isDiffusersMainForBase('z-image')(candidate),
          'Select a Diffusers Z-Image model to provide VAE and Qwen3 components.'
        ),
        {
          ...qwen3EncoderSlot('Required unless a Diffusers component source is available.', isNonAnimaQwen3Encoder),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a Qwen3 Encoder for Z-Image models.',
        },
        {
          ...vaeSlot('Required unless a Diffusers component source is available.', isVaeForBases(['flux'])),
          required: (ctx) => !isBundledOrDiffusersSourceSatisfied(ctx),
          missingMessage: 'Generate needs a VAE for Z-Image models.',
        },
      ]);
    case 'anima':
      return createPolicy(true, [
        {
          ...qwen3EncoderSlot('Required for Anima models.', isAnimaQwen3Encoder),
          required: () => true,
          missingMessage: 'Generate needs a Qwen3 Encoder for Anima models.',
        },
        {
          ...vaeSlot('Required for Anima models. Qwen Image and FLUX VAEs are supported.', isAnimaVae),
          required: () => true,
          missingMessage: 'Generate needs a VAE for Anima models.',
        },
      ]);
    default:
      return EMPTY_COMPONENT_POLICY;
  }
};

const getComponentPolicyContext = (model: GenerateModelConfig, settings: GenerateSettings): ComponentPolicyContext => ({
  model,
  settings,
  selectedComponents: {
    clipEmbedModel: settings.clipEmbedModel,
    clipGEmbedModel: settings.clipGEmbedModel,
    clipLEmbedModel: settings.clipLEmbedModel,
    componentSourceModel: settings.componentSourceModel,
    qwen3EncoderModel: settings.qwen3EncoderModel,
    qwenVLEncoderModel: settings.qwenVLEncoderModel,
    t5EncoderModel: settings.t5EncoderModel,
    vae: settings.vae,
  },
});

const COMPONENT_SETTING_LABELS: Record<GenerateComponentValueKey, string> = {
  clipEmbedModel: 'CLIP Embed',
  clipGEmbedModel: 'CLIP G',
  clipLEmbedModel: 'CLIP L',
  qwen3EncoderModel: 'Qwen3 Encoder',
  qwenVLEncoderModel: 'Qwen VL Encoder',
  t5EncoderModel: 'T5 Encoder',
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

type ReferenceModelCandidate = { base: string; key: string; name: string; type: string };

const isFluxKontextModel = (model: GenerateModelConfig | undefined): model is MainModelConfig =>
  Boolean(
    model &&
    model.type !== 'external_image_generator' &&
    model.base === 'flux' &&
    model.name.toLowerCase().includes('kontext')
  );

/**
 * Whether the model can consume reference images at all. Kept strict so the UI
 * only offers reference images where `getGenerationValidationReasons` would
 * accept them (e.g. Qwen Image requires the `edit` variant).
 */
export const isReferenceImageSupported = (model: GenerateModelConfig | undefined): boolean => {
  if (!model) {
    return false;
  }

  if (model.type === 'external_image_generator') {
    return model.capabilities?.supports_reference_images === true;
  }

  if (model.base === 'qwen-image') {
    return model.variant === 'edit';
  }

  return ['flux', 'flux2', 'sd-1', 'sdxl'].includes(model.base);
};

export const getMaxReferenceImages = (model: GenerateModelConfig | undefined): number => {
  if (!model || !isReferenceImageSupported(model)) {
    return 0;
  }

  if (model.type === 'external_image_generator' && typeof model.capabilities?.max_reference_images === 'number') {
    return Math.max(0, model.capabilities.max_reference_images);
  }

  return DEFAULT_REFERENCE_IMAGE_LIMIT;
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
 * Reference images carried over to `model`: dropped entirely when the model
 * does not support reference images, and re-targeted to a default config for
 * the new model (keeping the image and enabled state) when incompatible.
 * Returns the input array untouched when nothing changes.
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

export const getSettingsWithCompatibleModelSelections = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ReferenceModelCandidate[]
): GenerateSettingsCompatibilityResult => {
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
  const dimensions = getGenerationDimensions(model);

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

const getReferenceImageValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  const reasons: string[] = [];
  const enabled = settings.referenceImages.filter((referenceImage) => referenceImage.isEnabled);
  const maxReferenceImages =
    model.type === 'external_image_generator' && typeof model.capabilities?.max_reference_images === 'number'
      ? model.capabilities.max_reference_images
      : DEFAULT_REFERENCE_IMAGE_LIMIT;

  if (enabled.length > maxReferenceImages) {
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

export const getGenerationValidationReasons = (model: GenerateModelConfig, settings: GenerateSettings): string[] => {
  if (!isSupportedGenerateModel(model)) {
    return ['Generate needs a supported model before it can be invoked.'];
  }

  const reasons = [
    ...getDimensionValidationReasons(model, settings),
    ...getReferenceImageValidationReasons(model, settings),
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

  return reasons;
};

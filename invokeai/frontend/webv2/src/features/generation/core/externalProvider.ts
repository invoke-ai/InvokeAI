/**
 * Pure policy for hosted image providers: which provider-specific options a selected external model exposes, the
 * named sizes and fixed-size ratios it advertises, and the invocation fields those settings compile to.
 */

import type {
  ExternalGenerationMode,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalResolutionPreset,
  GeminiThinkingLevel,
  GenerateModelConfig,
  GenerateSettings,
  OpenAIImageBackground,
  OpenAIImageQuality,
  OpenAIInputFidelity,
} from './types';

export const OPENAI_IMAGE_QUALITIES: readonly OpenAIImageQuality[] = ['auto', 'high', 'medium', 'low'];
export const OPENAI_IMAGE_BACKGROUNDS: readonly OpenAIImageBackground[] = ['auto', 'transparent', 'opaque'];
export const OPENAI_INPUT_FIDELITIES: readonly OpenAIInputFidelity[] = ['low', 'high'];
export const GEMINI_THINKING_LEVELS: readonly GeminiThinkingLevel[] = ['minimal', 'high'];
/** `GeminiImageGenerationInvocation.temperature` is `ge=0, le=2`; the provider default is 1. */
export const GEMINI_TEMPERATURE_MIN = 0;
export const GEMINI_TEMPERATURE_MAX = 2;
export const DEFAULT_GEMINI_TEMPERATURE = 1;

export const isOpenAIImageQuality = (value: unknown): value is OpenAIImageQuality =>
  OPENAI_IMAGE_QUALITIES.includes(value as OpenAIImageQuality);
export const isOpenAIImageBackground = (value: unknown): value is OpenAIImageBackground =>
  OPENAI_IMAGE_BACKGROUNDS.includes(value as OpenAIImageBackground);
export const isOpenAIInputFidelity = (value: unknown): value is OpenAIInputFidelity =>
  OPENAI_INPUT_FIDELITIES.includes(value as OpenAIInputFidelity);
export const isGeminiThinkingLevel = (value: unknown): value is GeminiThinkingLevel =>
  GEMINI_THINKING_LEVELS.includes(value as GeminiThinkingLevel);
export const isExternalGenerationMode = (value: unknown): value is ExternalGenerationMode =>
  value === 'txt2img' || value === 'img2img' || value === 'inpaint';

/**
 * Seedream's batch endpoint counts inputs and outputs against one total. Mirrors `_SEEDREAM_BATCH_PREFIXES` and
 * `_SEEDREAM_BATCH_MAX_TOTAL_IMAGES` in `invokeai/app/services/external_generation/providers/seedream.py`.
 */
const SEEDREAM_BATCH_MODEL_PREFIXES = [
  'seedream-5',
  'seedream-4.5',
  'seedream-4.0',
  'seedream-4-5',
  'seedream-4-0',
  'seedream-5-0',
] as const;
export const SEEDREAM_BATCH_MAX_TOTAL_IMAGES = 15;

export const isSeedreamBatchModel = (model: GenerateModelConfig): boolean =>
  model.type === 'external_image_generator' &&
  model.provider_id === 'seedream' &&
  typeof model.provider_model_id === 'string' &&
  SEEDREAM_BATCH_MODEL_PREFIXES.some((prefix) => (model.provider_model_id as string).startsWith(prefix));

export type ExternalProviderOptionKey = keyof Pick<
  GenerateSettings,
  | 'openaiQuality'
  | 'openaiBackground'
  | 'openaiInputFidelity'
  | 'geminiTemperature'
  | 'geminiThinkingLevel'
  | 'seedreamWatermark'
  | 'seedreamOptimizePrompt'
>;

const getCapabilities = (model: GenerateModelConfig | undefined): ExternalModelCapabilities | null =>
  model?.type === 'external_image_generator' ? (model.capabilities ?? null) : null;

const getProviderId = (model: GenerateModelConfig | undefined): string | null =>
  model?.type === 'external_image_generator' && typeof model.provider_id === 'string' ? model.provider_id : null;

/** True when the model accepts image inputs in any form (an edit mode or reference images). */
const acceptsImageInputs = (capabilities: ExternalModelCapabilities | null): boolean =>
  capabilities?.supports_reference_images === true ||
  (capabilities?.modes ?? []).some((mode) => mode === 'img2img' || mode === 'inpaint');

/**
 * The provider options the selected model's invocation accepts, in display order. Each provider's node declares its
 * own fields, so the provider decides; OpenAI's input fidelity is only consulted for image inputs, so a model that
 * takes none does not offer it.
 */
export const getExternalProviderOptionKeys = (model: GenerateModelConfig | undefined): ExternalProviderOptionKey[] => {
  const capabilities = getCapabilities(model);

  switch (getProviderId(model)) {
    case 'openai':
      return [
        'openaiQuality',
        'openaiBackground',
        ...(acceptsImageInputs(capabilities) ? (['openaiInputFidelity'] as const) : []),
      ];
    case 'gemini':
      return ['geminiTemperature', 'geminiThinkingLevel'];
    case 'seedream':
      return ['seedreamWatermark', 'seedreamOptimizePrompt'];
    default:
      return [];
  }
};

export const getExternalResolutionPresets = (model: GenerateModelConfig | undefined): ExternalResolutionPreset[] => {
  const presets = getCapabilities(model)?.resolution_presets;

  return Array.isArray(presets) ? presets : [];
};

/** Fixed sizes per ratio, as `[ratio, size]` pairs in the provider's order; empty for free-form providers. */
export const getExternalAspectRatioSizes = (
  model: GenerateModelConfig | undefined
): Array<[ratio: string, size: ExternalImageSize]> => {
  const sizes = getCapabilities(model)?.aspect_ratio_sizes;

  return sizes ? Object.entries(sizes) : [];
};

/** Dimensions select the preset; the stored name breaks ties if the provider advertises duplicate sizes. */
export const findExternalResolutionPreset = (
  settings: Pick<GenerateSettings, 'externalImageSize' | 'height' | 'width'>,
  model: GenerateModelConfig | undefined
): ExternalResolutionPreset | null => {
  const presets = getExternalResolutionPresets(model).filter(
    (preset) => preset.width === settings.width && preset.height === settings.height
  );

  return presets.find((preset) => preset.image_size === settings.externalImageSize) ?? presets[0] ?? null;
};

/**
 * The named size to send: the selected model's preset the current size matches exactly, including a hand-typed size. A
 * pick left over from another model never overrides the dimensions at the provider.
 */
export const resolveExternalImageSize = (
  settings: Pick<GenerateSettings, 'externalImageSize' | 'height' | 'width'>,
  model: GenerateModelConfig | undefined
): string | null => findExternalResolutionPreset(settings, model)?.image_size ?? null;

/**
 * Provider-specific invocation fields for the selected model, keyed as the node declares them. Optional fields are
 * omitted when unset so the node applies its own default.
 */
export const getExternalProviderInvocationOptions = (
  settings: GenerateSettings,
  model: GenerateModelConfig
): Record<string, unknown> => {
  const keys = new Set(getExternalProviderOptionKeys(model));
  const options: Record<string, unknown> = {};

  if (keys.has('openaiQuality')) {
    options.quality = settings.openaiQuality;
  }
  if (keys.has('openaiBackground')) {
    options.background = settings.openaiBackground;
  }
  if (keys.has('openaiInputFidelity') && settings.openaiInputFidelity !== null) {
    options.input_fidelity = settings.openaiInputFidelity;
  }
  if (keys.has('geminiTemperature') && settings.geminiTemperature !== null) {
    options.temperature = settings.geminiTemperature;
  }
  if (keys.has('geminiThinkingLevel') && settings.geminiThinkingLevel !== null) {
    options.thinking_level = settings.geminiThinkingLevel;
  }
  if (keys.has('seedreamWatermark')) {
    options.watermark = settings.seedreamWatermark;
  }
  if (keys.has('seedreamOptimizePrompt')) {
    options.optimize_prompt = settings.seedreamOptimizePrompt;
  }

  return options;
};

/** Metadata keys mirror what the provider nodes record themselves, so Generate and Canvas outputs recall alike. */
export const EXTERNAL_METADATA_KEYS: Record<ExternalProviderOptionKey, string> = {
  geminiTemperature: 'gemini_temperature',
  geminiThinkingLevel: 'gemini_thinking_level',
  openaiBackground: 'openai_background',
  openaiInputFidelity: 'openai_input_fidelity',
  openaiQuality: 'openai_quality',
  seedreamOptimizePrompt: 'seedream_optimize_prompt',
  seedreamWatermark: 'seedream_watermark',
};

export const getExternalProviderMetadata = (
  settings: GenerateSettings,
  model: GenerateModelConfig
): Record<string, unknown> => {
  const metadata: Record<string, unknown> = {};
  const imageSize = resolveExternalImageSize(settings, model);

  if (imageSize !== null) {
    metadata.image_size = imageSize;
  }
  for (const key of getExternalProviderOptionKeys(model)) {
    const value = settings[key];

    if (value !== null) {
      metadata[EXTERNAL_METADATA_KEYS[key]] = value;
    }
  }

  return metadata;
};

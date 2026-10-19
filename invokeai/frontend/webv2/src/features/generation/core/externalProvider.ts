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
  GenerateReferenceImageAsset,
  GenerateSettings,
  OpenAIImageBackground,
  OpenAIImageQuality,
  OpenAIInputFidelity,
} from './types';

import { MAX_DIMENSION } from './dimensionLimits';

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

/** OpenAI applies its options only to these models; mirrors `_GPT_IMAGE_MODELS` in `providers/openai.py`. */
const OPENAI_GPT_IMAGE_MODEL_IDS: readonly unknown[] = [
  'gpt-image-1',
  'gpt-image-1.5',
  'gpt-image-1-mini',
  'gpt-image-2',
];

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
 * own fields, so the provider decides. OpenAI sends them only for GPT Image models (DALL·E takes none), and input
 * fidelity only with image inputs, so a model that takes none does not offer it.
 */
export const getExternalProviderOptionKeys = (model: GenerateModelConfig | undefined): ExternalProviderOptionKey[] => {
  const capabilities = getCapabilities(model);

  switch (getProviderId(model)) {
    case 'openai':
      if (!OPENAI_GPT_IMAGE_MODEL_IDS.includes(model?.provider_model_id)) {
        return [];
      }
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

/** A "W:H" ratio of whole numbers, the only form the backend sizes from (`_size_for_ratio`). */
const parseWholeRatio = (ratio: string): ExternalImageSize | null => {
  const match = /^(\d+):(\d+)$/.exec(ratio);
  const width = Number(match?.[1]);
  const height = Number(match?.[2]);

  return width > 0 && height > 0 ? { height, width } : null;
};

/**
 * The sizes a Canvas request may name, keyed by nominal ratio: the resolution presets, else the fixed size per ratio,
 * else per whole-number allowed ratio its smallest exact multiple covering the frame within the maximum size and the
 * generation limit. The backend's bucketing (`_bucket_request`) passes a fixed size (its own table entry) and an exact
 * multiple (it reduces to a declared ratio) through unchanged. A preset whose rounded size is off its exact ratio may
 * be nudged by a pixel or so, which is harmless: the provider sizes its output from the ratio and the preset's name,
 * and the compiler restores whatever comes back to the bbox.
 */
const getCanvasRequestSizes = (
  model: GenerateModelConfig,
  frame: ExternalImageSize
): Array<{ ratio: string; size: ExternalImageSize }> => {
  const presets = getExternalResolutionPresets(model);

  if (presets.length > 0) {
    return presets.map((preset) => ({ ratio: preset.aspect_ratio, size: preset }));
  }

  const sizes = getExternalAspectRatioSizes(model);

  if (sizes.length > 0) {
    return sizes.map(([ratio, size]) => ({ ratio, size }));
  }

  const capabilities = getCapabilities(model);
  const maxSize = capabilities?.max_image_size;

  return (capabilities?.allowed_aspect_ratios ?? []).flatMap((ratio) => {
    const unit = parseWholeRatio(ratio);

    if (!unit) {
      return [];
    }

    const scale = Math.min(
      Math.ceil(Math.max(frame.width / unit.width, frame.height / unit.height)),
      Math.floor(Math.min(maxSize?.width ?? MAX_DIMENSION, MAX_DIMENSION) / unit.width),
      Math.floor(Math.min(maxSize?.height ?? MAX_DIMENSION, MAX_DIMENSION) / unit.height)
    );

    return scale >= 1 ? [{ ratio, size: { height: unit.height * scale, width: unit.width * scale } }] : [];
  });
};

/**
 * The size a Canvas frame asks a hosted provider for. A provider limited to certain shapes gets one of its own sizes,
 * so the compiler knows the shape it gets back: the ratio closest to the frame's, then the smallest size of that ratio
 * covering the frame, else the largest. A free-form provider gets the frame.
 */
export const resolveExternalCanvasRequestSize = (
  model: GenerateModelConfig,
  frame: ExternalImageSize
): ExternalImageSize => {
  const candidates = getCanvasRequestSizes(model, frame);

  if (candidates.length === 0) {
    return { height: frame.height, width: frame.width };
  }

  const frameRatio = frame.width / frame.height;
  const ratioDistance = ({ size }: (typeof candidates)[number]) =>
    Math.abs(Math.log(size.width / size.height / frameRatio));
  const closest = candidates.reduce((best, candidate) =>
    ratioDistance(candidate) < ratioDistance(best) ? candidate : best
  );
  const area = ({ size }: (typeof candidates)[number]) => size.width * size.height;
  const sameRatio = candidates.filter((candidate) => candidate.ratio === closest.ratio);
  const covering = sameRatio.filter(({ size }) => size.width >= frame.width && size.height >= frame.height);
  const { size } =
    covering.length > 0
      ? covering.reduce((best, candidate) => (area(candidate) < area(best) ? candidate : best))
      : sameRatio.reduce((best, candidate) => (area(candidate) > area(best) ? candidate : best));

  return { height: size.height, width: size.width };
};

/** The reference images a request sends: none unless the model takes them. */
export const getExternalReferenceImages = (
  settings: Pick<GenerateSettings, 'referenceImages'>,
  model: GenerateModelConfig
): GenerateReferenceImageAsset[] =>
  getCapabilities(model)?.supports_reference_images === true
    ? settings.referenceImages.flatMap((referenceImage) =>
        referenceImage.isEnabled &&
        referenceImage.config.type === 'external_reference_image' &&
        referenceImage.config.image !== null
          ? [referenceImage.config.image]
          : []
      )
    : [];

/**
 * Canvas content a hosted model cannot take, worded once for readiness and for the invoke-time check that measures
 * pixel coverage. Raster content in the frame selects an edit mode, which a model without image-to-image cannot run.
 */
export const describeExternalCanvasRefusal = (
  model: Pick<GenerateModelConfig, 'name'>,
  refusal: { kind: 'control_layer' | 'regional_guidance'; layerName: string } | { kind: 'raster_content' }
): string => {
  switch (refusal.kind) {
    case 'control_layer':
      return `${model.name} does not support control layer "${refusal.layerName}". Disable it to generate.`;
    case 'regional_guidance':
      return `${model.name} does not support regional guidance layer "${refusal.layerName}". Disable it to generate.`;
    case 'raster_content':
      return `${model.name} cannot use image content in the generation frame. Hide or move raster content out of the frame to generate.`;
  }
};

/**
 * Whether a hosted model only generates from text on the Canvas. Image-to-image is the only provider edit mode the
 * Canvas compiles, so a model without it (even one advertising inpaint) cannot use content in the frame.
 */
export const isExternalTextOnlyCanvasModel = (model: GenerateModelConfig): boolean =>
  model.type === 'external_image_generator' && !(model.capabilities?.modes ?? ['txt2img']).includes('img2img');

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

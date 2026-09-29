import type { BackendGraphContract, GraphContract } from '@features/generation/core/contracts';
import type { DynamicPromptsSeedBehaviour } from '@features/generation/core/dynamicPrompts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { SeedMode } from '@platform/core/seed';

export type ModelIdentifierConfig = {
  key: string;
  name: string;
  base: string;
  type: string;
  format?: string;
  variant?: string | null;
  hash?: string;
  submodel_type?: string;
  [key: string]: unknown;
};

export type MainModelConfig = ModelIdentifierConfig & {
  type: 'main';
  default_settings?: {
    scheduler?: string | null;
    steps?: number | null;
    cfg_scale?: number | null;
    guidance?: number | null;
    cfg_rescale_multiplier?: number | null;
    width?: number | null;
    height?: number | null;
    vae?: string | null;
    vae_precision?: string | null;
  } | null;
};

export type ExternalImageGeneratorModelConfig = ModelIdentifierConfig & {
  base: 'external';
  type: 'external_image_generator';
  format: 'external_api';
  provider_id?: string;
  capabilities?: {
    modes?: string[];
    supports_seed?: boolean;
    supports_negative_prompt?: boolean;
    supports_reference_images?: boolean;
    max_reference_images?: number | null;
  } | null;
  default_settings?: {
    width?: number | null;
    height?: number | null;
    num_images?: number | null;
  } | null;
};

export type GenerateModelConfig = MainModelConfig | ExternalImageGeneratorModelConfig;

export type ComponentModelConfig = ModelIdentifierConfig;

export type VaeModelConfig = ModelIdentifierConfig & {
  type: 'vae';
};

export type LoraModelConfig = ModelIdentifierConfig & {
  type: 'lora';
  trigger_phrases?: string[] | null;
  default_settings?: {
    weight?: number | null;
  } | null;
  variant?: string | null;
};

export interface GenerateLora {
  isEnabled: boolean;
  model: LoraModelConfig;
  weight: number;
}

export interface ImageWithDims {
  image_name: string;
  width: number;
  height: number;
}

/**
 * What a widget suggests to Expand Prompt for its model family. Each part applies only until the
 * user picks a model or system prompt themselves.
 */
export interface ExpandPromptSuggestion {
  /** Source of the family's released enhancer, preselected when an installed text LLM has it. */
  modelSource: string | null;
  /** How the model is listed among the starter models, for the hint shown when it is not installed. */
  modelName: string | null;
  /** For a text-only rewrite. */
  systemPromptId: string | null;
  /** Replaces `systemPromptId` whenever `image` is actually sent; a prompt written for an image must not run without one. */
  imageSystemPromptId: string | null;
  /** An image the rewrite may describe from, such as a video's first frame. */
  image: ImageWithDims | null;
}

export interface CroppableImageWithDims {
  original: { image: ImageWithDims };
  crop?: {
    image: ImageWithDims;
    box: { x: number; y: number; width: number; height: number };
    ratio: number | null;
  };
}

/** Canonical persisted/metadata asset for global Generate reference images. */
export type GenerateReferenceImageAsset = CroppableImageWithDims;

export type ClipVisionModel = 'ViT-H' | 'ViT-G' | 'ViT-L';

export type IPAdapterMethod = 'full' | 'style' | 'composition' | 'style_strong' | 'style_precise';

export type FluxReduxImageInfluence = 'lowest' | 'low' | 'medium' | 'high' | 'highest';

type IPAdapterReferenceImageConfig<TImage> = {
  type: 'ip_adapter';
  image: TImage | null;
  model: ComponentModelConfig | null;
  weight: number;
  beginEndStepPct: [number, number];
  method: IPAdapterMethod;
  clipVisionModel: ClipVisionModel;
};

type FluxReduxReferenceImageConfig<TImage> = {
  type: 'flux_redux';
  image: TImage | null;
  model: ComponentModelConfig | null;
  imageInfluence: FluxReduxImageInfluence;
};

export type GenerateReferenceImageConfig =
  | IPAdapterReferenceImageConfig<GenerateReferenceImageAsset>
  | FluxReduxReferenceImageConfig<GenerateReferenceImageAsset>
  | {
      type: 'flux_kontext_reference_image';
      image: GenerateReferenceImageAsset | null;
      model: MainModelConfig | null;
    }
  | { type: 'flux2_reference_image'; image: GenerateReferenceImageAsset | null }
  | { type: 'qwen_image_reference_image'; image: GenerateReferenceImageAsset | null }
  | { type: 'external_reference_image'; image: GenerateReferenceImageAsset | null };

export interface GenerateReferenceImage {
  id: string;
  isEnabled: boolean;
  config: GenerateReferenceImageConfig;
}

export type VaePrecision = 'fp16' | 'fp32';

export type AspectRatioId =
  | 'Free'
  | '8:1'
  | '4:1'
  | '21:9'
  | '16:9'
  | '3:2'
  | '5:4'
  | '4:3'
  | '1:1'
  | '3:4'
  | '4:5'
  | '2:3'
  | '9:16'
  | '1:4'
  | '9:21'
  | '1:8';

/** GENERATE_UI_STATE_KEYS defines the UI-only keys; this module owns their types. */
export interface GenerateSettings {
  batchCount: number;
  modelKey: string;
  positivePrompt: string;
  positivePromptHeightPx: number;
  /** The text LLM last picked for Expand Prompt; unset or uninstalled falls back to the first installed one. */
  expandPromptModelKey: string | null;
  /** The vision model last picked for Image to Prompt, with the same fallback. */
  imageToPromptModelKey: string | null;
  negativePromptEnabled: boolean;
  negativePrompt: string;
  negativePromptHeightPx: number;
  /** Stored template snapshots allow pure submission without catalog lookup. */
  promptTemplate: PromptTemplateSnapshot | null;
  /** Show the merged prompt read-only instead of the authored text. */
  promptTemplateViewMode: boolean;
  /** Expand `{a|b}` into every combination; otherwise draw a random sample. */
  dynamicPromptsCombinatorial: boolean;
  /** Upper bound on expanded prompts (the sample size when not combinatorial). */
  dynamicPromptsMaxPrompts: number;
  /** Seeds the random sampler so the preview matches what generates. */
  dynamicPromptsSampleSeed: number;
  dynamicPromptsSeedBehaviour: DynamicPromptsSeedBehaviour;
  width: number;
  height: number;
  aspectRatioId: AspectRatioId;
  /** width / height ratio enforced while locked. Tracks the preset, or the captured ratio in Free mode. */
  aspectRatioValue: number;
  aspectRatioIsLocked: boolean;
  steps: number;
  cfgScale: number;
  cfgRescaleMultiplier: number;
  scheduler: string;
  clipSkip: number;
  colorCompensation: boolean;
  /** Enables HiDiffusion high-resolution denoising for supported SD models. */
  hiDiffusionEnabled: boolean;
  hiDiffusionRauNetEnabled: boolean;
  hiDiffusionWindowAttentionEnabled: boolean;
  hiDiffusionT1Ratio: number;
  hiDiffusionT2Ratio: number;
  seed: number;
  seedMode: SeedMode;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  /** Optional VAE override; null uses the VAE bundled with the main model. */
  vae: VaeModelConfig | null;
  vaePrecision: VaePrecision;
  loras: GenerateLora[];
  referenceImages: GenerateReferenceImage[];
  t5EncoderModel: ComponentModelConfig | null;
  clipEmbedModel: ComponentModelConfig | null;
  clipLEmbedModel: ComponentModelConfig | null;
  clipGEmbedModel: ComponentModelConfig | null;
  qwen3EncoderModel: ComponentModelConfig | null;
  /** FLUX.2 [dev]'s Mistral text encoder. */
  mistralEncoderModel: ComponentModelConfig | null;
  qwenVLEncoderModel: ComponentModelConfig | null;
  /** Krea-2's text encoder. Distinct from `qwenVLEncoderModel` (Qwen2.5-VL). */
  qwen3VLEncoderModel: ComponentModelConfig | null;
  /** Wan 2.2's UMT5-XXL text encoder. */
  wanT5EncoderModel: ComponentModelConfig | null;
  /** The low-noise expert is optional; the selected expert can span the full schedule. */
  wanLowNoiseModel: MainModelConfig | null;
  /**
   * Ideogram 4's unconditional transformer branch. Required with a single-file main, which holds
   * only the conditional branch; null for a diffusers pipeline, which bundles both.
   */
  ideogram4UnconditionalModel: MainModelConfig | null;
  /** Optional Diffusers main model used as a component source for split/quantized model families. */
  componentSourceModel: MainModelConfig | null;
  /** Guidance for the low-noise half of a Wan A14B schedule; null reuses `cfgScale`. */
  wanGuidanceScaleLowNoise: number | null;
  ideogram4SamplerPreset: Ideogram4SamplerPreset;
  /** Null lets the sampler preset decide. */
  ideogram4Steps: number | null;
  ideogram4GuidanceScale: number | null;
  ideogram4Mu: number | null;
  /** Free-text colour terms forwarded to the caption builder. */
  ideogram4ColorPalette: string[];
  /** Scales Krea-2 conditioning toward the prompt before denoise. Default off. */
  krea2RebalanceEnabled: boolean;
  krea2RebalanceMultiplier: number;
  /** Comma-separated per-layer weights, forwarded verbatim to the backend parser. */
  krea2RebalanceWeights: string;
  /** Perturbs Krea-2 conditioning for variety between seeds. Default off. */
  krea2SeedVarianceEnabled: boolean;
  krea2SeedVarianceStrength: number;
  krea2SeedVarianceRandomizePercent: number;
  /** PiD replaces VAE decode with caption-conditioned 4× output; requires a PiD decoder and Gemma-2 encoder. */
  pidMode: PidMode;
  /** PiD decoder checkpoint. Trained per backbone, so it must match the main model's base. */
  pidDecoderModel: ComponentModelConfig | null;
  /** The shared Gemma-2 caption encoder PiD conditions its decode on. */
  gemma2EncoderModel: ComponentModelConfig | null;
  pidSteps: number;
}

/**
 * - `off`: ordinary VAE decode.
 * - `fit`: generate at the requested size, decode 4x, downscale back to it.
 * - `native`: the requested size IS the 4x target; generate at size / 4 and keep the
 *   full 4x output.
 */
export type PidMode = 'off' | 'fit' | 'native';

/** Use preset defaults unless steps, guidance, or mu is explicitly overridden. */
export type Ideogram4SamplerPreset = 'V4_QUALITY_48' | 'V4_DEFAULT_20' | 'V4_TURBO_12';

export interface GenerateWidgetValues extends GenerateSettings {
  model: GenerateModelConfig;
}

export interface CompiledGenerateGraph {
  backendGraph: BackendGraphContract;
  graph: GraphContract;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
}

import type { BackendGraphContract, GeneratedImageContract, GraphContract, ResultDestination } from '@workbench/types';

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

/** Canvas regional guidance intentionally keeps its flat, URL-bearing image contract. */
export type RegionalGuidanceReferenceImageAsset = GeneratedImageContract;

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

export type RegionalGuidanceReferenceImageConfig =
  | IPAdapterReferenceImageConfig<RegionalGuidanceReferenceImageAsset>
  | FluxReduxReferenceImageConfig<RegionalGuidanceReferenceImageAsset>;

export interface RegionalGuidanceReferenceImage {
  id: string;
  isEnabled: boolean;
  config: RegionalGuidanceReferenceImageConfig;
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

export interface GenerateSettings {
  batchCount: number;
  modelKey: string;
  positivePrompt: string;
  positivePromptHeightPx: number;
  negativePromptEnabled: boolean;
  negativePrompt: string;
  negativePromptHeightPx: number;
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
  seed: number;
  shouldRandomizeSeed: boolean;
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
  qwenVLEncoderModel: ComponentModelConfig | null;
  /** Optional Diffusers main model used as a component source for split/quantized model families. */
  componentSourceModel: MainModelConfig | null;
}

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

export interface EnqueueGenerateResult {
  batchId?: string;
  enqueued: number;
  itemIds: number[];
  requested: number;
}

export interface QueueItemDTO {
  item_id: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'canceled';
  batch_id?: string;
  origin?: string | null;
  destination?: string | null;
  error_type?: string | null;
  error_message?: string | null;
  session?: {
    prepared_source_mapping?: Record<string, string>;
    results: Record<string, unknown>;
  };
}

/** A source-agnostic graph submission (used by the workflow source). */
export interface EnqueueWorkflowRequest {
  batchCount: number;
  destination: ResultDestination;
  graph: BackendGraphContract;
  projectId: string;
  sourceQueueItemId: string;
}

export interface EnqueueGenerateRequest {
  batchCount: number;
  destination: ResultDestination;
  graph: BackendGraphContract;
  negativePrompt: string;
  negativePromptNodeId: string;
  positivePrompt: string;
  positivePromptNodeId: string;
  seed: number;
  seedNodeId: string;
  shouldRandomizeSeed: boolean;
  projectId: string;
  sourceQueueItemId: string;
}

export interface ImageDTO extends GeneratedImageContract {
  isIntermediate: boolean;
}

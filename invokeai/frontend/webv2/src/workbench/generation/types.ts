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
  negativePrompt: string;
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
  itemIds: number[];
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
    results: Record<string, unknown>;
  };
}

/** A source-agnostic graph submission (used by the project-graph / workflow source). */
export interface EnqueueWorkflowRequest {
  batchCount: number;
  destination: ResultDestination;
  graph: BackendGraphContract;
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
  sourceQueueItemId: string;
}

export interface ImageDTO extends GeneratedImageContract {
  isIntermediate: boolean;
}

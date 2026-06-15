import type { BackendGraphContract, GeneratedImageContract, GraphContract, ResultDestination } from '@workbench/types';

export type SupportedGenerateBase = 'sd-1' | 'sd-2' | 'sdxl';

export type MainModelConfig = {
  key: string;
  name: string;
  base: string;
  type: 'main';
  format?: string;
  default_settings?: {
    scheduler?: string | null;
    steps?: number | null;
    cfg_scale?: number | null;
    cfg_rescale_multiplier?: number | null;
    width?: number | null;
    height?: number | null;
    vae?: string | null;
    vae_precision?: string | null;
  } | null;
  [key: string]: unknown;
};

export type VaeModelConfig = {
  key: string;
  name: string;
  base: string;
  type: 'vae';
  [key: string]: unknown;
};

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
  seed: number;
  shouldRandomizeSeed: boolean;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  /** Optional VAE override; null uses the VAE bundled with the main model. */
  vae: VaeModelConfig | null;
  vaePrecision: VaePrecision;
}

export interface GenerateWidgetValues extends GenerateSettings {
  model: MainModelConfig;
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

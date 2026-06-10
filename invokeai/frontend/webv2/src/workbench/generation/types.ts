import type { BackendGraphContract, GeneratedImageContract, GraphContract, ResultDestination } from '../types';

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
    vae_precision?: string | null;
  } | null;
  [key: string]: unknown;
};

export interface GenerateSettings {
  batchCount: number;
  modelKey: string;
  positivePrompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  cfgRescaleMultiplier: number;
  scheduler: string;
  seed: number;
  shouldRandomizeSeed: boolean;
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
  itemIds: number[];
}

export interface QueueItemDTO {
  item_id: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'canceled';
  error_type?: string | null;
  error_message?: string | null;
  session: {
    results: Record<string, unknown>;
  };
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
  sourceQueueItemId: string;
}

export interface ImageDTO extends GeneratedImageContract {
  isIntermediate: boolean;
}

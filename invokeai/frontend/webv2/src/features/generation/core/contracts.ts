/** Public, storage-independent contracts owned by Generation. */

export type ResultDestination = 'canvas' | 'gallery';

export interface GenerationProjectSettings {
  useCpuNoise: boolean;
}

export interface PromptHistoryItem {
  positivePrompt: string;
  negativePrompt: string | null;
}

export interface BackendInvocationContract {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface BackendGraphEdgeContract {
  source: { node_id: string; field: string };
  destination: { node_id: string; field: string };
}

export interface BackendGraphContract {
  id: string;
  nodes: Record<string, BackendInvocationContract>;
  edges: BackendGraphEdgeContract[];
}

export interface GraphNodeContract {
  id: string;
  type: string;
  inputs: Record<string, unknown>;
}

export interface GraphEdgeContract {
  id: string;
  sourceNodeId: string;
  sourceField: string;
  targetNodeId: string;
  targetField: string;
}

export interface GraphContract {
  id: string;
  version: 1;
  label: string;
  nodes: GraphNodeContract[];
  edges: GraphEdgeContract[];
  updatedAt: string;
  backendGraph?: BackendGraphContract;
}

export type KnownGenerationModelBase =
  | 'sd-1'
  | 'sd-2'
  | 'sd-3'
  | 'sdxl'
  | 'sdxl-refiner'
  | 'flux'
  | 'flux2'
  | 'cogview4'
  | 'qwen-image'
  | 'z-image'
  | 'anima'
  | 'external'
  | 'any'
  | 'unknown';

export type GenerationModelTaxonomyType = string;

export interface GenerationModelCatalogItem {
  key: string;
  name: string;
  base: string;
  type: string;
  format?: string;
  variant?: string | null;
  hash?: string;
  submodel_type?: string;
  trigger_phrases?: string[] | null;
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
    weight?: number | null;
  } | null;
  [key: string]: unknown;
}

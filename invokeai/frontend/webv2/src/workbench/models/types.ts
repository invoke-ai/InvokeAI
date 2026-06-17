/**
 * Contracts for the backend model manager (`/api/v2/models/*`). Shapes mirror
 * the pydantic models in `invokeai/backend/model_manager` and
 * `invokeai/app/services/model_install` (serialized as snake_case). Unions are
 * kept open (`| (string & {})`) so new backend architectures appear in the UI
 * without a frontend release — unknown values fall back to generic labels.
 */

/** Known model base architectures. Open union: new bases still render. */
export type ModelBase =
  | 'any'
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
  | 'unknown'
  | (string & {});

/** Known model types. Open union: new types still render. */
export type ModelTaxonomyType =
  | 'main'
  | 'vae'
  | 'lora'
  | 'control_lora'
  | 'controlnet'
  | 't2i_adapter'
  | 'ip_adapter'
  | 'embedding'
  | 'clip_embed'
  | 'clip_vision'
  | 't5_encoder'
  | 'qwen3_encoder'
  | 'qwen_vl_encoder'
  | 'siglip'
  | 'spandrel_image_to_image'
  | 'flux_redux'
  | 'llava_onevision'
  | 'text_llm'
  | 'external_image_generator'
  | 'onnx'
  | 'unknown'
  | (string & {});

export type ModelFileFormat =
  | 'checkpoint'
  | 'diffusers'
  | 'lycoris'
  | 'omi'
  | 'onnx'
  | 'olive'
  | 'embedding_file'
  | 'embedding_folder'
  | 'invokeai'
  | 't5_encoder'
  | 'qwen3_encoder'
  | 'qwen_vl_encoder'
  | 'bnb_quantized_int8b'
  | 'bnb_quantized_nf4b'
  | 'gguf_quantized'
  | 'external_api'
  | 'unknown'
  | (string & {});

export type PredictionType = 'epsilon' | 'v_prediction' | 'sample';

/** Per-model defaults that Generate can apply explicitly. */
export interface MainModelDefaultSettings {
  vae?: string | null;
  vae_precision?: 'fp16' | 'fp32' | null;
  scheduler?: string | null;
  steps?: number | null;
  cfg_scale?: number | null;
  cfg_rescale_multiplier?: number | null;
  width?: number | null;
  height?: number | null;
  guidance?: number | null;
  fp8_storage?: boolean | null;
}

export interface LoraModelDefaultSettings {
  weight?: number | null;
}

export interface ControlAdapterDefaultSettings {
  preprocessor?: string | null;
  fp8_storage?: boolean | null;
}

export type AnyModelDefaultSettings = MainModelDefaultSettings &
  LoraModelDefaultSettings &
  ControlAdapterDefaultSettings;

/**
 * A model config record. This is the union of fields across all backend config
 * classes; fields not applicable to a given type/format are absent.
 */
export interface ModelConfig {
  key: string;
  hash: string;
  path: string;
  file_size: number;
  name: string;
  description?: string | null;
  source: string;
  source_type: 'path' | 'url' | 'hf_repo_id' | (string & {});
  source_url?: string | null;
  cover_image?: string | null;
  base: ModelBase;
  type: ModelTaxonomyType;
  format: ModelFileFormat;
  variant?: string | null;
  prediction_type?: PredictionType | null;
  upcast_attention?: boolean;
  config_path?: string | null;
  trigger_phrases?: string[] | null;
  default_settings?: AnyModelDefaultSettings | null;
  submodels?: Record<string, unknown> | null;
  [key: string]: unknown;
}

/** Body for PATCH `/api/v2/models/i/{key}` — all fields optional. */
export interface ModelRecordChanges {
  name?: string;
  description?: string | null;
  source_url?: string | null;
  base?: ModelBase;
  type?: ModelTaxonomyType;
  format?: ModelFileFormat;
  variant?: string | null;
  prediction_type?: PredictionType | null;
  upcast_attention?: boolean;
  config_path?: string | null;
  trigger_phrases?: string[] | null;
  default_settings?: AnyModelDefaultSettings | null;
}

export type ModelInstallStatus =
  | 'waiting'
  | 'downloading'
  | 'downloads_done'
  | 'running'
  | 'paused'
  | 'completed'
  | 'error'
  | 'cancelled';

export interface ModelInstallDownloadPart {
  source?: string;
  url?: string;
  local_path?: string;
  bytes: number;
  total_bytes: number;
  status?: string;
}

/** A model install job as returned by `/api/v2/models/install`. */
export interface ModelInstallJob {
  id: number;
  status: ModelInstallStatus;
  /** Local path, URL, or HF repo id — may be a string or a structured source. */
  source: string | { repo_id?: string; url?: string; path?: string; type?: string; [key: string]: unknown };
  error?: string | null;
  error_reason?: string | null;
  error_traceback?: string | null;
  bytes?: number;
  total_bytes?: number;
  local_path?: string | null;
  inplace?: boolean;
  config_in?: ModelRecordChanges | null;
  config_out?: ModelConfig | null;
  download_parts?: ModelInstallDownloadPart[];
  [key: string]: unknown;
}

/** A model file found by `/api/v2/models/scan_folder`. */
export interface FoundModel {
  path: string;
  is_installed: boolean;
}

/** Result of `/api/v2/models/hugging_face` metadata lookup. */
export interface HuggingFaceModels {
  urls: string[] | null;
  is_diffusers: boolean;
}

export interface StarterModel {
  name: string;
  description: string;
  source: string;
  base: ModelBase;
  type: ModelTaxonomyType;
  format?: ModelFileFormat | null;
  variant?: string | null;
  is_installed: boolean;
  dependencies?: Omit<StarterModel, 'dependencies'>[] | null;
  previous_names?: string[];
}

export interface StarterModelBundle {
  name: string;
  models: StarterModel[];
}

export interface StarterModelResponse {
  starter_models: StarterModel[];
  starter_bundles: Record<string, StarterModelBundle>;
}

export type HFTokenStatus = 'valid' | 'invalid' | 'unknown';

/** A model directory on disk with no matching database record. */
export interface OrphanedModelInfo {
  path: string;
  absolute_path: string;
  files: string[];
  size_bytes: number;
}

export interface BulkDeleteModelsResponse {
  deleted: string[];
  failed: { key: string; error: string }[];
}

export interface DeleteOrphanedModelsResponse {
  deleted: string[];
  errors: Record<string, string>;
}

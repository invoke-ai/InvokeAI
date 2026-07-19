import type { ModelFileFormat, ModelInstallJob, ModelTaxonomyType } from './types';

/**
 * Display metadata for the model taxonomy. Open-union friendly: unknown bases,
 * types, and formats fall back to readable generic labels so a backend that
 * ships a new architecture never renders a blank or broken library.
 */

interface CategoryDefinition {
  type: ModelTaxonomyType;
  label: string;
  pluralLabel: string;
}

/** Library grouping order — most-used categories first, mirroring legacy web. */
export const MODEL_CATEGORIES: CategoryDefinition[] = [
  { label: 'Main Model', pluralLabel: 'Main Models', type: 'main' },
  { label: 'LoRA', pluralLabel: 'LoRAs', type: 'lora' },
  { label: 'Embedding', pluralLabel: 'Embeddings', type: 'embedding' },
  { label: 'ControlNet', pluralLabel: 'ControlNets', type: 'controlnet' },
  { label: 'Control LoRA', pluralLabel: 'Control LoRAs', type: 'control_lora' },
  { label: 'IP Adapter', pluralLabel: 'IP Adapters', type: 'ip_adapter' },
  { label: 'T2I Adapter', pluralLabel: 'T2I Adapters', type: 't2i_adapter' },
  { label: 'VAE', pluralLabel: 'VAEs', type: 'vae' },
  { label: 'T5 Encoder', pluralLabel: 'T5 Encoders', type: 't5_encoder' },
  { label: 'Qwen3 Encoder', pluralLabel: 'Qwen3 Encoders', type: 'qwen3_encoder' },
  { label: 'Qwen VL Encoder', pluralLabel: 'Qwen VL Encoders', type: 'qwen_vl_encoder' },
  { label: 'CLIP Embed', pluralLabel: 'CLIP Embeds', type: 'clip_embed' },
  { label: 'CLIP Vision', pluralLabel: 'CLIP Visions', type: 'clip_vision' },
  { label: 'SigLIP', pluralLabel: 'SigLIPs', type: 'siglip' },
  { label: 'FLUX Redux', pluralLabel: 'FLUX Reduxes', type: 'flux_redux' },
  { label: 'Image-to-Image', pluralLabel: 'Image-to-Image Models', type: 'spandrel_image_to_image' },
  { label: 'LLaVA OneVision', pluralLabel: 'LLaVA OneVision Models', type: 'llava_onevision' },
  { label: 'Text LLM', pluralLabel: 'Text LLMs', type: 'text_llm' },
  { label: 'External Generator', pluralLabel: 'External Generators', type: 'external_image_generator' },
  { label: 'ONNX', pluralLabel: 'ONNX Models', type: 'onnx' },
  { label: 'Unknown', pluralLabel: 'Unknown Models', type: 'unknown' },
];

const categoryByType = new Map(MODEL_CATEGORIES.map((category) => [category.type, category]));

const toTitleCase = (value: string): string =>
  value.replaceAll(/[_-]+/g, ' ').replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1));

export const getModelTypeLabel = (type: ModelTaxonomyType): string =>
  categoryByType.get(type)?.label ?? toTitleCase(type);

export const getModelTypePluralLabel = (type: ModelTaxonomyType): string =>
  categoryByType.get(type)?.pluralLabel ?? `${toTitleCase(type)} Models`;

/** Sort rank of a category for grouped views; unknown types sort last. */
export const getModelCategoryRank = (type: ModelTaxonomyType): number => {
  const index = MODEL_CATEGORIES.findIndex((category) => category.type === type);

  return index === -1 ? MODEL_CATEGORIES.length : index;
};

const FORMAT_LABELS: Record<string, string> = {
  bnb_quantized_int8b: 'BnB int8',
  bnb_quantized_nf4b: 'BnB nf4',
  checkpoint: 'Checkpoint',
  diffusers: 'Diffusers',
  embedding_file: 'Embedding File',
  embedding_folder: 'Embedding Folder',
  external_api: 'External API',
  gguf_quantized: 'GGUF',
  invokeai: 'InvokeAI',
  lycoris: 'LyCORIS',
  olive: 'Olive',
  omi: 'OMI',
  onnx: 'ONNX',
  qwen3_encoder: 'Qwen3 Encoder',
  qwen_vl_encoder: 'Qwen VL Encoder',
  t5_encoder: 'T5 Encoder',
  unknown: 'Unknown',
};

export const getModelFormatLabel = (format: ModelFileFormat): string => FORMAT_LABELS[format] ?? toTitleCase(format);

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

export const formatBytes = (bytes: number | null | undefined): string => {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
    return '—';
  }

  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${unitIndex === 0 ? value : value.toFixed(1)} ${BYTE_UNITS[unitIndex]}`;
};

/** Human-readable source for an install job (string or structured source). */
export const getInstallSourceLabel = (source: ModelInstallJob['source']): string => {
  if (typeof source === 'string') {
    return source;
  }

  return source.repo_id ?? source.url ?? source.path ?? JSON.stringify(source);
};

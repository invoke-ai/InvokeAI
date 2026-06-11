import type { ModelBase, ModelConfig, ModelFileFormat, ModelInstallJob, ModelTaxonomyType } from './types';

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

const BASE_LABELS: Record<string, string> = {
  anima: 'Anima',
  any: 'Any',
  cogview4: 'CogView4',
  external: 'External',
  flux: 'FLUX',
  flux2: 'FLUX.2',
  'qwen-image': 'Qwen Image',
  'sd-1': 'SD 1.x',
  'sd-2': 'SD 2.x',
  'sd-3': 'SD 3.x',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXL Refiner',
  unknown: 'Unknown',
  'z-image': 'Z-Image',
};

export const getModelBaseLabel = (base: ModelBase): string => BASE_LABELS[base] ?? toTitleCase(base);

/** Chakra color palette per base, so architecture is scannable at a glance. */
const BASE_COLOR_PALETTES: Record<string, string> = {
  anima: 'pink',
  any: 'gray',
  cogview4: 'red',
  external: 'gray',
  flux: 'yellow',
  flux2: 'orange',
  'qwen-image': 'cyan',
  'sd-1': 'green',
  'sd-2': 'teal',
  'sd-3': 'purple',
  sdxl: 'blue',
  'sdxl-refiner': 'blue',
  unknown: 'gray',
  'z-image': 'pink',
};

export const getModelBaseColorPalette = (base: ModelBase): string => BASE_COLOR_PALETTES[base] ?? 'gray';

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

/** Checkpoint→diffusers conversion is only supported for these bases. */
export const isConvertibleToDiffusers = (model: ModelConfig): boolean =>
  model.format === 'checkpoint' && model.type === 'main' && ['sd-1', 'sd-2', 'sdxl'].includes(model.base);

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

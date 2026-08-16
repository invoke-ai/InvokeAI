import type { ModelFileFormat, ModelTaxonomyType } from './types';

import { toTitleCase } from './baseIdentity';

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
  { label: 'Wan T5 Encoder', pluralLabel: 'Wan T5 Encoders', type: 'wan_t5_encoder' },
  { label: 'Qwen3 Encoder', pluralLabel: 'Qwen3 Encoders', type: 'qwen3_encoder' },
  { label: 'Qwen VL Encoder', pluralLabel: 'Qwen VL Encoders', type: 'qwen_vl_encoder' },
  { label: 'Qwen3 VL Encoder', pluralLabel: 'Qwen3 VL Encoders', type: 'qwen3_vl_encoder' },
  { label: 'Mistral Encoder', pluralLabel: 'Mistral Encoders', type: 'mistral_encoder' },
  { label: 'Gemma 2 Encoder', pluralLabel: 'Gemma 2 Encoders', type: 'gemma2_encoder' },
  { label: 'PiD Decoder', pluralLabel: 'PiD Decoders', type: 'pid_decoder' },
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

export const getModelTypeLabel = (type: ModelTaxonomyType): string =>
  categoryByType.get(type)?.label ?? toTitleCase(type);

export const getModelTypePluralLabel = (type: ModelTaxonomyType): string =>
  categoryByType.get(type)?.pluralLabel ?? `${toTitleCase(type)} Models`;

const categoryRankByType = new Map<ModelTaxonomyType, number>(
  MODEL_CATEGORIES.map((category, index) => [category.type, index])
);

/** Sort rank of a category for grouped views; unknown types sort last. */
export const getModelCategoryRank = (type: ModelTaxonomyType): number =>
  categoryRankByType.get(type) ?? MODEL_CATEGORIES.length;

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

/**
 * Formats a user may assign in the edit form (repairing a mis-detected
 * model). `unknown` is not a repair target and `external_api` would misroute
 * a local model, mirroring the base select's `external` exclusion. The PATCH
 * re-validates through the config factory, so an invalid combination is
 * rejected server-side rather than silently accepted.
 */
export const EDITABLE_MODEL_FORMATS: readonly string[] = Object.keys(FORMAT_LABELS).filter(
  (format) => format !== 'unknown' && format !== 'external_api'
);

/**
 * External page for a model's install source: the URL itself, or the
 * HuggingFace page for a repo-id source. Null for local paths and anything
 * else that has no linkable home.
 */
export const getModelSourceHref = (source: string, sourceType: string): string | null => {
  if (source.startsWith('https://') || source.startsWith('http://')) {
    return source;
  }

  if (sourceType === 'hf_repo_id') {
    // Strip the :variant[:path] qualifiers an install source may carry.
    return `https://huggingface.co/${source.split(':')[0]}`;
  }

  return null;
};

/** Long display names for every known variant value. */
export const MODEL_VARIANT_LABELS: Record<string, string> = {
  '5b': 'Wan 2.2 5B LoRA',
  a14b: 'Wan 2.2 A14B LoRA',
  cow_mistral3_small: 'cow-mistral3-small (FLUX.2)',
  depth: 'Depth',
  dev: 'FLUX Dev',
  dev_fill: 'FLUX Dev - Fill',
  edit: 'Qwen Image Edit',
  generate: 'Qwen Image',
  gigantic: 'CLIP G',
  i2v_a14b: 'Wan 2.2 I2V A14B',
  inpaint: 'Inpaint',
  klein_4b: 'FLUX.2 Klein 4B',
  klein_4b_base: 'FLUX.2 Klein 4B Base',
  klein_9b: 'FLUX.2 Klein 9B',
  klein_9b_base: 'FLUX.2 Klein 9B Base',
  krea2_base: 'Krea-2 Raw',
  krea2_turbo: 'Krea-2 Turbo',
  large: 'CLIP L',
  mistral3_24b: 'Mistral Small 3 (24B, FLUX.2)',
  normal: 'Normal',
  qwen3_06b: 'Qwen3 0.6B',
  qwen3_4b: 'Qwen3 4B',
  qwen3_8b: 'Qwen3 8B',
  res2k_sr4x: 'PiD 2K (4x SR)',
  res2kto4k_sr4x: 'PiD 4K (4x SR Upscale)',
  schnell: 'FLUX Schnell',
  t2v_a14b: 'Wan 2.2 T2V A14B',
  ti2v_5b: 'Wan 2.2 TI2V 5B',
  turbo: 'Z-Image Turbo',
  zbase: 'Z-Image Base',
};

export const getModelVariantLabel = (variant: string): string => MODEL_VARIANT_LABELS[variant] ?? toTitleCase(variant);

// Mirrors the backend's per-class variant enums (taxonomy.py): main models
// key their variants off the base, and a few non-main types carry their own.
const MAIN_VARIANTS_BY_BASE: Record<string, readonly string[]> = {
  flux: ['schnell', 'dev', 'dev_fill'],
  flux2: ['klein_4b', 'klein_4b_base', 'klein_9b', 'klein_9b_base', 'dev'],
  'krea-2': ['krea2_turbo', 'krea2_base'],
  'qwen-image': ['generate', 'edit'],
  'sd-1': ['normal', 'inpaint'],
  'sd-2': ['normal', 'inpaint', 'depth'],
  sdxl: ['normal', 'inpaint'],
  'sdxl-refiner': ['normal'],
  wan: ['t2v_a14b', 'i2v_a14b', 'ti2v_5b'],
  'z-image': ['turbo', 'zbase'],
};

const VARIANTS_BY_TYPE: Record<string, readonly string[]> = {
  clip_embed: ['large', 'gigantic'],
  mistral_encoder: ['cow_mistral3_small', 'mistral3_24b'],
  pid_decoder: ['res2k_sr4x', 'res2kto4k_sr4x'],
  qwen3_encoder: ['qwen3_4b', 'qwen3_8b', 'qwen3_06b'],
};

/**
 * Constrained variant choices for a base/type pair; empty means the pair has
 * no variant concept and the edit form falls back to free text.
 */
export const getVariantOptionsFor = (base: string, type: string): readonly string[] => {
  if (type === 'main') {
    return MAIN_VARIANTS_BY_BASE[base] ?? [];
  }

  if (type === 'lora') {
    return base === 'wan' ? ['a14b', '5b'] : [];
  }

  return VARIANTS_BY_TYPE[type] ?? [];
};

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

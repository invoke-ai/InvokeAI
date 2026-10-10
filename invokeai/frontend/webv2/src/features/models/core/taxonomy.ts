import type { ModelFileFormat, ModelTaxonomyType } from './types';

import { toTitleCase } from './baseIdentity';

/** Use readable fallback labels for unknown taxonomy values so new backend architectures remain usable. */

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
  { label: 'Qwen3.5 Encoder', pluralLabel: 'Qwen3.5 Encoders', type: 'qwen3_5_encoder' },
  { label: 'Mistral Encoder', pluralLabel: 'Mistral Encoders', type: 'mistral_encoder' },
  { label: 'Gemma 2 Encoder', pluralLabel: 'Gemma 2 Encoders', type: 'gemma2_encoder' },
  { label: 'Gemma 4 Encoder', pluralLabel: 'Gemma 4 Encoders', type: 'gemma4_encoder' },
  { label: 'PiD Decoder', pluralLabel: 'PiD Decoders', type: 'pid_decoder' },
  { label: 'LTX-2 Duration Head', pluralLabel: 'LTX-2 Duration Heads', type: 'ltx2_duration_head' },
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
  sdnq_quantized: 'SDNQ',
  t5_encoder: 'T5 Encoder',
  unknown: 'Unknown',
};

export const getModelFormatLabel = (format: ModelFileFormat): string => FORMAT_LABELS[format] ?? toTitleCase(format);

/** Exclude unknown/external_api as repair formats; the config factory validates remaining combinations server-side. */
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
  anima_qwen3: 'Anima',
  anima_qwen35: 'Anima + Qwen3.5 (Anima-3.8B)',
  cow_mistral3_small: 'cow-mistral3-small (FLUX.2)',
  depth: 'Depth',
  dev: 'FLUX Dev',
  dev_fill: 'FLUX Dev - Fill',
  edit: 'Qwen Image Edit',
  fl2va: 'MiniMax H3 FL2VA',
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
  ministral3_3b: 'Ministral 3B (ERNIE-Image)',
  mistral3_24b: 'Mistral Small 3 (24B, FLUX.2)',
  normal: 'Normal',
  qwen3_06b: 'Qwen3 0.6B',
  qwen3_4b: 'Qwen3 4B',
  qwen3_8b: 'Qwen3 8B',
  qwen3_5_4b: 'Qwen3.5 4B (Anima-3.8B)',
  qwen3_vl_4b: 'Qwen3-VL 4B (Krea-2)',
  qwen3_vl_8b: 'Qwen3-VL 8B (Ideogram 4, Qwen-Image-2.1)',
  qwen_image_21_base: 'Qwen-Image-2.1 Base',
  qwen_image_21_turbo: 'Qwen-Image-2.1 Turbo',
  ref2va: 'MiniMax H3 Ref2VA',
  res2k_sr4x: 'PiD 2K (4x SR)',
  res2kto4k_sr4x: 'PiD 4K (4x SR Upscale)',
  schnell: 'FLUX Schnell',
  ltx2_dev: 'LTX-2 Dev',
  ltx2_distilled: 'LTX-2 Distilled',
  t2v_a14b: 'Wan 2.2 T2V A14B',
  ti2v_5b: 'Wan 2.2 TI2V 5B',
  turbo: 'Z-Image Turbo',
  zbase: 'Z-Image Base',
};

export const getModelVariantLabel = (variant: string): string => MODEL_VARIANT_LABELS[variant] ?? toTitleCase(variant);

// Mirrors the backend's per-class variant enums (taxonomy.py): main models
// key their variants off the base, and a few non-main types carry their own.
const MAIN_VARIANTS_BY_BASE: Record<string, readonly string[]> = {
  anima: ['anima_qwen3', 'anima_qwen35'],
  flux: ['schnell', 'dev', 'dev_fill'],
  flux2: ['klein_4b', 'klein_4b_base', 'klein_9b', 'klein_9b_base', 'dev'],
  'krea-2': ['krea2_turbo', 'krea2_base'],
  'ltx-2': ['ltx2_dev', 'ltx2_distilled'],
  'minimax-h3': ['fl2va', 'ref2va'],
  'qwen-image': ['generate', 'edit'],
  'qwen-image-2-1': ['qwen_image_21_base', 'qwen_image_21_turbo'],
  'sd-1': ['normal', 'inpaint'],
  'sd-2': ['normal', 'inpaint', 'depth'],
  sdxl: ['normal', 'inpaint'],
  'sdxl-refiner': ['normal'],
  wan: ['t2v_a14b', 'i2v_a14b', 'ti2v_5b'],
  'z-image': ['turbo', 'zbase'],
};

const VARIANTS_BY_TYPE: Record<string, readonly string[]> = {
  clip_embed: ['large', 'gigantic'],
  mistral_encoder: ['cow_mistral3_small', 'mistral3_24b', 'ministral3_3b'],
  pid_decoder: ['res2k_sr4x', 'res2kto4k_sr4x'],
  qwen3_encoder: ['qwen3_4b', 'qwen3_8b', 'qwen3_06b'],
  qwen3_5_encoder: ['qwen3_5_4b'],
  // Required variant configs need explicit choices; a fallback None would fail database validation.
  qwen3_vl_encoder: ['qwen3_vl_4b', 'qwen3_vl_8b'],
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

  if (type === 'qwen3_vl_encoder') {
    // Only base-agnostic encoders carry variant; MiniMax H3's same-type encoder does not support size selection.
    return base === 'any' ? (VARIANTS_BY_TYPE[type] ?? []) : [];
  }

  return VARIANTS_BY_TYPE[type] ?? [];
};

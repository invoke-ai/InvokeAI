import type { AnyModelVariant, BaseModelType, ModelFormat, ModelType } from 'features/nodes/types/common';
import type { AnyModelConfig } from 'services/api/types';
import {
  isCLIPEmbedModelConfig,
  isCLIPVisionModelConfig,
  isControlLoRAModelConfig,
  isControlNetModelConfig,
  isFluxReduxModelConfig,
  isIPAdapterModelConfig,
  isLLaVAModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isQwen3EncoderModelConfig,
  isRefinerMainModelModelConfig,
  isSigLipModelConfig,
  isSpandrelImageToImageModelConfig,
  isT2IAdapterModelConfig,
  isT5EncoderModelConfig,
  isTIModelConfig,
  isUnknownModelConfig,
  isVAEModelConfig,
} from 'services/api/types';
import { objectEntries } from 'tsafe';

import type { FilterableModelType } from './store/modelManagerV2Slice';

export type ModelCategoryData = {
  category: FilterableModelType;
  i18nKey: string;
  filter: (config: AnyModelConfig) => boolean;
};

export const MODEL_CATEGORIES: Record<FilterableModelType, ModelCategoryData> = {
  unknown: {
    category: 'unknown',
    i18nKey: 'common.unknown',
    filter: isUnknownModelConfig,
  },
  main: {
    category: 'main',
    i18nKey: 'modelManager.main',
    filter: isNonRefinerMainModelConfig,
  },
  refiner: {
    category: 'refiner',
    i18nKey: 'sdxl.refiner',
    filter: isRefinerMainModelModelConfig,
  },
  lora: {
    category: 'lora',
    i18nKey: 'modelManager.loraModels',
    filter: isLoRAModelConfig,
  },
  embedding: {
    category: 'embedding',
    i18nKey: 'modelManager.textualInversions',
    filter: isTIModelConfig,
  },
  controlnet: {
    category: 'controlnet',
    i18nKey: 'ControlNet',
    filter: isControlNetModelConfig,
  },
  t2i_adapter: {
    category: 't2i_adapter',
    i18nKey: 'common.t2iAdapter',
    filter: isT2IAdapterModelConfig,
  },
  t5_encoder: {
    category: 't5_encoder',
    i18nKey: 'modelManager.t5Encoder',
    filter: isT5EncoderModelConfig,
  },
  qwen3_encoder: {
    category: 'qwen3_encoder',
    i18nKey: 'modelManager.qwen3Encoder',
    filter: isQwen3EncoderModelConfig,
  },
  control_lora: {
    category: 'control_lora',
    i18nKey: 'modelManager.controlLora',
    filter: isControlLoRAModelConfig,
  },
  clip_embed: {
    category: 'clip_embed',
    i18nKey: 'modelManager.clipEmbed',
    filter: isCLIPEmbedModelConfig,
  },
  spandrel_image_to_image: {
    category: 'spandrel_image_to_image',
    i18nKey: 'modelManager.spandrelImageToImage',
    filter: isSpandrelImageToImageModelConfig,
  },
  ip_adapter: {
    category: 'ip_adapter',
    i18nKey: 'common.ipAdapter',
    filter: isIPAdapterModelConfig,
  },
  vae: {
    category: 'vae',
    i18nKey: 'VAE',
    filter: isVAEModelConfig,
  },
  clip_vision: {
    category: 'clip_vision',
    i18nKey: 'CLIP Vision',
    filter: isCLIPVisionModelConfig,
  },
  siglip: {
    category: 'siglip',
    i18nKey: 'modelManager.sigLip',
    filter: isSigLipModelConfig,
  },
  flux_redux: {
    category: 'flux_redux',
    i18nKey: 'modelManager.fluxRedux',
    filter: isFluxReduxModelConfig,
  },
  llava_onevision: {
    category: 'llava_onevision',
    i18nKey: 'modelManager.llavaOnevision',
    filter: isLLaVAModelConfig,
  },
};

export const MODEL_CATEGORIES_AS_LIST = objectEntries(MODEL_CATEGORIES).map(([category, { i18nKey, filter }]) => ({
  category,
  i18nKey,
  filter,
}));

/**
 * Mapping of model base to its color
 */
export const MODEL_BASE_TO_COLOR: Record<BaseModelType, string> = {
  any: 'base',
  'sd-1': 'green',
  'sd-2': 'teal',
  'sd-3': 'purple',
  sdxl: 'invokeBlue',
  'sdxl-refiner': 'invokeBlue',
  flux: 'gold',
  cogview4: 'red',
  'z-image': 'cyan',
  unknown: 'red',
};

/**
 * Mapping of model type to human readable name
 */
export const MODEL_TYPE_TO_LONG_NAME: Record<ModelType, string> = {
  main: 'Main',
  vae: 'VAE',
  lora: 'LoRA',
  llava_onevision: 'LLaVA OneVision',
  control_lora: 'ControlLoRA',
  controlnet: 'ControlNet',
  t2i_adapter: 'T2I Adapter',
  ip_adapter: 'IP Adapter',
  embedding: 'Embedding',
  onnx: 'ONNX',
  clip_vision: 'CLIP Vision',
  spandrel_image_to_image: 'Spandrel (Image to Image)',
  t5_encoder: 'T5 Encoder',
  qwen3_encoder: 'Qwen3 Encoder',
  clip_embed: 'CLIP Embed',
  siglip: 'SigLIP',
  flux_redux: 'FLUX Redux',
  unknown: 'Unknown',
};

/**
 * Mapping of model base to human readable name
 */
export const MODEL_BASE_TO_LONG_NAME: Record<BaseModelType, string> = {
  any: 'Any',
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
  'sd-3': 'Stable Diffusion 3.x',
  sdxl: 'Stable Diffusion XL',
  'sdxl-refiner': 'Stable Diffusion XL Refiner',
  flux: 'FLUX',
  cogview4: 'CogView4',
  'z-image': 'Z-Image',
  unknown: 'Unknown',
};

/**
 * Mapping of model base to short human readable name
 */
export const MODEL_BASE_TO_SHORT_NAME: Record<BaseModelType, string> = {
  any: 'Any',
  'sd-1': 'SD1.X',
  'sd-2': 'SD2.X',
  'sd-3': 'SD3.X',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXLR',
  flux: 'FLUX',
  cogview4: 'CogView4',
  'z-image': 'Z-Image',
  unknown: 'Unknown',
};

export const MODEL_VARIANT_TO_LONG_NAME: Record<AnyModelVariant, string> = {
  normal: 'Normal',
  inpaint: 'Inpaint',
  depth: 'Depth',
  dev: 'FLUX Dev',
  dev_fill: 'FLUX Dev - Fill',
  schnell: 'FLUX Schnell',
  large: 'CLIP L',
  gigantic: 'CLIP G',
};

export const MODEL_FORMAT_TO_LONG_NAME: Record<ModelFormat, string> = {
  omi: 'OMI',
  diffusers: 'Diffusers',
  checkpoint: 'Checkpoint',
  lycoris: 'LyCORIS',
  onnx: 'ONNX',
  olive: 'Olive',
  embedding_file: 'Embedding (file)',
  embedding_folder: 'Embedding (folder)',
  invokeai: 'InvokeAI',
  t5_encoder: 'T5 Encoder',
  qwen3_encoder: 'Qwen3 Encoder',
  bnb_quantized_int8b: 'BNB Quantized (int8b)',
  bnb_quantized_nf4b: 'BNB Quantized (nf4b)',
  gguf_quantized: 'GGUF Quantized',
  unknown: 'Unknown',
};

export const SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS: BaseModelType[] = ['flux', 'sd-3', 'z-image'];

export const SUPPORTS_REF_IMAGES_BASE_MODELS: BaseModelType[] = ['sd-1', 'sdxl', 'flux'];

export const SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sdxl',
  'cogview4',
  'sd-3',
  'z-image',
];

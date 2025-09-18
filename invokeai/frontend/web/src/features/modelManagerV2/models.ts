import type { BaseModelType } from 'features/nodes/types/common';
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
  isRefinerMainModelModelConfig,
  isSigLipModelConfig,
  isSpandrelImageToImageModelConfig,
  isT2IAdapterModelConfig,
  isT5EncoderModelConfig,
  isTIModelConfig,
  isUnknownModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

type ModelCategoryData = {
  i18nKey: string;
  filter: (config: AnyModelConfig) => boolean;
};

export const MODEL_CATEGORIES: Record<string, ModelCategoryData> = {
  main: {
    i18nKey: 'model_manager.category.main_models',
    filter: isNonRefinerMainModelConfig,
  },
  refiner: {
    i18nKey: 'model_manager.category.refiner_models',
    filter: isRefinerMainModelModelConfig,
  },
  lora: {
    i18nKey: 'model_manager.category.lora_models',
    filter: isLoRAModelConfig,
  },
  embedding: {
    i18nKey: 'model_manager.category.embedding_models',
    filter: isTIModelConfig,
  },
  controlnet: {
    i18nKey: 'model_manager.category.controlnet_models',
    filter: isControlNetModelConfig,
  },
  t2i_adapter: {
    i18nKey: 'model_manager.category.t2i_adapter_models',
    filter: isT2IAdapterModelConfig,
  },
  t5_encoder: {
    i18nKey: 'model_manager.category.t5_encoder_models',
    filter: isT5EncoderModelConfig,
  },
  control_lora: {
    i18nKey: 'model_manager.category.control_lora_models',
    filter: isControlLoRAModelConfig,
  },
  clip_embed: {
    i18nKey: 'model_manager.category.clip_embed_models',
    filter: isCLIPEmbedModelConfig,
  },
  spandrel: {
    i18nKey: 'model_manager.category.spandrel_image_to_image_models',
    filter: isSpandrelImageToImageModelConfig,
  },
  ip_adapter: {
    i18nKey: 'model_manager.category.ip_adapter_models',
    filter: isIPAdapterModelConfig,
  },
  vae: {
    i18nKey: 'model_manager.category.vae_models',
    filter: isVAEModelConfig,
  },
  clip_vision: {
    i18nKey: 'model_manager.category.clip_vision_models',
    filter: isCLIPVisionModelConfig,
  },
  siglip: {
    i18nKey: 'model_manager.category.siglip_models',
    filter: isSigLipModelConfig,
  },
  flux_redux: {
    i18nKey: 'model_manager.category.flux_redux_models',
    filter: isFluxReduxModelConfig,
  },
  llava_one_vision: {
    i18nKey: 'model_manager.category.llava_one_vision_models',
    filter: isLLaVAModelConfig,
  },
  unknown: {
    i18nKey: 'model_manager.category.unknown_models',
    filter: isUnknownModelConfig,
  },
};

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
  imagen3: 'pink',
  imagen4: 'pink',
  'chatgpt-4o': 'pink',
  'flux-kontext': 'pink',
  'gemini-2.5': 'pink',
  veo3: 'purple',
  runway: 'green',
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
  imagen3: 'Imagen3',
  imagen4: 'Imagen4',
  'chatgpt-4o': 'ChatGPT 4o',
  'flux-kontext': 'Flux Kontext',
  'gemini-2.5': 'Gemini 2.5',
  veo3: 'Veo3',
  runway: 'Runway',
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
  imagen3: 'Imagen3',
  imagen4: 'Imagen4',
  'chatgpt-4o': 'ChatGPT 4o',
  'flux-kontext': 'Flux Kontext',
  'gemini-2.5': 'Gemini 2.5',
  veo3: 'Veo3',
  runway: 'Runway',
};

/**
 * List of base models that make API requests
 */
export const API_BASE_MODELS: BaseModelType[] = ['imagen3', 'imagen4', 'chatgpt-4o', 'flux-kontext', 'gemini-2.5'];

export const SUPPORTS_SEED_BASE_MODELS: BaseModelType[] = ['sd-1', 'sd-2', 'sd-3', 'sdxl', 'flux', 'cogview4'];

export const SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS: BaseModelType[] = ['flux', 'sd-3'];

export const SUPPORTS_REF_IMAGES_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sdxl',
  'flux',
  'flux-kontext',
  'chatgpt-4o',
  'gemini-2.5',
];

export const SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sdxl',
  'cogview4',
  'sd-3',
  'imagen3',
  'imagen4',
];

export const SUPPORTS_PIXEL_DIMENSIONS_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sd-3',
  'sdxl',
  'flux',
  'cogview4',
];

export const SUPPORTS_ASPECT_RATIO_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sd-3',
  'sdxl',
  'flux',
  'cogview4',
  'imagen3',
  'imagen4',
  'flux-kontext',
  'chatgpt-4o',
];

export const VIDEO_BASE_MODELS = ['veo3', 'runway'];

export const REQUIRES_STARTING_FRAME_BASE_MODELS = ['runway'];

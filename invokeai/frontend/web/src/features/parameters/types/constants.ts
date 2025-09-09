import type { ComboboxOption } from '@invoke-ai/ui-library';
import type { BaseModelType } from 'services/api/types';

/**
 * Mapping of base model to human readable name
 */
export const MODEL_TYPE_MAP: Record<BaseModelType, string> = {
  any: 'Any',
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
  'sd-3': 'Stable Diffusion 3.x',
  sdxl: 'Stable Diffusion XL',
  'sdxl-refiner': 'Stable Diffusion XL Refiner',
  flux: 'FLUX',
  cogview4: 'CogView4',
  'qwen-image': 'Qwen-Image',
  imagen3: 'Imagen3',
  imagen4: 'Imagen4',
  'chatgpt-4o': 'ChatGPT 4o',
  'flux-kontext': 'Flux Kontext',
  'gemini-2.5': 'Gemini 2.5',
  veo3: 'Veo3',
  runway: 'Runway',
};

/**
 * Mapping of base model to (short) human readable name
 */
export const MODEL_TYPE_SHORT_MAP: Record<BaseModelType, string> = {
  any: 'Any',
  'sd-1': 'SD1.X',
  'sd-2': 'SD2.X',
  'sd-3': 'SD3.X',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXLR',
  flux: 'FLUX',
  cogview4: 'CogView4',
  'qwen-image': 'Qwen',
  imagen3: 'Imagen3',
  imagen4: 'Imagen4',
  'chatgpt-4o': 'ChatGPT 4o',
  'flux-kontext': 'Flux Kontext',
  'gemini-2.5': 'Gemini 2.5',
  veo3: 'Veo3',
  runway: 'Runway',
};

/**
 * Mapping of base model to CLIP skip parameter constraints
 */
export const CLIP_SKIP_MAP: { [key in BaseModelType]?: { maxClip: number; markers: number[] } } = {
  any: {
    maxClip: 0,
    markers: [],
  },
  'sd-1': {
    maxClip: 12,
    markers: [0, 1, 2, 3, 4, 8, 12],
  },
  'sd-2': {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  'sd-3': {
    maxClip: 0,
    markers: [],
  },
  sdxl: {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  'sdxl-refiner': {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  flux: {
    maxClip: 0,
    markers: [],
  },
  cogview4: {
    maxClip: 0,
    markers: [],
  },
  imagen3: {
    maxClip: 0,
    markers: [],
  },
  imagen4: {
    maxClip: 0,
    markers: [],
  },
  'chatgpt-4o': {
    maxClip: 0,
    markers: [],
  },
  'flux-kontext': {
    maxClip: 0,
    markers: [],
  },
  'gemini-2.5': {
    maxClip: 0,
    markers: [],
  },
};

/**
 * Mapping of schedulers to human readable name
 */
export const SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'ddim', label: 'DDIM' },
  { value: 'ddpm', label: 'DDPM' },
  { value: 'deis', label: 'DEIS' },
  { value: 'deis_k', label: 'DEIS Karras' },
  { value: 'dpmpp_2s', label: 'DPM++ 2S' },
  { value: 'dpmpp_2s_k', label: 'DPM++ 2S Karras' },
  { value: 'dpmpp_2m', label: 'DPM++ 2M' },
  { value: 'dpmpp_2m_k', label: 'DPM++ 2M Karras' },
  { value: 'dpmpp_2m_sde', label: 'DPM++ 2M SDE' },
  { value: 'dpmpp_2m_sde_k', label: 'DPM++ 2M SDE Karras' },
  { value: 'dpmpp_3m', label: 'DPM++ 3M' },
  { value: 'dpmpp_3m_k', label: 'DPM++ 3M Karras' },
  { value: 'dpmpp_sde', label: 'DPM++ SDE' },
  { value: 'dpmpp_sde_k', label: 'DPM++ SDE Karras' },
  { value: 'euler', label: 'Euler' },
  { value: 'euler_k', label: 'Euler Karras' },
  { value: 'euler_a', label: 'Euler Ancestral' },
  { value: 'heun', label: 'Heun' },
  { value: 'heun_k', label: 'Heun Karras' },
  { value: 'kdpm_2', label: 'KDPM 2' },
  { value: 'kdpm_2_k', label: 'KDPM 2 Karras' },
  { value: 'kdpm_2_a', label: 'KDPM 2 Ancestral' },
  { value: 'kdpm_2_a_k', label: 'KDPM 2 Ancestral Karras' },
  { value: 'lcm', label: 'LCM' },
  { value: 'lms', label: 'LMS' },
  { value: 'lms_k', label: 'LMS Karras' },
  { value: 'pndm', label: 'PNDM' },
  { value: 'tcd', label: 'TCD' },
  { value: 'unipc', label: 'UniPC' },
  { value: 'unipc_k', label: 'UniPC Karras' },
];

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

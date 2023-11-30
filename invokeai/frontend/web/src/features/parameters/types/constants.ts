import { SchedulerField } from 'features/nodes/types/common';
import { LoRAModelFormat } from 'services/api/types';

/**
 * Mapping of model type to human readable name
 */
export const MODEL_TYPE_MAP = {
  any: 'Any',
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
  sdxl: 'Stable Diffusion XL',
  'sdxl-refiner': 'Stable Diffusion XL Refiner',
};

/**
 * Mapping of model type to (short) human readable name
 */
export const MODEL_TYPE_SHORT_MAP = {
  any: 'Any',
  'sd-1': 'SD1',
  'sd-2': 'SD2',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXLR',
};

/**
 * Mapping of model type to CLIP skip parameter constraints
 */
export const CLIP_SKIP_MAP = {
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
  sdxl: {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  'sdxl-refiner': {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
};

/**
 * Mapping of LoRA format to human readable name
 */
export const LORA_MODEL_FORMAT_MAP: {
  [key in LoRAModelFormat]: string;
} = {
  lycoris: 'LyCORIS',
  diffusers: 'Diffusers',
};

/**
 * Mapping of schedulers to human readable name
 */
export const SCHEDULER_LABEL_MAP: Record<SchedulerField, string> = {
  euler: 'Euler',
  deis: 'DEIS',
  ddim: 'DDIM',
  ddpm: 'DDPM',
  dpmpp_sde: 'DPM++ SDE',
  dpmpp_2s: 'DPM++ 2S',
  dpmpp_2m: 'DPM++ 2M',
  dpmpp_2m_sde: 'DPM++ 2M SDE',
  heun: 'Heun',
  kdpm_2: 'KDPM 2',
  lms: 'LMS',
  pndm: 'PNDM',
  unipc: 'UniPC',
  euler_k: 'Euler Karras',
  dpmpp_sde_k: 'DPM++ SDE Karras',
  dpmpp_2s_k: 'DPM++ 2S Karras',
  dpmpp_2m_k: 'DPM++ 2M Karras',
  dpmpp_2m_sde_k: 'DPM++ 2M SDE Karras',
  heun_k: 'Heun Karras',
  lms_k: 'LMS Karras',
  euler_a: 'Euler Ancestral',
  kdpm_2_a: 'KDPM 2 Ancestral',
  lcm: 'LCM',
};

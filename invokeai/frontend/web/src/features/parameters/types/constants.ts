import type { ComboboxOption } from '@invoke-ai/ui-library';

/**
 * Mapping of base model to human readable name
 */
export const MODEL_TYPE_MAP = {
  any: 'Any',
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
  sdxl: 'Stable Diffusion XL',
  'sdxl-refiner': 'Stable Diffusion XL Refiner',
};

/**
 * Mapping of base model to (short) human readable name
 */
export const MODEL_TYPE_SHORT_MAP = {
  any: 'Any',
  'sd-1': 'SD1.X',
  'sd-2': 'SD2.X',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXLR',
};

/**
 * Mapping of base model to CLIP skip parameter constraints
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
 * Mapping of schedulers to human readable name
 */
export const SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'deis', label: 'DEIS' },
  { value: 'ddim', label: 'DDIM' },
  { value: 'ddpm', label: 'DDPM' },
  { value: 'dpmpp_sde', label: 'DPM++ SDE' },
  { value: 'dpmpp_2s', label: 'DPM++ 2S' },
  { value: 'dpmpp_2m', label: 'DPM++ 2M' },
  { value: 'dpmpp_2m_sde', label: 'DPM++ 2M SDE' },
  { value: 'heun', label: 'Heun' },
  { value: 'kdpm_2', label: 'KDPM 2' },
  { value: 'lms', label: 'LMS' },
  { value: 'pndm', label: 'PNDM' },
  { value: 'unipc', label: 'UniPC' },
  { value: 'euler_k', label: 'Euler Karras' },
  { value: 'dpmpp_sde_k', label: 'DPM++ SDE Karras' },
  { value: 'dpmpp_2s_k', label: 'DPM++ 2S Karras' },
  { value: 'dpmpp_2m_k', label: 'DPM++ 2M Karras' },
  { value: 'dpmpp_2m_sde_k', label: 'DPM++ 2M SDE Karras' },
  { value: 'heun_k', label: 'Heun Karras' },
  { value: 'lms_k', label: 'LMS Karras' },
  { value: 'euler_a', label: 'Euler Ancestral' },
  { value: 'kdpm_2_a', label: 'KDPM 2 Ancestral' },
  { value: 'lcm', label: 'LCM' },
].sort((a, b) => a.label.localeCompare(b.label));

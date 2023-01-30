// TODO: use Enums?

import { InProgressImageType } from 'features/system/store/systemSlice';

// Valid samplers
export const SAMPLERS: Array<string> = [
  'ddim',
  'plms',
  'k_lms',
  'k_dpm_2',
  'k_dpm_2_a',
  'k_dpmpp_2',
  'k_dpmpp_2_a',
  'k_euler',
  'k_euler_a',
  'k_heun',
];

// Valid image widths
export const WIDTHS: Array<number> = [
  64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
  1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792,
  1856, 1920, 1984, 2048,
];

// Valid image heights
export const HEIGHTS: Array<number> = [
  64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
  1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792,
  1856, 1920, 1984, 2048,
];

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<{ key: string; value: number }> = [
  { key: '2x', value: 2 },
  { key: '4x', value: 4 },
];

export const NUMPY_RAND_MIN = 0;

export const NUMPY_RAND_MAX = 4294967295;

export const FACETOOL_TYPES = ['gfpgan', 'codeformer'] as const;

export const IN_PROGRESS_IMAGE_TYPES: Array<{
  key: string;
  value: InProgressImageType;
}> = [
  { key: 'None', value: 'none' },
  { key: 'Fast', value: 'latents' },
  { key: 'Accurate', value: 'full-res' },
];

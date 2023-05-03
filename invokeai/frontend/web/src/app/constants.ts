// TODO: use Enums?

export const DIFFUSERS_SCHEDULERS: Array<string> = [
  'ddim',
  'plms',
  'k_lms',
  'dpmpp_2',
  'k_dpm_2',
  'k_dpm_2_a',
  'k_dpmpp_2',
  'k_euler',
  'k_euler_a',
  'k_heun',
];

// Valid image widths
export const WIDTHS: Array<number> = Array.from(Array(64)).map(
  (_x, i) => (i + 1) * 64
);

// Valid image heights
export const HEIGHTS: Array<number> = Array.from(Array(64)).map(
  (_x, i) => (i + 1) * 64
);

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<{ key: string; value: number }> = [
  { key: '2x', value: 2 },
  { key: '4x', value: 4 },
];

export const NUMPY_RAND_MIN = 0;

export const NUMPY_RAND_MAX = 2147483647;

export const FACETOOL_TYPES = ['gfpgan', 'codeformer'] as const;

export const NODE_MIN_WIDTH = 250;

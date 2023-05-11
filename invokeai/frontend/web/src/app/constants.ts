// TODO: use Enums?

export const DIFFUSERS_SCHEDULERS: Array<string> = [
  'ddim',
  'ddpm',
  'deis',
  'lms',
  'pndm',
  'heun',
  'euler',
  'euler_k',
  'euler_a',
  'kdpm_2',
  'kdpm_2_a',
  'dpmpp_2s',
  'dpmpp_2m',
  'dpmpp_2m_k',
  'unipc',
];

export const IMG2IMG_DIFFUSERS_SCHEDULERS = DIFFUSERS_SCHEDULERS.filter(
  (scheduler) => {
    return scheduler !== 'dpmpp_2s';
  }
);

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

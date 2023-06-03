// TODO: use Enums?

export const SCHEDULERS = [
  'ddim',
  'lms',
  'euler',
  'euler_k',
  'euler_a',
  'dpmpp_2s',
  'dpmpp_2m',
  'dpmpp_2m_k',
  'kdpm_2',
  'kdpm_2_a',
  'deis',
  'ddpm',
  'pndm',
  'heun',
  'heun_k',
  'unipc',
] as const;

export type Scheduler = (typeof SCHEDULERS)[number];

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<{ key: string; value: number }> = [
  { key: '2x', value: 2 },
  { key: '4x', value: 4 },
];
export const NUMPY_RAND_MIN = 0;

export const NUMPY_RAND_MAX = 2147483647;

export const FACETOOL_TYPES = ['gfpgan', 'codeformer'] as const;

export const NODE_MIN_WIDTH = 250;

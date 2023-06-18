import { SelectItem } from '@mantine/core';

// TODO: use Enums?
export const SCHEDULERS: SelectItem[] = [
  { label: 'euler', value: 'euler', group: 'Standard' },
  { label: 'deis', value: 'deis', group: 'Standard' },
  { label: 'ddim', value: 'ddim', group: 'Standard' },
  { label: 'ddpm', value: 'ddpm', group: 'Standard' },
  { label: 'dpmpp_2s', value: 'dpmpp_2s', group: 'Standard' },
  { label: 'dpmpp_2m', value: 'dpmpp_2m', group: 'Standard' },
  { label: 'heun', value: 'heun', group: 'Standard' },
  { label: 'kdpm_2', value: 'kdpm_2', group: 'Standard' },
  { label: 'lms', value: 'lms', group: 'Standard' },
  { label: 'pndm', value: 'pndm', group: 'Standard' },
  { label: 'unipc', value: 'unipc', group: 'Standard' },
  { label: 'euler_k', value: 'euler_k', group: 'Karras' },
  { label: 'dpmpp_2s_k', value: 'dpmpp_2s_k', group: 'Karras' },
  { label: 'dpmpp_2m_k', value: 'dpmpp_2m_k', group: 'Karras' },
  { label: 'heun_k', value: 'heun_k', group: 'Karras' },
  { label: 'lms_k', value: 'lms_k', group: 'Karras' },
  { label: 'euler_a', value: 'euler_a', group: 'Ancestral' },
  { label: 'kdpm_2_a', value: 'kdpm_2_a', group: 'Ancestral' },
];

export const SCHEDULER_ITEMS = [
  'ddim',
  'lms',
  'lms_k',
  'euler',
  'euler_k',
  'euler_a',
  'dpmpp_2s',
  'dpmpp_2s_k',
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

export type Scheduler = (typeof SCHEDULER_ITEMS)[number];

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<{ label: string; value: string }> = [
  { label: '2x', value: '2' },
  { label: '4x', value: '4' },
];
export const NUMPY_RAND_MIN = 0;

export const NUMPY_RAND_MAX = 2147483647;

export const FACETOOL_TYPES = ['gfpgan', 'codeformer'] as const;

export const NODE_MIN_WIDTH = 250;

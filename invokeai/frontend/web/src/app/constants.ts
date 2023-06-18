import { SelectItem } from '@mantine/core';

// TODO: use Enums?
export const SCHEDULERS: SelectItem[] = [
  { label: 'Euler', value: 'euler', group: 'Standard' },
  { label: 'DEIS', value: 'deis', group: 'Standard' },
  { label: 'DDIM', value: 'ddim', group: 'Standard' },
  { label: 'DDPM', value: 'ddpm', group: 'Standard' },
  { label: 'DPM++ 2S', value: 'dpmpp_2s', group: 'Standard' },
  { label: 'DPM++ 2M', value: 'dpmpp_2m', group: 'Standard' },
  { label: 'Heun', value: 'heun', group: 'Standard' },
  { label: 'KDPM 2', value: 'kdpm_2', group: 'Standard' },
  { label: 'LMS', value: 'lms', group: 'Standard' },
  { label: 'PNDM', value: 'pndm', group: 'Standard' },
  { label: 'UniPC', value: 'unipc', group: 'Standard' },
  { label: 'Euler Karras', value: 'euler_k', group: 'Karras' },
  { label: 'DPM++ 2S Karras', value: 'dpmpp_2s_k', group: 'Karras' },
  { label: 'DPM++ 2M Karras', value: 'dpmpp_2m_k', group: 'Karras' },
  { label: 'Heun Karras', value: 'heun_k', group: 'Karras' },
  { label: 'LMS Karras', value: 'lms_k', group: 'Karras' },
  { label: 'Euler Ancestral', value: 'euler_a', group: 'Ancestral' },
  { label: 'KDPM 2 Ancestral', value: 'kdpm_2_a', group: 'Ancestral' },
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

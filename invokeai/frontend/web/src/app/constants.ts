import { SelectItem } from '@mantine/core';

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

// zod needs the array to be `as const` to infer the type correctly
export const SCHEDULER_NAMES_AS_CONST = [
  'euler',
  'deis',
  'ddim',
  'ddpm',
  'dpmpp_2s',
  'dpmpp_2m',
  'heun',
  'kdpm_2',
  'lms',
  'pndm',
  'unipc',
  'euler_k',
  'dpmpp_2s_k',
  'dpmpp_2m_k',
  'heun_k',
  'lms_k',
  'euler_a',
  'kdpm_2_a',
] as const;

export const SCHEDULER_NAMES = [...SCHEDULER_NAMES_AS_CONST];

export type Scheduler = (typeof SCHEDULER_NAMES)[number];

// Valid upscaling levels
export const UPSCALING_LEVELS: Array<{ label: string; value: string }> = [
  { label: '2x', value: '2' },
  { label: '4x', value: '4' },
];
export const NUMPY_RAND_MIN = 0;

export const NUMPY_RAND_MAX = 2147483647;

export const FACETOOL_TYPES = ['gfpgan', 'codeformer'] as const;

export const NODE_MIN_WIDTH = 250;

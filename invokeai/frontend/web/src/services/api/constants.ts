import { BaseModelType } from './types';

export const ALL_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sdxl',
  'sdxl-refiner',
];

export const NON_REFINER_BASE_MODELS: BaseModelType[] = [
  'sd-1',
  'sd-2',
  'sdxl',
];

export const SDXL_MAIN_MODELS: BaseModelType[] = ['sdxl'];
export const NON_SDXL_MAIN_MODELS: BaseModelType[] = ['sd-1', 'sd-2'];

export const REFINER_BASE_MODELS: BaseModelType[] = ['sdxl-refiner'];

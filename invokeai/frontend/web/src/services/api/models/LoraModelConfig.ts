/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type LoRAModelConfig = {
  path: string;
  description?: string;
  format: ('lycoris' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};

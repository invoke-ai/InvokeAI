/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type VAEModelConfig = {
  path: string;
  description?: string;
  format: ('checkpoint' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};

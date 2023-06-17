/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type VaeModelConfig = {
  path: string;
  description?: string;
  format: ('checkpoint' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};

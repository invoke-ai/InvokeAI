/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';

export type StableDiffusion1DiffusersModelConfig = {
  path: string;
  description?: string;
  format: 'diffusers';
  default?: boolean;
  error?: ModelError;
  vae?: string;
  variant: ModelVariantType;
};

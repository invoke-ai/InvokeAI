/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';

export type StableDiffusion1CheckpointModelConfig = {
  path: string;
  description?: string;
  format: 'checkpoint';
  default?: boolean;
  error?: ModelError;
  vae?: string;
  config?: string;
  variant: ModelVariantType;
};

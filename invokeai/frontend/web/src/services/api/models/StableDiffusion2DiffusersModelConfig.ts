/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';
import type { SchedulerPredictionType } from './SchedulerPredictionType';

export type StableDiffusion2DiffusersModelConfig = {
  path: string;
  description?: string;
  format: 'diffusers';
  default?: boolean;
  error?: ModelError;
  vae?: string;
  variant: ModelVariantType;
  prediction_type: SchedulerPredictionType;
  upcast_attention: boolean;
};

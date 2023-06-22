/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';
import type { SchedulerPredictionType } from './SchedulerPredictionType';

export type StableDiffusion2ModelDiffusersConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'pipeline';
  path: string;
  description?: string;
  model_format: 'diffusers';
  error?: ModelError;
  vae?: string;
  variant: ModelVariantType;
  prediction_type: SchedulerPredictionType;
  upcast_attention: boolean;
};


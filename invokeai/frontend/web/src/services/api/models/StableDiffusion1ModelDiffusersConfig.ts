/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';

export type StableDiffusion1ModelDiffusersConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'pipeline';
  path: string;
  description?: string;
  model_format: 'diffusers';
  error?: ModelError;
  vae?: string;
  variant: ModelVariantType;
};


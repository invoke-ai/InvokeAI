/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';

export type StableDiffusion1ModelCheckpointConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'pipeline';
  path: string;
  description?: string;
  model_format: 'checkpoint';
  error?: ModelError;
  vae?: string;
  config?: string;
  variant: ModelVariantType;
};


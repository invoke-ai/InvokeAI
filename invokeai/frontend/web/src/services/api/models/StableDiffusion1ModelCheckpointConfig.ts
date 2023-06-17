/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelType } from './ModelType';
import type { ModelVariantType } from './ModelVariantType';

export type StableDiffusion1ModelCheckpointConfig = {
  name: string;
  base_model: BaseModelType;
  type: ModelType;
  path: string;
  description?: string;
  format: 'checkpoint';
  default?: boolean;
  error?: ModelError;
  vae?: string;
  config?: string;
  variant: ModelVariantType;
};

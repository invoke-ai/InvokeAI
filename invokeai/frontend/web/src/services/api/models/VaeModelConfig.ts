/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { VaeModelFormat } from './VaeModelFormat';

export type VaeModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'vae';
  path: string;
  description?: string;
  model_format: VaeModelFormat;
  error?: ModelError;
};


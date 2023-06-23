/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ControlNetModelFormat } from './ControlNetModelFormat';
import type { ModelError } from './ModelError';

export type ControlNetModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'controlnet';
  path: string;
  description?: string;
  model_format: ControlNetModelFormat;
  error?: ModelError;
};


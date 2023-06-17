/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelType } from './ModelType';

export type ControlNetModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: ModelType;
  path: string;
  description?: string;
  format: ('checkpoint' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};

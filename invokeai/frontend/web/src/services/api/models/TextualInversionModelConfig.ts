/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';

export type TextualInversionModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'embedding';
  path: string;
  description?: string;
  model_format: null;
  error?: ModelError;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { LoRAModelFormat } from './LoRAModelFormat';
import type { ModelError } from './ModelError';

export type LoRAModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'lora';
  path: string;
  description?: string;
  model_format: LoRAModelFormat;
  error?: ModelError;
};


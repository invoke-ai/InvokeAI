/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type invokeai__backend__model_management__models__lora__LoRAModel__Config = {
  path: string;
  description?: string;
  format: ('lycoris' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};


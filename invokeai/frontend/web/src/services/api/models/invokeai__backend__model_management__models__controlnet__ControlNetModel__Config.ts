/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type invokeai__backend__model_management__models__controlnet__ControlNetModel__Config = {
  path: string;
  description?: string;
  format: ('checkpoint' | 'diffusers');
  default?: boolean;
  error?: ModelError;
};


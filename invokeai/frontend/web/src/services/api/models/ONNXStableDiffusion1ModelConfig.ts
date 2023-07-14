/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelError } from './ModelError';
import type { ModelVariantType } from './ModelVariantType';

export type ONNXStableDiffusion1ModelConfig = {
  name: string;
  base_model: BaseModelType;
  type: 'onnx';
  path: string;
  description?: string;
  error?: ModelError;
  variant: ModelVariantType;
};

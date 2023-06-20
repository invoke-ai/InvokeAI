/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LoraInfo } from './LoraInfo';
import type { ModelInfo } from './ModelInfo';

export type UNetField = {
  /**
   * Info to load unet submodel
   */
  unet: ModelInfo;
  /**
   * Info to load scheduler submodel
   */
  scheduler: ModelInfo;
  /**
   * Loras to apply on model loading
   */
  loras: Array<LoraInfo>;
};


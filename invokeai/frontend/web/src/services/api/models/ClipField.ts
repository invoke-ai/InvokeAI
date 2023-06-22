/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LoraInfo } from './LoraInfo';
import type { ModelInfo } from './ModelInfo';

export type ClipField = {
  /**
   * Info to load tokenizer submodel
   */
  tokenizer: ModelInfo;
  /**
   * Info to load text_encoder submodel
   */
  text_encoder: ModelInfo;
  /**
   * Loras to apply on model loading
   */
  loras: Array<LoraInfo>;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ClipField } from './ClipField';
import type { UNetField } from './UNetField';

/**
 * Model loader output
 */
export type LoraLoaderOutput = {
  type?: 'lora_loader_output';
  /**
   * UNet submodel
   */
  unet?: UNetField;
  /**
   * Tokenizer and text_encoder submodels
   */
  clip?: ClipField;
};


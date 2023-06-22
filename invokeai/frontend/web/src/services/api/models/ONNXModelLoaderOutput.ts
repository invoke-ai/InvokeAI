/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ClipField } from './ClipField';
import type { UNetField } from './UNetField';
import type { VaeField } from './VaeField';

/**
 * Model loader output
 */
export type ONNXModelLoaderOutput = {
  type?: 'model_loader_output_onnx';
  /**
   * UNet submodel
   */
  unet?: UNetField;
  /**
   * Tokenizer and text_encoder submodels
   */
  clip?: ClipField;
  /**
   * Vae submodel
   */
  vae_decoder?: VaeField;
  /**
   * Vae submodel
   */
  vae_encoder?: VaeField;
};

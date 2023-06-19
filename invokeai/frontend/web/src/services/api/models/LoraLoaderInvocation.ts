/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ClipField } from './ClipField';
import type { UNetField } from './UNetField';

/**
 * Apply selected lora to unet and text_encoder.
 */
export type LoraLoaderInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'lora_loader';
  /**
   * Lora model name
   */
  lora_name: string;
  /**
   * With what weight to apply lora
   */
  weight?: number;
  /**
   * UNet model for applying lora
   */
  unet?: UNetField;
  /**
   * Clip model for applying lora
   */
  clip?: ClipField;
};


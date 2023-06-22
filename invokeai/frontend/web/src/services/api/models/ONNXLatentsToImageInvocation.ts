/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';
import type { VaeField } from './VaeField';

/**
 * Generates an image from latents.
 */
export type ONNXLatentsToImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'l2i_onnx';
  /**
   * The latents to generate an image from
   */
  latents?: LatentsField;
  /**
   * Vae submodel
   */
  vae?: VaeField;
};

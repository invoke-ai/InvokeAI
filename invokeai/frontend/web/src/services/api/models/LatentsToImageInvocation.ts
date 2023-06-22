/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';
import type { VaeField } from './VaeField';

/**
 * Generates an image from latents.
 */
export type LatentsToImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'l2i';
  /**
   * The latents to generate an image from
   */
  latents?: LatentsField;
  /**
   * Vae submodel
   */
  vae?: VaeField;
  /**
   * Decode latents by overlaping tiles(less memory consumption)
   */
  tiled?: boolean;
};


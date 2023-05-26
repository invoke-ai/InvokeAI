/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Extracts the alpha channel of an image as a mask.
 */
export type MaskFromAlphaInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'tomask';
  /**
   * The image to create the mask from
   */
  image?: ImageField;
  /**
   * Whether or not to invert the mask
   */
  invert?: boolean;
};


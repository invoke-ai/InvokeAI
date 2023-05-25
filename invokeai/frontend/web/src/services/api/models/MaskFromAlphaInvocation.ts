/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Extracts the alpha channel of an image as a mask.
 */
export type MaskFromAlphaInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'tomask';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to create the mask from
   */
  image?: ImageField;
  /**
   * Whether or not to invert the mask
   */
  invert?: boolean;
};


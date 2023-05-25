/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Encodes an image into latents.
 */
export type ImageToLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'i2l';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to encode
   */
  image?: ImageField;
  /**
   * The model to use
   */
  model?: string;
};


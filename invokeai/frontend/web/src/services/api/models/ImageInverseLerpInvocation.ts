/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Inverse linear interpolation of all pixels of an image
 */
export type ImageInverseLerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_ilerp';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to lerp
   */
  image?: ImageField;
  /**
   * The minimum input value
   */
  min?: number;
  /**
   * The maximum input value
   */
  max?: number;
};


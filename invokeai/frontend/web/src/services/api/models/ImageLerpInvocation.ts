/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Linear interpolation of all pixels of an image
 */
export type ImageLerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_lerp';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to lerp
   */
  image?: ImageField;
  /**
   * The minimum output value
   */
  min?: number;
  /**
   * The maximum output value
   */
  max?: number;
};


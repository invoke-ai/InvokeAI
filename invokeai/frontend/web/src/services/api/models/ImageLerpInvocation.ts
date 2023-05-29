/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Linear interpolation of all pixels of an image
 */
export type ImageLerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_lerp';
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


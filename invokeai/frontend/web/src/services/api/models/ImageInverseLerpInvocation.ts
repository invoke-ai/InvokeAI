/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Inverse linear interpolation of all pixels of an image
 */
export type ImageInverseLerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_ilerp';
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


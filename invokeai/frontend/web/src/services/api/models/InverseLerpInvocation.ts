/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Inverse linear interpolation of all pixels of an image
 */
export type InverseLerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'ilerp';
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


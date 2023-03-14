/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Linear interpolation of all pixels of an image
 */
export type LerpInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'lerp';
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


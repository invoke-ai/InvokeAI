/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Multiplies two images together using `PIL.ImageChops.multiply()`.
 */
export type ImageMultiplyInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_mul';
  /**
   * The first image to multiply
   */
  image1?: ImageField;
  /**
   * The second image to multiply
   */
  image2?: ImageField;
};


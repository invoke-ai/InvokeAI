/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Multiplies two images together using `PIL.ImageChops.multiply()`.
 */
export type ImageMultiplyInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_mul';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The first image to multiply
   */
  image1?: ImageField;
  /**
   * The second image to multiply
   */
  image2?: ImageField;
};


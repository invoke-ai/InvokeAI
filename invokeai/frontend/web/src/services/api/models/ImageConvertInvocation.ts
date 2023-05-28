/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Converts an image to a different mode.
 */
export type ImageConvertInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_conv';
  /**
   * The image to convert
   */
  image?: ImageField;
  /**
   * The mode to convert to
   */
  mode?: 'L' | 'RGB' | 'RGBA' | 'CMYK' | 'YCbCr' | 'LAB' | 'HSV' | 'I' | 'F';
};


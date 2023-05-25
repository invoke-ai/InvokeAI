/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Converts an image to a different mode.
 */
export type ImageConvertInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_conv';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to convert
   */
  image?: ImageField;
  /**
   * The mode to convert to
   */
  mode?: 'L' | 'RGB' | 'RGBA' | 'CMYK' | 'YCbCr' | 'LAB' | 'HSV' | 'I' | 'F';
};


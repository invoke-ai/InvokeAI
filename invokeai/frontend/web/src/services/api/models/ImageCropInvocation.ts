/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Crops an image to a specified box. The box can be outside of the image.
 */
export type ImageCropInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_crop';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to crop
   */
  image?: ImageField;
  /**
   * The left x coordinate of the crop rectangle
   */
  'x'?: number;
  /**
   * The top y coordinate of the crop rectangle
   */
  'y'?: number;
  /**
   * The width of the crop rectangle
   */
  width?: number;
  /**
   * The height of the crop rectangle
   */
  height?: number;
};


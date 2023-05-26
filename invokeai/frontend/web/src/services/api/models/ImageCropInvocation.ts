/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Crops an image to a specified box. The box can be outside of the image.
 */
export type ImageCropInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_crop';
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


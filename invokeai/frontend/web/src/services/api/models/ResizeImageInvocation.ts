/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Crops an image to a specified box. The box can be outside of the image.
 */
export type ResizeImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'resize';
  /**
   * The image to resize
   */
  image?: ImageField;
  /**
   * The size of the resized images' longest side
   */
  size: number;
};


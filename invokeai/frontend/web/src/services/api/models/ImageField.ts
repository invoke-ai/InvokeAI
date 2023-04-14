/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageType } from './ImageType';

/**
 * An image field used for passing image objects between invocations
 */
export type ImageField = {
  /**
   * The type of the image
   */
  image_type: ImageType;
  /**
   * The name of the image
   */
  image_name: string;
  /**
   * The width of the image in pixels
   */
  width: number;
  /**
   * The height of the image in pixels
   */
  height: number;
  /**
   * The image mode (ie pixel format)
   */
  mode: string;
  /**
   * The image file's metadata
   */
  info: any;
};


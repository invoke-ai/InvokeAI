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
};


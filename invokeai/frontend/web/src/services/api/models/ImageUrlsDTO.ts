/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageType } from './ImageType';

/**
 * The URLs for an image and its thumbnaill
 */
export type ImageUrlsDTO = {
  /**
   * The name of the image.
   */
  image_name: string;
  /**
   * The type of the image.
   */
  image_type: ImageType;
  /**
   * The URL of the image.
   */
  image_url: string;
  /**
   * The thumbnail URL of the image.
   */
  thumbnail_url: string;
};


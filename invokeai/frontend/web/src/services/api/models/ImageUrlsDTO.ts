/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ResourceOrigin } from './ResourceOrigin';

/**
 * The URLs for an image and its thumbnail.
 */
export type ImageUrlsDTO = {
  /**
   * The unique name of the image.
   */
  image_name: string;
  /**
   * The type of the image.
   */
  image_origin: ResourceOrigin;
  /**
   * The URL of the image.
   */
  image_url: string;
  /**
   * The URL of the image's thumbnail.
   */
  thumbnail_url: string;
};


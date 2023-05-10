/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageResponseMetadata } from './ImageResponseMetadata';
import type { ImageType } from './ImageType';

/**
 * The response type for images
 */
export type ImageResponse = {
  /**
   * The type of the image
   */
  image_type: ImageType;
  /**
   * The name of the image
   */
  image_name: string;
  /**
   * The url of the image
   */
  image_url: string;
  /**
   * The url of the image's thumbnail
   */
  thumbnail_url: string;
  /**
   * The image's metadata
   */
  metadata: ImageResponseMetadata;
};


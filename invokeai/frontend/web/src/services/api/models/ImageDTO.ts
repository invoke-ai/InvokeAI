/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageCategory } from './ImageCategory';
import type { ImageMetadata } from './ImageMetadata';
import type { ImageType } from './ImageType';

/**
 * Deserialized image record with URLs.
 */
export type ImageDTO = {
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
  /**
   * The category of the image.
   */
  image_category: ImageCategory;
  /**
   * The created timestamp of the image.
   */
  created_at: string;
  /**
   * The session ID.
   */
  session_id?: string;
  /**
   * The node ID.
   */
  node_id?: string;
  /**
   * The image's metadata.
   */
  metadata?: ImageMetadata;
};


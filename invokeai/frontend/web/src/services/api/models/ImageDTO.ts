/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageCategory } from './ImageCategory';
import type { ImageMetadata } from './ImageMetadata';
import type { ResourceOrigin } from './ResourceOrigin';

/**
 * Deserialized image record, enriched for the frontend with URLs.
 */
export type ImageDTO = {
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
  /**
   * The category of the image.
   */
  image_category: ImageCategory;
  /**
   * The width of the image in px.
   */
  width: number;
  /**
   * The height of the image in px.
   */
  height: number;
  /**
   * The created timestamp of the image.
   */
  created_at: string;
  /**
   * The updated timestamp of the image.
   */
  updated_at: string;
  /**
   * The deleted timestamp of the image.
   */
  deleted_at?: string;
  /**
   * Whether this is an intermediate image.
   */
  is_intermediate: boolean;
  /**
   * The session ID that generated this image, if it is a generated image.
   */
  session_id?: string;
  /**
   * The node ID that generated this image, if it is a generated image.
   */
  node_id?: string;
  /**
   * A limited subset of the image's generation metadata. Retrieve the image's session for full metadata.
   */
  metadata?: ImageMetadata;
};


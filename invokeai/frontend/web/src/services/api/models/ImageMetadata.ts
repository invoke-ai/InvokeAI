/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * An image's metadata
 */
export type ImageMetadata = {
  /**
   * The creation timestamp of the image
   */
  timestamp: number;
  /**
   * The width of the image in pixels
   */
  width: number;
  /**
   * The width of the image in pixels
   */
  height: number;
  /**
   * The image's SD-specific metadata
   */
  sd_metadata?: any;
};


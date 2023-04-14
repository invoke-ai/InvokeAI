/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvokeAIMetadata } from './InvokeAIMetadata';

/**
 * An image's general metadata
 */
export type ImageMetadata = {
  /**
   * The creation timestamp of the image
   */
  created: number;
  /**
   * The width of the image in pixels
   */
  width: number;
  /**
   * The height of the image in pixels
   */
  height: number;
  /**
   * The image's InvokeAI-specific metadata
   */
  invokeai?: InvokeAIMetadata;
};


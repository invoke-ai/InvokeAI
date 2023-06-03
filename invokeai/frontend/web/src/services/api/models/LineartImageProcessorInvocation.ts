/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies line art processing to image
 */
export type LineartImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'lineart_image_processor';
  /**
   * image to process
   */
  image?: ImageField;
  /**
   * pixel resolution for edge detection
   */
  detect_resolution?: number;
  /**
   * pixel resolution for output image
   */
  image_resolution?: number;
  /**
   * whether to use coarse mode
   */
  coarse?: boolean;
};


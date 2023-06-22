/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies content shuffle processing to image
 */
export type ContentShuffleImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'content_shuffle_image_processor';
  /**
   * The image to process
   */
  image?: ImageField;
  /**
   * The pixel resolution for detection
   */
  detect_resolution?: number;
  /**
   * The pixel resolution for the output image
   */
  image_resolution?: number;
  /**
   * Content shuffle `h` parameter
   */
  'h'?: number;
  /**
   * Content shuffle `w` parameter
   */
  'w'?: number;
  /**
   * Content shuffle `f` parameter
   */
  'f'?: number;
};


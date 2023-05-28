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
   * content shuffle h parameter
   */
  'h'?: number;
  /**
   * content shuffle w parameter
   */
  'w'?: number;
  /**
   * cont
   */
  'f'?: number;
};


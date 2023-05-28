/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Canny edge detection for ControlNet
 */
export type CannyImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'canny_image_processor';
  /**
   * image to process
   */
  image?: ImageField;
  /**
   * low threshold of Canny pixel gradient
   */
  low_threshold?: number;
  /**
   * high threshold of Canny pixel gradient
   */
  high_threshold?: number;
};


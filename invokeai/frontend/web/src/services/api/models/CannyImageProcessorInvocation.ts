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
   * The image to process
   */
  image?: ImageField;
  /**
   * The low threshold of the Canny pixel gradient (0-255)
   */
  low_threshold?: number;
  /**
   * The high threshold of the Canny pixel gradient (0-255)
   */
  high_threshold?: number;
};


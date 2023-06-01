/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies Openpose processing to image
 */
export type OpenposeImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'openpose_image_processor';
  /**
   * image to process
   */
  image?: ImageField;
  /**
   * whether to use hands and face mode
   */
  hand_and_face?: boolean;
  /**
   * pixel resolution for edge detection
   */
  detect_resolution?: number;
  /**
   * pixel resolution for output image
   */
  image_resolution?: number;
};


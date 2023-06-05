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
   * The image to process
   */
  image?: ImageField;
  /**
   * Whether to use hands and face mode
   */
  hand_and_face?: boolean;
  /**
   * The pixel resolution for detection
   */
  detect_resolution?: number;
  /**
   * The pixel resolution for the output image
   */
  image_resolution?: number;
};


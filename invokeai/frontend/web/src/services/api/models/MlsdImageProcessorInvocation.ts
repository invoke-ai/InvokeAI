/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies MLSD processing to image
 */
export type MlsdImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'mlsd_image_processor';
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
   * MLSD parameter `thr_v`
   */
  thr_v?: number;
  /**
   * MLSD parameter `thr_d`
   */
  thr_d?: number;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies Midas depth processing to image
 */
export type MidasDepthImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'midas_depth_image_processor';
  /**
   * image to process
   */
  image?: ImageField;
  /**
   * Midas parameter a = amult * PI
   */
  a_mult?: number;
  /**
   * Midas parameter bg_th
   */
  bg_th?: number;
};


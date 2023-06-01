/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Base class for invocations that preprocess images for ControlNet
 */
export type ImageProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'image_processor';
  /**
   * image to process
   */
  image?: ImageField;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Displays a provided image, and passes it forward in the pipeline.
 */
export type ShowImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'show_image';
  /**
   * The image to show
   */
  image?: ImageField;
};


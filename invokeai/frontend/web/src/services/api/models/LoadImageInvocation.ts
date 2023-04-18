/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageType } from './ImageType';

/**
 * Load an image and provide it as output.
 */
export type LoadImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'load_image';
  /**
   * The type of the image
   */
  image_type: ImageType;
  /**
   * The name of the image
   */
  image_name: string;
};


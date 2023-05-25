/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

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
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to load
   */
  image?: ImageField;
};


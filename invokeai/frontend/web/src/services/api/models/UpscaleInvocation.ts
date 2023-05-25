/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Upscales an image.
 */
export type UpscaleInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'upscale';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The input image
   */
  image?: ImageField;
  /**
   * The strength
   */
  strength?: number;
  /**
   * The upscale level
   */
  level?: 2 | 4;
};


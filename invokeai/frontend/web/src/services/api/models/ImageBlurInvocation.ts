/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Blurs an image
 */
export type ImageBlurInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_blur';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to blur
   */
  image?: ImageField;
  /**
   * The blur radius
   */
  radius?: number;
  /**
   * The type of blur
   */
  blur_type?: 'gaussian' | 'box';
};


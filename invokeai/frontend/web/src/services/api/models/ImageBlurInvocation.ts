/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Blurs an image
 */
export type ImageBlurInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_blur';
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


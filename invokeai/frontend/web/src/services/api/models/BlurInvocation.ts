/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Blurs an image
 */
export type BlurInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'blur';
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


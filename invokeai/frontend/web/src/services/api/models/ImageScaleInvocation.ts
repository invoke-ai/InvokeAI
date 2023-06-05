/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Scales an image by a factor
 */
export type ImageScaleInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_scale';
  /**
   * The image to scale
   */
  image?: ImageField;
  /**
   * The factor by which to scale the image
   */
  scale_factor: number;
  /**
   * The resampling mode
   */
  resample_mode?: 'nearest' | 'box' | 'bilinear' | 'hamming' | 'bicubic' | 'lanczos';
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Resizes an image to specific dimensions
 */
export type ImageResizeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_resize';
  /**
   * The image to resize
   */
  image?: ImageField;
  /**
   * The width to resize to (px)
   */
  width: number;
  /**
   * The height to resize to (px)
   */
  height: number;
  /**
   * The resampling mode
   */
  resample_mode?: 'nearest' | 'box' | 'bilinear' | 'hamming' | 'bicubic' | 'lanczos';
};


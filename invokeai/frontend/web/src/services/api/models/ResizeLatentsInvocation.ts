/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8.
 */
export type ResizeLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'lresize';
  /**
   * The latents to resize
   */
  latents?: LatentsField;
  /**
   * The width to resize to (px)
   */
  width: number;
  /**
   * The height to resize to (px)
   */
  height: number;
  /**
   * The interpolation mode
   */
  mode?: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact';
  /**
   * Whether or not to antialias (applied in bilinear and bicubic modes only)
   */
  antialias?: boolean;
};


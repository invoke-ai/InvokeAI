/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Scales latents by a given factor.
 */
export type ScaleLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'lscale';
  /**
   * The latents to scale
   */
  latents?: LatentsField;
  /**
   * The factor by which to scale the latents
   */
  scale_factor: number;
  /**
   * The interpolation mode
   */
  mode?: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact';
  /**
   * Whether or not to antialias (applied in bilinear and bicubic modes only)
   */
  antialias?: boolean;
};


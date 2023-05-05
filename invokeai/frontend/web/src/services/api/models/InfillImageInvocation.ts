/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ColorField } from './ColorField';
import type { ImageField } from './ImageField';

/**
 * Infills transparent areas of an image
 */
export type InfillImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'infill';
  /**
   * The image to infill
   */
  image?: ImageField;
  /**
   * The method used to infill empty regions (px)
   */
  infill_method?: 'patchmatch' | 'tile' | 'solid';
  /**
   * The solid infill method color
   */
  inpaint_fill?: ColorField;
  /**
   * The tile infill method size (px)
   */
  tile_size?: number;
  /**
   * The seed to use (-1 for a random seed)
   */
  seed?: number;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Infills transparent areas of an image with tiles of the image
 */
export type InfillTileInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'infill_tile';
  /**
   * The image to infill
   */
  image?: ImageField;
  /**
   * The tile size (px)
   */
  tile_size?: number;
  /**
   * The seed to use for tile generation (omit for random)
   */
  seed?: number;
};


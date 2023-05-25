/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Infills transparent areas of an image with tiles of the image
 */
export type InfillTileInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'infill_tile';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
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


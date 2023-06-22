/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ColorField } from './ColorField';
import type { ImageField } from './ImageField';

/**
 * Infills transparent areas of an image with a solid color
 */
export type InfillColorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'infill_rgba';
  /**
   * The image to infill
   */
  image?: ImageField;
  /**
   * The color to use to infill
   */
  color?: ColorField;
};


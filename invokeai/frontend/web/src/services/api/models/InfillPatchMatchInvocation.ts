/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Infills transparent areas of an image using the PatchMatch algorithm
 */
export type InfillPatchMatchInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'infill_patchmatch';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to infill
   */
  image?: ImageField;
};


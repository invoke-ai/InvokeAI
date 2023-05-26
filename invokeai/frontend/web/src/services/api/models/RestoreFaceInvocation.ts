/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Restores faces in an image.
 */
export type RestoreFaceInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'restore_face';
  /**
   * The input image
   */
  image?: ImageField;
  /**
   * The strength of the restoration
   */
  strength?: number;
};


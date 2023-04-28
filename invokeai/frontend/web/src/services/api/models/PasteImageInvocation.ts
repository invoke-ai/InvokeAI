/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Pastes an image into another image.
 */
export type PasteImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'paste';
  /**
   * The base image
   */
  base_image?: ImageField;
  /**
   * The image to paste
   */
  image?: ImageField;
  /**
   * The mask to use when pasting
   */
  mask?: ImageField;
  /**
   * The left x coordinate at which to paste the image
   */
  'x'?: number;
  /**
   * The top y coordinate at which to paste the image
   */
  'y'?: number;
};


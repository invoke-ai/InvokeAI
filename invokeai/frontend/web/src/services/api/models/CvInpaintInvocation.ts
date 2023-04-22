/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Simple inpaint using opencv.
 */
export type CvInpaintInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'cv_inpaint';
  /**
   * The image to inpaint
   */
  image?: ImageField;
  /**
   * The mask to use when inpainting
   */
  mask?: ImageField;
};


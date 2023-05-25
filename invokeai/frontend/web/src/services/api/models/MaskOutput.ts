/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Base class for invocations that output a mask
 */
export type MaskOutput = {
  type: 'mask';
  /**
   * The output mask
   */
  mask: ImageField;
  /**
   * The width of the mask in pixels
   */
  width?: number;
  /**
   * The height of the mask in pixels
   */
  height?: number;
};


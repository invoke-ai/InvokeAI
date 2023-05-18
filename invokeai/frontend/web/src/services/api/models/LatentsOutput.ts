/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Base class for invocations that output latents
 */
export type LatentsOutput = {
  type?: 'latents_output';
  /**
   * The output latents
   */
  latents?: LatentsField;
  /**
   * The width of the latents in pixels
   */
  width: number;
  /**
   * The height of the latents in pixels
   */
  height: number;
};


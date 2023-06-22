/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Invocation noise output
 */
export type NoiseOutput = {
  type?: 'noise_output';
  /**
   * The output noise
   */
  noise?: LatentsField;
  /**
   * The width of the noise in pixels
   */
  width: number;
  /**
   * The height of the noise in pixels
   */
  height: number;
};


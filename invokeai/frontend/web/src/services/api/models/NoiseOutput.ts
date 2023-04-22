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
};


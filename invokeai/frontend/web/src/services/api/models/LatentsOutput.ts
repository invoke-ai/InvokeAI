/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Base class for invocations that output latents
 */
export type LatentsOutput = {
  type?: 'latent_output';
  /**
   * The output latents
   */
  latents?: LatentsField;
};


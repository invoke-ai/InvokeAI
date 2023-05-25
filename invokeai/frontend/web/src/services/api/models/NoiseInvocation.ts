/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Generates latent noise.
 */
export type NoiseInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'noise';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The seed to use
   */
  seed?: number;
  /**
   * The width of the resulting noise
   */
  width?: number;
  /**
   * The height of the resulting noise
   */
  height?: number;
};


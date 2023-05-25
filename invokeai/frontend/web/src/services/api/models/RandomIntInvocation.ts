/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Outputs a single random integer.
 */
export type RandomIntInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'rand_int';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The inclusive low value
   */
  low?: number;
  /**
   * The exclusive high value
   */
  high?: number;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Creates a range of numbers from start to stop with step
 */
export type RangeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'range';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The start of the range
   */
  start?: number;
  /**
   * The stop of the range
   */
  stop?: number;
  /**
   * The step of the range
   */
  step?: number;
};


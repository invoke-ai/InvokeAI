/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Iterates over a list of items
 */
export type IterateInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'iterate';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The list of items to iterate over
   */
  collection?: Array<any>;
  /**
   * The index, will be provided on executed iterators
   */
  index?: number;
};


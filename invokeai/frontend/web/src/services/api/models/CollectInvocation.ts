/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Collects values into a collection
 */
export type CollectInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'collect';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The item to collect (all inputs must be of the same type)
   */
  item?: any;
  /**
   * The collection, will be provided on execution
   */
  collection?: Array<any>;
};


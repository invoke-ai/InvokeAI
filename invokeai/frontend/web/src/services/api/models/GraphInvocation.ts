/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { Graph } from './Graph';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Execute a graph
 */
export type GraphInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'graph';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The graph to run
   */
  graph?: Graph;
};


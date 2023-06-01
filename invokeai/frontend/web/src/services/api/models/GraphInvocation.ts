/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { Graph } from './Graph';

/**
 * Execute a graph
 */
export type GraphInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'graph';
  /**
   * The graph to run
   */
  graph?: Graph;
};


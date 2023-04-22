/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { Graph } from './Graph';

/**
 * A node to process inputs and produce outputs.
 * May use dependency injection in __init__ to receive providers.
 */
export type GraphInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'graph';
  /**
   * The graph to run
   */
  graph?: Graph;
};


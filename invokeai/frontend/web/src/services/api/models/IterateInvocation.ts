/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * A node to process inputs and produce outputs.
 * May use dependency injection in __init__ to receive providers.
 */
export type IterateInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'iterate';
  /**
   * The list of items to iterate over
   */
  collection?: Array<any>;
  /**
   * The index, will be provided on executed iterators
   */
  index?: number;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Iterates over a list of items
 */
export type IterateInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
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


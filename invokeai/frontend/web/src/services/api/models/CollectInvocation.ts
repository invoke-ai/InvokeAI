/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

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
   * The item to collect (all inputs must be of the same type)
   */
  item?: any;
  /**
   * The collection, will be provided on execution
   */
  collection?: Array<any>;
};


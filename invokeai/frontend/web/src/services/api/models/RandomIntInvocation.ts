/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Outputs a single random integer.
 */
export type RandomIntInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'rand_int';
  /**
   * The inclusive low value
   */
  low?: number;
  /**
   * The exclusive high value
   */
  high?: number;
};


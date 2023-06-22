/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Adds two numbers
 */
export type AddInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'add';
  /**
   * The first number
   */
  'a'?: number;
  /**
   * The second number
   */
  'b'?: number;
};


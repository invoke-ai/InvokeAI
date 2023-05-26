/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Multiplies two numbers
 */
export type MultiplyInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'mul';
  /**
   * The first number
   */
  'a'?: number;
  /**
   * The second number
   */
  'b'?: number;
};


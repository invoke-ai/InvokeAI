/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Divides two numbers
 */
export type DivideInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'div';
  /**
   * The first number
   */
  'a'?: number;
  /**
   * The second number
   */
  'b'?: number;
};


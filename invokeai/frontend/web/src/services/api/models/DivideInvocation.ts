/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

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
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The first number
   */
  'a'?: number;
  /**
   * The second number
   */
  'b'?: number;
};


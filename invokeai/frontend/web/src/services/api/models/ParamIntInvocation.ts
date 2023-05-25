/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * An integer parameter
 */
export type ParamIntInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'param_int';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The integer value
   */
  'a'?: number;
};


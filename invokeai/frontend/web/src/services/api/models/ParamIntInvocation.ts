/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * An integer parameter
 */
export type ParamIntInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'param_int';
  /**
   * The integer value
   */
  'a'?: number;
};


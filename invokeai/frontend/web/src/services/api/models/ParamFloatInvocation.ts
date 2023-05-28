/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * A float parameter
 */
export type ParamFloatInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'param_float';
  /**
   * The float value
   */
  param?: number;
};


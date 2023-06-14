/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Creates a range
 */
export type FloatLinearRangeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'float_range';
  /**
   * The first value of the range
   */
  start?: number;
  /**
   * The last value of the range
   */
  stop?: number;
  /**
   * number of values to interpolate over (including start and stop)
   */
  steps?: number;
};


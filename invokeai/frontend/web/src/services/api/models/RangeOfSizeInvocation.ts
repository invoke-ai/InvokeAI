/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Creates a range from start to start + size with step
 */
export type RangeOfSizeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'range_of_size';
  /**
   * The start of the range
   */
  start?: number;
  /**
   * The number of values
   */
  size?: number;
  /**
   * The step of the range
   */
  step?: number;
};


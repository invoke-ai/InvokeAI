/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Creates a collection of random numbers
 */
export type RandomRangeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'random_range';
  /**
   * The inclusive low value
   */
  low?: number;
  /**
   * The exclusive high value
   */
  high?: number;
  /**
   * The number of values to generate
   */
  size?: number;
  /**
   * The seed for the RNG (omit for random)
   */
  seed?: number;
};


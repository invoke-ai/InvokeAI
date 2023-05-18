/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Creates a range
 */
export type RangeInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'range';
  /**
   * The start of the range
   */
  start?: number;
  /**
   * The stop of the range
   */
  stop?: number;
  /**
   * The step of the range
   */
  step?: number;
};


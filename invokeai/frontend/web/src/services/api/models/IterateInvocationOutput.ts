/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Used to connect iteration outputs. Will be expanded to a specific output.
 */
export type IterateInvocationOutput = {
  type: 'iterate_output';
  /**
   * The item being iterated over
   */
  item: any;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Base class for all invocation outputs
 */
export type CollectInvocationOutput = {
  type: 'collect_output';
  /**
   * The collection of input items
   */
  collection: Array<any>;
};


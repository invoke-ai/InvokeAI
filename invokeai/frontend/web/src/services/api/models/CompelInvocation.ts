/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Parse prompt using compel package to conditioning.
 */
export type CompelInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'compel';
  /**
   * Prompt
   */
  prompt?: string;
  /**
   * Model to use
   */
  model?: string;
};


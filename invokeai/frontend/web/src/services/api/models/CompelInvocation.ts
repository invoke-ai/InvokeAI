/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { InvocationMeta } from './InvocationMeta';

/**
 * Parse prompt using compel package to conditioning.
 */
export type CompelInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'compel';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * Prompt
   */
  prompt?: string;
  /**
   * Model to use
   */
  model?: string;
};

